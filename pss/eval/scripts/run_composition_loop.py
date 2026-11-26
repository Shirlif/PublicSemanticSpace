"""Minimal composition loop: typed atoms, best-first search, plan cache.

Three domains:

- ``laundry``: fact-state tasks built from the reference 12-step plan.  Multi
  step goals require composing conditional actions under hand capacity 1.
- ``spatial``: a fixed set of candidate placements around a reference object.
  The goal is a directional relation *in a named frame*, so the agent must
  compose the frame atom with a placement and a check.  For non-world goals the
  task is filtered so that the first placement satisfying the world frame does
  not satisfy the goal frame; a frame-blind composer therefore fails.
- ``selection``: pick the red cup that fits the size interval and is not
  occupied.  Each of the four decoys survives all filters but one, so the
  select/project/constraint chain is necessary.

Strategies compared:

- ``direct``: no composition (one laundry action / the first candidate spot /
  the first candidate).
- ``typed``: best-first over type-compatible atoms.
- ``typed_cache``: typed search plus verified plan reuse (cold cache).
- ``typed_warm``: a second pass over the same tasks with the populated cache.
- ``frame_blind``: fixed pipeline that always uses the world frame.  This is
  the stand-in for a model that proposes compositions without type checks; no
  LLM is called in this harness.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Hashable

import numpy as np

from pss._console import ensure_utf8
from pss.eval.spatial import predicates as P
from pss.eval.spatial.generator import generate_suite
from pss.reasoning.composition import (
    Candidate,
    ConstraintAtom,
    PickAtom,
    PlanCache,
    ProjectAtom,
    SelectAtom,
    Signature,
    best_first,
)
from pss.world_model.actions import hand_written_plan, initial_laundry_facts
from pss.world_model.conditional import ResourcePool, laundry_conditional_library

DIRECTIONAL = ("left_of", "right_of", "front_of", "behind")
HAND_CAPACITY = {"hand": 1}
SPOT_RADIUS = 0.35
SPOT_OFFSETS = ((1, 1), (1, -1), (-1, 1), (-1, -1))

CONTRADICTIONS = (
    (("door", "open"), ("door", "closed")),
    (("machine", "running"), ("machine", "stopped")),
    (("water", "in", "washer"), ("water", "drained")),
    (("clothes", "in", "basket"), ("clothes", "in", "washer")),
    (("clothes", "held"), ("clothes", "in", "washer")),
    (("clothes", "held"), ("clothes", "on", "drying_rack")),
)


# ── atoms ───────────────────────────────────────────────────────────────────


class LaundryAtom:
    def __init__(self, operator, library):
        self.op = operator
        self.library = library
        self.name = operator.name
        self.signature = Signature("state", "state", cost=float(operator.cost))

    @staticmethod
    def _pool(snapshot) -> ResourcePool:
        pool = ResourcePool(HAND_CAPACITY)
        if snapshot:
            pool.restore(snapshot)
        return pool

    def applicable(self, payload: dict) -> bool:
        pool = self._pool(payload.get("pool"))
        return self.op.applicable(payload["facts"]) and self.op.resource_ok(pool)

    def apply(self, payload: dict) -> dict:
        pool = self._pool(payload.get("pool"))
        facts = self.library.apply(self.op.name, payload["facts"], pool)
        out = dict(payload)
        out["type"] = "state"
        out["facts"] = facts
        out["pool"] = pool.snapshot()
        return out


class SetFrameAtom:
    def __init__(self, frame: str):
        self.frame = frame
        self.name = f"frame:{frame}"
        self.signature = Signature("goal", "framed_goal", frame=frame, cost=0.1)

    def applicable(self, payload: dict) -> bool:
        return payload.get("type") == "goal"

    def apply(self, payload: dict) -> dict:
        out = dict(payload)
        out["type"] = "framed_goal"
        out["frame"] = self.frame
        return out


class PlaceAtom:
    """Move the subject to one of the fixed candidate placements."""

    def __init__(self, subject: str, spot_id: str, position):
        self.subject = subject
        self.spot_id = spot_id
        self.position = np.asarray(position, dtype=float)
        self.name = f"place:{spot_id}"
        self.signature = Signature("framed_goal", "state", cost=1.0)

    def applicable(self, payload: dict) -> bool:
        return payload.get("frame") is not None

    def apply(self, payload: dict) -> dict:
        positions = dict(payload["positions"])
        positions[self.subject] = self.position.copy()
        out = dict(payload)
        out["type"] = "state"
        out["spot"] = self.spot_id
        out["positions"] = positions
        return out


class CheckAtom:
    def __init__(self, verify: Callable[[dict], bool]):
        self.verify = verify
        self.name = "check"
        self.signature = Signature("state", "verified", cost=0.1)

    def applicable(self, payload: dict) -> bool:
        return payload.get("type") == "state"

    def apply(self, payload: dict) -> dict:
        ok = bool(self.verify(payload))
        out = dict(payload)
        out["type"] = "verified"
        out["verified"] = ok
        out["residual"] = 0.0 if ok else 1.0
        return out


# ── tasks ───────────────────────────────────────────────────────────────────


@dataclass
class Task:
    task_id: str
    domain: str
    initial: dict
    goal: Any
    atoms: list
    atom_map: dict
    goal_test: Callable[[dict], bool]
    goal_key: Hashable
    subject: str | None = None
    verify: Callable[[dict], bool] | None = None
    verify_world: Callable[[dict], bool] | None = None
    meta: dict = field(default_factory=dict)


def _scene_with_positions(scene, positions):
    objects = tuple(scene.obj(obj.id).moved(positions[obj.id]) for obj in scene.objects)
    return scene.with_objects(objects)


def _spatial_key(payload: dict):
    positions = tuple(
        sorted(
            (name, tuple(np.round(np.asarray(pos, dtype=float), 6)))
            for name, pos in payload["positions"].items()
        )
    )
    return (
        payload.get("type"),
        payload.get("frame"),
        payload.get("verified"),
        payload.get("spot"),
        positions,
    )


def _laundry_key(payload: dict):
    return (
        payload.get("type"),
        frozenset(payload.get("facts", ())),
        tuple(sorted(payload.get("pool", {}).items())),
    )


def _selection_key(payload: dict):
    ids = tuple(candidate.id for candidate in payload.get("candidates", ()))
    return (payload.get("type"), ids, payload.get("selected"))


def _laundry_verified(payload: dict, goal) -> bool:
    facts = payload.get("facts", frozenset())
    if goal not in facts:
        return False
    return not any(a in facts and b in facts for a, b in CONTRADICTIONS)


def _laundry_tasks(n: int = 25) -> list[Task]:
    library = laundry_conditional_library()
    facts = frozenset(initial_laundry_facts())
    states = [facts]
    for name in hand_written_plan():
        name = "wait_done" if name == "wait" else name
        if name not in library.operators:
            continue
        op = library.get(name)
        if op.applicable(facts):
            facts = library.apply(name, facts)
            states.append(facts)

    pairs = []
    for start in range(len(states) - 1):
        for depth in (1, 2, 3, 4, 6):
            end = start + depth
            if end >= len(states):
                continue
            new = states[end] - states[start]
            if new:
                pairs.append((start, end, sorted(new)[0]))
    if not pairs:
        return []
    step = max(1, len(pairs) // n)
    chosen = pairs[::step][:n]

    atoms = [LaundryAtom(op, library) for op in library.operators.values()]
    atom_map = {atom.name: atom for atom in atoms}
    tasks = []
    for k, (start, end, goal) in enumerate(chosen):
        initial = {"type": "state", "facts": states[start], "pool": {}}

        def goal_test(payload, goal=goal):
            return goal in payload.get("facts", frozenset())

        tasks.append(
            Task(
                task_id=f"laundry_{k:03d}",
                domain="laundry",
                initial=initial,
                goal=goal,
                atoms=atoms,
                atom_map=atom_map,
                goal_test=goal_test,
                goal_key=("laundry", goal),
                verify=lambda payload, goal=goal: _laundry_verified(payload, goal),
                meta={"start": start, "end": end, "depth": end - start},
            )
        )
    return tasks


def _selection_tasks(n: int = 6) -> list[Task]:
    """Pick the red cup that fits the size interval and is not occupied.

    Each task has four decoys, each surviving all filters but one, so every
    filter is necessary.  The target is never first, so a direct pick fails.
    """
    tasks = []
    for k in range(n):
        target = f"t{k}"
        def attributes(category="cup", color="red", size=0.10, occupied=False):
            return {
                "category": category,
                "color": color,
                "size": size,
                "occupied": occupied,
            }

        candidates = [
            Candidate(target, attributes()),
            Candidate(f"d{k}_blue", attributes(color="blue")),
            Candidate(f"d{k}_bowl", attributes(category="bowl")),
            Candidate(f"d{k}_big", attributes(size=0.30)),
            Candidate(f"d{k}_used", attributes(occupied=True)),
        ]
        rotate = (k + 1) % len(candidates)
        candidates = candidates[rotate:] + candidates[:rotate]

        atoms = [
            SelectAtom("select:cup", "category", "cup"),
            SelectAtom("select:red", "color", "red"),
            ProjectAtom(
                "project:size",
                lambda candidate: 0.08 <= float(candidate.get("size", 0.0)) <= 0.12,
            ),
            ConstraintAtom(
                "constraint:free",
                lambda candidate: not bool(candidate.get("occupied")),
            ),
            PickAtom("pick", require_unique=True),
        ]
        atom_map = {atom.name: atom for atom in atoms}

        def goal_test(payload, target=target):
            return payload.get("type") == "selected" and payload.get("selected") == target

        tasks.append(
            Task(
                task_id=f"selection_{k:03d}",
                domain="selection",
                initial={"type": "candidates", "candidates": candidates},
                goal=target,
                atoms=atoms,
                atom_map=atom_map,
                goal_test=goal_test,
                goal_key=("selection", "red"),
                verify=goal_test,
                meta={"target": target},
            )
        )
    return tasks


def _relation_at(scene, subject_id, reference_id, spot, predicate, frame) -> bool:
    objects = tuple(obj.moved(spot) if obj.id == subject_id else obj for obj in scene.objects)
    moved = scene.with_objects(objects)
    return P.relation(
        moved,
        moved.obj(subject_id),
        moved.obj(reference_id),
        predicate,
        frame,
    )


def _spatial_tasks(n: int = 25, seed: int = 0) -> list[Task]:
    items = generate_suite(
        n_per_family=max(1, (n + 1) // 2),
        seed=seed,
        split="test",
        ood="id",
        families=("frame", "perspective"),
    )[:n]
    tasks = []
    for item in items:
        meta = item.query.meta
        subject_id = meta["subject"]
        reference_id = meta["object"]
        goal_frame = str(meta["frame"])
        scene = item.scene
        subject = scene.obj(subject_id)
        reference = scene.obj(reference_id)

        spots = {
            f"s{i}": np.array(
                [
                    float(reference.pos[0]) + dx * SPOT_RADIUS,
                    float(reference.pos[1]) + dy * SPOT_RADIUS,
                    float(subject.pos[2]),
                ]
            )
            for i, (dx, dy) in enumerate(SPOT_OFFSETS)
        }

        candidates = [str(meta["predicate"])]
        candidates += [pred for pred in DIRECTIONAL if pred != candidates[0]]
        chosen = None
        for predicate in candidates:
            goal_ok = [
                sid
                for sid, spot in spots.items()
                if _relation_at(scene, subject_id, reference_id, spot, predicate, goal_frame)
            ]
            if not goal_ok:
                continue
            if goal_frame != "world":
                world_ok = [
                    sid
                    for sid, spot in spots.items()
                    if _relation_at(scene, subject_id, reference_id, spot, predicate, "world")
                ]
                # The frame-blind baseline picks the first world-satisfying
                # spot; skip the item if that spot also satisfies the goal.
                if not world_ok or world_ok[0] in goal_ok:
                    continue
            chosen = predicate
            break
        if chosen is None:
            continue

        positions = {obj.id: np.asarray(obj.pos, dtype=float) for obj in scene.objects}

        def verify(payload, scene=scene, subject=subject_id, reference=reference_id):
            moved = _scene_with_positions(scene, payload["positions"])
            return P.relation(
                moved,
                moved.obj(subject),
                moved.obj(reference),
                str(payload["goal_predicate"]),
                str(payload["frame"]),
            )

        def verify_goal(payload, scene=scene, subject=subject_id, reference=reference_id):
            moved = _scene_with_positions(scene, payload["positions"])
            return P.relation(
                moved,
                moved.obj(subject),
                moved.obj(reference),
                str(payload["goal_predicate"]),
                str(payload["goal_frame"]),
            )

        def verify_world(payload, scene=scene, subject=subject_id, reference=reference_id):
            moved = _scene_with_positions(scene, payload["positions"])
            return P.relation(
                moved,
                moved.obj(subject),
                moved.obj(reference),
                str(payload["goal_predicate"]),
                "world",
            )

        frames = sorted({"world", "viewer", "viewer:cam1", goal_frame})
        atoms = [SetFrameAtom(frame) for frame in frames]
        atoms += [PlaceAtom(subject_id, sid, pos) for sid, pos in spots.items()]
        atoms.append(CheckAtom(verify))
        atom_map = {atom.name: atom for atom in atoms}

        initial = {
            "type": "goal",
            "scene": scene,
            "subject": subject_id,
            "reference": reference_id,
            "goal_predicate": chosen,
            "goal_frame": goal_frame,
            "positions": positions,
            "frame": None,
            "verified": False,
            "spot": None,
        }

        def goal_test(payload):
            return (
                payload.get("type") == "verified"
                and bool(payload.get("verified"))
                and payload.get("frame") == payload.get("goal_frame")
            )

        tasks.append(
            Task(
                task_id=f"spatial_{len(tasks):03d}",
                domain="spatial",
                initial=initial,
                goal=(subject_id, chosen, reference_id, goal_frame),
                atoms=atoms,
                atom_map=atom_map,
                goal_test=goal_test,
                goal_key=("spatial", chosen, goal_frame),
                subject=subject_id,
                verify=lambda payload, verify_goal=verify_goal: verify_goal(payload),
                verify_world=lambda payload, verify_world=verify_world: verify_world(payload),
                meta={"predicate": chosen, "frame": goal_frame},
            )
        )
    return tasks


# ── strategies ──────────────────────────────────────────────────────────────


def _run_direct(task: Task) -> dict:
    """No composition: one laundry action, or the first candidate spot."""
    started = time.perf_counter()
    if task.domain == "laundry":
        goal = task.goal
        for atom in task.atoms:
            adds = any(goal in effect.add for effect in atom.op.effects)
            if adds and atom.applicable(task.initial):
                payload = atom.apply(dict(task.initial))
                ok = bool(task.verify(payload)) if task.verify else False
                return _row(task, 1, 0, ok, True, started, payload)
        return _row(task, 0, 0, False, False, started, None)

    if task.domain == "selection":
        payload = dict(task.initial)
        pick = next(atom for atom in task.atoms if atom.name == "pick")
        payload = pick.apply(payload)
        ok = bool(task.verify(payload)) if task.verify else False
        return _row(task, 1, 0, ok, True, started, payload)

    payload = dict(task.initial)
    payload["type"] = "framed_goal"
    payload["frame"] = "world"
    atom = next(atom for atom in task.atoms if atom.name == "place:s0")
    payload = atom.apply(payload)
    ok = bool(task.verify(payload)) if task.verify else False
    return _row(task, 2, 0, ok, True, started, payload)


def _run_frame_blind(task: Task) -> dict:
    """Fixed composition that always uses the world frame (untyped stand-in)."""
    if task.domain in {"laundry", "selection"}:
        return _run_direct(task)
    started = time.perf_counter()
    payload = dict(task.initial)
    payload["type"] = "framed_goal"
    payload["frame"] = "world"
    chosen = None
    for atom in task.atoms:
        if not atom.name.startswith("place:"):
            continue
        candidate = atom.apply(payload)
        if task.verify_world(candidate):
            chosen = candidate
            break
    if chosen is None:
        chosen = next(atom for atom in task.atoms if atom.name == "place:s0").apply(payload)
    ok = bool(task.verify(chosen)) if task.verify else False
    return _row(task, 2, 0, ok, True, started, chosen)


def _run_typed(task: Task, cache: PlanCache | None = None) -> dict:
    started = time.perf_counter()
    if cache is not None:
        hit = cache.get(
            task.goal_key,
            task.atom_map,
            task.initial,
            task.goal_test,
            subject=task.subject,
        )
        if hit is not None:
            ok = bool(task.verify(hit.payload)) if task.verify and hit.payload else hit.success
            cache.record(task.goal_key, hit.steps, verified=ok)
            return _row(
                task,
                len(hit.steps),
                hit.expansions,
                ok,
                True,
                started,
                hit.payload,
                cache_hit=True,
            )

    if task.domain == "laundry":
        state_key, max_steps, max_exp = _laundry_key, 12, 5000
    elif task.domain == "spatial":
        state_key, max_steps, max_exp = _spatial_key, 4, 400
    else:
        state_key, max_steps, max_exp = _selection_key, 6, 300
    result = best_first(
        task.initial,
        task.atoms,
        task.goal_test,
        max_expansions=max_exp,
        max_steps=max_steps,
        check_types=True,
        state_key=state_key,
    )
    ok = bool(task.verify(result.payload)) if task.verify and result.payload else result.success
    if cache is not None:
        cache.record(
            task.goal_key,
            result.steps if result.success else (),
            verified=bool(ok and result.success),
            subject=task.subject,
        )
    return _row(
        task,
        len(result.steps),
        result.expansions,
        ok and result.success,
        result.success,
        started,
        result.payload,
    )


def _row(task, steps, expansions, ok, claimed, started, payload, cache_hit=False) -> dict:
    return {
        "task_id": task.task_id,
        "domain": task.domain,
        "verified": bool(ok),
        "claimed": bool(claimed),
        "false_claim": bool(claimed and not ok),
        "frame_sensitive": task.domain == "spatial" and task.meta.get("frame") != "world",
        "steps": int(steps),
        "expansions": int(expansions),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "cache_hit": bool(cache_hit),
    }


# ── metrics ─────────────────────────────────────────────────────────────────


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    verified = sum(1 for row in rows if row["verified"])
    false_claims = sum(1 for row in rows if row["false_claim"])
    latencies = np.asarray([row["latency_ms"] for row in rows], dtype=float)
    return {
        "n": n,
        "success": verified,
        "success_rate": verified / n,
        "false_claims": false_claims,
        "false_claim_rate": false_claims / n,
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_expansions": float(np.mean([row["expansions"] for row in rows])),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }


def _by_domain(rows: list[dict]) -> dict:
    grouped: dict[str, list] = defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row)
    return {domain: _summary(group) for domain, group in grouped.items()}


def run(n_laundry: int = 25, n_spatial: int = 25, seed: int = 0) -> dict:
    tasks = _laundry_tasks(n_laundry) + _spatial_tasks(n_spatial, seed=seed) + _selection_tasks(6)
    strategies = {"direct": [], "frame_blind": [], "typed": [], "typed_cache": []}
    cache = PlanCache()

    for task in tasks:
        strategies["direct"].append(_run_direct(task))
        strategies["frame_blind"].append(_run_frame_blind(task))
        strategies["typed"].append(_run_typed(task, cache=None))
        strategies["typed_cache"].append(_run_typed(task, cache=cache))

    # Second pass over the same tasks with the cache from pass one.
    strategies["typed_warm"] = [_run_typed(task, cache=cache) for task in tasks]

    cache_payload = cache.stats.to_dict()
    cache_payload["entries"] = len(cache.entries)
    cache_payload["coverage"] = cache.coverage([task.goal_key for task in tasks])

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "seed": seed,
        "tasks": {
            "laundry": sum(1 for task in tasks if task.domain == "laundry"),
            "spatial": sum(1 for task in tasks if task.domain == "spatial"),
            "selection": sum(1 for task in tasks if task.domain == "selection"),
        },
        "strategies": {
            name: {"overall": _summary(rows), "by_domain": _by_domain(rows)}
            for name, rows in strategies.items()
        },
        "subsets": {
            name: {
                "spatial_non_world": _summary(
                    [row for row in rows if row.get("frame_sensitive")]
                )
            }
            for name, rows in strategies.items()
        },
        "cache": cache_payload,
    }


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _report(metrics: dict) -> str:
    lines = [
        "# Minimal composition loop",
        "",
        f"> Generated: {metrics['generated_at']}",
        f"> Tasks: {metrics['tasks']['laundry']} laundry + "
        f"{metrics['tasks']['spatial']} spatial + {metrics['tasks']['selection']} selection",
        "",
        "Strategies: `direct` (no composition), `frame_blind` (untyped fixed "
        "world-frame pipeline), ",
        "`typed` (best-first with signatures), `typed_cache` (typed + verified "
        "plan reuse), `typed_warm` (second pass with the populated cache).",
        "",
        "## Overall",
        "",
        "| strategy | success | false claims | mean steps | mean expansions | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in metrics["strategies"].items():
        row = payload["overall"]
        lines.append(
            f"| {name} | {_fmt(row['success_rate'])} | {_fmt(row['false_claim_rate'])} | "
            f"{_fmt(row['mean_steps'])} | {_fmt(row['mean_expansions'])} | "
            f"{_fmt(row['latency_p50_ms'])} | {_fmt(row['latency_p95_ms'])} |"
        )

    lines += [
        "",
        "## By domain",
        "",
        "| strategy | domain | n | success | false claims |",
        "|---|---|---:|---:|---:|",
    ]
    for name, payload in metrics["strategies"].items():
        for domain, row in payload["by_domain"].items():
            lines.append(
                f"| {name} | {domain} | {row['n']} | {_fmt(row['success_rate'])} | "
                f"{_fmt(row['false_claim_rate'])} |"
            )

    lines += [
        "",
        "## Frame-sensitive subset (spatial, non-world goals)",
        "",
        "These tasks were filtered so the first world-frame placement does not satisfy the goal.",
        "",
        "| strategy | n | success | false claims |",
        "|---|---:|---:|---:|",
    ]
    for name, payload in metrics["subsets"].items():
        row = payload["spatial_non_world"]
        lines.append(
            f"| {name} | {row.get('n', 0)} | {_fmt(row.get('success_rate'))} | "
            f"{_fmt(row.get('false_claim_rate'))} |"
        )

    cache = metrics["cache"]
    typed = metrics["strategies"]["typed"]["overall"]
    cached = metrics["strategies"]["typed_cache"]["overall"]
    warm = metrics["strategies"]["typed_warm"]["overall"]
    direct = metrics["strategies"]["direct"]["overall"]
    blind = metrics["strategies"]["frame_blind"]["overall"]
    reduction = None
    if typed["mean_expansions"]:
        reduction = 1.0 - cached["mean_expansions"] / typed["mean_expansions"]
    warm_reduction = None
    if typed["mean_expansions"]:
        warm_reduction = 1.0 - warm["mean_expansions"] / typed["mean_expansions"]
    lines += [
        "",
        "## Cache",
        "",
        f"- hits: {cache['hits']}, stale: {cache['stale']}, misses: {cache['misses']}, "
        f"stores: {cache['stores']}",
        f"- verified writes: {cache['verified']}, rejected attempts: {cache['rejected']}, "
        f"entries: {cache['entries']}, coverage: {_fmt(cache['coverage'])}",
        f"- mean expansions: typed {_fmt(typed['mean_expansions'])} -> "
        f"cache {_fmt(cached['mean_expansions'])} ({_fmt(reduction)}) -> "
        f"warm {_fmt(warm['mean_expansions'])} ({_fmt(warm_reduction)})",
        "",
        "## Decision",
        "",
        f"- `direct` succeeds {_fmt(direct['success_rate'])} with "
        f"{_fmt(direct['false_claim_rate'])} false claims.",
        f"- `frame_blind` succeeds {_fmt(blind['success_rate'])} with "
        f"{_fmt(blind['false_claim_rate'])} false claims.",
        f"- `typed` succeeds {_fmt(typed['success_rate'])} with "
        f"{_fmt(typed['false_claim_rate'])} false claims.",
        f"- The verified cache reaches coverage {_fmt(cache['coverage'])} after one pass and "
        f"cuts mean expansions from {_fmt(typed['mean_expansions'])} to "
        f"{_fmt(warm['mean_expansions'])} on the warm pass ({_fmt(warm_reduction)}), with "
        f"{cache['stale']} stale replays rejected.",
        "- Conclusion: the typed layer is viable.  The verifier must stay the goal test, "
        "and cache replay must re-check applicability and the goal.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(description="Minimal typed composition loop")
    parser.add_argument("--out", default="reports/claim_gaps/composition_loop")
    parser.add_argument("--n-laundry", type=int, default=25)
    parser.add_argument("--n-spatial", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    metrics = run(args.n_laundry, args.n_spatial, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    print(_report(metrics))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
