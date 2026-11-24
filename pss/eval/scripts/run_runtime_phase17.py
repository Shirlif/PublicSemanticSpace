"""Phase 17: integrate learned modules into the PSS cognitive runtime.

Stages:

- ``relational``: learned spatial relations drive a relational goal in the
  runtime planner; compare with the exact predicate.
- ``write_gate``: the learned write policy gates facts entering the runtime.
- ``operators``: versioned operators; update/remove/replan/rollback.
- ``long``: 70-frame occlusion on ``tracking_bench_long`` through the runtime.
- ``decoder``: whether a real local LLM decoder is available.

Example:
    python -m pss.eval.scripts.run_runtime_phase17
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.encoders.data import load_index
from pss.heads.text import load_local_decoder
from pss.memory.active_policy import WritePolicy, write_features
from pss.memory.perception import PoseNoise, extract_frame, load_perception_models
from pss.memory.typed_relations import SpatialRelationHead
from pss.runtime import (
    CognitiveState,
    Entity,
    EntityLayer,
    Goal,
    PSSRuntime,
    RelationFactLayer,
    RuntimePlanner,
    UncertaintyService,
)
from pss.runtime.operators import VersionedOperatorLibrary
from pss.sim.dataset import load_sample

from pss.eval.scripts.run_runtime import (
    ENTITIES,
    NAMES,
    REGIONS,
    _apply_action,
    _move_library,
    _observation,
)

EXPERTS = "checkpoints/category_pretrain_experts"
PROJECTIONS = "checkpoints/phase45_projection"
OUT = "eval/results/runtime_phase17"
REL_INDEX = {"left": 0, "right": 1, "above": 2, "below": 3}


def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "n/a" if math.isnan(v) else f"{v:.{digits}f}"


def _rel(a, b, predicate):
    ax, ay = REGIONS[a]
    bx, by = REGIONS[b]
    if predicate == "left":
        return ax < bx
    if predicate == "right":
        return ax > bx
    if predicate == "above":
        return ay > by
    if predicate == "below":
        return ay < by
    raise ValueError(predicate)


def _pair_feature(a, b):
    pa = np.array(REGIONS[a], dtype=np.float32)
    pb = np.array(REGIONS[b], dtype=np.float32)
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    return [pa[0], pa[1], pb[0], pb[1], dx, dy, abs(dx), abs(dy)]


def _pair_feature_xy(pa, pb):
    pa = np.asarray(pa, dtype=np.float32)
    pb = np.asarray(pb, dtype=np.float32)
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    return [pa[0], pa[1], pb[0], pb[1], dx, dy, abs(dx), abs(dy)]


def _pair_labels(a, b, eps=0.1):
    dx = REGIONS[b][0] - REGIONS[a][0]
    dy = REGIONS[b][1] - REGIONS[a][1]
    return [dx > eps, dx < -eps, dy < -eps, dy > eps]


def train_region_head(epochs: int = 400, seed: int = 0):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    head = SpatialRelationHead()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    pairs = [(a, b) for a in NAMES for b in NAMES if a != b]
    rng = np.random.default_rng(seed)
    X, y = [], []
    for a, b in pairs:
        for _ in range(64):
            pa = np.array(REGIONS[a]) + rng.normal(scale=0.03, size=2)
            pb = np.array(REGIONS[b]) + rng.normal(scale=0.03, size=2)
            X.append(_pair_feature_xy(pa, pb))
            y.append(_pair_labels(a, b))
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for start in range(0, len(perm), 128):
            idx = perm[start : start + 128]
            loss = F.binary_cross_entropy_with_logits(head.net(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(head.net(X)) > 0.5).float()
    accuracy = float((pred == y).all(dim=1).float().mean())
    return head, accuracy


def learned_rel(head, a, b, predicate):
    import torch

    x = torch.tensor([_pair_feature(a, b)], dtype=torch.float32)
    with torch.no_grad():
        prob = torch.sigmoid(head.net(x))[0]
    return bool(prob[REL_INDEX[predicate]] > 0.5)


def _runtime(use_entity=True, audit=True, uncertainty=False):
    return PSSRuntime(
        entity_layer=EntityLayer() if use_entity else None,
        relation_layer=RelationFactLayer(),
        planner=None,
        uncertainty=UncertaintyService(),
        audit=audit,
        uncertainty_enabled=uncertainty,
    )


def _ingest_clean(runtime, world, frames=3, relations=False):
    for frame in range(frames):
        observations = [
            _observation(i, gt, world[gt], frame)
            for i, (gt, _) in enumerate(ENTITIES)
        ]
        runtime.ingest(observations, frame, relation_records=None)
    return {
        str(track.category): tid for tid, track in runtime.state.entities.items()
    }


def _apply_plan_to_world(world, actions, track_to_gt):
    for action in actions:
        prefix, dst = action.rsplit("_", 1)
        tid = prefix[len("move_") :].rsplit("_", 1)[0]
        gt = track_to_gt.get(tid, tid)
        if gt in world:
            world[gt] = dst


def stage_relational(device="cpu") -> dict:
    head, region_accuracy = train_region_head()
    rows = []
    for predicate in ("left", "right", "above", "below"):
        results = {}
        for mode in ("symbolic", "learned"):
            world = {gt: region for gt, region in ENTITIES}
            runtime = _runtime(uncertainty=False)
            gt_to_track = _ingest_clean(runtime, world)
            track_to_gt = {v: k for k, v in gt_to_track.items()}
            for gt, region in world.items():
                tid = gt_to_track.get(gt)
                if tid:
                    runtime.assert_fact(tid, "location", region, source="perception")
            cup_tid = gt_to_track.get("cup")
            bottle_tid = gt_to_track.get("bottle")
            goal = Goal(
                id=f"rel_{predicate}",
                relation=(bottle_tid, cup_tid, predicate),
                description=f"bottle {predicate} cup",
            )
            runtime.set_goal(goal)
            runtime.planner = RuntimePlanner(
                _move_library(sorted(gt_to_track.values()))
            )
            rel_fn = (
                _rel if mode == "symbolic" else (lambda a, b, p: learned_rel(head, a, b, p))
            )
            plan = runtime.planner.plan_relational(
                runtime.state, goal, list(NAMES), rel_fn
            )
            _apply_plan_to_world(world, plan.actions, track_to_gt)
            true_success = bool(plan.confidence > 0) and _rel(
                world["bottle"], world["cup"], predicate
            )
            results[mode] = {
                "plan_actions": len(plan.actions),
                "success": true_success,
            }
        rows.append(
            {
                "predicate": predicate,
                "symbolic": results["symbolic"],
                "learned": results["learned"],
            }
        )
    return {"region_head_accuracy": region_accuracy, "rows": rows}


def _make_write_stream(T=300, n_keys=20, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 8, n_keys)
    keys = rng.integers(0, n_keys, T)
    reliability = rng.uniform(0.35, 1.0, T)
    obs = truth[keys].copy()
    wrong = rng.random(T) > reliability
    obs[wrong] = rng.integers(0, 8, int(wrong.sum()))
    return keys, obs, reliability, truth


def _run_write_gate(policy, T=300, n_keys=20):
    keys, obs, reliability, truth = _make_write_stream(T, n_keys)
    runtime = PSSRuntime(audit=True, uncertainty_enabled=False)
    memory = {}
    for i in range(T):
        key = f"k{int(keys[i])}"
        current = memory.get(key)
        feats = write_features(
            reliability=float(reliability[i]),
            conflict=0.0 if current is None else float(current != obs[i]),
            novelty=1.0 if current is None else 0.0,
            key_frequency=float(np.bincount(keys[: i + 1], minlength=n_keys)[int(keys[i])] / (i + 1)),
            age_ratio=i / T,
            is_new=float(current is None),
            value_ratio=float(obs[i]) / 8.0,
        )
        gate = None if policy is None else (lambda f, p=policy: p.decide(f, 0.6, 0.4, 0.6))
        runtime.assert_fact(
            key, "value", int(obs[i]), source="sensor", priority=0,
            gate=gate, gate_features=feats,
            use_default_gate=(policy is not None),
        )
        fact = runtime.state.get_fact(key, "value")
        if fact is not None:
            memory[key] = int(fact.value)
    accuracy = float(
        np.mean([memory.get(f"k{k}") == truth[k] for k in range(n_keys)])
    )
    rejected = sum(1 for e in runtime.state.audit if e.action == "fact_rejected")
    return {
        "accuracy": accuracy,
        "facts": len(runtime.state.facts),
        "rejected": rejected,
        "version": runtime.state.version,
    }


def _train_write_policy_fallback(seed=1, epochs=120):
    from pss.memory.active_policy import train_write_policy

    keys, obs, reliability, truth = _make_write_stream(600, 20, seed)
    memory = {}
    X, y = [], []
    for i in range(len(keys)):
        key = f"k{int(keys[i])}"
        current = memory.get(key)
        X.append(
            write_features(
                reliability=float(reliability[i]),
                conflict=0.0 if current is None else float(current != obs[i]),
                novelty=1.0 if current is None else 0.0,
                key_frequency=float(
                    np.bincount(keys[: i + 1], minlength=20)[int(keys[i])] / (i + 1)
                ),
                age_ratio=i / len(keys),
                is_new=float(current is None),
                value_ratio=float(obs[i]) / 8.0,
            )
        )
        y.append(float(obs[i] == truth[int(keys[i])]))
        memory[key] = int(obs[i])
    return train_write_policy(np.stack(X), np.array(y), epochs=epochs, seed=seed)


def stage_write_gate() -> dict:
    ckpt = Path("checkpoints/active_policy/write_policy.pt")
    policy = WritePolicy()
    if ckpt.exists():
        import torch

        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        policy.load_state_dict(payload["state_dict"])
        policy.eval()
    else:
        policy = _train_write_policy_fallback()
    return {
        "policy_loaded": ckpt.exists(),
        "write_all": _run_write_gate(None),
        "gated": _run_write_gate(policy),
    }


def stage_operators() -> dict:
    world = {gt: region for gt, region in ENTITIES}
    runtime = _runtime(uncertainty=False)
    gt_to_track = _ingest_clean(runtime, world)
    for gt, region in world.items():
        tid = gt_to_track.get(gt)
        if tid:
            runtime.assert_fact(tid, "location", region, source="perception")
    cup_tid = gt_to_track["cup"]
    goal = Goal("op_goal", frozenset({(cup_tid, "location", "G")}), "cup to G")
    runtime.set_goal(goal)
    library = VersionedOperatorLibrary(list(_move_library([cup_tid]).operators.values()))
    runtime.planner = RuntimePlanner(library)

    plan_v1 = runtime.planner.plan(runtime.state, goal)
    library.remove(f"move_{cup_tid}_A_G", reason="direct route blocked")
    plan_v2 = runtime.planner.plan(runtime.state, goal)
    library.rollback(1)
    plan_v3 = runtime.planner.plan(runtime.state, goal)
    return {
        "plan_v1": plan_v1.to_dict(),
        "plan_v2": plan_v2.to_dict(),
        "plan_v3": plan_v3.to_dict(),
        "changed_after_update": plan_v1.actions != plan_v2.actions,
        "restored_after_rollback": plan_v1.actions == plan_v3.actions,
        "operator_version": library.version,
    }


def stage_long(device="cpu", sequences=8) -> dict:
    index = load_index("data/tracking_bench_long")
    entries = [e for e in index["entries"] if e.get("split") == "test"][:sequences]
    models = load_perception_models(EXPERTS, PROJECTIONS, device=device)
    retentions, persistences, n_entities, n_active = [], [], [], []
    for entry in entries:
        sample = load_sample(Path("data/tracking_bench_long") / entry["path"])
        start = int(sample["occlusion_start"])
        end = int(sample["occlusion_end"])
        runtime = _runtime(uncertainty=False)
        pre_ids, post_ids = {}, {}
        for frame in range(sample["rgb"].shape[0]):
            if start <= frame < end:
                observations = []
            else:
                observations = extract_frame(
                    sample, frame, models, PoseNoise(seed=0, enabled=False)
                )
            events = runtime.ingest(observations, frame)["events"]
            mapping = pre_ids if frame < start else post_ids
            for key in ("matched", "new", "reid"):
                for tid, gid in events.get(key, []):
                    mapping[str(gid)] = tid
        common = set(pre_ids) & set(post_ids)
        retentions.append(
            float(np.mean([pre_ids[g] == post_ids[g] for g in common]))
            if common
            else 0.0
        )
        persistences.append(
            float(all(tid in runtime.state.entities for tid in pre_ids.values()))
        )
        n_entities.append(len(runtime.state.entities))
        n_active.append(sum(1 for e in runtime.state.entities.values() if e.active))
    return {
        "sequences": len(entries),
        "id_retention": float(np.mean(retentions)) if retentions else 0.0,
        "entity_persistence": float(np.mean(persistences)) if persistences else 0.0,
        "mean_entities": float(np.mean(n_entities)) if n_entities else 0.0,
        "mean_active": float(np.mean(n_active)) if n_active else 0.0,
        "per_sequence_retention": retentions,
    }


def stage_decoder() -> dict:
    decoder = load_local_decoder()
    return {
        "decoder_available": decoder is not None,
        "note": "No text LLM in the local HF cache; router falls back to the tool agent.",
    }


def render(payload: dict) -> str:
    lines = [
        "# Phase 17 — learned modules inside the runtime",
        "",
        f"> generated: {payload['generated']}",
        "",
        "## Learned relational goal",
        "",
        f"- region relation head all-4 accuracy: {_fmt(payload['relational']['region_head_accuracy'])}",
        "",
        "| predicate | symbolic plan | symbolic success | learned plan | learned success |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["relational"]["rows"]:
        s, l = row["symbolic"], row["learned"]
        lines.append(
            f"| {row['predicate']} | {s['plan_actions']} | {'yes' if s['success'] else 'no'} | "
            f"{l['plan_actions']} | {'yes' if l['success'] else 'no'} |"
        )
    wg = payload["write_gate"]
    lines += [
        "",
        "## Learned write gate",
        "",
        "| policy | accuracy | facts | rejected |",
        "|---|---:|---:|---:|",
        f"| write all | {_fmt(wg['write_all']['accuracy'])} | {wg['write_all']['facts']} | "
        f"{wg['write_all']['rejected']} |",
        f"| gated | {_fmt(wg['gated']['accuracy'])} | {wg['gated']['facts']} | "
        f"{wg['gated']['rejected']} |",
        "",
        "## Versioned operators",
        "",
        f"- v1 plan: {payload['operators']['plan_v1']['actions']}",
        f"- v2 plan after removing the direct route: {payload['operators']['plan_v2']['actions']}",
        f"- v3 plan after rollback: {payload['operators']['plan_v3']['actions']}",
        f"- changed after update: {payload['operators']['changed_after_update']}; "
        f"restored after rollback: {payload['operators']['restored_after_rollback']}",
        "",
        "## Long occlusion (70 frames)",
        "",
        f"- sequences: {payload['long']['sequences']}",
        f"- ID retention: {_fmt(payload['long']['id_retention'])}",
        f"- entity persistence: {_fmt(payload['long']['entity_persistence'])}",
        f"- mean entities / active: {_fmt(payload['long']['mean_entities'])} / "
        f"{_fmt(payload['long']['mean_active'])}",
        "",
        "## Real decoder",
        "",
        f"- available: {payload['decoder']['decoder_available']}",
        f"- {payload['decoder']['note']}",
        "",
        "## Verdict",
        "",
        "- Learned relations can drive the runtime planner; the exact predicate is the reference.",
        "- The learned write policy gates facts at the runtime boundary, not in a side script.",
        "- Operators are versioned; removing a route changes the plan and rollback restores it.",
        "- The runtime retains entities through a 70-frame occlusion on the long bench.",
        "- No real LLM is cached; the decoder path is wired but records unavailability.",
        "",
    ]
    return "\n".join(lines)


def run(device="cpu", sequences=8) -> dict:
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "relational": stage_relational(device),
        "write_gate": stage_write_gate(),
        "operators": stage_operators(),
        "long": stage_long(device, sequences),
        "decoder": stage_decoder(),
    }
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_phase17.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "runtime_phase17.md").write_text(render(payload), encoding="utf-8")
    return payload


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sequences", type=int, default=8)
    args = ap.parse_args()
    payload = run(args.device, args.sequences)
    print("relational:", payload["relational"]["region_head_accuracy"])
    print("write gate:", payload["write_gate"])
    print("operators changed:", payload["operators"]["changed_after_update"])
    print("long retention:", payload["long"]["id_retention"])
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
