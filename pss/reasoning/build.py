"""Phase 7 evaluation: relations, planner, conflict, world model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss.encoders.data import load_index
from pss.sim.dataset import load_sample
from pss.sim.scene import CATEGORY_SPECS
from pss.world_model.actions import (
    hand_written_plan,
    initial_laundry_facts,
    laundry_goal,
    laundry_library,
    simulate_plan,
    spill_anomaly,
)
from pss.world_model.model import train_world_model
from pss.world_model.multi import train_multi_world_model
from pss.world_model.set_model import train_set_world_model

from .conflict import detect_conflict, should_replan
from .planner import AStarPlanner
from .relations import precision_recall_f1, relations_from_frame

PREDICATES = ("in", "contact", "near")
SYMMETRIC_PREDICATES = {"contact", "near"}


def _truth_set(rows) -> set[tuple]:
    """Canonicalize simulator truth for symmetric predicates.

    ``SensorRig.relations`` stores one direction for contact/near. The
    relation layer stores both because queries can start from either entity.
    Comparing the raw sets makes every symmetric fact look like a false
    positive, so symmetrize the truth side.
    """
    out: set[tuple] = set()
    for row in rows:
        fact = tuple(row)
        out.add(fact)
        if fact[1] in SYMMETRIC_PREDICATES:
            out.add((fact[2], fact[1], fact[0]))
    return out


def evaluate_relations(data_dir: str | Path, max_episodes: int = 8) -> dict:
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    entries = [e for e in index["entries"] if e["kind"] == "episode"][:max_episodes]
    sizes = {c: spec["size"] for c, spec in CATEGORY_SPECS.items()}
    overall = {"tp": 0, "fp": 0, "fn": 0}
    per = {p: {"tp": 0, "fp": 0, "fn": 0} for p in PREDICATES}
    frames = 0

    for entry in entries:
        sample = load_sample(data_dir / entry["path"])
        for t in range(sample["rgb"].shape[0]):
            computed = {r.as_tuple() for r in relations_from_frame(sample, t, sizes)}
            truth = _truth_set(sample["relations"][t])
            for key in overall:
                overall[key] += precision_recall_f1(computed, truth)[key]
            for predicate in PREDICATES:
                c = {f for f in computed if f[1] == predicate}
                g = {f for f in truth if f[1] == predicate}
                prf = precision_recall_f1(c, g)
                for key in per[predicate]:
                    per[predicate][key] += prf[key]
            frames += 1

    def finalize(counts):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    return {
        "episodes": len(entries),
        "frames": frames,
        "overall": finalize(overall),
        "per_predicate": {p: finalize(counts) for p, counts in per.items()},
    }


def run_planner_demo() -> dict:
    library = laundry_library()
    planner = AStarPlanner(library)
    initial = initial_laundry_facts()
    goal = laundry_goal()

    hand = hand_written_plan()
    facts_hand, _log_hand, ok_hand = simulate_plan(hand, initial, library)
    astar = planner.plan(initial, goal)
    facts_astar, _log_astar, ok_astar = (
        simulate_plan(astar.actions, initial, library) if astar.success else (None, [], False)
    )

    blocked_initial = frozenset(initial - {("detergent", "on", "counter")})
    blocked = planner.plan(blocked_initial, goal)

    facts_anom, _log_anom, ok_anom = simulate_plan(
        hand, initial, library, anomaly_step=4, anomaly_fn=spill_anomaly
    )
    replan = planner.plan(facts_anom, goal)
    facts_replan, _log_replan, ok_replan = (
        simulate_plan(replan.actions, facts_anom, library) if replan.success else (None, [], False)
    )

    return {
        "hand_written": {
            "actions": len(hand),
            "goal_reached": goal <= facts_hand,
            "valid": ok_hand,
        },
        "astar": {
            "success": astar.success,
            "actions": len(astar.actions),
            "nodes": astar.nodes,
            "cost": astar.cost,
            "plan": astar.actions,
            "goal_reached": goal <= facts_astar if facts_astar is not None else False,
            "valid": ok_astar,
        },
        "blocked": {
            "success": blocked.success,
            "reason": blocked.reason,
            "nodes": blocked.nodes,
        },
        "anomaly": {
            "injected_at": "pour_detergent",
            "post_state": sorted(["::".join(f) for f in facts_anom]),
            "replan_success": replan.success,
            "replan_actions": replan.actions,
            "goal_reached": goal <= facts_replan if facts_replan is not None else False,
        },
    }


def run_soft_precondition_demo() -> dict:
    library = laundry_library()
    initial = initial_laundry_facts()
    goal = laundry_goal()
    high = {fact: 0.95 for fact in initial}
    low = dict(high)
    low[("detergent", "on", "counter")] = 0.20
    planner = AStarPlanner(library)
    high_plan = planner.plan(initial, goal, confidence=high, threshold=0.5)
    low_plan = planner.plan(initial, goal, confidence=low, threshold=0.5)
    return {
        "threshold": 0.5,
        "high_confidence_success": high_plan.success,
        "high_confidence_actions": len(high_plan.actions),
        "low_confidence_success": low_plan.success,
        "low_confidence_reason": low_plan.reason,
        "low_confidence_nodes": low_plan.nodes,
    }


def run_conflict_demo() -> dict:
    predicted = {("detergent", "in", "washer"), ("door", "open")}
    observed = {("detergent", "on", "floor"), ("door", "open")}
    critical = {("detergent", "in", "washer")}
    result = detect_conflict(observed, predicted, critical)
    return {
        "conflict": result.conflict,
        "missing": [list(x) for x in result.missing],
        "unexpected": [list(x) for x in result.unexpected],
        "score": result.score,
        "should_replan": should_replan(result, critical),
    }


def _write_report(path: Path, metrics: dict) -> None:
    rel = metrics["relations"]
    plan = metrics["planner"]
    conflict = metrics["conflict"]
    world = metrics["world_model"]
    lines = [
        "# Phase 7 report: relations, planning, world model",
        "",
        f"- data: `{metrics['data_dir']}`",
        "",
        "## Relation computation vs simulation GT",
        "",
        f"- episodes: {rel['episodes']}, frames: {rel['frames']}",
        "",
        "| scope | precision | recall | F1 |",
        "|---|---:|---:|---:|",
        f"| overall | {rel['overall']['precision']:.3f} | {rel['overall']['recall']:.3f} | {rel['overall']['f1']:.3f} |",
    ]
    for predicate, vals in rel["per_predicate"].items():
        lines.append(
            f"| {predicate} | {vals['precision']:.3f} | {vals['recall']:.3f} | {vals['f1']:.3f} |"
        )
    lines += [
        "",
        "`contact` uses collision geometry when `object_specs` is present; older samples fall back to a radius proxy.",
        "",
        "## Planner",
        "",
        "| route | success | actions | nodes | goal reached |",
        "|---|---|---:|---:|---|",
        f"| hand-written | {plan['hand_written']['valid']} | {plan['hand_written']['actions']} | — | {plan['hand_written']['goal_reached']} |",
        f"| A* | {plan['astar']['success']} | {plan['astar']['actions']} | {plan['astar']['nodes']} | {plan['astar']['goal_reached']} |",
        f"| blocked (no detergent) | {plan['blocked']['success']} | — | {plan['blocked']['nodes']} | — |",
        "",
        f"A* plan: {' → '.join(plan['astar']['plan']) or '—'}",
        "",
        "### Anomaly recovery",
        "",
        f"- injected: detergent spills after `pour_detergent`",
        f"- replan success: {plan['anomaly']['replan_success']}",
        f"- replan: {' → '.join(plan['anomaly']['replan_actions']) or '—'}",
        f"- goal reached after replan: {plan['anomaly']['goal_reached']}",
        "",
        "## Soft preconditions",
        "",
        f"- threshold: {metrics['soft_preconditions']['threshold']:.2f}",
        f"- high confidence: success={metrics['soft_preconditions']['high_confidence_success']} "
        f"actions={metrics['soft_preconditions']['high_confidence_actions']}",
        f"- low confidence: success={metrics['soft_preconditions']['low_confidence_success']} "
        f"reason=`{metrics['soft_preconditions']['low_confidence_reason']}`",
        "",
        "## Conflict detection",
        "",
        f"- conflict: {conflict['conflict']}, score: {conflict['score']}",
        f"- missing: {conflict['missing']}",
        f"- unexpected: {conflict['unexpected']}",
        f"- should replan: {conflict['should_replan']}",
        "",
        "## World model",
        "",
        f"- train / val transitions: {world['n_train']} / {world['n_val']}",
        "",
        "| baseline | 1-step pos error | rollout pos error |",
        "|---|---:|---:|",
    ]
    for name in ("zero", "analytic", "learned"):
        lines.append(
            f"| {name} | {world['one_step_pos_error'][name]:.4f} | "
            f"{world['rollout_pos_error'][name]:.4f} |"
        )

    multi = metrics.get("world_model_multi")
    if multi:
        lines += [
            "",
            "## Multi-entity world model",
            "",
            f"- train / val transitions: {multi['n_train']} / {multi['n_val']}",
            f"- entities per sample: {multi['n_entities']}",
            "",
            "| scope | baseline | 1-step pos error | rollout pos error |",
            "|---|---|---:|---:|",
        ]
        for scope, one_key, roll_key in (
            ("all entities", "one_step_pos_error", "rollout_pos_error"),
            ("target only", "one_step_target_pos_error", "rollout_target_pos_error"),
        ):
            for name in ("zero", "analytic", "learned"):
                lines.append(
                    f"| {scope} | {name} | {multi[one_key][name]:.4f} | "
                    f"{multi[roll_key][name]:.4f} |"
                )
        lines.append("")

    set_world = metrics.get("world_model_set")
    if set_world:
        lines += [
            "",
            "## Multi-entity world model (per-entity set)",
            "",
            f"- train entity samples / val transitions: {set_world['n_train']} / {set_world['n_val']}",
            f"- entities per sample: {set_world['n_entities']}",
            "",
            "| scope | baseline | 1-step pos error | rollout pos error |",
            "|---|---|---:|---:|",
        ]
        for scope, one_key, roll_key in (
            ("all entities", "one_step_pos_error", "rollout_pos_error"),
            ("target only", "one_step_target_pos_error", "rollout_target_pos_error"),
        ):
            for name in ("zero", "analytic", "learned"):
                lines.append(
                    f"| {scope} | {name} | {set_world[one_key][name]:.4f} | "
                    f"{set_world[roll_key][name]:.4f} |"
                )
        lines.append("")

    lines += [
        "",
        "## Caveats",
        "",
        "- Relation `contact` uses collision geometry; simulation GT uses MuJoCo contacts.",
        "- S1 closed loop is run separately by `pss.reasoning.scripts.run_closed_loop`.",
        "- Planner states are discrete relation facts; soft preconditions are exercised in the demo below.",
        "- Soft-precondition confidence values are synthetic; perception confidence is not wired yet.",
        "- A* heuristic counts missing goals and is not admissible when one action adds several facts.",
        "- World models are trained on a small episode set; teleport actions dominate.",
        "- The multi-entity model predicts all entity deltas; its rollout rebuilds features from its own predicted state.",
        "- All results are smoke tests on primitive-proxy data.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase7(
    data_dir: str | Path,
    out_dir: str | Path,
    max_episodes: int = 8,
    epochs: int = 100,
    seed: int = 0,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    world = train_world_model(data_dir, out / "world_model", epochs=epochs, seed=seed)
    multi_world = train_multi_world_model(
        data_dir, out / "world_model_multi", epochs=epochs, seed=seed
    )
    set_world = train_set_world_model(
        data_dir, out / "world_model_set", epochs=epochs, seed=seed
    )
    metrics = {
        "data_dir": str(data_dir),
        "relations": evaluate_relations(data_dir, max_episodes),
        "planner": run_planner_demo(),
        "soft_preconditions": run_soft_precondition_demo(),
        "conflict": run_conflict_demo(),
        "world_model": world,
        "world_model_multi": multi_world,
        "world_model_set": set_world,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "REPORT.md", metrics)
    return metrics
