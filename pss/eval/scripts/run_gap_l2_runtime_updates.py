"""Claim-gap E19: L2 cognitive-layer updates at runtime.

The Isaac S4-PSS demo has oracle poses and no occlusion, so it cannot test
mid-task updates or module swaps.  This script exercises the same
``PSSRuntime`` on a deterministic tabletop state machine, without a simulator:

- ``kb_update``: the goal changes mid-task; the old plan is executed as a
  baseline and fails the new goal, while ``set_goal`` + replan succeeds.
- ``planner_swap``: a fact invalidates the direct operator mid-task; the old
  plan fails its precondition, while swapping ``runtime.planner`` and
  replanning succeeds.

This is L2 runtime behavior (state, facts, planning, audit), not perception.
The perception-side conflict/abstention evidence is in E15 and
``run_runtime.py``; the L2 memory ablation is E12.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.runtime import Entity, Goal, PSSRuntime, RuntimePlanner
from pss.world_model.actions import Operator, OperatorLibrary


def _move(name: str, src: str, dst: str, extra_pre=()):
    pre = frozenset({("cup", "location", src), *extra_pre})
    return Operator(
        name,
        pre,
        frozenset({("cup", "location", dst)}),
        frozenset({("cup", "location", src)}),
    )


def _runtime(library) -> PSSRuntime:
    runtime = PSSRuntime(planner=RuntimePlanner(library))
    runtime.state.add_entity(
        Entity("cup", "cup", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    )
    return runtime


def _simulate(actions, facts, library):
    cur = frozenset(facts)
    for name in actions:
        op = library.get(name)
        if not op.applicable(cur):
            return False, cur
        cur = op.apply(cur)
    return True, cur


def _at(facts, region: str) -> bool:
    return ("cup", "location", region) in facts


def _kb_update(seed: int) -> dict:
    library = OperatorLibrary(
        [_move("move_A_G", "A", "G"), _move("move_A_B", "A", "B")]
    )
    runtime = _runtime(library)
    runtime.assert_fact("cup", "location", "A", source="perception")
    runtime.set_goal(Goal("g1", frozenset({("cup", "location", "G")}), "cup to G"))
    plan1 = runtime.plan("g1")

    # Mid-task: the user changes the target to B.
    runtime.set_goal(Goal("g2", frozenset({("cup", "location", "B")}), "cup to B"))
    plan2 = runtime.plan("g2")

    facts = runtime.state.fact_set()
    baseline_ok, baseline_facts = _simulate(plan1.actions, facts, library)
    updated_ok, updated_facts = _simulate(plan2.actions, facts, library)
    return {
        "seed": seed,
        "scenario": "kb_update",
        "plan_before": list(plan1.actions),
        "plan_after": list(plan2.actions),
        "plan_changed": list(plan1.actions) != list(plan2.actions),
        "baseline_exec_ok": baseline_ok,
        "baseline_goal_ok": _at(baseline_facts, "B"),
        "updated_exec_ok": updated_ok,
        "updated_goal_ok": _at(updated_facts, "B"),
        "version_after_update": int(runtime.state.version),
    }


def _planner_swap(seed: int) -> dict:
    direct = OperatorLibrary(
        [
            _move(
                "move_A_G_direct",
                "A",
                "G",
                extra_pre=(("cup", "direct_ok", "yes"),),
            )
        ]
    )
    detour = OperatorLibrary(
        [
            _move(
                "move_A_D",
                "A",
                "D",
                extra_pre=(("cup", "detour_ok", "yes"),),
            ),
            _move("move_D_G", "D", "G"),
        ]
    )
    runtime = _runtime(direct)
    runtime.assert_fact("cup", "location", "A", source="perception")
    runtime.assert_fact("cup", "direct_ok", "yes", source="perception")
    runtime.assert_fact("cup", "detour_ok", "no", source="perception")
    runtime.set_goal(Goal("g", frozenset({("cup", "location", "G")}), "cup to G"))
    plan1 = runtime.plan("g")

    # Mid-task: the direct route is blocked; swap in the detour planner.
    runtime.assert_fact("cup", "direct_ok", "no", source="runtime", priority=5)
    runtime.assert_fact("cup", "detour_ok", "yes", source="runtime", priority=5)
    runtime.planner = RuntimePlanner(detour)
    plan2 = runtime.plan("g")

    facts = runtime.state.fact_set()
    baseline_ok, baseline_facts = _simulate(plan1.actions, facts, direct)
    updated_ok, updated_facts = _simulate(plan2.actions, facts, detour)
    return {
        "seed": seed,
        "scenario": "planner_swap",
        "plan_before": list(plan1.actions),
        "plan_after": list(plan2.actions),
        "plan_changed": list(plan1.actions) != list(plan2.actions),
        "baseline_exec_ok": baseline_ok,
        "baseline_goal_ok": _at(baseline_facts, "G"),
        "updated_exec_ok": updated_ok,
        "updated_goal_ok": _at(updated_facts, "G"),
        "version_after_update": int(runtime.state.version),
    }


def run(seeds: int = 3) -> dict:
    rows = []
    for seed in range(seeds):
        rows.append(_kb_update(seed))
        rows.append(_planner_swap(seed))
    summary = {}
    for scenario in ("kb_update", "planner_swap"):
        per = [r for r in rows if r["scenario"] == scenario]
        summary[scenario] = {
            "n": len(per),
            "plan_changed": float(np.mean([r["plan_changed"] for r in per])),
            "baseline_goal_ok": float(np.mean([r["baseline_goal_ok"] for r in per])),
            "updated_goal_ok": float(np.mean([r["updated_goal_ok"] for r in per])),
            "baseline_exec_ok": float(np.mean([r["baseline_exec_ok"] for r in per])),
            "updated_exec_ok": float(np.mean([r["updated_exec_ok"] for r in per])),
        }
    return {"seeds": int(seeds), "rows": rows, "summary": summary}


def _report(m: dict) -> str:
    lines = [
        "# Claim-gap E19: L2 runtime mid-task updates",
        "",
        f"> seeds: {m['seeds']}",
        "",
        "| scenario | plan changed | baseline goal | updated goal | baseline exec | updated exec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, r in m["summary"].items():
        lines.append(
            f"| {scenario} | {r['plan_changed']:.0%} | {r['baseline_goal_ok']:.0%} | "
            f"{r['updated_goal_ok']:.0%} | {r['baseline_exec_ok']:.0%} | "
            f"{r['updated_exec_ok']:.0%} |"
        )
    lines += [
        "",
        "## Scenarios",
        "",
        "- `kb_update`: goal changes from G to B mid-task. The old plan moves to G and",
        "  misses the new goal; `set_goal` + replan moves to B.",
        "- `planner_swap`: the direct operator is blocked mid-task. The old plan fails",
        "  its precondition; swapping `runtime.planner` to a detour library and",
        "  replanning succeeds without restarting the runtime.",
        "",
        "## Scope",
        "",
        "This is runtime-level L2 evidence (state, facts, planning, audit), not a",
        "sensory simulator. S4-PSS cannot test these because its observations are",
        "oracle poses with no occlusion and no raw features to swap.",
        "",
        "Conflict/abstention is covered by `run_runtime.py` (`no_uncertainty` drops",
        "the conflict flag) and by E15 (modality conflict); L2 memory is E12.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="reports/claim_gaps/l2_runtime_updates")
    args = ap.parse_args()
    metrics = run(seeds=args.seeds)
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    for scenario, r in metrics["summary"].items():
        print(
            f"{scenario:14s} changed={r['plan_changed']:.0%} "
            f"baseline={r['baseline_goal_ok']:.0%} updated={r['updated_goal_ok']:.0%}"
        )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
