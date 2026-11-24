"""Row 5: learned cost-to-go heuristic for A*.

Keeps A*/BFS as the search and learns the heuristic from optimal A* costs on a
tabletop region domain with distance-dependent move costs. Compares hand
heuristic vs learned vs weighted learned on success, cost, expansions and
latency, then wires the learned heuristic into ``RuntimePlanner``.

Example:
    python -m pss.eval.scripts.run_runtime_planning
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.reasoning.planner import AStarPlanner
from pss.reasoning.planner.learned import (
    DEFAULT_ENTITIES,
    DEFAULT_REGIONS,
    make_dataset,
    save_heuristic,
    train_heuristic,
)
from pss.runtime import CognitiveState, Entity, Fact, Goal, PSSRuntime, RuntimePlanner

OUT = "eval/results/runtime_planning"
CKPT = "checkpoints/planner/learned_heuristic.pt"


def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _evaluate(planner, states, goals, optimal):
    successes, costs, nodes, times, suboptimal = [], [], [], [], []
    for state, goal, best in zip(states, goals, optimal):
        t0 = time.perf_counter()
        result = planner.plan(state, goal)
        times.append((time.perf_counter() - t0) * 1000.0)
        successes.append(float(result.success))
        nodes.append(result.nodes)
        if result.success:
            costs.append(result.cost)
            suboptimal.append((result.cost - best) / max(1e-9, best))
        else:
            costs.append(float("inf"))
            suboptimal.append(float("inf"))
    return {
        "success": float(np.mean(successes)),
        "mean_cost": float(np.mean([c for c in costs if np.isfinite(c)])) if any(np.isfinite(costs)) else None,
        "mean_expansions": float(np.mean(nodes)),
        "mean_latency_ms": float(np.mean(times)),
        "mean_suboptimality": float(np.mean([s for s in suboptimal if np.isfinite(s)]))
        if any(np.isfinite(suboptimal))
        else None,
    }


def run(device="cpu", n_train=4000, n_test=400, epochs=200) -> dict:
    train_states, train_goals, train_costs, _ = make_dataset(n_train, seed=0)
    test_states, test_goals, test_costs, library = make_dataset(n_test, seed=1)
    model = train_heuristic(
        train_states, train_goals, train_costs, epochs=epochs, seed=0
    )
    save_heuristic(
        CKPT,
        model,
        meta={"entities": list(DEFAULT_ENTITIES), "regions": list(DEFAULT_REGIONS)},
    )

    hand = AStarPlanner(library)
    learned = AStarPlanner(library, heuristic=model.predict)
    weighted = AStarPlanner(library, heuristic=model.predict, heuristic_weight=1.5)
    results = {
        "hand": _evaluate(hand, test_states, test_goals, test_costs),
        "learned": _evaluate(learned, test_states, test_goals, test_costs),
        "weighted_learned": _evaluate(weighted, test_states, test_goals, test_costs),
    }

    # Runtime integration: the learned heuristic is attached to RuntimePlanner.
    state = CognitiveState()
    for entity, _attribute, region in test_states[0]:
        state.add_entity(
            Entity(entity, entity, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        state.set_fact(Fact(entity, "location", region))
    goal = Goal("learned_goal", frozenset(test_goals[0]), "learned heuristic demo")
    runtime = PSSRuntime(
        planner=RuntimePlanner(library, heuristic=model.predict), state=state
    )
    runtime.set_goal(goal)
    plan = runtime.plan("learned_goal")
    summary = runtime.summary()

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "counts": {"train": len(train_states), "test": len(test_states)},
        "results": results,
        "runtime": {
            "plan_actions": len(plan.actions),
            "plan_success": plan.confidence > 0,
            "planning": summary.get("planning"),
        },
    }
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_planning.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "runtime_planning.md").write_text(render(payload), encoding="utf-8")
    return payload


def render(payload: dict) -> str:
    lines = [
        "# Row 5 — learned cost-to-go heuristic for A*",
        "",
        f"> generated: {payload['generated']}",
        f"> train / test: {payload['counts']['train']} / {payload['counts']['test']}",
        "",
        "| planner | success | mean cost | expansions | latency | suboptimality |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in payload["results"].items():
        lines.append(
            f"| {name} | {_fmt(r['success'])} | {_fmt(r['mean_cost'])} | "
            f"{_fmt(r['mean_expansions'], 1)} | {_fmt(r['mean_latency_ms'], 2)} ms | "
            f"{_fmt(r['mean_suboptimality'])} |"
        )
    lines += [
        "",
        f"- runtime plan actions: {payload['runtime']['plan_actions']}; "
        f"heuristic: {payload['runtime']['planning']}",
        "",
        "## Verdict",
        "",
        "- A* remains the search; only the heuristic changed.",
        "- The learned heuristic is trained on optimal A* costs, so it can reduce",
        "  expansions relative to the hand `missing-facts` heuristic when move costs",
        "  vary. It may be inadmissible, so suboptimality is reported.",
        "- The runtime planner accepts the learned heuristic; the hand heuristic is",
        "  the default/fallback.",
        "",
    ]
    return "\n".join(lines)


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    payload = run(args.device, args.train, args.test, args.epochs)
    for name, r in payload["results"].items():
        print(
            f"{name}: success={r['success']:.3f} cost={_fmt(r['mean_cost'])} "
            f"exp={r['mean_expansions']:.1f} {r['mean_latency_ms']:.2f} ms"
        )
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
