"""CLI: run Phase 7 relations, planner, conflict, and world model.

Example:
    python -m pss.reasoning.scripts.run_phase7 \
        --data data/v1 --out checkpoints/phase7 --episodes 8 --epochs 100
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.reasoning.build import run_phase7


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--out", default="checkpoints/phase7")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    metrics = run_phase7(args.data, args.out, args.episodes, args.epochs, args.seed)
    rel = metrics["relations"]["overall"]
    plan = metrics["planner"]
    world = metrics["world_model"]
    multi = metrics["world_model_multi"]
    set_world = metrics["world_model_set"]
    print("Phase 7 complete:")
    print(f"  relation F1: {rel['f1']:.3f} (P={rel['precision']:.3f}, R={rel['recall']:.3f})")
    print(f"  A*: success={plan['astar']['success']} actions={plan['astar']['actions']} "
          f"nodes={plan['astar']['nodes']}")
    print(f"  blocked: {plan['blocked']['reason']}")
    print(f"  anomaly replan: {plan['anomaly']['replan_success']}")
    soft = metrics["soft_preconditions"]
    print(f"  soft preconditions: high={soft['high_confidence_success']} "
          f"low={soft['low_confidence_success']} ({soft['low_confidence_reason']})")
    print(f"  world model 1-step: {world['one_step_pos_error']}")
    print(f"  multi-entity 1-step: {multi['one_step_pos_error']}")
    print(f"  multi-entity rollout: {multi['rollout_pos_error']}")
    print(f"  set-entity 1-step: {set_world['one_step_pos_error']}")
    print(f"  set-entity rollout: {set_world['rollout_pos_error']}")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
