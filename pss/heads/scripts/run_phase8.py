"""CLI: run Phase 8 readout-head evaluation.

Example:
    python -m pss.heads.scripts.run_phase8 \
        --data data/v1 --experts checkpoints/expert_v1 \
        --projections checkpoints/projection_v1 --prototypes checkpoints/prototypes_v1 \
        --out checkpoints/phase8 --trials 10
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.heads.build import run_phase8


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--experts", default="checkpoints/expert_v1")
    ap.add_argument("--projections", default="checkpoints/projection_v1")
    ap.add_argument("--prototypes", default="checkpoints/prototypes_v1")
    ap.add_argument("--out", default="checkpoints/phase8")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    metrics = run_phase8(
        args.data,
        args.experts,
        args.projections,
        args.prototypes,
        args.out,
        n_trials=args.trials,
        seed=args.seed,
    )
    action = metrics["action"]
    learned = metrics["action_learned"]
    ablation = metrics["ablation"]
    print("Phase 8 complete:")
    print(f"  action head: success={action['success_rate']:.3f} "
          f"final_dist={action['mean_final_distance']:.3f} "
          f"pushes={action['mean_pushes']:.1f}")
    print(f"  learned head: success={learned['success_rate']:.3f} "
          f"final_dist={learned['mean_final_distance']:.3f} "
          f"pushes={learned['mean_pushes']:.1f}")
    print(f"  text head: {metrics['text']['parameters']} parameters")
    print(f"  value ranking: {metrics['value']['ranking'][:3]}")
    print(f"  value focus: {metrics['value']['focus_ids']}")
    value_learned = metrics["value_learned"]
    print(f"  value pairwise: rule={value_learned['rule']['pairwise_accuracy']:.3f} "
          f"learned={value_learned['learned']['pairwise_accuracy']:.3f}")
    print(f"  value top-1: rule={value_learned['rule']['top1_accuracy']:.3f} "
          f"learned={value_learned['learned']['top1_accuracy']:.3f}")
    print(f"  ablation: acc {ablation['acc_before']:.4f} → {ablation['acc_after']:.4f} "
          f"(delta {ablation['delta']:+.4f})")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
