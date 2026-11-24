"""CLI: build the Phase 5 prototype bank and evaluation report.

Example:
    python -m pss.prototypes.scripts.build_prototypes \
        --data data/expert_v1 --experts checkpoints/expert_v1 \
        --projections checkpoints/projection_v1 --out checkpoints/prototypes_v1
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.prototypes.build import run_phase5


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_v1")
    ap.add_argument("--experts", default="checkpoints/expert_v1")
    ap.add_argument("--projections", default="checkpoints/projection_v1")
    ap.add_argument("--out", default="checkpoints/prototypes_v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    metrics = run_phase5(
        args.data, args.experts, args.projections, args.out, seed=args.seed
    )
    fusion = metrics["fusion"]
    print("Prototype bank built:")
    print(f"  categories: {metrics['classes']}")
    print(f"  fusion: val={fusion['val_acc']:.3f} test={fusion['test_acc']:.3f}")
    for m, vals in metrics["single_modality"].items():
        print(f"  {m} only: val={vals['val_acc']:.3f} test={vals['test_acc']:.3f}")
    print(f"  open-set threshold: {metrics['open_set']['threshold']:.3f}")
    print(
        f"  rejection: seen={metrics['open_set']['seen_false_reject_rate']:.3f} "
        f"unseen={metrics['open_set']['unseen_reject_rate']:.3f}"
    )
    conflict = metrics["conflict_injection"]
    print(
        f"  conflict: normal={conflict['baseline']:.3f} "
        f"vision_inj={conflict.get('vision_injected', conflict['injected']):.3f} "
        f"geometry_inj={conflict.get('geometry_injected', conflict['injected']):.3f}"
    )
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
