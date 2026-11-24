"""CLI: train Phase 4 projection heads.

Example:
    python -m pss.projection.scripts.train_projections \
        --data data/expert_v1 --experts checkpoints/expert_v1 \
        --out checkpoints/projection_v1 --epochs 60 --batch-size 64 --temp 0.07
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.projection.train import train_projections


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_v1")
    ap.add_argument("--experts", default="checkpoints/expert_v1")
    ap.add_argument("--out", default="checkpoints/projection_v1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--level", choices=["category", "instance"], default="category")
    ap.add_argument("--material-weight", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    metrics = train_projections(
        args.data,
        args.experts,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temp=args.temp,
        level=args.level,
        seed=args.seed,
        material_weight=args.material_weight,
    )
    print("Projection training complete:")
    print(f"  device: {metrics['device']}")
    for split in ("val", "test", "test_category"):
        print(f"  [{split}]")
        for direction, vals in metrics["retrieval"][split].items():
            print(
                f"    {direction}: category R@1={vals['category_r1']:.3f} "
                f"pair R@1={vals['pair_r1']:.3f}"
            )
    print(f"  checkpoints: {args.out}")


if __name__ == "__main__":
    main()
