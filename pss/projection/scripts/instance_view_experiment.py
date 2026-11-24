"""CLI: instance-level view holdout (simulation proxy for RD2).

Example:
    python -m pss.projection.scripts.instance_view_experiment \
        --data data/category_pretrain \
        --experts checkpoints/category_pretrain_experts \
        --projections checkpoints/phase45_projection \
        --out checkpoints/instance_view \
        --epochs 60
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.projection.instance_view import run_instance_view_experiment


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/category_pretrain")
    parser.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    parser.add_argument("--projections", default="checkpoints/phase45_projection")
    parser.add_argument("--out", default="checkpoints/instance_view")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--instance-weight", type=float, default=1.0)
    parser.add_argument("--category-weight", type=float, default=0.5)
    parser.add_argument("--material-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-views", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--test-views", nargs="+", type=int, default=[2, 3])
    args = parser.parse_args()

    metrics = run_instance_view_experiment(
        args.data,
        args.experts,
        args.projections,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temp=args.temp,
        instance_weight=args.instance_weight,
        category_weight=args.category_weight,
        material_weight=args.material_weight,
        seed=args.seed,
        train_views=tuple(args.train_views),
        test_views=tuple(args.test_views),
    )
    base = metrics["baseline"]["instance_retrieval"]
    fine = metrics["fine_tuned"]["instance_retrieval"]
    print("Instance view holdout complete:")
    for pair in ("vision-vision", "vision-tactile", "vision-geometry", "geometry-geometry"):
        print(
            f"  {pair}: baseline R@1={base[pair]['r1']:.3f} "
            f"-> fine-tuned R@1={fine[pair]['r1']:.3f}"
        )
    print(f"  checkpoint: {args.out}")
    print(f"  report: {args.out}/INSTANCE_VIEW_REPORT.md")


if __name__ == "__main__":
    main()
