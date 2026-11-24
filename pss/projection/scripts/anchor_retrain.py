"""CLI: retrain one modality's projection head against frozen anchors.

Example:
    python -m pss.projection.scripts.anchor_retrain \
        --data data/category_pretrain \
        --experts checkpoints/category_pretrain_experts \
        --projections checkpoints/phase45_projection \
        --prototypes checkpoints/phase45_prototypes \
        --modality vision \
        --out checkpoints/anchor_retrain_vision \
        --epochs 60
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.projection.anchor import run_anchor_retrain


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/category_pretrain")
    parser.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    parser.add_argument("--projections", default="checkpoints/phase45_projection")
    parser.add_argument("--prototypes", default="checkpoints/phase45_prototypes")
    parser.add_argument("--modality", choices=["vision", "tactile", "geometry"], required=True)
    parser.add_argument("--new-expert", default=None)
    parser.add_argument("--out", default="checkpoints/anchor_retrain")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--material-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-from-old", choices=["true", "false"], default="true")
    args = parser.parse_args()

    metrics = run_anchor_retrain(
        args.data,
        args.experts,
        args.projections,
        args.prototypes,
        args.modality,
        args.out,
        new_expert=args.new_expert,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temp=args.temp,
        material_weight=args.material_weight,
        seed=args.seed,
        init_from_old=args.init_from_old == "true",
    )
    print("Anchor retraining complete:")
    print(f"  modality: {metrics['modality']}")
    print(f"  test category R@1: {metrics['retrieval']['test']['category_r1']:.3f}")
    print(f"  attribute retention: {metrics['attribute_retention']['mean_retention']:.1%}")
    print(f"  prototype drift mean cosine: {metrics['prototype_drift']['mean_cosine']:.3f}")
    print(f"  frozen artifacts unchanged: {metrics['frozen_integrity']['unchanged']}")
    print(f"  report: {args.out}/ANCHOR_REPORT.md")


if __name__ == "__main__":
    main()
