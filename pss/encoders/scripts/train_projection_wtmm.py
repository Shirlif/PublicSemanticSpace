"""Retrain projections with the WT-MM-pretrained tactile encoder.

Example:
    python -m pss.encoders.scripts.train_projection_wtmm \
        --data data/expert_v1 --experts checkpoints/expert_v1 \
        --tactile checkpoints/tactile_wtmm/tactile_wtmm.pt \
        --out checkpoints/projection_wtmm --epochs 60
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.encoders.tactile.wtmm import WTMMTactileExpert, load_wtmm_tactile
from pss.encoders.vision import load_vision_expert
from pss.projection.train import train_projections


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_v1")
    ap.add_argument("--experts", default="checkpoints/expert_v1")
    ap.add_argument("--tactile", default="checkpoints/tactile_wtmm/tactile_wtmm.pt")
    ap.add_argument("--out", default="checkpoints/projection_wtmm")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    vision = load_vision_expert(f"{args.experts}/vision.pt", device=device)
    encoder = load_wtmm_tactile(args.tactile, device=device)
    tactile = WTMMTactileExpert(encoder=encoder, device=device, input_dim=64)
    metrics = train_projections(
        args.data,
        args.experts,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        temp=args.temp,
        seed=args.seed,
        vision_expert=vision,
        tactile_expert=tactile,
    )
    retrieval = metrics["retrieval"]["test"]
    print("WT-MM tactile projection retrained:")
    for direction in ("vision->tactile", "tactile->vision", "vision->geometry", "tactile->geometry"):
        vals = retrieval.get(direction, {})
        print(f"  {direction}: category R@1={vals.get('category_r1', 0):.3f} "
              f"pair R@1={vals.get('pair_r1', 0):.3f}")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
