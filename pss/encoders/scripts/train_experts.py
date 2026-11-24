"""CLI: train Phase 3 sensory experts.

Example:
    python -m pss.encoders.scripts.train_experts --data data/expert_v1 \
        --out checkpoints/expert_v1 --epochs 30 --size 64
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.encoders.train import train_all


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_v1")
    ap.add_argument("--out", default="checkpoints/expert_v1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--tactile-arch",
        default="mlp",
        choices=("mlp", "conv"),
        help="tactile encoder architecture; conv writes tactile_conv.pt",
    )
    args = ap.parse_args()

    metrics = train_all(
        args.data,
        args.out,
        epochs=args.epochs,
        image_size=args.size,
        seed=args.seed,
        lr=args.lr,
        tactile_arch=args.tactile_arch,
    )
    print("Training complete:")
    print(f"  device: {metrics['device']}")
    print(f"  vision: train={metrics['vision']['train_acc']:.3f} "
          f"val={metrics['vision']['val_acc']:.3f} "
          f"test={metrics['vision']['test_acc']:.3f}")
    print(f"  tactile: train={metrics['tactile']['train_acc']:.3f} "
          f"val={metrics['tactile']['val_acc']:.3f} "
          f"test={metrics['tactile']['test_acc']:.3f}")
    print(f"  geometry: train={metrics['geometry']['train_acc']:.3f} "
          f"val={metrics['geometry']['val_acc']:.3f} "
          f"test={metrics['geometry']['test_acc']:.3f}")
    print(f"  checkpoints: {args.out}")


if __name__ == "__main__":
    main()
