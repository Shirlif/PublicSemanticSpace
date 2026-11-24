"""CLI: build the PhysiCLeAR cache and train the R2 tactile encoder.

Example:
    python -m pss.encoders.scripts.train_physiclear \
        --videos _external/octopi/dataset \
        --repo-root _external/octopi \
        --data data/physiclear \
        --out checkpoints/tactile_physiclear \
        --epochs 30
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.encoders.tactile.physiclear import (
    prepare_physiclear,
    train_physiclear_encoder,
)


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", default="_external/octopi/dataset")
    parser.add_argument("--repo-root", default="_external/octopi")
    parser.add_argument("--data", default="data/physiclear")
    parser.add_argument("--out", default="checkpoints/tactile_physiclear")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    index = prepare_physiclear(
        args.videos,
        args.repo_root,
        args.data,
        num_frames=args.frames,
        size=args.size,
        force=args.force,
    )
    print(f"PhysiCLeAR cache: {index['counts']} clips -> {args.data}")
    if args.prepare_only:
        return

    metrics = train_physiclear_encoder(
        args.data,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dim=args.out_dim,
        seed=args.seed,
        device=args.device,
    )
    print("R2 tactile encoder trained:")
    print(f"  clips: {metrics['clips']}")
    print(f"  best epoch: {metrics['best_epoch']} val={metrics['best_val_loss']:.4f}")
    print(f"  property acc: {metrics['history'][metrics['best_epoch'] - 1]['val']['property_acc']}")
    print(f"  checkpoint: {args.out}/tactile_real.pt")


if __name__ == "__main__":
    main()
