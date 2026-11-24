"""Pretrain a tactile encoder on WT-MM (12x32 contact arrays).

Example:
    python -m pss.encoders.scripts.train_tactile_wtmm \
        --pair-root data/public/wt_mm_pair --extra-root data/public/wt_mm \
        --video data/public/wt_mm_pair/videos/observation.images.cam/chunk-000/file-005.mp4 \
        --out checkpoints/tactile_wtmm --epochs 30
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.encoders.tactile.wtmm import train_wtmm_tactile


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-root", default="data/public/wt_mm_pair")
    ap.add_argument("--extra-root", default="data/public/wt_mm")
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default="checkpoints/tactile_wtmm")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    metrics = train_wtmm_tactile(
        args.pair_root,
        args.extra_root,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        out_dim=args.out_dim,
        seed=args.seed,
        video_path=args.video,
    )
    print("WT-MM tactile encoder:")
    print(f"  frames={metrics['frames']} episodes={metrics['episodes']} "
          f"holdout={metrics['holdout_episode']}")
    print(f"  reconstruction MSE: {metrics['reconstruction_mse']:.6f}")
    print(f"  contact probe: {metrics['contact_probe']}")
    print(f"  temporal retrieval: {metrics['temporal_retrieval']}")
    if "video_tactile_alignment" in metrics:
        print(f"  video alignment: {metrics['video_tactile_alignment']}")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
