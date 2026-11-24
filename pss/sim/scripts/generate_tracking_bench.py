"""CLI: generate the PSS vs Transformer tracking benchmark (P1).

Example:
    python -m pss.sim.scripts.generate_tracking_bench \
        --out data/tracking_bench_v1 --train 200 --val 40 --test 60
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.sim.tracking_bench import generate_tracking_bench


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tracking_bench_v1")
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--val", type=int, default=40)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--objects", type=int, default=3)
    ap.add_argument("--min-separation", type=float, default=0.15)
    ap.add_argument("--min-pixels", type=int, default=20)
    ap.add_argument("--scale", default="tracking_bench_v1")
    ap.add_argument(
        "--legacy-motion",
        action="store_true",
        help="use the old swap/+0.12 motion instead of randomized motion",
    )
    ap.add_argument(
        "--legacy-camera",
        action="store_true",
        help="use the fixed camera instead of camera jitter",
    )
    args = ap.parse_args()
    summary = generate_tracking_bench(
        args.out,
        train_count=args.train,
        val_count=args.val,
        test_count=args.test,
        seed=args.seed,
        width=args.width,
        height=args.height,
        frames=args.frames,
        n_objects=args.objects,
        min_separation=args.min_separation,
        min_pixels=args.min_pixels,
        data_scale=args.scale,
        randomize_motion=not args.legacy_motion,
        camera_jitter=not args.legacy_camera,
    )
    print("tracking_bench generated:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
