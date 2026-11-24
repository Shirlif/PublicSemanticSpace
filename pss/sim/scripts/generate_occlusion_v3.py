"""CLI: generate the fully-occluded occlusion_v3 benchmark.

Example:
    python -m pss.sim.scripts.generate_occlusion_v3 \
        --out data/occlusion_v3 --sequences 6 --frames 24
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.sim.generator import generate_occlusion_v3_dataset


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/occlusion_v3")
    ap.add_argument("--sequences", type=int, default=6)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--visible-head", type=int, default=5)
    ap.add_argument("--visible-tail", type=int, default=5)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument(
        "--category",
        default=None,
        help="build N instances of one category (same-category variant)",
    )
    ap.add_argument(
        "--variation-specs",
        action="store_true",
        help="sample per-instance scale/color/material variation",
    )
    ap.add_argument(
        "--near-identical",
        action="store_true",
        help="same-category instances with identical size/color/material",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    summary = generate_occlusion_v3_dataset(
        args.out,
        sequences=args.sequences,
        frames=args.frames,
        seed=args.seed,
        width=args.width,
        height=args.height,
        visible_head=args.visible_head,
        visible_tail=args.visible_tail,
        category=args.category,
        variation_specs=args.variation_specs,
        near_identical=args.near_identical,
    )
    print("occlusion_v3 generated:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
