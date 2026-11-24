"""CLI: generate a category-recognition pretraining set.

Example:
    python -m pss.sim.scripts.generate_category_pretrain \
        --out data/category_pretrain --variations 24 --views 4 --seed 0
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.sim.generator import generate_category_pretrain


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/category_pretrain")
    ap.add_argument("--variations", type=int, default=24)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument(
        "--categories",
        default="",
        help="comma-separated subset; default = all CATEGORY_SPECS",
    )
    args = ap.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()] or None
    summary = generate_category_pretrain(
        args.out,
        categories=categories,
        variations=args.variations,
        views=args.views,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )
    print("Category pretrain dataset generated:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
