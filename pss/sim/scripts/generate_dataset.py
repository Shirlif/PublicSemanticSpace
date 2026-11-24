"""CLI: generate the Phase 2 tabletop dataset.

Example:
    python -m pss.sim.scripts.generate_dataset --out data/v1 --scale small --seed 0
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.sim.generator import SCALES, generate_dataset


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v1")
    ap.add_argument("--scale", choices=sorted(SCALES), default="small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument(
        "--tactile-mode",
        choices=("contact", "material"),
        default="contact",
        help="contact: contact-derived maps (empty without contact); "
        "material: material/texture proxy per object",
    )
    args = ap.parse_args()

    summary = generate_dataset(
        args.out,
        scale=args.scale,
        seed=args.seed,
        width=args.width,
        height=args.height,
        tactile_mode=args.tactile_mode,
    )
    print("Dataset generated:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
