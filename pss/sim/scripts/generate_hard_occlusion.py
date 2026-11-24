"""Generate the hard (full) occlusion benchmark.

Example:
    python -m pss.sim.scripts.generate_hard_occlusion --out data/hard_occlusion
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.sim.generator import generate_hard_occlusion_dataset


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hard_occlusion")
    ap.add_argument("--sequences", type=int, default=6)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    args = ap.parse_args()
    summary = generate_hard_occlusion_dataset(
        args.out,
        sequences=args.sequences,
        frames=args.frames,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )
    print("Hard occlusion dataset generated:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
