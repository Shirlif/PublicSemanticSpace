"""CLI: symmetric vision/tactile projection on ObjectFolder 2.0.

Example:
    python -m pss.rd2.scripts.train_objectfolder_symmetric \
        --renders eval/results/objectfolder/renders \
        --out eval/results/objectfolder_symmetric \
        --checkpoint-dir checkpoints/objectfolder_symmetric \
        --epochs 40 --folds 5
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.rd2.objectfolder_symmetric import run_symmetric


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", default="eval/results/objectfolder/renders")
    parser.add_argument("--out", default="eval/results/objectfolder_symmetric")
    parser.add_argument("--checkpoint-dir", default="checkpoints/objectfolder_symmetric")
    parser.add_argument("--tactile-baseline", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = run_symmetric(
        args.renders,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temp=args.temp,
        folds=args.folds,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        tactile_baseline_path=args.tactile_baseline,
    )
    print("Symmetric projection:")
    for direction in ("vision_to_touch", "touch_to_vision"):
        seen = results["seen"][direction]
        print(f"  seen {direction}: R@1={seen['r1']:.3f} R@5={seen['r5']:.3f}")
    for direction in ("vision_to_touch", "touch_to_vision"):
        r1 = results["heldout"][f"{direction}_r1"]
        r5 = results["heldout"][f"{direction}_r5"]
        print(
            f"  heldout {direction}: R@1={r1['mean']:.3f}±{r1['std']:.3f} "
            f"R@5={r5['mean']:.3f}±{r5['std']:.3f}"
        )
    print(f"  report: {args.out}/SYMMETRIC_REPORT.md")


if __name__ == "__main__":
    main()
