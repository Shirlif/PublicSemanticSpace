"""World-model training CLI.

Example:
    python -m pss.world_model.scripts.train_world_model \
        --data data/v1 --out checkpoints/world_model_v1 --epochs 100
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.world_model.model import train_world_model


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--out", default="checkpoints/world_model_v1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    metrics = train_world_model(args.data, args.out, args.epochs, args.hidden, args.seed)
    print(f"World model trained: train={metrics['n_train']} val={metrics['n_val']}")
    print(f"  1-step pos error: {metrics['one_step_pos_error']}")
    print(f"  rollout pos error: {metrics['rollout_pos_error']}")


if __name__ == "__main__":
    main()
