"""CLI: generate mock RD2 episodes with domain shift."""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.rd2.mock import make_mock_dataset


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rd2_mock")
    ap.add_argument("--data", default="data/expert_v1")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    summary = make_mock_dataset(args.out, args.data, args.episodes, args.seed)
    print(summary)


if __name__ == "__main__":
    main()
