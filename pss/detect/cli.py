"""``pss detect`` and ``pss replay`` commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from .oracle import detect_oracle, write_detections
from .replay import replay_bundle


def _cmd_detect(args: argparse.Namespace) -> int:
    if args.tier != "oracle":
        raise NotImplementedError(f"tier {args.tier!r} lands in W4; only oracle is available")
    rows = detect_oracle(args.bundle, min_visibility=args.min_visibility)
    path = write_detections(args.bundle, rows)
    print(f"[detect] {len(rows)} rows -> {path}")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    trace, pss = replay_bundle(args.bundle, write=not args.no_write)
    print(f"[replay] {len(trace)} assignments, {len(pss)} pss_trace rows")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pss detect", description="Detection and replay")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="write obs_detections.parquet for a bundle")
    detect.add_argument("bundle")
    detect.add_argument("--tier", default="oracle", choices=("oracle",))
    detect.add_argument("--min-visibility", type=float, default=0.001)
    detect.set_defaults(func=_cmd_detect)

    replay = sub.add_parser("replay", help="run PSS over detections and write traces")
    replay.add_argument("bundle")
    replay.add_argument("--no-write", action="store_true")
    replay.set_defaults(func=_cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
