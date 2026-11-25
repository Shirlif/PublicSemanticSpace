"""``pss report`` and ``pss verify-recompute``."""

from __future__ import annotations

import argparse
import json

from .report import build_report, verify_recompute


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pss report")
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("build", help="assemble reports/<tag>/ (A10)")
    report.add_argument("--tag", required=True)
    report.add_argument("--out", default="reports")
    report.add_argument("--bundles-root", default="bundles")
    report.add_argument("--scorecards", default="scorecards")
    report.add_argument("--prereg", default="eval/prereg_isaac_v1.md")
    report.add_argument("--freeze", default="freeze/p7-w11.json")
    report.add_argument("--notes", default="notes/w13_status.md")
    report.set_defaults(func=_cmd_build)

    verify = sub.add_parser("verify", help="recompute scorecards from bundles (A9)")
    verify.add_argument("--tag", required=True)
    verify.add_argument("--bundles-root", default="bundles")
    verify.add_argument("--scorecards", default="scorecards")
    verify.add_argument("--out", default="verify")
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=_cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


def _cmd_build(args: argparse.Namespace) -> int:
    out = build_report(
        args.tag,
        out_root=args.out,
        bundles_root=args.bundles_root,
        scorecards_dir=args.scorecards,
        prereg=args.prereg,
        freeze=args.freeze,
        notes=args.notes,
    )
    print(f"wrote {out}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    payload = verify_recompute(
        args.tag,
        bundles_root=args.bundles_root,
        scorecards_dir=args.scorecards,
        out_dir=args.out,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
