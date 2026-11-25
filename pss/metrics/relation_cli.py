"""``pss score-relations`` command."""

from __future__ import annotations

import argparse
import json

from .relation_score import score_relations_suite, write_relation_scorecard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pss score-relations")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--bundles-root", default="bundles")
    parser.add_argument("--device", choices=("dev", "official"), default=None)
    parser.add_argument("--out", default="scorecards")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = score_relations_suite(args.bundles_root, args.suite, device=args.device)
    json_path, md_path = write_relation_scorecard(report, args.out)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"wrote {json_path} and {md_path}")
        print(f"  macro_f1 {report['macro_f1']:.4f}  micro_f1 {report['micro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
