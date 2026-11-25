"""``pss score``: turn bundle traces into a scorecard."""

from __future__ import annotations

import argparse
import json

from .scorecard import score_suite, write_scorecard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pss score")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--bundles-root", default="bundles")
    parser.add_argument("--device", choices=("dev", "official"), default=None)
    parser.add_argument("--out", default="scorecards")
    parser.add_argument("--label", help="extra basename, e.g. S1_oracle")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = score_suite(args.bundles_root, args.suite, device=args.device)
    json_path, md_path = write_scorecard(report, args.out)
    if args.label:
        import shutil

        shutil.copyfile(json_path, json_path.with_name(f"{args.label}.json"))
        shutil.copyfile(md_path, md_path.with_name(f"{args.label}.md"))
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"wrote {json_path} and {md_path}")
        for name, stats in report["metrics"].items():
            print(f"  {name:<20} {stats['mean']:.4f} ±{stats['ci95']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
