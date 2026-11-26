"""Command-line entry point: ``python -m pss_eval``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .fixtures import make_gaussian_stream
from .properties import default_properties
from .runner import make_default_systems, run_suite, write_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pss_eval")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--stream-seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=None,
        help="directory for report.json and REPORT.md (omitted: print only)",
    )
    args = parser.parse_args(argv)

    if args.seeds <= 0:
        parser.error("--seeds must be positive")

    stream = make_gaussian_stream(n=args.n, dim=args.dim, seed=args.stream_seed)
    report = run_suite(
        make_default_systems(dim=args.dim),
        default_properties(),
        stream,
        seeds=tuple(range(args.seeds)),
    )

    for system, summary in report["summary"]["systems"].items():
        statuses = ", ".join(
            f"{prop}={status}" for prop, status in summary["statuses"].items()
        )
        print(f"{system}: {statuses}")
        print(f"  smoke_pass={summary['smoke_pass']}")

    check = report["preregistration"]
    print(f"preregistration: {'OK' if check['ok'] else 'MISMATCH'}")
    for mismatch in check["mismatches"]:
        print(f"  - {mismatch}")

    if args.out:
        paths = write_report(report, args.out)
        print(f"report json: {paths['json']}")
        print(f"report markdown: {paths['markdown']}")

    smoke_ok = all(
        summary["smoke_pass"] for summary in report["summary"]["systems"].values()
    )
    return 0 if check["ok"] and smoke_ok else 1


if __name__ == "__main__":
    sys.exit(main())
