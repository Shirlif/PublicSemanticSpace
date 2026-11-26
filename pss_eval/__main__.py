"""Command-line entry point: ``python -m pss_eval``.

Two suites:

* ``--suite external`` (default) -- the original F5 external baseline suite on
  a synthetic Gaussian stream: E5 flat + transformer-state, four properties.
* ``--suite protocol`` -- the three-way comparison (E5 flat, transformer-state,
  PSS runtime) with the extended property suite.  Add ``--bundle`` to run the
  real development bundles instead of the synthetic stream.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .bundles import DEFAULT_BUNDLE_NAMES, DEFAULT_BUNDLE_ROOT, load_bundle_streams
from .fixtures import make_gaussian_stream
from .preregistration import protocol_expectations
from .properties import default_properties, extended_properties
from .runner import (
    make_default_systems,
    make_protocol_systems,
    run_bundle_suite,
    run_suite,
    write_report,
)


def _print_report(report: dict) -> None:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pss_eval")
    parser.add_argument("--suite", choices=("external", "protocol"), default="external")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--stream-seed", type=int, default=0)
    parser.add_argument(
        "--bundles",
        action="store_true",
        help=(
            "protocol suite: run the real development bundles instead of the "
            f"synthetic stream (default names: {', '.join(DEFAULT_BUNDLE_NAMES)})"
        ),
    )
    parser.add_argument(
        "--bundle",
        action="append",
        default=None,
        help="protocol suite: explicit bundle name to load (repeatable)",
    )
    parser.add_argument(
        "--bundle-root",
        default=str(DEFAULT_BUNDLE_ROOT),
        help="root directory holding the development bundles",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="cap frames per bundle"
    )
    parser.add_argument(
        "--with-gt-identities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "build harness-side identity labels from gt.parquet (never passed "
            "to a system); --no-gt-identities disables it"
        ),
    )
    parser.add_argument("--budget-ms", type=float, default=50.0)
    parser.add_argument("--public-dim", type=int, default=32)
    parser.add_argument("--tier", default="frozen_real")
    parser.add_argument(
        "--out",
        default=None,
        help="directory for report.json and REPORT.md (omitted: print only)",
    )
    args = parser.parse_args(argv)

    if args.seeds <= 0:
        parser.error("--seeds must be positive")

    if args.suite == "protocol":
        properties = extended_properties()
        budget_enabled = args.budget_ms is not None and args.budget_ms > 0
        if args.bundles or args.bundle:
            streams = load_bundle_streams(
                args.bundle_root,
                args.bundle,
                with_gt_identities=args.with_gt_identities,
                max_frames=args.max_frames,
            )
            if not streams:
                parser.error(f"no bundles with embeddings found under {args.bundle_root}")
            dim = next(iter(streams.values())).dim
            systems = make_protocol_systems(
                dim=dim,
                input_dim=dim,
                public_dim=args.public_dim,
                budget_ms=args.budget_ms,
                tier=args.tier,
            )
            report = run_bundle_suite(
                systems,
                properties,
                streams,
                seeds=tuple(range(args.seeds)),
                preregistration_expectations=protocol_expectations(
                    bundle_stream=True, budget_enabled=budget_enabled
                ),
            )
        else:
            stream = make_gaussian_stream(
                n=args.n, dim=args.dim, seed=args.stream_seed
            )
            systems = make_protocol_systems(
                dim=args.dim,
                input_dim=args.dim,
                public_dim=args.public_dim,
                budget_ms=args.budget_ms,
                tier=args.tier,
            )
            report = run_suite(
                systems,
                properties,
                stream,
                seeds=tuple(range(args.seeds)),
                preregistration_expectations=protocol_expectations(
                    bundle_stream=False, budget_enabled=budget_enabled
                ),
            )
    else:
        stream = make_gaussian_stream(n=args.n, dim=args.dim, seed=args.stream_seed)
        report = run_suite(
            make_default_systems(dim=args.dim),
            default_properties(),
            stream,
            seeds=tuple(range(args.seeds)),
        )

    _print_report(report)

    if args.out:
        paths = write_report(report, args.out)
        print(f"report json: {paths['json']}")
        print(f"report markdown: {paths['markdown']}")

    smoke_ok = all(
        summary["smoke_pass"] for summary in report["summary"]["systems"].values()
    )
    return 0 if report["preregistration"]["ok"] and smoke_ok else 1


if __name__ == "__main__":
    sys.exit(main())
