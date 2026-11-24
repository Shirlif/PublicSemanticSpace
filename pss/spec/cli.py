"""Command line entry point for world specs.

    python -m pss.spec.cli lint specs/s1_occlusion.yaml
    python -m pss.spec.cli schema --out specs/schema_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys

from .loader import DEFAULT_REGISTRY, lint_file, register
from .schema import json_schema


def _cmd_lint(args: argparse.Namespace) -> int:
    status = 0
    for path in args.paths:
        result = lint_file(path)
        for warning in result.warnings:
            print(f"[WARN] {path}: {warning}", file=sys.stderr)
        if not result.ok:
            for error in result.errors:
                print(f"[FAIL] {path}: {error}", file=sys.stderr)
            status = 1
            continue

        registered = False
        if not args.no_registry:
            registered = register(result, path, registry=args.registry)

        suffix = "registry=skipped" if args.no_registry else (
            "registry=appended" if registered else "registry=unchanged"
        )
        if args.registry and not args.no_registry:
            suffix += f" ({args.registry})"
        print(f"[OK] {result.spec_id} {result.sha256} {path} {suffix}")
        if args.json:
            print(
                json.dumps(
                    {
                        "spec_id": result.spec_id,
                        "sha256": result.sha256,
                        "path": str(path),
                        "errors": result.errors,
                        "warnings": result.warnings,
                        "registered": registered,
                    },
                    ensure_ascii=False,
                )
            )
    return status


def _cmd_schema(args: argparse.Namespace) -> int:
    text = json.dumps(json_schema(), indent=2, ensure_ascii=False)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pss spec", description="PSS world spec tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    lint = sub.add_parser("lint", help="validate spec files and register their hashes")
    lint.add_argument("paths", nargs="+", metavar="SPEC")
    lint.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY),
        help=f"registry jsonl path (default: {DEFAULT_REGISTRY})",
    )
    lint.add_argument("--no-registry", action="store_true", help="validate only, do not append")
    lint.add_argument("--json", action="store_true", help="also emit a machine-readable line")
    lint.set_defaults(func=_cmd_lint)

    schema = sub.add_parser("schema", help="emit the JSON Schema for spec_version 1")
    schema.add_argument("--out", help="write to a file instead of stdout")
    schema.set_defaults(func=_cmd_schema)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
