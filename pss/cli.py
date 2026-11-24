"""Top-level ``pss`` command.

    pss spec lint specs/s1_occlusion.yaml
    pss sim build specs/s1_occlusion.yaml
    pss sim run specs/s1_occlusion.yaml --episodes 1 --seed 0 --device dev
    pss sim verify bundles/dev/s1_occlusion/<hash8>/seed0

The console script is registered by ``pyproject.toml``; without installation use
``python -m pss ...``.
"""

from __future__ import annotations

import sys

from pss import __version__


def _help() -> str:
    return (
        f"pss {__version__}\n"
        "usage: pss <group> <command> [options]\n\n"
        "groups:\n"
        "  spec   lint world specs, write specs/registry.jsonl, export JSON Schema\n"
        "  sim    build USD scenes, run episodes, verify bundles (Isaac Sim)\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(_help())
        return 0
    if args[0] == "--version":
        print(__version__)
        return 0

    group, rest = args[0], args[1:]
    if group == "spec":
        from pss.spec.cli import main as spec_main

        return spec_main(rest)
    if group == "sim":
        from pss.isaac.cli import main as sim_main

        return sim_main(rest)
    print(f"unknown command group {group!r}\n", file=sys.stderr)
    print(_help(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
