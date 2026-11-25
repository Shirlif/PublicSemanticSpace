#!/usr/bin/env python
"""Run the L0 property suites and write a scorecard.

Usage:
    python scripts/run_property_l0.py --seed 0 --n-observations 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pss.property.sandbox import run_all_l0  # noqa: E402


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# PSS property scorecard — L0",
        "",
        f"- seed: `{report['seed']}`",
        f"- observations: `{report['n_observations']}`",
        "",
        "## Metrics",
        "",
        "| metric | value | threshold | pass |",
        "|---|---:|---:|---|",
    ]
    for name, value in report["metrics"].items():
        threshold = report["thresholds"].get(name)
        passed = report["pass"].get(name)
        lines.append(
            f"| {name} | {value:.6g} | {'' if threshold is None else f'{threshold:.6g}'} | "
            f"{'' if passed is None else ('✅' if passed else '❌')} |"
        )
    lines += ["", "## Suites", ""]
    for suite in report["suites"]:
        lines.append(f"### {suite['name']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(suite["metrics"], indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-observations", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("scorecards/property"))
    args = parser.parse_args()

    report = run_all_l0(n_observations=args.n_observations, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"l0_seed{args.seed}.json"
    md_path = args.out / f"l0_seed{args.seed}.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_markdown(report, md_path)

    failed = [name for name, ok in report["pass"].items() if not ok]
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"metrics: {len(report['metrics'])}  failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
