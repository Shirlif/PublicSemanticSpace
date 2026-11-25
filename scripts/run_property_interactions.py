#!/usr/bin/env python
"""Run cross-property interaction experiments and negative controls (P3).

Usage:
    python scripts/run_property_interactions.py --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pss.property.controls import run_negative_controls  # noqa: E402
from pss.property.sandbox import run_interaction_suite  # noqa: E402


def _aggregate(rows: list[dict]) -> dict:
    keys = set().union(*(r["metrics"] for r in rows))
    out = {}
    for key in sorted(keys):
        values = [float(r["metrics"][key]) for r in rows if key in r["metrics"]]
        out[key] = {
            "mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "n": len(values),
        }
    return out


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# PSS interaction + negative-control scorecard — P3",
        "",
        f"- seeds: `{report['seeds']}`",
        "",
        "## Interactions",
        "",
        "| metric | mean | min | max |",
        "|---|---:|---:|---:|",
    ]
    for key, row in report["aggregate"].items():
        lines.append(
            f"| {key} | {row['mean']:.4f} | {row['min']:.4f} | {row['max']:.4f} |"
        )
    lines += ["", "## Negative controls", "", "| id | name | expected | passed |", "|---|---|---|---|"]
    for control in report["controls"]["controls"]:
        lines.append(
            f"| {control['id']} | {control['name']} | {control['expected']} | "
            f"{'✅' if control['passed'] else '❌'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out", type=Path, default=Path("scorecards/property"))
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows = [run_interaction_suite(seed=seed).to_dict() for seed in seeds]
    controls = run_negative_controls(seed=seeds[0] if seeds else 0)
    report = {
        "seeds": seeds,
        "rows": rows,
        "aggregate": _aggregate(rows),
        "controls": controls,
        "all_passed": controls["all_passed"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "interactions.json"
    md_path = args.out / "interactions.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"negative controls all passed: {controls['all_passed']}")


if __name__ == "__main__":
    main()
