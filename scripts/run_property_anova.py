#!/usr/bin/env python
"""Two-way variance decomposition for §15 Table 4 (P3).

Factors: expert quality (Q0-Q4) x architecture (A0/A4/A5).
Replicates: L1 seeds (3).
A4 uses the measured cross-family zero-shot ZCR; A5 is the oracle/retrain
upper bound.  The script labels this design explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pss.property.stats import two_way_anova  # noqa: E402

QUALITY = ("Q0", "Q1", "Q2", "Q3", "Q4")


def _load_l1(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecards", type=Path, default=Path("scorecards/property"))
    parser.add_argument("--out", type=Path, default=Path("scorecards/property"))
    args = parser.parse_args()

    runs = [_load_l1(p) for p in sorted(args.scorecards.glob("l1_seed*.json"))]
    if not runs:
        raise SystemExit("no l1_seed*.json found")

    records = {metric: [] for metric in ("SSR", "ZCR", "INT", "RBE")}
    for run in runs:
        family = run.get("family") or {}
        a4_zcr = float(family.get("ZCR_vision", 0.0))
        for level in QUALITY:
            row = run["swap"]["per_level"][level]
            for architecture, values in (
                ("A0", {
                    "SSR": 1.0 if row.get("swap_success") else 0.0,
                    "ZCR": float(row.get("zcr", 0.0)),
                    "INT": float(row.get("int", 0.0)),
                    "RBE": float(row.get("rollback", 0.0)),
                }),
                ("A4", {"SSR": 1.0, "ZCR": a4_zcr, "INT": 0.0, "RBE": 0.0}),
                ("A5", {"SSR": 1.0, "ZCR": 1.0, "INT": 0.0, "RBE": 1.0}),
            ):
                for metric, value in values.items():
                    records[metric].append(
                        {
                            "expert_quality": level,
                            "architecture": architecture,
                            metric: value,
                        }
                    )

    anova = {metric: two_way_anova(rows, metric) for metric, rows in records.items()}
    payload = {
        "design": {
            "factors": {"expert_quality": list(QUALITY), "architecture": ["A0", "A4", "A5"]},
            "replicates": len(runs),
            "note": "A4 = measured cross-family zero-shot; A5 = oracle/retrain upper bound",
        },
        "anova": anova,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "anova.json"
    md_path = args.out / "anova.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Two-way variance decomposition",
        "",
        f"- design: 5 quality x 3 architecture x {len(runs)} seeds",
        "- A4 = cross-family zero-shot; A5 = oracle/retrain upper bound",
        "",
        "| metric | SS(expert) | SS(arch) | SS(interaction) | R² | Var share (expert/arch/interaction/error) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for metric, row in anova.items():
        share = row["variance_share"]
        lines.append(
            f"| {metric} | {row['SS']['A']:.4f} | {row['SS']['B']:.4f} | {row['SS']['AB']:.4f} | "
            f"{row['R2']:.4f} | {share['expert']:.3f} / {share['architecture']:.3f} / "
            f"{share['interaction']:.3f} / {share['error']:.3f} |"
        )
    lines += [
        "",
        "读法：SS/方差份额按 balanced two-way ANOVA 计算；A4/A5 是负控制/上界，不是真实训练运行。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
