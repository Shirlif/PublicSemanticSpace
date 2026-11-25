#!/usr/bin/env python
"""Run the L1 property suite with real experts and frozen projections.

Usage:
    python scripts/run_property_l1.py --seed 0 --max-samples 400
    python scripts/run_property_l1.py --skip-family
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pss.property.l1 import (  # noqa: E402
    load_l1_assets,
    run_l1_family_calibration,
    run_l1_family_swap,
    run_l1_swap,
)


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# PSS property scorecard — L1",
        "",
        f"- seed: `{report['seed']}`",
        f"- data: `{report['swap']['metadata']['data_dir']}`",
        f"- experts: `{report['swap']['metadata']['expert_dir']}`",
        f"- samples: `{report['swap']['metadata']['n_samples']}`",
        "",
        "## Swap metrics (real embeddings)",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for name, value in report["swap"]["metrics"].items():
        lines.append(f"| {name} | {value:.6g} |")
    lines += ["", "## Quality sweep", "", "| level | target acc | ZCR | ratio vs expected | rollback | swap ok |", "|---|---:|---:|---:|---:|---|"]
    for level, row in report["swap"]["per_level"].items():
        lines.append(
            f"| {level} | {row['accuracy_target']:.2f} | {row['zcr']:.4f} | "
            f"{row.get('zcr_ratio', float('nan')):.3f} | {row['rollback']:.2f} | {row['swap_success']} |"
        )
    if report.get("family"):
        fam = report["family"]
        lines += [
            "",
            "## Expert-family swap (frozen shared space, H7)",
            "",
            f"- alt data: `{fam['alt_data']}`",
            f"- alt experts: `{fam['alt_experts']}`",
            f"- samples: {fam['n_samples']}",
            f"- vision ZCR: {fam.get('ZCR_vision')}",
            f"- tactile ZCR: {fam.get('ZCR_tactile')}",
            f"- chance: {fam.get('chance')}",
        ]
    if report.get("calibration"):
        cal = report["calibration"]
        lines += [
            "",
            "## Few-shot calibration (H7 follow-up)",
            "",
            f"- test samples: {cal['n_test']}",
            f"- chance: {cal['chance']}",
            "",
            "| k | vision ZCR | tactile ZCR |",
            "|---:|---:|---:|",
        ]
        for k in cal["ks"]:
            v = cal["per_modality"]["vision"].get(str(k))
            t = cal["per_modality"]["tactile"].get(str(k))
            lines.append(f"| {k} | {v:.4f} | {t:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/expert_v1")
    parser.add_argument("--experts", default="checkpoints/expert_v1")
    parser.add_argument("--projections", default="checkpoints/projection_v1")
    parser.add_argument("--prototypes", default="checkpoints/prototypes_v1")
    parser.add_argument("--alt-data", default="data/category_pretrain")
    parser.add_argument("--alt-experts", default="checkpoints/category_pretrain_experts")
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--max-per-class", type=int, default=40)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-family", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--ks", default="0,1,5,20")
    parser.add_argument("--out", type=Path, default=Path("scorecards/property"))
    args = parser.parse_args()

    assets = load_l1_assets(
        data_dir=args.data,
        expert_dir=args.experts,
        projection_dir=args.projections,
        prototype_dir=args.prototypes,
        max_samples=args.max_samples,
        split=args.split,
    )
    report = {"seed": args.seed, "swap": run_l1_swap(assets, seed=args.seed)}
    if not args.skip_family:
        report["family"] = run_l1_family_swap(
            assets,
            alt_data_dir=args.alt_data,
            alt_expert_dir=args.alt_experts,
            max_per_class=args.max_per_class,
            seed=args.seed,
            split=args.split,
        )
        if not args.skip_calibration:
            report["calibration"] = run_l1_family_calibration(
                assets,
                alt_data_dir=args.alt_data,
                alt_expert_dir=args.alt_experts,
                ks=tuple(int(x) for x in args.ks.split(",") if x.strip()),
                max_test_per_class=args.max_per_class,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"l1_seed{args.seed}.json"
    md_path = args.out / f"l1_seed{args.seed}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(json.dumps(report["swap"]["metrics"], indent=2))


if __name__ == "__main__":
    main()
