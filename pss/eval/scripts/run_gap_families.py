"""Claim-gap E11: contract across more expert families.

The L1 suite already swaps to one alternative family.  This script repeats the
frozen-space family swap and few-shot calibration for three independently
trained expert families:

- ``category_pretrain``
- ``category_pretrain_instance``
- ``category_pretrain_instance8``

All three output the same vision/tactile widths as the base expert, so the
frozen projection heads can be reused.  The question is whether the contract
holds across families, not which family is more accurate.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.property.l1 import (
    load_l1_assets,
    run_l1_family_calibration,
    run_l1_family_swap,
)

FAMILIES = (
    ("category_pretrain", "data/category_pretrain", "checkpoints/category_pretrain_experts"),
    (
        "category_pretrain_instance",
        "data/category_pretrain_instance",
        "checkpoints/category_pretrain_instance_experts",
    ),
    (
        "category_pretrain_instance8",
        "data/category_pretrain_instance8",
        "checkpoints/category_pretrain_instance8_experts",
    ),
)


def run(
    base_data: str = "data/expert_v1",
    base_experts: str = "checkpoints/expert_v1",
    base_projections: str = "checkpoints/projection_v1",
    base_prototypes: str = "checkpoints/prototypes_v1",
    max_per_class: int = 40,
    ks=(0, 1, 5),
    seed: int = 0,
) -> dict:
    assets = load_l1_assets(
        data_dir=base_data,
        expert_dir=base_experts,
        projection_dir=base_projections,
        prototype_dir=base_prototypes,
        max_samples=400,
        split="test",
    )
    rows = []
    for name, data_dir, expert_dir in FAMILIES:
        swap = run_l1_family_swap(
            assets,
            alt_data_dir=data_dir,
            alt_expert_dir=expert_dir,
            max_per_class=max_per_class,
            seed=seed,
            split="test",
        )
        calibration = run_l1_family_calibration(
            assets,
            alt_data_dir=data_dir,
            alt_expert_dir=expert_dir,
            ks=tuple(ks),
            max_test_per_class=max_per_class,
        )
        rows.append(
            {
                "family": name,
                "data": data_dir,
                "experts": expert_dir,
                "n_samples": swap["n_samples"],
                "chance": swap["chance"],
                "ZCR_vision": swap["ZCR_vision"],
                "ZCR_tactile": swap["ZCR_tactile"],
                "calibration": calibration["per_modality"],
                "n_test": calibration["n_test"],
            }
        )
    return {
        "base": {
            "data": base_data,
            "experts": base_experts,
            "projections": base_projections,
            "prototypes": base_prototypes,
        },
        "ks": list(ks),
        "rows": rows,
        "summary": {
            "vision_zero_shot_mean": float(np.mean([r["ZCR_vision"] for r in rows])),
            "tactile_zero_shot_mean": float(np.mean([r["ZCR_tactile"] for r in rows])),
            "vision_k1_mean": float(
                np.mean([r["calibration"]["vision"].get("1", np.nan) for r in rows])
            ),
            "tactile_k1_mean": float(
                np.mean([r["calibration"]["tactile"].get("1", np.nan) for r in rows])
            ),
        },
    }


def _report(m: dict) -> str:
    lines = [
        "# Claim-gap E11: contract across expert families",
        "",
        f"> base: `{m['base']['experts']}` + `{m['base']['projections']}`",
        f"> ks: {m['ks']}",
        "",
        "| family | samples | chance | vision ZCR | tactile ZCR | vision k=1 | tactile k=1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in m["rows"]:
        v1 = r["calibration"]["vision"].get("1", float("nan"))
        t1 = r["calibration"]["tactile"].get("1", float("nan"))
        lines.append(
            f"| {r['family']} | {r['n_samples']} | {r['chance']:.3f} | "
            f"{r['ZCR_vision']:.3f} | {r['ZCR_tactile']:.3f} | {v1:.3f} | {t1:.3f} |"
        )
    lines += [
        "",
        "## Full calibration curves",
        "",
    ]
    for r in m["rows"]:
        lines.append(f"### {r['family']}")
        lines.append("")
        lines.append("| k | vision | tactile |")
        lines.append("|---:|---:|---:|")
        for k in m["ks"]:
            lines.append(
                f"| {k} | {r['calibration']['vision'].get(str(k), float('nan')):.3f} | "
                f"{r['calibration']['tactile'].get(str(k), float('nan')):.3f} |"
            )
        lines.append("")
    lines += [
        "## How to read this",
        "",
        "Same-modality swap should recover with a little calibration (k=1) regardless",
        "of family. Tactile is expected to stay limited by the tactile expert itself.",
        "A family where vision does not recover means the frozen space is not a",
        "calibratable interface for that family.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=40)
    ap.add_argument("--ks", default="0,1,5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/claim_gaps/families")
    args = ap.parse_args()
    metrics = run(
        max_per_class=args.max_per_class,
        ks=tuple(int(x) for x in args.ks.split(",") if x.strip()),
        seed=args.seed,
    )
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    for r in metrics["rows"]:
        print(
            f"{r['family']:30s} vision={r['ZCR_vision']:.3f} "
            f"tactile={r['ZCR_tactile']:.3f} "
            f"v_k1={r['calibration']['vision'].get('1', float('nan')):.3f}"
        )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
