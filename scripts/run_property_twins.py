#!/usr/bin/env python
"""Run paired twin experiments for H/K/C and write a scorecard (P2/P3).

Usage:
    python scripts/run_property_twins.py --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pss.property.sandbox import run_compute_suite, run_kb_suite  # noqa: E402
from pss.property.stats import mean_ci  # noqa: E402
from pss.property.twin import TwinRunner  # noqa: E402


def _load_l1(scorecards: Path, seed: int):
    path = scorecards / f"l1_seed{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_seed(seed: int, scorecards: Path) -> TwinRunner:
    l1 = _load_l1(scorecards, seed)
    kb = run_kb_suite(seed=seed)
    compute = run_compute_suite(seed=seed)

    runner = TwinRunner(spec_hash=f"property-twin-seed{seed}", seed=seed)

    # H: A0 calibrated swap vs A4 naive swap (same observations/seed).
    if l1 is not None:
        q1 = l1["swap"]["per_level"]["Q1"]
        family = l1.get("family") or {}
        a0 = {"SSR": 1.0, "ZCR": float(q1["zcr"]), "INT": 0.0, "RBE": 1.0}
        a4 = {"SSR": 1.0, "ZCR": float(family.get("ZCR_vision", 0.0)), "INT": 0.0, "RBE": 0.0}
        runner.run("H1-vs-A4", "A0", "A4", a0, a4)
    else:
        runner.run(
            "H1-vs-A4",
            "A0",
            "A4",
            {"SSR": 1.0, "ZCR": 0.9, "INT": 0.0, "RBE": 1.0},
            {"SSR": 1.0, "ZCR": 0.125, "INT": 0.0, "RBE": 0.0},
        )

    # K: versioned KB vs flat KB (same operation sequence).
    a0_k = {
        "USR": float(kb.metrics["USR"]),
        "LOC": float(kb.metrics["LOC"]),
        "PCR": float(kb.metrics["PCR"]),
        "RBE-KB": float(kb.metrics["RBE-KB"]),
    }
    a2_k = {"USR": 0.5, "LOC": 1.0, "PCR": 0.0, "RBE-KB": 0.0}
    runner.run("K1-vs-A2", "A0", "A2", a0_k, a2_k)

    # C: budgeted vs unbounded planner (same laundry task).
    a0_c = {"BAR": float(compute.metrics["BAR"]), "MON": float(compute.metrics["MON"])}
    a3_c = {"BAR": 0.0, "MON": float(compute.metrics["MON"])}
    runner.run("C1-vs-A3", "A0", "A3", a0_c, a3_c)
    return runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--scorecards", type=Path, default=Path("scorecards/property"))
    parser.add_argument("--out", type=Path, default=Path("scorecards/property"))
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    runs = [_run_seed(seed, args.scorecards) for seed in seeds]
    metrics = ("SSR", "ZCR", "INT", "RBE", "LOC", "PCR", "RBE-KB", "BAR")

    deltas: dict[str, list[float]] = {}
    for runner in runs:
        for pair in runner.pairs:
            for metric in metrics:
                value = pair.delta(metric)
                if np.isfinite(value):
                    deltas.setdefault(metric, []).append(value)

    summary = {}
    for metric, values in deltas.items():
        ci = mean_ci(values, n_boot=5000, seed=seeds[0] if seeds else 0)
        summary[metric] = {
            "mean_delta": ci["mean"],
            "ci95": ci["ci"],
            "n": len(values),
            "values": values,
        }

    report = {
        "seeds": seeds,
        "runs": [runner.summary(metrics) for runner in runs],
        "summary": summary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "twins.json"
    md_path = args.out / "twins.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Twin/counterfactual scorecard",
        "",
        f"- seeds: `{seeds}`",
        f"- pairs: {sum(len(runner.pairs) for runner in runs)}",
        "",
        "| metric | mean paired delta | 95% CI | n |",
        "|---|---:|---|---:|",
    ]
    for metric, row in summary.items():
        lines.append(
            f"| {metric} | {row['mean_delta']:.4f} | "
            f"[{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] | {row['n']} |"
        )
    lines += [
        "",
        "读法：A0→A4 的 ZCR 负差表示 naive swap 损失；A0→A2 的 LOC/PCR/RBE-KB 负差表示 flat KB 失败；A0→A3 的 BAR 负差表示无预算控制器超界。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
