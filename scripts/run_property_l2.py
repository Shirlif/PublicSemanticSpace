#!/usr/bin/env python
"""L2 external-validity check on existing real bundles (P4).

The bundle traces expose per-frame entity/fact counts, planner expansions and
latency.  They do not exercise hot-swap or KB edits, so this script checks only
the bounded-inference contract on real replay data.  Swap/KB external validity
is reported by the L1 real-expert suite.

Usage:
    python scripts/run_property_l2.py --bundles bundles/dev
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
import pandas as pd  # noqa: E402

from pss.metrics.property import (  # noqa: E402
    budget_adherence_rate,
    cost_attribution_error,
    percentile,
    summarize,
)


def _per_trace(path: Path, max_entities: int, max_latency_ms: float) -> dict:
    df = pd.read_parquet(path)
    if df.empty:
        return {"path": str(path), "rows": 0}
    entities = df["entities_n"].astype(float).to_numpy()
    latency = df["latency_ms"].astype(float).to_numpy()
    expansions = df["expansions"].astype(float).to_numpy() if "expansions" in df else np.zeros_like(latency)
    facts = df["facts_n"].astype(float).to_numpy() if "facts_n" in df else np.zeros_like(latency)

    pairs = list(zip(entities, [max_entities] * len(entities))) + list(
        zip(latency, [max_latency_ms] * len(latency))
    )
    bar = budget_adherence_rate(pairs)

    # Cost attribution: sum per-layer latency against total latency.
    if "layer" in df:
        grouped = df.groupby("layer")["latency_ms"].sum().to_dict()
        total = float(df["latency_ms"].sum())
        cat = cost_attribution_error(list(grouped.values()), total)
    else:
        grouped = {"all": float(df["latency_ms"].sum())}
        cat = 0.0

    return {
        "path": str(path),
        "rows": int(len(df)),
        "BAR": float(bar),
        "p95_ms": percentile(latency, 95),
        "p99_ms": percentile(latency, 99),
        "max_entities": float(entities.max()),
        "max_latency_ms": float(latency.max()),
        "max_expansions": float(expansions.max()),
        "max_facts": float(facts.max()),
        "CAT": float(cat),
        "layers": {str(k): float(v) for k, v in grouped.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, default=Path("bundles/dev"))
    parser.add_argument("--max-entities", type=int, default=50)
    parser.add_argument("--max-latency-ms", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=Path("scorecards/property"))
    args = parser.parse_args()

    paths = sorted(args.bundles.rglob("pss_trace.parquet"))
    rows = [_per_trace(p, args.max_entities, args.max_latency_ms) for p in paths]
    rows = [r for r in rows if r.get("rows", 0) > 0]
    if not rows:
        raise SystemExit(f"no pss_trace.parquet under {args.bundles}")

    report = {
        "bundles": str(args.bundles),
        "max_entities": args.max_entities,
        "max_latency_ms": args.max_latency_ms,
        "n_traces": len(rows),
        "traces": rows,
        "aggregate": {
            "BAR_mean": float(np.mean([r["BAR"] for r in rows])),
            "BAR_min": float(np.min([r["BAR"] for r in rows])),
            "p99_ms": summarize([r["p99_ms"] for r in rows]),
            "max_entities": float(np.max([r["max_entities"] for r in rows])),
            "max_latency_ms": float(np.max([r["max_latency_ms"] for r in rows])),
            "CAT_mean": float(np.mean([r["CAT"] for r in rows])),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "l2_bundles.json"
    md_path = args.out / "l2_bundles.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# PSS property scorecard — L2 bundle replay",
        "",
        f"- bundles: `{report['bundles']}`",
        f"- traces: {report['n_traces']}",
        f"- bounds: entities ≤ {args.max_entities}, latency ≤ {args.max_latency_ms} ms",
        "",
        "| trace | rows | BAR | p99 ms | max entities | max latency ms | max expansions | max facts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {Path(r['path']).parent.parent.name}/{Path(r['path']).parent.name} | {r['rows']} | "
            f"{r['BAR']:.3f} | {r['p99_ms']:.3f} | {r['max_entities']:.0f} | "
            f"{r['max_latency_ms']:.3f} | {r['max_expansions']:.0f} | {r['max_facts']:.0f} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        f"- BAR mean/min: {report['aggregate']['BAR_mean']:.4f} / {report['aggregate']['BAR_min']:.4f}",
        f"- p99 distribution: {report['aggregate']['p99_ms']}",
        f"- CAT mean: {report['aggregate']['CAT_mean']:.4f}",
        "",
        "> Bundle traces do not exercise hot-swap or KB edits; those are covered by L1.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"traces={len(rows)} BAR_mean={report['aggregate']['BAR_mean']:.4f}")


if __name__ == "__main__":
    main()
