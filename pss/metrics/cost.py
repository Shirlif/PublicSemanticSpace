"""S8 cost metrics from pss_trace (T38)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def cost_suite(bundles_root: str | Path, suite: str) -> dict[str, Any]:
    root = Path(bundles_root)
    paths = sorted(root.glob(f"**/{suite}/**/seed*/pss_trace.parquet"))
    rows = []
    for path in paths:
        frame = pd.read_parquet(path)
        if not frame.empty:
            rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"no pss_trace for {suite}")
    trace = pd.concat(rows, ignore_index=True)
    latency = trace["latency_ms"].astype(float)
    return {
        "suite": suite,
        "episodes": len(paths),
        "calls": int(len(trace)),
        "latency_ms": {
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
            "mean": float(latency.mean()),
            "max": float(latency.max()),
        },
        "entities_mean": float(trace["entities_n"].mean()),
        "facts_mean": float(trace["facts_n"].mean()),
        "expansions_mean": float(trace["expansions"].mean()),
        "memory_footprint_mb": None,
        "tokens": 0,
        "note": "no LLM in the replay; footprint not instrumented",
    }


def write_costcard(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "S8.json"
    md_path = out / "S8.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# S8 cost (dev)",
        "",
        f"- suite: `{report['suite']}`  episodes: {report['episodes']}  calls: {report['calls']}",
        f"- latency p50: {report['latency_ms']['p50']:.3f} ms",
        f"- latency p95: {report['latency_ms']['p95']:.3f} ms",
        f"- entities mean: {report['entities_mean']:.2f}",
        f"- facts mean: {report['facts_mean']:.2f}",
        f"- memory footprint: {report['memory_footprint_mb']}",
        f"- tokens: {report['tokens']}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
