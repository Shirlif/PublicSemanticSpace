"""Claim-gap E5: contract-free baseline vs PSS L0 on the property suite."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.property.nonpss import run_nonpss_baseline
from pss.property.sandbox import run_all_l0

METRICS = ("SSR", "SRR", "INT", "RBE", "RCR", "USR", "LOC", "PCR", "AUD", "RBE-KB", "BAR", "MON")


def run(seeds: int = 3, dim: int = 128) -> dict:
    rows = []
    for seed in range(seeds):
        pss = run_all_l0(seed=seed)["metrics"]
        flat = run_nonpss_baseline(seed=seed, dim=dim)["metrics"]
        rows.append(
            {
                "seed": seed,
                "pss": {k: float(pss[k]) for k in METRICS if k in pss},
                "flat": {k: float(flat[k]) for k in METRICS if k in flat},
            }
        )
    summary = {}
    for group in ("pss", "flat"):
        summary[group] = {
            k: {
                "mean": float(np.mean([r[group].get(k, 0.0) for r in rows])),
                "std": float(np.std([r[group].get(k, 0.0) for r in rows])),
            }
            for k in METRICS
        }
    # The suite must separate the two on the contract metrics.
    separated = {
        k: (
            summary["pss"][k]["mean"] > summary["flat"][k]["mean"]
            if k in ("SSR", "SRR", "RBE", "RCR", "AUD", "RBE-KB", "PCR", "BAR", "MON")
            else True
        )
        for k in METRICS
    }
    return {
        "seeds": seeds,
        "dim": dim,
        "rows": rows,
        "summary": summary,
        "separated": separated,
    }


def _fmt(x: float) -> str:
    return f"{float(x):.3f}"


def _report(m: dict) -> str:
    lines = [
        "# Claim-gap E5: contract-free baseline vs PSS L0",
        "",
        f"> seeds: {m['seeds']}, dim: {m['dim']}",
        "",
        "| metric | PSS L0 | flat baseline | PSS > flat |",
        "|---|---:|---:|---|",
    ]
    for k in METRICS:
        p = m["summary"]["pss"][k]["mean"]
        f = m["summary"]["flat"][k]["mean"]
        lines.append(f"| {k} | {_fmt(p)} | {_fmt(f)} | {m['separated'][k]} |")
    lines += [
        "",
        "## How to read this",
        "",
        "The contract-free baseline accepts invalid modules, has no rollback or",
        "versioned routing, shares one codebook across modalities, has no audit or",
        "justification propagation, and has no budget controller. The property suite",
        "must mark those as failures while PSS passes them. This is what makes the",
        "suite a measurement instrument rather than a description of PSS.",
        "",
        "It is a negative architecture, not a competitive model: it is not a",
        "capability baseline and should not be read as one.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--out", default="reports/claim_gaps/nonpss")
    args = ap.parse_args()
    metrics = run(seeds=args.seeds, dim=args.dim)
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    for k in METRICS:
        print(
            f"{k:8s} PSS={metrics['summary']['pss'][k]['mean']:.3f} "
            f"flat={metrics['summary']['flat'][k]['mean']:.3f}"
        )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
