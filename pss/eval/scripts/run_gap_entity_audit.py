"""Claim-gap E1: entity-state audit properties.

Example:
    python -m pss.eval.scripts.run_gap_entity_audit --seeds 5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pss._console import ensure_utf8
from pss.property.entity_audit import run_entity_audit


def _fmt(v: float) -> str:
    return f"{float(v):.3f}"


def _report(metrics: dict) -> str:
    lines = [
        "# Claim-gap E1: entity-state audit",
        "",
        f"> seeds: {metrics['seeds']}, dim: {metrics['dim']}",
        "> L0 structured observations; real `Entity` / `build_entity` / `rebuild_for_slot` paths.",
        "",
        "| property | mean | min | std | threshold | pass |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, row in metrics["summary"].items():
        threshold = metrics["thresholds"].get(key, "n/a")
        ok = metrics["pass"].get(key, "n/a")
        lines.append(
            f"| {key} | {_fmt(row['mean'])} | {_fmt(row['min'])} | "
            f"{_fmt(row['std'])} | {threshold} | {ok} |"
        )
    lines += [
        "",
        f"**All pass: {metrics['all_pass']}**",
        "",
        "## Per-seed",
        "",
        "| seed | provenance | retraction | rollback | rebind | tamper | chain | entries |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["rows"]:
        lines.append(
            f"| {row['seed']} | {_fmt(row['provenance_completeness'])} | "
            f"{_fmt(row['retraction_correctness'])} | {_fmt(row['id_rollback_exactness'])} | "
            f"{_fmt(row['rebind_identity'])} | {_fmt(row['tamper_detection'])} | "
            f"{_fmt(row['audit_chain_valid'])} | {row['n_audit_entries']} |"
        )
    lines += [
        "",
        "## Negative controls",
        "",
        "| control | targeted metric | mean | detected |",
        "|---|---|---:|---|",
    ]
    for name, row in metrics.get("controls", {}).items():
        lines.append(
            f"| {name} | {row['metric']} | {_fmt(row['mean'])} | {row['detected']} |"
        )
    lines += [
        "",
        f"**Controls valid: {metrics.get('controls_valid')}**",
        "",
        "## What this closes",
        "",
        "The cognitive-runtime claim says entity state is inspectable. This suite measures it:",
        "every state change carries provenance, retractions are exact, rebind preserves identity,",
        "rollback is bit-exact, and an edited audit entry is detected. It does not measure",
        "perception; it measures the state layer.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--out", default="reports/claim_gaps/entity_audit")
    args = ap.parse_args()

    metrics = run_entity_audit(seeds=args.seeds, dim=args.dim)
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")

    for key, row in metrics["summary"].items():
        print(f"{key:26s} mean={row['mean']:.3f} min={row['min']:.3f}")
    print(f"all_pass: {metrics['all_pass']}")
    print(f"controls_valid: {metrics.get('controls_valid')}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
