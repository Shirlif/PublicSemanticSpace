"""Aggregate the claim-gap experiments into one report.

Run the experiments first (or pass ``--run``):

    python scripts/run_claim_gaps.py --run
    python scripts/run_claim_gaps.py

Output: ``reports/claim_gaps/REPORT.md``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "claim_gaps"

EXPERIMENTS = {
    "entity_audit": "pss.eval.scripts.run_gap_entity_audit",
    "tactile_binding": "pss.eval.scripts.run_gap_tactile_binding",
    "hybrid_relations": "pss.eval.scripts.run_gap_hybrid_relations",
    "fact_active": "pss.eval.scripts.run_gap_fact_active",
}

ARGS = {
    "entity_audit": ["--seeds", "5"],
    "tactile_binding": ["--sequences", "60", "--seeds", "3"],
    "hybrid_relations": ["--seeds", "3"],
    "fact_active": ["--seeds", "3", "--ks", "4,8,16"],
}


def _load(name: str) -> dict | None:
    path = OUT / name / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_all() -> None:
    for name, module in EXPERIMENTS.items():
        print(f"== {name} ==")
        cmd = [sys.executable, "-m", module, *ARGS[name]]
        subprocess.run(cmd, cwd=ROOT, check=True)


def _render(payloads: dict[str, dict | None]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Claim-gap validation",
        "",
        f"> generated: {now}",
        "> design: `notes/CLAIM_GAP_EXPERIMENTS.md`",
        "> §10 follow-ups: `reports/claim_gaps/FOLLOWUPS.md`",
        "",
        "## Verdict table",
        "",
        "| gap | experiment | key result | status |",
        "|---|---|---|---|",
    ]

    e1 = payloads.get("entity_audit")
    if e1:
        status = "pass" if e1.get("all_pass") and e1.get("controls_valid") else "fail"
        checked = [e1["summary"][k]["min"] for k in e1["pass"]]
        lines.append(
            f"| C1-a auditable state | E1 entity audit | provenance/retraction/rollback/"
            f"tamper all min={min(checked):.2f} | {status} |"
        )
    else:
        lines.append("| C1-a auditable state | E1 entity audit | (not run) | — |")

    e2 = payloads.get("tactile_binding")
    if e2:
        d = e2["deltas"]
        lines.append(
            f"| C1-b tactile confound | E2 tactile ablation | "
            f"tactile {d['tactile_contribution']:+.3f}, state {d['state_contribution']:+.3f} | measured |"
        )
    else:
        lines.append("| C1-b tactile confound | E2 tactile ablation | (not run) | — |")

    e3 = payloads.get("hybrid_relations")
    if e3:
        best = max(e3["noise_levels"])
        v = e3["verdict"][str(best)]
        lines.append(
            f"| C2 hybrid relation | E3 hybrid under noise | "
            f"hybrid_diag vs geometry(tau) at noise={best}: "
            f"{e3['summary'][str(best)]['hybrid_diag']['mean']:.3f} vs "
            f"{e3['summary'][str(best)]['geometry_eps0.25']['mean']:.3f} | "
            f"{'supported' if v['hybrid_diag_gt_geometry_tau'] else 'not supported'} |"
        )
    else:
        lines.append("| C2 hybrid relation | E3 hybrid under noise | (not run) | — |")

    e4 = payloads.get("fact_active")
    if e4:
        k = str(e4["ks"][0])
        row = e4["summary"][k]
        lines.append(
            f"| C3 fact active set | E4 query-conditioned set | "
            f"k={k}: hot {row['hot']['mean']:.3f}, old {row['learned_old']['mean']:.3f}, "
            f"query {row['query_recency']['mean']:.3f}/{row['learned_query']['mean']:.3f} | "
            f"{'fixed' if e4['verdict'][k]['learned_query_beats_hot'] else 'still negative'} |"
        )
    else:
        lines.append("| C3 fact active set | E4 query-conditioned set | (not run) | — |")

    lines += [
        "",
        "## E1: entity-state audit",
        "",
    ]
    if e1:
        lines += [
            "| property | mean | min | pass |",
            "|---|---:|---:|---|",
        ]
        for key, row in e1["summary"].items():
            lines.append(
                f"| {key} | {row['mean']:.3f} | {row['min']:.3f} | "
                f"{e1['pass'].get(key, 'inverse metric')} |"
            )
        lines += ["", "| control | targeted metric | mean | detected |", "|---|---|---:|---|"]
        for name, row in e1.get("controls", {}).items():
            lines.append(f"| {name} | {row['metric']} | {row['mean']:.3f} | {row['detected']} |")
    else:
        lines.append("(not run)")

    lines += ["", "## E2: tactile binding ablation", ""]
    if e2:
        lines += [
            "| variant | ID retention | switches | reappearance |",
            "|---|---:|---:|---:|",
        ]
        for name, row in e2["results"].items():
            o = row["overall"]
            lines.append(
                f"| {name} | {o['id_retention']:.3f} | {o['id_switches']:.2f} | {o['reappearance']:.3f} |"
            )
        lines += ["", "| delta | value |", "|---|---:|"]
        for name, value in e2["deltas"].items():
            lines.append(f"| {name} | {value:+.3f} |")
    else:
        lines.append("(not run)")

    lines += ["", "## E3: hybrid relation under noise", ""]
    if e3:
        noises = [str(n) for n in e3["noise_levels"]]
        lines += [
            "| method | " + " | ".join(f"noise={n}" for n in noises) + " |",
            "|---|" + "---:|" * len(noises),
        ]
        for method in ("geometry_eps0.1", "geometry_eps0.25", "learned", "hybrid", "hybrid_diag"):
            cells = [f"{e3['summary'][n][method]['mean']:.3f}" for n in noises]
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
    else:
        lines.append("(not run)")

    lines += ["", "## E4: query-conditioned fact active set", ""]
    if e4:
        ks = [str(k) for k in e4["ks"]]
        lines += [
            "| method | " + " | ".join(f"k={k}" for k in ks) + " |",
            "|---|" + "---:|" * len(ks),
        ]
        for method in e4["summary"][ks[0]]:
            cells = [f"{e4['summary'][k][method]['mean']:.3f}" for k in ks]
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
    else:
        lines.append("(not run)")

    lines += [
        "",
        "## What changes for the paper",
        "",
        "- **C1**: add an entity-audit property (E1). State the persistence mechanism and the",
        "  signal decomposition separately: E2 shows tactile is the largest sensory signal,",
        "  while the state layer's gain over a no-memory proxy is much larger. Do not claim",
        "  the state layer alone carries identity.",
        "- **C2**: narrow the hybrid claim. Learned relations are a calibration channel only",
        "  inside the regime they were trained on; on aligned/held-out axes, exact geometry",
        "  owns the decision. The naive hybrid (learned on every ambiguous axis) fails.",
        "- **C3**: the old negative result was a query-conditioning bug, not a learning limit.",
        "  Report that the simple query-recency rule matches the learned policy; the claim is",
        "  about the query-conditioned interface, not about learned scoring.",
        "- **C5 (deferred)**: the property suite has not been run on a non-PSS baseline.",
        "",
        "## Files",
        "",
        "- `reports/claim_gaps/entity_audit/`",
        "- `reports/claim_gaps/tactile_binding/`",
        "- `reports/claim_gaps/hybrid_relations/`",
        "- `reports/claim_gaps/fact_active/`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="run all four experiments first")
    args = ap.parse_args()
    if args.run:
        _run_all()
    payloads = {name: _load(name) for name in EXPERIMENTS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text(_render(payloads), encoding="utf-8")
    print(f"output: {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
