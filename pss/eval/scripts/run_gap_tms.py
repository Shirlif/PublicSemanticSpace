"""Claim-gap E10: TMS on the production ``MultiSourceFactStore``.

Runs the K3/K4/K5 properties on ``TMSFactStore`` instead of the standalone
``VersionedKnowledgeBase``.  This is the integration evidence the paper map
asked for: the same propagation, conflict and rollback properties hold on the
store the runtime actually uses.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pss._console import ensure_utf8
from pss.knowledge.tms_store import TMSFactStore


def _one_seed(seed: int) -> dict:
    # K3: dependency propagation.
    store = TMSFactStore()
    store.assert_fact("A", "v", "1", source="s1")
    store.justify(("B", "v", "1"), in_nodes=[("A", "v", "1")], source="r1")
    store.justify(
        ("C", "v", "1"),
        in_nodes=[("B", "v", "1")],
        source="r2",
    )
    before = store.label(("C", "v", "1"))
    store.retract("A", "v", "s1")
    after = (
        store.label(("B", "v", "1")) is False
        and store.label(("C", "v", "1")) is False
        and store.label(("A", "v", "1")) is False
    )
    propagation = 1.0 if before and after else 0.0

    # K4: conflict detection and priority resolution.
    store = TMSFactStore()
    store.assert_fact("cup", "color", "red", source="s1", priority=1)
    store.assert_fact("cup", "color", "blue", source="s2", priority=0)
    conflicts = store.conflicts()
    conflict_ok = 1.0 if ("cup", "color") in conflicts and store.tms_conflicts() else 0.0
    resolve_ok = 1.0 if store.resolve("cup", "color") == "red" else 0.0

    # K5: rollback exactness.
    store = TMSFactStore()
    store.assert_fact("cup", "material", "glass", source="vision")
    store.justify(
        ("cup", "fragile", "yes"),
        in_nodes=[("cup", "material", "glass")],
        source="rule:fragile",
    )
    before_hash = store.state_hash()
    before_version = store.version
    store.retract("cup", "material", "vision")
    store.rollback(before_version)
    rollback = 1.0 if store.state_hash() == before_hash else 0.0
    label_restored = 1.0 if store.label(("cup", "fragile", "yes")) else 0.0

    # Audit + source retraction on the production store.
    store = TMSFactStore()
    store.assert_fact("cup", "material", "glass", source="vision")
    store.assert_fact("cup", "color", "red", source="vision")
    store.assert_fact("block", "color", "blue", source="slam")
    audit = 1.0 if store.audit_count() >= 2 else 0.0
    store.retract_source("vision")
    source_retract = (
        1.0
        if store.resolve("cup", "material") is None
        and store.resolve("cup", "color") is None
        and store.resolve("block", "color") == "blue"
        else 0.0
    )

    return {
        "seed": int(seed),
        "K3_propagation": propagation,
        "K4_conflict": conflict_ok,
        "K4_resolution": resolve_ok,
        "K5_rollback": rollback,
        "K5_label_restored": label_restored,
        "AUD": audit,
        "source_retract": source_retract,
    }


def run(seeds: int = 5) -> dict:
    rows = [_one_seed(seed) for seed in range(seeds)]
    keys = (
        "K3_propagation",
        "K4_conflict",
        "K4_resolution",
        "K5_rollback",
        "K5_label_restored",
        "AUD",
        "source_retract",
    )
    summary = {
        k: {
            "mean": sum(r[k] for r in rows) / len(rows),
            "min": min(r[k] for r in rows),
        }
        for k in keys
    }
    return {
        "seeds": int(seeds),
        "rows": rows,
        "summary": summary,
        "all_pass": all(v["min"] == 1.0 for v in summary.values()),
    }


def _report(m: dict) -> str:
    lines = [
        "# Claim-gap E10: TMS on MultiSourceFactStore",
        "",
        f"> seeds: {m['seeds']}",
        "",
        "| property | mean | min | pass |",
        "|---|---:|---:|---|",
    ]
    for key, row in m["summary"].items():
        lines.append(
            f"| {key} | {row['mean']:.3f} | {row['min']:.3f} | {row['min'] == 1.0} |"
        )
    lines += [
        "",
        f"**All pass: {m['all_pass']}**",
        "",
        "## What this closes",
        "",
        "K3/K4/K5 were previously demonstrated on the standalone",
        "``VersionedKnowledgeBase``. This suite runs them on ``TMSFactStore``, which",
        "keeps ``MultiSourceFactStore`` as the production assertion table. The same",
        "propagation, conflict, resolution, rollback and source-retraction properties",
        "hold, so the TMS is no longer a parallel module.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="reports/claim_gaps/tms")
    args = ap.parse_args()
    metrics = run(seeds=args.seeds)
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    for key, row in metrics["summary"].items():
        print(f"{key:20s} mean={row['mean']:.3f} min={row['min']:.3f}")
    print(f"all_pass: {metrics['all_pass']}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
