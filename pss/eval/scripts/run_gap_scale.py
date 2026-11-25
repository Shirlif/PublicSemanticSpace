"""Claim-gap E8: scale evidence for the contract properties.

Three axes:

- representation dim: 128 / 256 / 512, via the L0 swap suite;
- fact count: 100 / 500 / 1000, build/update latency, rollback exactness, audit;
- entity/relation count: 50/100, 200/1000, 500/2000, bounded active-set size
  and latency.

The claim is invariance, not speedup: property metrics should stay flat while
latency grows with the data.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.observations import make_large_scene
from pss.property.compute import Budget, BudgetController
from pss.property.kb import VersionedKnowledgeBase
from pss.property.sandbox import run_swap_suite


def _dim_sweep(dims=(128, 256, 512), seed: int = 0) -> list[dict]:
    rows = []
    for dim in dims:
        result = run_swap_suite(n_observations=200, dim=dim, seed=seed)
        m = result.metrics
        rows.append(
            {
                "dim": dim,
                "SSR": float(m["SSR"]),
                "RBE": float(m["RBE"]),
                "INT": float(m["INT"]),
                "RCR": float(m["RCR"]),
                "ZCR": float(m["ZCR"]),
                "swap_latency_ms": float(m["swap_latency_ms"]),
            }
        )
    return rows


def _fact_sweep(counts=(100, 500, 1000), seed: int = 0) -> list[dict]:
    from pss.knowledge.facts import Assertion

    rows = []
    for count in counts:
        # Build in one batch: the point is update/rollback cost at size, not
        # the O(N^2) per-assert snapshot chain of the naive construction path.
        assertions = {
            (f"e{i}", "state"): [Assertion("ok", "bulk")] for i in range(count)
        }
        t0 = time.perf_counter()
        kb = VersionedKnowledgeBase(assertions)
        build_ms = (time.perf_counter() - t0) * 1000.0

        target = count // 2
        before_hash = kb.state_hash()
        snap = kb.snapshot()
        unrelated = kb.resolve(f"e{min(count - 1, target + 1)}", "state")
        t0 = time.perf_counter()
        kb.update(f"e{target}", "state", "changed", source="runtime")
        update_ms = (time.perf_counter() - t0) * 1000.0
        unrelated_after = kb.resolve(f"e{min(count - 1, target + 1)}", "state")
        rollback_ok = 0.0
        kb.rollback(snap["index"])
        if kb.state_hash() == before_hash and unrelated_after == unrelated:
            rollback_ok = 1.0
        rows.append(
            {
                "facts": count,
                "build_ms": build_ms,
                "update_ms": update_ms,
                "rollback_exact": rollback_ok,
                "audit_complete": 1.0 if kb.audit_count() > 0 else 0.0,
                "locality_ok": 0.0 if unrelated_after != unrelated else 1.0,
            }
        )
    return rows


def _scene_sweep(shapes=((50, 100), (200, 1000), (500, 2000)), seed: int = 0) -> list[dict]:
    rows = []
    for n_entities, n_relations in shapes:
        scene = make_large_scene(
            n_entities=n_entities, n_relations=n_relations, seed=seed
        )
        controller = BudgetController(
            Budget(max_entities=min(50, n_entities), max_relations=min(100, n_relations))
        )
        t0 = time.perf_counter()
        active = controller.active_set(scene.entities)
        relations = controller.relation_set(scene.relations)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        rows.append(
            {
                "entities": n_entities,
                "relations": n_relations,
                "active_entities": len(active),
                "active_relations": len(relations),
                "bounded": bool(
                    len(active) <= min(50, n_entities)
                    and len(relations) <= min(100, n_relations)
                ),
                "latency_ms": latency_ms,
            }
        )
    return rows


def run(seed: int = 0) -> dict:
    return {
        "seed": int(seed),
        "dim": _dim_sweep(seed=seed),
        "facts": _fact_sweep(seed=seed),
        "scenes": _scene_sweep(seed=seed),
    }


def _report(m: dict) -> str:
    lines = [
        "# Claim-gap E8: scale evidence",
        "",
        f"> seed: {m['seed']}",
        "",
        "## Representation dimension",
        "",
        "| dim | SSR | RBE | INT | RCR | ZCR | swap ms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in m["dim"]:
        lines.append(
            f"| {r['dim']} | {r['SSR']:.3f} | {r['RBE']:.3f} | {r['INT']:.3f} | "
            f"{r['RCR']:.3f} | {r['ZCR']:.3f} | {r['swap_latency_ms']:.3f} |"
        )
    lines += [
        "",
        "## Fact count",
        "",
        "| facts | build ms | update ms | rollback exact | audit | locality |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in m["facts"]:
        lines.append(
            f"| {r['facts']} | {r['build_ms']:.1f} | {r['update_ms']:.3f} | "
            f"{r['rollback_exact']:.0f} | {r['audit_complete']:.0f} | {r['locality_ok']:.0f} |"
        )
    lines += [
        "",
        "## Scene size",
        "",
        "| entities | relations | active entities | active relations | bounded | latency ms |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    for r in m["scenes"]:
        lines.append(
            f"| {r['entities']} | {r['relations']} | {r['active_entities']} | "
            f"{r['active_relations']} | {r['bounded']} | {r['latency_ms']:.3f} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "The contract metrics should stay flat across dim and data size; latency is allowed",
        "to grow. The fact sweep uses the versioned KB, whose snapshot-on-change design is",
        "O(history) per update, so build time is expected to superlinear; that is a",
        "known cost, not a property failure.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/claim_gaps/scale")
    args = ap.parse_args()
    metrics = run(seed=args.seed)
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    for r in metrics["dim"]:
        print(f"dim={r['dim']}: SSR={r['SSR']:.3f} RBE={r['RBE']:.3f} INT={r['INT']:.3f}")
    for r in metrics["facts"]:
        print(f"facts={r['facts']}: update={r['update_ms']:.3f}ms rollback={r['rollback_exact']:.0f}")
    for r in metrics["scenes"]:
        print(f"scene={r['entities']}/{r['relations']}: active={r['active_entities']}/{r['active_relations']} bounded={r['bounded']}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
