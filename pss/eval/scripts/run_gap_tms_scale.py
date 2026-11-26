"""Claim-gap F7: TMS scale at 100 / 1k / 10k facts.

Three measurements:

1. **Snapshot cost.**  Build, update and rollback latency plus retained memory
   for the three ``TMSFactStore`` snapshot modes:
   - ``full``: deep-copy the assertion table and the whole justification graph
     on every change (the original behaviour);
   - ``cow``: copy-on-write shallow snapshots of immutable structures;
   - ``delta``: record only the touched keys/nodes, with a periodic COW
     checkpoint (the runtime default).
2. **Incremental propagation.**  A three-level dependency chain per fact
   (assertion -> d1 -> d2).  Updating one assertion should re-evaluate only the
   dirty subgraph; ``propagate_full()`` is the reference full recomputation.
   Labels from both paths are compared on every sample.
3. **Scale curve.**  The same axes at 100 / 1k / 10k facts so the report can
   name where the superlinear cost comes from instead of only reporting it.

The property metrics (K3/K4/K5/AUD) are covered by ``run_gap_tms``; this script
measures cost, not correctness.  Rollback exactness is still checked for every
case.

Run:
    python -m pss.eval.scripts.run_gap_tms_scale
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

from pss._console import ensure_utf8
from pss.knowledge.facts import Assertion
from pss.knowledge.tms_store import TMSFactStore

MODES = ("full", "cow", "delta")
COUNTS = (100, 1000, 10000)
CHECKPOINT_INTERVAL = 256


def _updates_for(count: int, mode: str) -> int:
    # The full-copy mode is O(N) per update; cap its sample count at 10k so
    # the benchmark stays bounded while still giving a median.
    if mode == "full" and count >= 10000:
        return 5
    return 10


def _build(count: int, mode: str) -> tuple[TMSFactStore, float]:
    assertions = {
        (f"e{i}", "state"): [Assertion("ok", "bulk")] for i in range(count)
    }
    started = time.perf_counter()
    store = TMSFactStore(
        assertions,
        snapshot_mode=mode,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )
    build_ms = (time.perf_counter() - started) * 1000.0
    return store, build_ms


def _target(count: int, step: int) -> int:
    return (step * 7919) % count


def _timing_case(count: int, mode: str) -> dict:
    store, build_ms = _build(count, mode)
    updates = _updates_for(count, mode)
    before_version = int(store.version)
    before_hash = store.state_hash()

    update_ms: list[float] = []
    for step in range(updates):
        target = _target(count, step)
        started = time.perf_counter()
        store.update(
            f"e{target}", "state", f"v{step}", source="bulk", reason="scale"
        )
        update_ms.append((time.perf_counter() - started) * 1000.0)

    started = time.perf_counter()
    store.rollback(before_version)
    rollback_ms = (time.perf_counter() - started) * 1000.0
    # Restoring the graph clears labels; the first query pays a relabel. Time
    # it separately so the rollback number is not hiding that cost.
    started = time.perf_counter()
    store.tms.propagate()
    relabel_ms = (time.perf_counter() - started) * 1000.0
    rollback_exact = 1.0 if store.state_hash() == before_hash else 0.0

    return {
        "facts": count,
        "mode": mode,
        "updates": updates,
        "build_ms": build_ms,
        "update_ms_median": statistics.median(update_ms),
        "update_ms_p90": sorted(update_ms)[max(0, int(0.9 * len(update_ms)) - 1)],
        "rollback_ms": rollback_ms,
        "post_rollback_relabel_ms": relabel_ms,
        "rollback_exact": rollback_exact,
    }


def _memory_case(count: int, mode: str) -> dict:
    updates = _updates_for(count, mode)
    gc.collect()
    tracemalloc.start()
    store, _ = _build(count, mode)
    gc.collect()
    base, peak_build = tracemalloc.get_traced_memory()
    for step in range(updates):
        target = _target(count, step)
        store.update(
            f"e{target}", "state", f"v{step}", source="bulk", reason="scale"
        )
    gc.collect()
    after, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    retained = after - base
    return {
        "facts": count,
        "mode": mode,
        "updates": updates,
        "build_retained_mb": base / 1e6,
        "update_retained_mb": retained / 1e6,
        "update_retained_kb_per_update": retained / max(1, updates) / 1e3,
        "peak_mb": peak / 1e6,
        "build_peak_mb": peak_build / 1e6,
    }


def _propagation_case(count: int, samples: int = 5) -> dict:
    store = TMSFactStore(
        snapshot_mode="delta", checkpoint_interval=CHECKPOINT_INTERVAL
    )
    for i in range(count):
        node = (f"e{i}", "state", "ok")
        store.assert_fact(node[0], node[1], node[2], source="bulk")
        store.justify(
            (f"e{i}", "derived", "d1"),
            in_nodes=[node],
            source="rule:d1",
            rule="d1",
        )
        store.justify(
            (f"e{i}", "derived", "d2"),
            in_nodes=[(f"e{i}", "derived", "d1")],
            source="rule:d2",
            rule="d2",
        )
    started = time.perf_counter()
    store.tms.propagate()
    initial_ms = (time.perf_counter() - started) * 1000.0

    incremental_ms: list[float] = []
    full_ms: list[float] = []
    labels_equal = True
    for step in range(samples):
        target = _target(count, step)
        store.update(
            f"e{target}", "state", f"v{step}", source="bulk", reason="scale"
        )
        started = time.perf_counter()
        incremental = store.tms.propagate(copy_labels=False)
        incremental_ms.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        full = store.tms.propagate_full()
        full_ms.append((time.perf_counter() - started) * 1000.0)
        for node in set(incremental) | set(full):
            if bool(incremental.get(node, False)) != bool(full.get(node, False)):
                labels_equal = False
                break

    inc_median = statistics.median(incremental_ms)
    full_median = statistics.median(full_ms)
    return {
        "facts": count,
        "nodes": 3 * count,
        "initial_propagate_ms": initial_ms,
        "incremental_ms": inc_median,
        "full_ms": full_median,
        "speedup": (full_median / inc_median) if inc_median > 0 else 0.0,
        "labels_equal": 1.0 if labels_equal else 0.0,
    }


def run(
    counts=COUNTS,
    modes=MODES,
    *,
    with_memory: bool = True,
    with_propagation: bool = True,
) -> dict:
    scale = []
    for count in counts:
        for mode in modes:
            row = _timing_case(count, mode)
            if with_memory:
                row.update(
                    {
                        key: value
                        for key, value in _memory_case(count, mode).items()
                        if key not in ("facts", "mode", "updates")
                    }
                )
            scale.append(row)
    propagation = (
        [_propagation_case(count) for count in counts] if with_propagation else []
    )
    return {
        "counts": [int(c) for c in counts],
        "modes": list(modes),
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "scale": scale,
        "propagation": propagation,
    }


def _report(m: dict) -> str:
    lines = [
        "# Claim-gap F7: TMS scale (100 / 1k / 10k facts)",
        "",
        f"> generated: {m.get('generated', '')}",
        f"> checkpoint interval: {m['checkpoint_interval']}",
        "",
        "## Runtime integration",
        "",
        "`RelationFactLayer` (the runtime's fact layer; `EntityLayer` owns object",
        "slots, not facts) now defaults to `TMSFactStore` with",
        "`snapshot_mode='delta'`. The constructor still accepts an injected",
        "`MultiSourceFactStore` or `TMSFactStore`, and `components()['fact_store']`",
        "reports which store is active.",
        "",
        "K3/K4/K5/AUD and the E12/E19/E20 regressions are recorded in",
        "`REGRESSION.md` in this directory.",
        "",
        "## Snapshot cost: build / update / rollback",
        "",
        "`update` is one retract+assert of a single source value (the same path",
        "as `VersionedKnowledgeBase.update`). Times are wall-clock medians over",
        "the per-mode sample count shown in the table.",
        "",
        "| facts | mode | updates | build ms | update ms (median) | update ms (p90) | rollback ms | post-rollback relabel ms | rollback exact |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in m["scale"]:
        lines.append(
            f"| {row['facts']} | {row['mode']} | {row['updates']} | "
            f"{row['build_ms']:.2f} | {row['update_ms_median']:.3f} | "
            f"{row['update_ms_p90']:.3f} | {row['rollback_ms']:.3f} | "
            f"{row['post_rollback_relabel_ms']:.3f} | "
            f"{row['rollback_exact']:.0f} |"
        )
    if any("update_retained_mb" in row for row in m["scale"]):
        lines += [
            "",
            "## Retained memory",
            "",
            "Measured with `tracemalloc`; `update retained` is the additional",
            "memory kept after the update loop, divided by the number of updates.",
            "",
            "| facts | mode | store after build MB | update retained MB | per update KB | peak MB |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for row in m["scale"]:
            if "update_retained_mb" not in row:
                continue
            lines.append(
                f"| {row['facts']} | {row['mode']} | {row['build_retained_mb']:.2f} | "
                f"{row['update_retained_mb']:.2f} | "
                f"{row['update_retained_kb_per_update']:.1f} | {row['peak_mb']:.2f} |"
            )
    if m["propagation"]:
        lines += [
            "",
            "## Incremental vs full propagation",
            "",
            "Each fact is an assertion with a two-level derived chain",
            "(`assertion -> d1 -> d2`), so `nodes = 3 x facts`. One assertion is",
            "updated per sample; the incremental path should touch only that",
            "fact's chain.",
            "",
            "| facts | nodes | initial propagate ms | incremental ms | full ms | speedup | labels equal |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in m["propagation"]:
            lines.append(
                f"| {row['facts']} | {row['nodes']} | "
                f"{row['initial_propagate_ms']:.2f} | {row['incremental_ms']:.3f} | "
                f"{row['full_ms']:.3f} | {row['speedup']:.1f}x | "
                f"{row['labels_equal']:.0f} |"
            )
    lines += [
        "",
        "## Where the superlinear cost comes from",
        "",
        "- **Build.** `MultiSourceFactStore` copies the assertion table once and",
        "  the TMS bulk-loads justifications without per-item history, so build",
        "  is linear in the fact count for all three modes. The constructor's",
        "  initial checkpoint(s) are the only O(N) snapshots.",
        "- **Update, `full`.** Every mutation deep-copies the whole assertion",
        "  table *and* the whole justification graph. An update is a retract plus",
        "  an assert, so it pays four O(N) deep copies: two store snapshots and",
        "  two TMS snapshots. This is the original cost and it is linear per",
        "  update, hence quadratic over a sequence of updates.",
        "- **Update, `cow`.** Same O(N) asymptotic because the snapshot still",
        "  copies one pointer per node, but the constant is much smaller: no",
        "  recursive deep copy of `Assertion` / `Justification` objects.",
        "- **Update, `delta`.** O(changed keys + changed nodes) per update, with a",
        "  COW checkpoint every `checkpoint_interval` changes. The periodic",
        "  checkpoint is the only O(N) term and it amortises to O(N / interval)",
        "  per update.",
        "- **Rollback.** `full` / `cow` restore a snapshot directly (O(N));",
        "  `delta` replays from the nearest checkpoint (O(interval + changed)).",
        "  All modes then clear labels, so the first query after a rollback pays",
        "  an O(N) relabel; the table reports that separately instead of hiding",
        "  it in the rollback number.",
        "- **Propagation.** `propagate_full()` re-evaluates every node, so its",
        "  cost grows with the whole graph. Incremental `propagate()` evaluates",
        "  only the changed nodes and their transitive dependents, so the cost is",
        "  independent of total fact count for a fixed fan-out. This is the",
        "  reason the 10k-fact update latency is dominated by snapshotting, not",
        "  by truth maintenance.",
        "",
        "## How to read this",
        "",
        "The incremental propagation number uses `propagate(copy_labels=False)`.",
        "Materialising every label with the default `propagate()` is an O(N) dict",
        "copy by definition; it is not part of the dirty re-evaluation. A",
        "single-node query (`label()`) also skips that copy.",
        "",
        "The contract properties (K3/K4/K5/AUD) are unchanged; this suite only",
        "moves cost. `delta` is the runtime default. The 10k-fact update cost is",
        "a measured number, not an extrapolation from 1k.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--counts",
        default=",".join(str(c) for c in COUNTS),
        help="comma-separated fact counts",
    )
    ap.add_argument(
        "--modes",
        default=",".join(MODES),
        help="comma-separated snapshot modes",
    )
    ap.add_argument("--out", default="reports/claim_gaps/tms_scale")
    ap.add_argument("--skip-memory", action="store_true")
    ap.add_argument("--skip-propagation", action="store_true")
    args = ap.parse_args()
    counts = [int(x) for x in args.counts.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    metrics = run(
        counts,
        modes,
        with_memory=not args.skip_memory,
        with_propagation=not args.skip_propagation,
    )
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    for row in metrics["scale"]:
        print(
            f"facts={row['facts']:>5} mode={row['mode']:<5} "
            f"build={row['build_ms']:8.2f}ms update={row['update_ms_median']:8.3f}ms "
            f"rollback={row['rollback_ms']:8.3f}ms exact={row['rollback_exact']:.0f}"
        )
    for row in metrics["propagation"]:
        print(
            f"prop facts={row['facts']:>5} inc={row['incremental_ms']:8.3f}ms "
            f"full={row['full_ms']:8.3f}ms speedup={row['speedup']:.1f}x "
            f"equal={row['labels_equal']:.0f}"
        )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
