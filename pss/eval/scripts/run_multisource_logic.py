"""Step 1-2: multi-source facts + versioned rules.

Benchmarks PSS knowledge-layer governance:
- priority resolution under conflicting sources;
- conflict detection;
- cascade delete and rollback;
- rule update without touching facts.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.knowledge import (
    FactStore,
    MultiSourceFactStore,
    Rule,
    RuleEngine,
    VersionedRuleEngine,
)

MATERIALS = ("glass", "ceramic", "metal", "plastic")
LOCATIONS = ("shelf", "floor", "table")
COLORS = ("red", "blue", "green")
ATTRS = ("material", "location", "color")


def _ground_truth(store, entities):
    return {(e, a): store.resolve(e, a) for e in entities for a in ATTRS}


def _flat_accuracy(flat, truth):
    correct = 0
    for key, value in truth.items():
        if flat.get(key) == value:
            correct += 1
    return correct / max(1, len(truth))


def run(out_dir="eval/results/pss_vs_transformer", n_entities=100, seed=0):
    rng = np.random.default_rng(seed)
    entities = [f"obj_{i:03d}" for i in range(n_entities)]
    store = MultiSourceFactStore()
    flat = {}
    sources = (("sensor_a", 0), ("sensor_b", 1), ("sensor_c", 2))
    for entity in entities:
        for attr in ATTRS:
            values = (
                MATERIALS if attr == "material" else LOCATIONS if attr == "location" else COLORS
            )
            for source, priority in sources:
                value = str(rng.choice(values))
                store.assert_fact(entity, attr, value, source, priority)
                flat[(entity, attr)] = value  # last write wins (low priority last if order changed)

    # Inject conflicts: high-priority source disagrees with low-priority source.
    for entity in entities[:30]:
        for attr in ("material", "location"):
            values = MATERIALS if attr == "material" else LOCATIONS
            high, low = rng.choice(values, size=2, replace=False)
            store.assert_fact(entity, attr, str(high), "oracle", priority=10)
            store.assert_fact(entity, attr, str(low), "legacy", priority=-1)
            flat[(entity, attr)] = str(low)

    truth = _ground_truth(store, entities)
    multi_acc = sum(store.resolve(e, a) == truth[(e, a)] for e, a in truth) / len(truth)
    flat_acc = _flat_accuracy(flat, truth)
    conflicts = store.conflicts()

    # Cascade delete + rollback.
    v_before = store.version
    t0 = time.perf_counter()
    for entity in entities[:10]:
        store.delete_entity(entity)
    delete_ms = (time.perf_counter() - t0) * 1000.0
    after_delete = sum(store.resolve(e, a) is None for e in entities[:10] for a in ATTRS)
    survivors = sum(store.resolve(e, a) is not None for e in entities[10:] for a in ATTRS)
    store.rollback(v_before)
    restored = sum(store.resolve(e, a) is not None for e in entities[:10] for a in ATTRS)

    # Rule hot update: fragile = glass -> glass or ceramic.
    base_rules = [
        Rule("fragile", lambda s, e: s.get(e, "material") == "glass", "fragile"),
        Rule(
            "needs_padding",
            lambda s, e: s.get(e, "material") == "glass" and s.get(e, "location") == "shelf",
            "needs_padding",
        ),
    ]
    engine = VersionedRuleEngine(base_rules)
    old_derived = {
        (e, a): engine.answer(store, e, a)
        for e in entities
        for a in ("fragile", "needs_padding")
    }
    t0 = time.perf_counter()
    engine.update(
        Rule(
            "fragile",
            lambda s, e: s.get(e, "material") in ("glass", "ceramic"),
            "fragile",
        ),
        reason="add ceramic",
    )
    engine.update(
        Rule(
            "needs_padding",
            lambda s, e: s.get(e, "material") in ("glass", "ceramic")
            and s.get(e, "location") == "shelf",
            "needs_padding",
        ),
        reason="add ceramic",
    )
    rule_edit_ms = (time.perf_counter() - t0) * 1000.0
    new_derived = {
        (e, a): engine.answer(store, e, a)
        for e in entities
        for a in ("fragile", "needs_padding")
    }
    direct_after = all(store.resolve(e, a) == truth[(e, a)] for e in entities for a in ATTRS)
    changed = sum(old_derived[k] != new_derived[k] for k in old_derived)
    engine.rollback(0)
    rollback_derived = {
        (e, a): engine.answer(store, e, a)
        for e in entities
        for a in ("fragile", "needs_padding")
    }
    rollback_ok = rollback_derived == old_derived

    metrics = {
        "n_entities": n_entities,
        "n_facts": len(truth),
        "multi_source_accuracy": multi_acc,
        "flat_last_write_accuracy": flat_acc,
        "conflict_count": len(conflicts),
        "cascade_delete": {
            "deleted_entities_cleared": after_delete,
            "survivors_kept": survivors,
            "restored_after_rollback": restored,
            "delete_ms": delete_ms,
        },
        "rule_update": {
            "changed_queries": changed,
            "rule_edit_ms": rule_edit_ms,
            "direct_facts_retained": direct_after,
            "rollback_ok": rollback_ok,
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "multisource_logic.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "multisource_logic.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# Multi-source facts + versioned rules",
        "",
        f"- entities: {metrics['n_entities']}, fact slots: {metrics['n_facts']}",
        "",
        "| store | accuracy vs priority resolver |",
        "|---|---:|",
        f"| Multi-source + priority | **{metrics['multi_source_accuracy']:.3f}** |",
        f"| Flat last-write | {metrics['flat_last_write_accuracy']:.3f} |",
        "",
        f"- conflicts detected: {metrics['conflict_count']}",
        f"- cascade delete: {metrics['cascade_delete']['deleted_entities_cleared']} cleared, "
        f"{metrics['cascade_delete']['survivors_kept']} survivors kept, "
        f"{metrics['cascade_delete']['restored_after_rollback']} restored; "
        f"{metrics['cascade_delete']['delete_ms']:.2f} ms",
        "",
        "## Rule hot update",
        "",
        f"- changed derived queries: {metrics['rule_update']['changed_queries']}",
        f"- direct facts retained: {metrics['rule_update']['direct_facts_retained']}",
        f"- rule edit time: {metrics['rule_update']['rule_edit_ms']:.3f} ms",
        f"- rollback exact: {metrics['rule_update']['rollback_ok']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    ap.add_argument("--entities", type=int, default=100)
    args = ap.parse_args()
    m = run(args.out, args.entities)
    print(
        f"multi {m['multi_source_accuracy']:.3f} vs flat {m['flat_last_write_accuracy']:.3f}; "
        f"conflicts {m['conflict_count']}; rule edit {m['rule_update']['rule_edit_ms']:.3f} ms"
    )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
