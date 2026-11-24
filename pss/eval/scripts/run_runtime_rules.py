"""Row 4: learnable, versioned relation rules.

Fits the geometric relation thresholds on phase7 episodes, compares them with
the hand-written thresholds, exercises version/rollback, and wires the learned
rule set into the runtime's ``RelationFactLayer``.

Example:
    python -m pss.eval.scripts.run_runtime_rules
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.encoders.data import load_index
from pss.reasoning.relations import geometric_relations, precision_recall_f1
from pss.reasoning.relations.rules import RelationRuleSet, _instances, _truth
from pss.runtime import CognitiveState, RelationFactLayer
from pss.sim.dataset import load_sample

OUT = "eval/results/runtime_rules"
DATA = "data/phase7_v2"
RULES_CKPT = "checkpoints/relation_rules/rules.json"
PREDICATES = ("in", "contact", "near")


def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _episodes():
    index = load_index(DATA)
    entries = [e for e in index["entries"] if e.get("kind") == "episode"]
    out = []
    for entry in entries:
        sample = load_sample(Path(DATA) / entry["path"])
        for frame in range(sample["rgb"].shape[0]):
            out.append((sample, frame))
    return out


def _evaluate(rule_fn, episodes):
    counts = {p: {"tp": 0, "fp": 0, "fn": 0} for p in PREDICATES}
    for sample, frame in episodes:
        instances = _instances(sample, frame)
        computed = {r.as_tuple() for r in rule_fn(instances)}
        truth = _truth(sample["relations"][frame])
        for predicate in PREDICATES:
            prf = precision_recall_f1(
                {f for f in computed if f[1] == predicate},
                {f for f in truth if f[1] == predicate},
            )
            for key in counts[predicate]:
                counts[predicate][key] += prf[key]
    out = {}
    for predicate, c in counts.items():
        support = c["tp"] + c["fn"]
        if support == 0 and c["fp"] == 0:
            out[predicate] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "tp": c["tp"],
                "fp": c["fp"],
                "fn": c["fn"],
                "support": 0,
            }
            continue
        p = c["tp"] / max(1, c["tp"] + c["fp"])
        r = c["tp"] / max(1, c["tp"] + c["fn"])
        out[predicate] = {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / max(1e-9, p + r),
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
            "support": support,
        }
    return out


def run() -> dict:
    episodes = _episodes()
    split = max(1, len(episodes) // 2)
    train, test = episodes[:split], episodes[split:]
    hand = RelationRuleSet()
    learned = RelationRuleSet.fit(train)

    Path(RULES_CKPT).parent.mkdir(parents=True, exist_ok=True)
    Path(RULES_CKPT).write_text(
        json.dumps(learned.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    hand_metrics = _evaluate(lambda inst: geometric_relations(inst), test)
    learned_metrics = _evaluate(learned.relations, test)

    # Runtime integration.
    state = CognitiveState()
    layer = RelationFactLayer(rule_set=learned)
    sample, frame = test[0]
    layer.add_geometry(_instances(sample, frame), state)
    components = layer.rule_set.snapshot()

    # Version / rollback.
    snapshot = learned.snapshot()
    version_before = learned.version
    near_before = learned.near_dist
    learned.update(near_dist=min(0.5, near_before + 0.05), reason="rule edit")
    version_after = learned.version
    learned.rollback(snapshot)
    restored = learned.version == version_before and abs(learned.near_dist - near_before) < 1e-9

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "counts": {"train_frames": len(train), "test_frames": len(test)},
        "hand_rules": hand.snapshot(),
        "learned_rules": learned.snapshot(),
        "hand_metrics": hand_metrics,
        "learned_metrics": learned_metrics,
        "runtime": {
            "relations": len(state.relations),
            "rule_component": components,
        },
        "versioning": {
            "version_before": version_before,
            "version_after_update": version_after,
            "restored": restored,
        },
    }
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_rules.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "runtime_rules.md").write_text(render(payload), encoding="utf-8")
    return payload


def render(payload: dict) -> str:
    lines = [
        "# Row 4 — learnable, versioned relation rules",
        "",
        f"> generated: {payload['generated']}",
        f"> frames: train {payload['counts']['train_frames']}, test {payload['counts']['test_frames']}",
        "",
        "| threshold | hand | learned |",
        "|---|---:|---:|",
    ]
    for key in ("near_dist", "contact_pad", "bin_half", "bin_z"):
        lines.append(
            f"| {key} | {_fmt(payload['hand_rules'][key])} | "
            f"{_fmt(payload['learned_rules'][key])} |"
        )
    lines += [
        "",
        "| predicate | hand P/R/F1 | learned P/R/F1 |",
        "|---|---:|---:|",
    ]
    for predicate in PREDICATES:
        h = payload["hand_metrics"][predicate]
        l = payload["learned_metrics"][predicate]
        lines.append(
            f"| {predicate} (support {h['support']}) | "
            f"{_fmt(h['precision'])} / {_fmt(h['recall'])} / {_fmt(h['f1'])} | "
            f"{_fmt(l['precision'])} / {_fmt(l['recall'])} / {_fmt(l['f1'])} |"
        )
    lines += [
        "",
        f"- runtime relation count: {payload['runtime']['relations']}",
        f"- rule version before/after: {payload['versioning']['version_before']} / "
        f"{payload['versioning']['version_after_update']}; rollback restored: "
        f"{payload['versioning']['restored']}",
        "",
        "## Verdict",
        "",
        "- The relation thresholds are now fitted on labelled frames and stored",
        "  with a version; the hand thresholds remain the default/fallback.",
        "- The runtime consumes the learned rule set through `RelationFactLayer`;",
        "  `components.relation_rules` records version/source/thresholds.",
        "- Rule edits are audit-friendly: update increments the version and a",
        "  snapshot restores the previous thresholds.",
        "- This is rule calibration, not a new relation model; the typed relation",
        "  heads remain the learned relation channel from Phase 12/18.",
        "",
    ]
    return "\n".join(lines)


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    payload = run()
    print("hand   ", {p: payload["hand_metrics"][p]["f1"] for p in PREDICATES})
    print("learned", {p: payload["learned_metrics"][p]["f1"] for p in PREDICATES})
    print("rules", payload["learned_rules"])
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
