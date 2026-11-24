"""Phase 6 decomposition-rule demo (plan 7.4).

Shows all four branches:
- witnessed fragmentation -> deterministic;
- result-only, consistent shards -> probabilistic and applied;
- result-only, weak evidence -> uncertain, graph untouched;
- retraction of an applied hypothesis -> parent restored.

Example:
    python -m pss.memory.scripts.run_decomposition_demo
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.memory import DecompositionRule, SceneGraph, Track


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _track(track_id: str, pos, vision, category: str | None = None) -> Track:
    return Track(
        id=track_id,
        pos=np.asarray(pos, dtype=float),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        raw={"vision": _unit(vision)},
        appearance=_unit(vision),
        category=category,
    )


def _strong_case():
    graph = SceneGraph()
    parent = _track("cup", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], "cup")
    graph.add(parent)
    shards = [
        _track("shard0", [0.03, 0.0, 0.0], [1.0, 0.0, 0.0]),
        _track("shard1", [0.04, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ]
    rule = DecompositionRule(graph)
    hypothesis = rule.infer_result_only(parent, shards, t=5)
    return graph, rule, hypothesis


def _weak_case():
    graph = SceneGraph()
    parent = _track("cup", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], "cup")
    graph.add(parent)
    shards = [
        _track("shard0", [0.50, 0.0, 0.0], [0.0, 1.0, 0.0]),
        _track("shard1", [0.51, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ]
    rule = DecompositionRule(graph)
    hypothesis = rule.infer_result_only(parent, shards, t=5)
    return graph, rule, hypothesis


def _witness_case():
    graph = SceneGraph()
    parent = _track("cup", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], "cup")
    graph.add(parent)
    shards = [
        _track("shard0", [0.02, 0.0, 0.0], [1.0, 0.0, 0.0]),
        _track("shard1", [0.03, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ]
    rule = DecompositionRule(graph)
    hypothesis = rule.witness("cup", shards, t=4)
    return graph, rule, hypothesis


def main() -> None:
    ensure_utf8()
    strong_graph, strong_rule, strong = _strong_case()
    strong_status = strong.status
    strong_confidence = strong.confidence
    strong_applied = strong.applied
    shard_relation_before = strong_graph.get("shard0").provenance.get("relation")
    shard_status_before = strong_graph.get("shard0").provenance.get("decomposition_status")
    strong_retracted = strong_rule.retract(
        strong.id, reason="parent observed intact again", t=9
    )

    weak_graph, _weak_rule, weak = _weak_case()
    witness_graph, _witness_rule, witnessed = _witness_case()

    metrics = {
        "witnessed": {
            "status": witnessed.status,
            "confidence": witnessed.confidence,
            "applied": witnessed.applied,
            "parent_status": witness_graph.get("cup").status,
        },
        "result_only_strong": {
            "status_before_retract": strong_status,
            "confidence_before_retract": strong_confidence,
            "applied_before_retract": strong_applied,
            "status_after_retract": strong_retracted.status,
            "confidence_after_retract": strong_retracted.confidence,
            "parent_status_after_retract": strong_graph.get("cup").status,
            "shard_relation_before_retract": shard_relation_before,
            "shard_status_before_retract": shard_status_before,
            "shard_status_after_retract": strong_graph.get("shard0").provenance.get(
                "decomposition_status"
            ),
        },
        "result_only_weak": {
            "status": weak.status,
            "confidence": weak.confidence,
            "applied": weak.applied,
            "parent_status": weak_graph.get("cup").status,
        },
    }

    out = Path("eval/results/decomposition_demo")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Phase 6 decomposition-rule demo",
        "",
        "| branch | status | confidence | applied | parent after |",
        "|---|---|---:|---|---|",
        f"| witnessed | {witnessed.status} | {witnessed.confidence:.3f} | "
        f"{witnessed.applied} | {metrics['witnessed']['parent_status']} |",
        f"| result-only strong | {strong_status} -> {strong_retracted.status} | "
        f"{strong_confidence:.3f} -> {strong_retracted.confidence:.3f} | "
        f"{strong_applied} -> False | "
        f"{metrics['result_only_strong']['parent_status_after_retract']} |",
        f"| result-only weak | {weak.status} | {weak.confidence:.3f} | "
        f"{weak.applied} | {metrics['result_only_weak']['parent_status']} |",
        "",
        "Strong evidence: shards within 4 cm, prototype cosine 1.0. "
        "Weak evidence: shards 50 cm away with orthogonal features. "
        "The applied hypothesis is retracted after the parent is seen intact again.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    for name, row in metrics.items():
        if name == "result_only_strong":
            print(
                f"{name:20s} status={row['status_before_retract']:14s} "
                f"confidence={row['confidence_before_retract']:.3f} "
                f"applied={row['applied_before_retract']} "
                f"-> {row['status_after_retract']}"
            )
        else:
            print(
                f"{name:20s} status={row['status']:14s} "
                f"confidence={row['confidence']:.3f} applied={row['applied']}"
            )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
