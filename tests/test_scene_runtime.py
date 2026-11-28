"""Tests for the S4 3D scene model -> PSS runtime bridge."""

from __future__ import annotations

import numpy as np

from pss.reasoning.relations.bayes_predicate import PREDICATES, AxisPredicate
from pss.runtime import PSSRuntime, RelationFactLayer
from pss.scene.bench_api import BenchConfig, build_examples
from pss.scene.model import ObjectState, SceneModel
from pss.scene.optimizer import solve_scene
from pss.scene.pss_bridge import (
    apply_scene_to_runtime,
    positional_confidence,
    scene_relation_records,
    scene_to_entities,
)


def _scene():
    return SceneModel(
        scene_id="synth0",
        objects={
            "a": ObjectState(
                "a", np.array([0.0, 0.0, 0.35]), shape="cube", radius=0.35
            ),
            "b": ObjectState(
                "b", np.array([0.10, 0.0, 0.35]), shape="sphere", radius=0.35
            ),
            "c": ObjectState(
                "c", np.array([0.80, 0.80, 0.35]), shape="cylinder", radius=0.35
            ),
        },
        table_z=0.0,
    )


def _solve(scene):
    config = BenchConfig(n_views=3, sigma_px=0.0, sigma_depth=0.0, seed=0)
    example = build_examples([scene], config)[0]
    return solve_scene(example, config)


def _records(scene, output, **kwargs):
    return scene_relation_records(
        scene,
        output.positions,
        output.covariances,
        eps_xy=0.12,
        eps_z=0.05,
        eps_c=0.02,
        table_z=0.0,
        **kwargs,
    )


def test_scene_to_entities_carries_pose_covariance_and_provenance():
    scene = _scene()
    output = _solve(scene)
    entities = scene_to_entities(scene, output.positions, output.covariances)

    assert {entity.id for entity in entities} == {"a", "b", "c"}
    for entity in entities:
        cov = np.asarray(entity.attributes["covariance"], dtype=float)
        assert cov.shape == (3, 3)
        assert np.isfinite(cov).all()
        assert entity.attributes["covariance_source"] == "linearised"
        assert entity.provenance["source"] == "scene_model"
        assert entity.provenance["scene_id"] == "synth0"
        assert 0.0 <= entity.confidence <= 1.0
        assert entity.quat.shape == (4,)
    # The solved poses stay close to the synthetic ground truth.
    positions = {entity.id: entity.pos for entity in entities}
    for oid, obj in scene.objects.items():
        assert np.linalg.norm(positions[oid] - obj.pos) < 1e-2


def test_scene_to_entities_without_covariance_is_explicit():
    scene = _scene()
    entities = scene_to_entities(scene, scene.positions(), covariances=None)
    for entity in entities:
        assert entity.attributes["covariance"] == [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        assert entity.attributes["covariance_source"] == "missing"
        assert entity.confidence == 1.0
    assert positional_confidence(np.eye(3) * 1e-6) > 0.9
    assert positional_confidence(np.eye(3) * 1.0) < 0.1


def test_scene_relation_records_cover_3d_predicates_and_override():
    scene = _scene()
    output = _solve(scene)
    records = _records(scene, output)
    triples = {(record.subject, record.predicate, record.object) for record in records}

    # a and b are aligned on x, same level, and their spheres overlap.
    assert ("a", "aligned_x", "b") in triples
    assert ("b", "aligned_x", "a") in triples
    assert ("a", "same_level", "b") in triples
    assert ("a", "contact", "b") in triples
    assert ("a", "collision", "b") in triples
    # Every object rests on the table.
    for oid in ("a", "b", "c"):
        assert (oid, "support", "table") in triples
    # c is far from a/b on both x and y.
    assert ("a", "aligned_y", "c") not in triples
    assert ("a", "aligned_x", "c") not in triples
    assert all(0.0 <= record.confidence <= 1.0 for record in records)
    assert all(record.source == "scene_model" for record in records)

    override = _records(
        scene, output, probabilities={("a", "contact", "b"): 0.123}
    )
    hit = [
        record
        for record in override
        if (record.subject, record.predicate, record.object) == ("a", "contact", "b")
    ]
    assert len(hit) == 1
    assert hit[0].confidence == 0.123


def test_apply_scene_to_runtime_writes_state_and_audit():
    scene = _scene()
    output = _solve(scene)
    runtime = PSSRuntime()
    summary = apply_scene_to_runtime(
        runtime,
        scene,
        output.positions,
        output.covariances,
        eps_xy=0.12,
        eps_z=0.05,
        eps_c=0.02,
        table_z=0.0,
        actor="scene_model",
    )

    assert summary["scene_id"] == "synth0"
    assert summary["entities"] == 3
    assert summary["relations"] > 0
    assert summary["audit_action"] == "scene_ingest"
    assert summary["entity_ids"] == ["a", "b", "c"]
    assert set(runtime.state.entities) == {"a", "b", "c"}
    assert len(runtime.state.relations) == summary["relations"]
    assert all(r.origin == "scene_model" for r in runtime.state.relations)
    assert all(r.relation_type == "scene_geometry" for r in runtime.state.relations)
    assert all(0.0 <= r.probability <= 1.0 for r in runtime.state.relations)
    assert runtime.state.version == summary["version"] == 1
    assert len(runtime.state.audit) == 1
    entry = runtime.state.audit[0]
    assert entry.action == "scene_ingest"
    assert entry.target == "synth0"
    assert entry.actor == "scene_model"
    assert entry.after["entities"] == 3
    assert entry.after["relations"] == summary["relations"]
    assert entry.after["source"] == "scene_model"


def test_apply_scene_without_commit_skips_audit_entry():
    scene = _scene()
    output = _solve(scene)
    runtime = PSSRuntime()
    summary = apply_scene_to_runtime(
        runtime,
        scene,
        output.positions,
        output.covariances,
        commit=False,
    )
    assert summary["audit_action"] is None
    assert runtime.state.version == 0
    assert runtime.state.audit == []
    assert len(runtime.state.entities) == 3


def test_1d_axis_fallback_still_available_and_opt_in():
    predicate = AxisPredicate(mode="hard", eps=0.02)
    records = predicate.records(
        {
            "a": {"pos": np.array([0.0, 0.0, 0.0]), "radius": 0.05},
            "b": {"pos": np.array([0.5, 0.0, 0.0]), "radius": 0.05},
        }
    )
    assert records
    assert {record.predicate for record in records} <= set(PREDICATES)
    assert {record.predicate for record in records} == {"left", "right"}

    # The scene bridge is opt-in and does not switch the runtime axis mode.
    layer = RelationFactLayer()
    assert layer.rule_set.axis_mode == "off"
    runtime = PSSRuntime()
    scene = _scene()
    apply_scene_to_runtime(runtime, scene, scene.positions())
    assert runtime.relations.rule_set.axis_mode == "off"
