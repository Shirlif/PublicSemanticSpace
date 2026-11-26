"""Integration tests for the runtime wiring: frames, derivation, planner, history."""

from __future__ import annotations

import numpy as np

from pss.memory.perception import Observation
from pss.memory.scene_graph import SceneGraph, Track
from pss.reasoning.composition import Signature
from pss.reasoning.relations import RelationRecord
from pss.reasoning.relations.inference import derive
from pss.runtime import (
    CognitiveState,
    EntityLayer,
    Fact,
    Goal,
    PSSRuntime,
    RelationFactLayer,
    RuntimePlanner,
)
from pss.world_model.actions import Operator, OperatorLibrary


def test_relation_record_carries_frame():
    record = RelationRecord("a", "left_of", "b", frame="viewer")
    assert record.frame == "viewer"
    assert record.as_tuple() == ("a", "left_of", "b")
    assert record.as_frame_tuple() == ("a", "left_of", "b", "viewer")


def test_derive_groups_by_frame():
    records = [
        RelationRecord("a", "left_of", "b", frame="world"),
        RelationRecord("b", "left_of", "c", frame="world"),
        RelationRecord("x", "left_of", "y", frame="viewer"),
        RelationRecord("y", "left_of", "z", frame="viewer"),
    ]
    result = derive(records)
    assert ("a", "left_of", "c") in result.facts()
    assert ("x", "left_of", "z") in result.facts()
    assert {fact.frame for fact in result.derived} == {"world", "viewer"}

    world = derive(records, frame="world")
    assert {fact.frame for fact in world.derived} == {"world"}
    assert ("x", "left_of", "z") not in world.facts()


def test_cross_frame_chain_is_not_derived():
    records = [
        RelationRecord("a", "left_of", "b", frame="world"),
        RelationRecord("b", "left_of", "c", frame="viewer"),
    ]
    assert derive(records).facts() == set()


def test_relation_fact_layer_asserts_derived_facts():
    layer = RelationFactLayer()
    state = CognitiveState()
    layer.add_symbolic(
        [
            RelationRecord("a", "left_of", "b"),
            RelationRecord("b", "left_of", "c"),
        ],
        state,
    )
    derived = layer.derive_facts(frame="world")
    assert [fact.as_tuple() for fact in derived] == [("a", "left_of", "c")]

    assert layer.assert_derived(derived, state) == 1
    assert layer.resolve("a", "left_of") == "c"
    assert any(relation.origin == "inference" for relation in state.relations)
    assert any(relation.frame == "world" for relation in state.relations)


def test_entity_layer_frame_relation_and_membership():
    graph = SceneGraph()
    yaw90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    graph.add(Track("a", np.array([0.0, 0.0, 0.0]), yaw90))
    graph.add(Track("b", np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])))
    layer = EntityLayer(graph=graph)

    frame = layer.frame_for("a")
    np.testing.assert_allclose(
        frame.apply_direction(np.array([1.0, 0.0, 0.0])),
        np.array([0.0, 1.0, 0.0]),
        atol=1e-9,
    )
    assert layer.relation_in_frame("a", "b", frame_id="a") == ["front_of"]

    track = graph.get("a")
    track.raw = {"vision": np.array([1.0, 0.0, 0.0])}
    bank = {"cup": {"vision": np.array([1.0, 0.0, 0.0])}}
    assert layer.membership("a", ["cup"], bank) > 0.99
    assert layer.type_constraint(["cup"], bank).membership(track.raw) > 0.99


def test_runtime_ingest_derives_relations():
    runtime = PSSRuntime(entity_layer=EntityLayer(), uncertainty_enabled=False)
    observation = Observation(
        gt_id="a",
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        appearance=np.array([1.0, 0.0, 0.0]),
        raw={"vision": np.ones(4), "geometry": np.ones(4), "tactile": np.ones(4)},
        confidence=0.9,
        category="cup",
        frame=0,
        gt_category="cup",
    )
    result = runtime.ingest(
        [observation],
        0,
        relation_records=[
            RelationRecord("a", "left_of", "b"),
            RelationRecord("b", "left_of", "c"),
        ],
        derive_frame="world",
    )
    assert result["derived"] == [("a", "left_of", "c")]


def test_runtime_planner_uses_composition_for_fact_goals():
    library = OperatorLibrary(
        [
            Operator(
                "move",
                frozenset({("robot", "at", "a")}),
                frozenset({("robot", "at", "b")}),
                frozenset({("robot", "at", "a")}),
            )
        ]
    )
    planner = RuntimePlanner(library)
    state = CognitiveState()
    state.set_fact(Fact("robot", "at", "a"))
    goal = Goal("g1", facts=frozenset({("robot", "at", "b")}))

    plan = planner.plan(state, goal)
    assert plan.actions == ["move"]
    assert plan.confidence == 1.0


def test_runtime_planner_typed_cache_uses_goal_id():
    class Step:
        name = "step"
        signature = Signature("start", "done")

        def applicable(self, payload):
            return True

        def apply(self, payload):
            out = dict(payload)
            out["type"] = "done"
            return out

    planner = RuntimePlanner(OperatorLibrary([]))
    goal = Goal("g-typed", facts=frozenset())
    first = planner.plan_typed(
        goal,
        {"type": "start"},
        [Step()],
        lambda payload: payload.get("type") == "done",
    )
    assert first.success
    assert first.actions == ["step"]

    second = planner.plan_typed(
        goal,
        {"type": "start"},
        [Step()],
        lambda payload: payload.get("type") == "done",
    )
    assert second.success
    assert second.reason == "cache"
    assert planner.cache.stats.hits == 1


def test_state_history_tracks_facts_by_version():
    state = CognitiveState()
    state.set_fact(Fact("key", "location", "drawer"))
    state.commit("move", "key")
    state.set_fact(Fact("key", "location", "table"))
    state.commit("move", "key")

    assert state.fact_at(1) == {("key", "location", "drawer")}
    assert state.fact_at(2) == {("key", "location", "table")}
    assert ("key", "location", "drawer") in state.history.removed_between(1.0, 2.0)
    assert state.history.where_was("key", 1.0, location_predicates=("location",)) == {
        "drawer"
    }
