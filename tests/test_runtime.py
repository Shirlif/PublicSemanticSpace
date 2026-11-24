"""Tests for the PSS cognitive runtime (state, layers, services, outputs)."""

from __future__ import annotations

import numpy as np

from pss.memory.perception import Observation
from pss.runtime import (
    CognitiveState,
    Entity,
    EntityLayer,
    Fact,
    Goal,
    OutputRouter,
    PSSRuntime,
    RelationFactLayer,
    RuntimePlanner,
    UncertaintyService,
)
from pss.world_model.actions import Operator, OperatorLibrary


def _obs(gt, pos, category, frame=0, appearance=None):
    return Observation(
        gt_id=gt,
        pos=np.asarray(pos, dtype=float),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        appearance=np.asarray(appearance if appearance is not None else np.eye(3)[0]),
        raw={"vision": np.ones(4), "geometry": np.ones(4), "tactile": np.ones(4)},
        confidence=0.9,
        category=category,
        frame=frame,
        gt_category=category,
    )


def test_state_commit_rollback():
    state = CognitiveState()
    state.add_entity(Entity("a", "cup", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
    state.commit("add_entity", "a")
    state.commit("noop", "x")
    version = state.version
    state.set_fact(Fact("a", "x", 1))
    state.commit("set_fact", "a.x")
    state.rollback(version)
    assert state.get_fact("a", "x") is None
    assert state.version == version


def test_entity_layer_persistence_and_sync():
    layer = EntityLayer()
    layer.ingest([_obs("cup", [0, 0, 0], "cup")], 0)
    layer.ingest([], 1)  # one missed frame
    state = CognitiveState()
    layer.sync(state)
    assert len(state.entities) == 1
    entity = next(iter(state.entities.values()))
    assert entity.category == "cup"
    assert entity.status in ("active", "dormant")


def test_relation_fact_conflict_and_uncertainty():
    facts = RelationFactLayer()
    facts.assert_fact("cup", "location", "A", source="vision", priority=0)
    facts.assert_fact("cup", "location", "B", source="human", priority=2)
    assert ("cup", "location") in facts.conflicts()
    assert facts.resolve("cup", "location") == "B"
    state = CognitiveState()
    facts.sync(state)
    assert state.get_fact("cup", "location").value == "B"
    report = UncertaintyService().assess(state, fact_conflicts=facts.conflicts())
    assert "fact_conflict" in report.flags
    assert report.recommendation in ("review", "abstain")


def _library(ids=("cup",)):
    ops = []
    for tid in ids:
        for src in ("A", "B", "G"):
            for dst in ("A", "B", "G"):
                if src == dst:
                    continue
                ops.append(
                    Operator(
                        f"move_{tid}_{src}_{dst}",
                        frozenset({(tid, "location", src)}),
                        frozenset({(tid, "location", dst)}),
                        frozenset({(tid, "location", src)}),
                    )
                )
    return OperatorLibrary(ops)


def test_runtime_plan_explain_rollback():
    runtime = PSSRuntime(planner=RuntimePlanner(_library()))
    runtime.state.add_entity(
        Entity("cup", "cup", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    )
    runtime.assert_fact("cup", "location", "A", source="perception")
    runtime.set_goal(
        Goal("g", frozenset({("cup", "location", "G")}), "cup to G")
    )
    plan = runtime.plan("g")
    assert plan.actions
    result = runtime.explain("cup 的 location 是什么？")
    assert "location" in result.language or "fact" in result.language.lower()

    before = runtime.state.get_fact("cup", "location").value
    version = runtime.state.version
    runtime.assert_fact("cup", "location", "B", source="bad", priority=3)
    runtime.rollback(version)
    assert runtime.state.get_fact("cup", "location").value == before


def test_audit_disabled_rollback_unavailable():
    runtime = PSSRuntime(audit=False)
    runtime.assert_fact("cup", "location", "A")
    try:
        runtime.rollback(0)
        assert False, "expected rollback to fail"
    except RuntimeError:
        pass
    assert runtime.state.version == 0


def test_output_router_safety_veto():
    router = OutputRouter()
    state = CognitiveState()
    state.add_entity(Entity("cup", "cup", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
    result = router.route(
        state,
        "cup 的 location 是什么？",
        action={"kind": "push", "target_is_human": True},
        safety_state={"human_distance": 0.2},
    )
    assert result.safety["allowed"] is False
    assert result.safety["hard_veto"] is True
