"""Tests for Phase 17 integration: learned relations, write gate, operators."""

from __future__ import annotations

import json

import numpy as np

from pss.heads.text import load_local_decoder
from pss.runtime import (
    CognitiveState,
    Entity,
    Fact,
    Goal,
    OutputRouter,
    PSSRuntime,
    RuntimePlanner,
)
from pss.runtime.operators import VersionedOperatorLibrary
from pss.world_model.actions import Operator, OperatorLibrary

REGIONS = {
    "A": (-0.24, 0.00),
    "B": (0.24, 0.00),
    "C": (0.00, 0.18),
    "D": (0.00, -0.18),
    "G": (0.00, 0.00),
}


def _rel(a, b, predicate):
    ax, ay = REGIONS[a]
    bx, by = REGIONS[b]
    return {
        "left": ax < bx,
        "right": ax > bx,
        "above": ay > by,
        "below": ay < by,
    }[predicate]


def _state():
    state = CognitiveState()
    for eid, loc in (("cup", "A"), ("bottle", "D")):
        state.add_entity(Entity(eid, eid, np.zeros(3), np.array([1.0, 0, 0, 0])))
        state.set_fact(Fact(eid, "location", loc))
    return state


def _library(ids=("cup", "bottle")):
    ops = []
    for tid in ids:
        for src in REGIONS:
            for dst in REGIONS:
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


def test_versioned_operators_update_and_rollback():
    library = VersionedOperatorLibrary(list(_library(("cup",)).operators.values()))
    assert "move_cup_A_G" in library.operators
    library.remove("move_cup_A_G", reason="blocked")
    assert "move_cup_A_G" not in library.operators
    library.rollback(1)
    assert "move_cup_A_G" in library.operators
    assert library.version == 1


def test_relational_plan_symbolic():
    state = _state()
    goal = Goal("rel", relation=("bottle", "cup", "left"), description="bottle left of cup")
    planner = RuntimePlanner(_library())
    plan = planner.plan_relational(state, goal, list(REGIONS), _rel)
    assert plan.confidence > 0
    # Apply the plan's actions to the fact state and check the true relation.
    facts = state.fact_set()
    for action in plan.actions:
        prefix, dst = action.rsplit("_", 1)
        tid = prefix[len("move_") :].rsplit("_", 1)[0]
        src = prefix[len("move_") :].rsplit("_", 1)[1]
        facts.discard((tid, "location", src))
        facts.add((tid, "location", dst))
    locations = {f[0]: f[2] for f in facts if f[1] == "location"}
    assert _rel(locations["bottle"], locations["cup"], "left")


def test_write_gate_skips_low_quality_fact():
    runtime = PSSRuntime(audit=True)
    features = np.zeros(8, dtype=np.float32)
    runtime.assert_fact(
        "cup", "location", "A", source="sensor",
        gate=lambda f: "skip", gate_features=features,
    )
    assert runtime.state.get_fact("cup", "location") is None
    assert any(e.action == "fact_rejected" for e in runtime.state.audit)


def test_output_router_uses_decoder_when_provided():
    def decoder(prompt: str) -> str:
        if "TOOL_RESULTS" not in prompt:
            return json.dumps({"tool": "get_objects", "args": {}})
        return "ok"

    state = CognitiveState()
    state.add_entity(Entity("cup", "cup", np.zeros(3), np.array([1.0, 0, 0, 0])))
    result = OutputRouter(decoder=decoder).route(state, "列出所有对象")
    assert result.language == "ok"
    assert result.safety["allowed"] is True


def test_local_decoder_is_optional():
    decoder = load_local_decoder(model_id="nonexistent/model-not-cached")
    assert decoder is None
