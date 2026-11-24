"""Tests for runtime permission profiles (row 7)."""

from __future__ import annotations

import numpy as np
import pytest

from pss.runtime import (
    CognitiveState,
    Entity,
    Fact,
    Goal,
    PSSRuntime,
    PermissionDenied,
    PermissionService,
    RuntimePlanner,
)
from pss.world_model.actions import Operator, OperatorLibrary


def _library():
    ops = []
    for src in ("A", "G"):
        for dst in ("A", "G"):
            if src == dst:
                continue
            ops.append(
                Operator(
                    f"move_cup_{src}_{dst}",
                    frozenset({("cup", "location", src)}),
                    frozenset({("cup", "location", dst)}),
                    frozenset({("cup", "location", src)}),
                )
            )
    return OperatorLibrary(ops)


def _runtime(profile):
    state = CognitiveState()
    state.add_entity(
        Entity("cup", "cup", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    )
    state.set_fact(Fact("cup", "location", "A"))
    return PSSRuntime(
        state=state, planner=RuntimePlanner(_library()), permissions=profile
    )


def test_permission_matrix():
    assert PermissionService("read_only").allows("read")
    assert not PermissionService("read_only").allows("ingest")
    assert PermissionService("service").allows("ingest")
    assert not PermissionService("service").allows("write_facts")
    assert PermissionService("operator").allows("act")
    assert not PermissionService("operator").allows("rollback")
    assert PermissionService("admin").allows("rollback")


def test_read_only_denies_write_and_audits():
    runtime = _runtime("read_only")
    assert runtime.summary()["permissions"]["name"] == "read_only"
    with pytest.raises(PermissionDenied):
        runtime.assert_fact("cup", "location", "B")
    assert any(e.action == "permission_denied" for e in runtime.state.audit)


def test_operator_can_write_plan_act_but_not_rollback():
    runtime = _runtime("operator")
    runtime.assert_fact("cup", "location", "A")
    runtime.set_goal(Goal("g", frozenset({("cup", "location", "G")}), "cup to G"))
    plan = runtime.plan("g")
    assert plan.actions
    runtime.act({"kind": "push", "target_is_human": False})
    with pytest.raises(PermissionDenied):
        runtime.rollback(0)


def test_admin_can_rollback():
    runtime = _runtime("admin")
    version = runtime.state.commit("setup", "init")
    runtime.assert_fact("cup", "location", "B", source="test")
    runtime.rollback(version)
    assert runtime.state.get_fact("cup", "location").value == "A"
