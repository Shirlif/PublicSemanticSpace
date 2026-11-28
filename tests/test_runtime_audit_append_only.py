"""U5b regression: rollback must not truncate the audit log."""

from __future__ import annotations

import numpy as np

from pss.runtime.state import AuditEntry, CognitiveState, Entity, Fact


def test_rollback_keeps_permission_denied_entry():
    state = CognitiveState()
    state.add_entity(Entity("cup", "cup", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
    state.commit("add_entity", "cup")
    target = state.version

    state.set_fact(Fact("cup", "location", "B"))
    state.commit("set_fact", "cup.location")
    denied = AuditEntry(state.version, "permission_denied", "cup.location", None, None, "service", "denied")
    state.audit.append(denied)

    state.rollback(target)

    assert state.get_fact("cup", "location") is None
    actions = [entry.action for entry in state.audit]
    assert "permission_denied" in actions
    assert actions[-1] == "rollback"
    assert any(entry.reason == f"to v{target}" for entry in state.audit)


def test_rollback_remains_state_exact_and_append_only():
    state = CognitiveState()
    state.add_entity(Entity("cup", "cup", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
    state.commit("add_entity", "cup")
    target = state.version
    state.set_fact(Fact("cup", "location", "B"))
    state.commit("set_fact", "cup.location")

    state.rollback(target)
    assert state.version == target
    assert state.get_fact("cup", "location") is None

    # A later commit must not resurrect the rolled-back fact.
    state.set_fact(Fact("cup", "location", "C"))
    state.commit("set_fact", "cup.location")
    assert state.get_fact("cup", "location").value == "C"
    assert [e.action for e in state.audit].count("rollback") == 1
