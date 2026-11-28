"""Characterization tests for runtime semantics that are known gaps.

These tests deliberately assert the **current** behaviour, not the ideal one.
They exist so that changing any of these semantics is a conscious, reviewable
decision instead of an accidental side effect.  Each test names the gap it
locks; when a gap is fixed, update the test and the comment together.

Locked gaps:
1. ``PSSRuntime`` defaults to the ``admin`` permission profile.
2. ``PSSRuntime.act`` commits even when an action would hit a hard safety veto,
   because it never forwards the signals the blacklist needs.
3. ``EntityLayer.active_ids(capacity=...)`` ignores the caller's capacity.
4. ``RelationFactLayer.sync`` writes facts even when the TMS labels them OUT.
5. Symbolic relation contradictions are invisible to ``UncertaintyService``.
6. Derived facts are not added to the relation-record set used by later
   derivations.
7. The ``no_tools`` variant is behaviourally identical to ``full``.
8. ``PSSRuntime`` has no planner by default.
"""

from __future__ import annotations

import numpy as np
import pytest

from pss.knowledge.tms_store import TMSFactStore
from pss.memory.perception import Observation
from pss.reasoning.relations import RelationRecord
from pss.runtime import (
    CognitiveState,
    EntityLayer,
    PSSRuntime,
    RelationFactLayer,
    UncertaintyService,
)
from pss.runtime.variants import build_runtime
from pss.safety import GuardrailEvaluator


def _obs(gt: str, pos, category: str = "cup", frame: int = 0) -> Observation:
    return Observation(
        gt_id=gt,
        pos=np.asarray(pos, dtype=float),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        appearance=np.ones(4),
        raw={"vision": np.ones(4), "geometry": np.ones(4), "tactile": np.ones(4)},
        confidence=0.9,
        category=category,
        frame=frame,
        gt_category=category,
    )


def _runtime() -> PSSRuntime:
    return PSSRuntime(use_default_relations=False, use_default_active=False)


# ── 1. default authority ────────────────────────────────────────────────────


def test_default_permissions_are_admin():
    runtime = _runtime()
    assert runtime.permissions.profile.name == "admin"
    assert runtime.permissions.allows("rollback") is True


# ── 2. safety is advisory, and unreachable from act() ───────────────────────


def test_guardrail_hard_veto_works_when_state_is_supplied():
    action = {"kind": "push", "params": {"force": 10.0}}
    verdict = GuardrailEvaluator().evaluate(
        action, {"target_category": "tool"}, candidate={}
    )
    assert verdict.allowed is False
    assert verdict.hard_veto is True


def test_act_commits_an_unsafe_action_because_signals_are_dropped():
    runtime = _runtime()
    action = {"kind": "push", "params": {"force": 10.0}, "category": "tool"}
    before = runtime.state.version
    result = runtime.act(action)
    # The blacklist needs ``target_category`` in safety_state; ``act`` passes
    # an empty state, so the hard veto never fires and the action commits.
    assert result.safety["allowed"] is True
    assert runtime.state.version == before + 1


# ── 3. active-set capacity is fixed ─────────────────────────────────────────


def test_active_ids_ignores_the_capacity_argument():
    layer = EntityLayer(
        use_default_relations=False,
        use_default_active=False,
        active_capacity=4,
    )
    layer.ingest(
        [
            _obs("a", [0.0, 0.0, 0.0]),
            _obs("b", [5.0, 0.0, 0.0]),
            _obs("c", [10.0, 0.0, 0.0]),
        ],
        0,
    )
    assert len(layer.graph.active()) == 3
    # Caller asks for one slot; the layer keeps its constructor capacity.
    assert len(layer.active_ids(capacity=1)) == 3


# ── 4. TMS labels are ignored by sync ──────────────────────────────────────


def test_sync_writes_a_fact_even_when_tms_labels_it_out():
    store = TMSFactStore()
    store.assert_fact("cup", "state", "full", source="vision", priority=1)
    # Remove the support without removing the assertion: the store still has
    # the value, but the TMS node is no longer IN.
    store.tms.remove_source_for(("cup", "state", "full"), "vision")
    assert store.resolve("cup", "state") is None
    assert store.label(("cup", "state", "full")) is False

    layer = RelationFactLayer(facts=store, use_default_write=False)
    state = CognitiveState()
    layer.sync(state)
    # ``sync`` selects by priority/timestamp and ignores the label above.
    assert state.get_fact("cup", "state").value == "full"


# ── 5. symbolic relation conflicts are invisible ────────────────────────────


def test_symbolic_relation_conflict_is_not_flagged():
    layer = RelationFactLayer(use_default_write=False)
    state = CognitiveState()
    layer.add_symbolic(
        [
            RelationRecord("a", "left", "b", confidence=0.9),
            RelationRecord("a", "right", "b", confidence=0.9),
        ],
        state,
    )
    report = UncertaintyService().assess(state)
    # Only ``relation_type == "spatial"`` is scanned; these are "symbolic".
    assert "relation_conflict" not in report.flags
    assert report.relation_conflicts == []


# ── 6. derived facts do not feed later derivations ─────────────────────────


def test_derived_facts_are_not_added_to_the_record_set():
    layer = RelationFactLayer(use_default_write=False)
    state = CognitiveState()
    layer.add_symbolic(
        [
            RelationRecord("a", "left", "b"),
            RelationRecord("b", "left", "c"),
        ],
        state,
    )
    before = len(layer.records())
    derived = layer.derive_facts(frame="world", max_rounds=2, min_conf=0.5)
    layer.assert_derived(derived, state)
    # ``assert_derived`` writes to the fact store/state, not to ``_records``.
    assert len(layer.records()) == before


# ── 7. no_tools is not behaviourally distinct ───────────────────────────────


def test_no_tools_variant_is_identical_to_full():
    full = build_runtime("full")
    no_tools = build_runtime("no_tools")
    assert type(no_tools.entities) is type(full.entities)
    assert type(no_tools.relations) is type(full.relations)


# ── 8. planner is not default-wired ─────────────────────────────────────────


def test_planner_is_not_default_wired():
    runtime = _runtime()
    assert runtime.planner is None
    with pytest.raises(RuntimeError, match="no planner"):
        runtime.plan("anything")
