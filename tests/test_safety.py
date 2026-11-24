import numpy as np
import pytest

from pss.safety import (
    GuardrailEvaluator,
    HardSafetyBlacklist,
    ScenarioValuePrototype,
    ScenarioValueStore,
    ValueVector,
)
from pss.safety.audit import AuditLog, AuditRecord
from pss.safety.degradation import HealthLevel, HealthMonitor


def test_hard_safety_blacklist_vetoes():
    blacklist = HardSafetyBlacklist()
    assert not blacklist.check(
        {"kind": "push", "params": {"force": 20.0}, "target_is_human": True},
        {"human_distance": 0.2},
    ).allowed
    assert blacklist.check(
        {"kind": "push", "params": {"force": 2.0}},
        {"human_distance": 1.0, "target_category": "cup"},
    ).allowed
    assert not blacklist.check(
        {"kind": "push", "params": {"force": 8.0}},
        {"human_distance": 1.0, "target_category": "tool"},
    ).allowed


def test_value_vector_and_scenario_modifier():
    vv = ValueVector()
    assert vv.score(
        {"similarity": 1.0, "value_domain": "task_fidelity", "safety_coeff": 1.0}
    ) == pytest.approx(0.6)
    store = ScenarioValueStore(
        [ScenarioValuePrototype("night", {"human_comfort": 0.1})]
    )
    assert store.apply("night", {"human_comfort": 0.85})["human_comfort"] == pytest.approx(0.95)


def test_guardrail_hard_veto_and_safety_coeff():
    guard = GuardrailEvaluator()
    hard = guard.evaluate(
        {"kind": "push", "params": {"force": 20.0}, "target_is_human": True},
        {"human_distance": 0.2},
        {"id": "x", "similarity": 1.0, "value_domain": "task_fidelity"},
    )
    assert hard.hard_veto and not hard.allowed

    unsafe = guard.evaluate(
        {"kind": "push", "params": {"force": 2.0}},
        {"human_distance": 1.0},
        {
            "id": "x",
            "similarity": 1.0,
            "value_domain": "task_fidelity",
            "safety_coeff": 0.0,
        },
    )
    assert not unsafe.allowed
    assert "safety_coeff_zero" in unsafe.reasons


def test_guardrail_threshold_calibration():
    from pss.safety import calibrate_min_score, synthetic_guardrail_cases

    calibration = calibrate_min_score(synthetic_guardrail_cases(seed=0, n=200))
    assert calibration["balanced_accuracy"] > 0.9
    assert calibration["safe_mean_score"] > calibration["unsafe_mean_score"]
    assert 0.0 <= calibration["min_score"] <= 1.0


def test_degradation_fsm_debounce_and_capabilities():
    monitor = HealthMonitor(debounce=2)
    assert monitor.check({}).level == HealthLevel.NORMAL

    monitor.check({"projection_drift": 0.2})
    warning = monitor.check({"projection_drift": 0.2})
    assert warning.level == HealthLevel.WARNING
    assert warning.proposal_allowed
    assert warning.exploration_allowed

    monitor.check({"camera_ok": False})
    degraded = monitor.check({"camera_ok": False})
    assert degraded.level == HealthLevel.DEGRADED
    assert not degraded.proposal_allowed
    assert not degraded.exploration_allowed
    assert degraded.motion_allowed

    monitor.check({"temperature": 90})
    emergency = monitor.check({"temperature": 90})
    assert emergency.level == HealthLevel.EMERGENCY
    assert not emergency.motion_allowed
    assert emergency.manual_required

    monitor.check({})
    normal = monitor.check({})
    assert normal.level == HealthLevel.NORMAL

    offline = monitor.check({"offline": True})
    assert offline.offline and not offline.proposal_allowed


def test_chaos_fault_sequence_degrades_without_crash():
    monitor = HealthMonitor(debounce=2)
    levels = []
    for step in range(120):
        phase = step % 20
        if phase < 10:
            signals = {}
        elif phase < 14:
            signals = {"projection_drift": 0.2}
        elif phase < 17:
            signals = {"camera_ok": False}
        else:
            signals = {"temperature": 90}
        status = monitor.check(signals)
        levels.append(status.level)
        assert status.level in (
            HealthLevel.NORMAL,
            HealthLevel.WARNING,
            HealthLevel.DEGRADED,
            HealthLevel.EMERGENCY,
        )
        if status.level == HealthLevel.EMERGENCY:
            assert not status.motion_allowed
            assert status.manual_required
        if status.offline:
            assert not status.proposal_allowed

    assert HealthLevel.EMERGENCY in levels
    for _ in range(3):
        recovered = monitor.check({})
    assert recovered.level == HealthLevel.NORMAL


def test_audit_replay_recomputes_guardrail(tmp_path):
    action = {"kind": "push", "params": {"force": 2.0}}
    state = {"human_distance": 1.0, "target_category": "cup"}
    candidate = {
        "id": "safe",
        "similarity": 0.9,
        "value_domain": "task_fidelity",
        "safety_coeff": 1.0,
    }
    guard = GuardrailEvaluator().evaluate(action, state, candidate)
    record = AuditRecord(
        observation={"scene": "x"},
        entities=["e1"],
        emotion={"valence": 0.5},
        motivation={"goal": "g"},
        candidates=[candidate],
        decision="safe",
        scores={"final": guard.score},
        value_vector=ValueVector().weights,
        guardrail={"hard_veto": guard.hard_veto, "allowed": guard.allowed},
        health="normal",
        guardrail_inputs={"action": action, "state": state, "candidate": candidate},
    )
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(record)
    replayed = log.replay()
    assert AuditLog.verify(replayed)
    replayed[0]["guardrail"]["hard_veto"] = not replayed[0]["guardrail"]["hard_veto"]
    assert not AuditLog.verify(replayed)


def test_audit_replay_recomputes_plan(tmp_path):
    from pss.world_model.actions import (
        hand_written_plan,
        initial_laundry_facts,
        laundry_goal,
    )

    candidate = {
        "id": "plan",
        "similarity": 1.0,
        "value_domain": "task_fidelity",
        "safety_coeff": 1.0,
    }
    record = AuditRecord(
        observation={"scene": "laundry"},
        entities=["clothes", "washer"],
        emotion={},
        motivation={"goal": "laundry"},
        candidates=[candidate],
        decision="plan",
        scores={"final": 0.6},
        value_vector=ValueVector().weights,
        guardrail={"hard_veto": False},
        health="normal",
        plan_audit={
            "initial_facts": sorted([list(f) for f in initial_laundry_facts()]),
            "plan": hand_written_plan(),
            "goal": sorted([list(f) for f in laundry_goal()]),
            "library": "laundry",
        },
    )
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(record)
    replayed = log.replay()
    assert AuditLog.verify(replayed)
    replayed[0]["plan_audit"]["plan"][0] = "not_an_action"
    assert not AuditLog.verify(replayed)


def test_audit_replay_recomputes_relation_chain(tmp_path):
    from pss.reasoning.relations import relations_from_frame
    from pss.sim.scene import CATEGORY_SPECS

    box_size = list(CATEGORY_SPECS["box"]["size"])
    sample = {
        "poses": np.array(
            [
                [
                    [0.0, 0.0, box_size[2], 1.0, 0.0, 0.0, 0.0],
                    [0.10, 0.0, box_size[2], 1.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=float,
        ),
        "instance_ids": ["box_00", "box_01"],
        "categories": ["box", "box"],
        "object_specs": [
            {"geom_type": "box", "size": box_size},
            {"geom_type": "box", "size": box_size},
        ],
    }
    sizes = {c: spec["size"] for c, spec in CATEGORY_SPECS.items()}
    chain = {
        "poses": sample["poses"].tolist(),
        "instance_ids": sample["instance_ids"],
        "categories": sample["categories"],
        "object_specs": sample["object_specs"],
        "relations": [
            list(fact.as_tuple()) for fact in relations_from_frame(sample, 0, sizes)
        ],
    }
    candidate = {
        "id": "plan",
        "similarity": 1.0,
        "value_domain": "task_fidelity",
        "safety_coeff": 1.0,
    }
    record = AuditRecord(
        observation={"scene": "chain"},
        entities=["box_00", "box_01"],
        emotion={},
        motivation={"goal": "g"},
        candidates=[candidate],
        decision="plan",
        scores={"final": 0.6},
        value_vector=ValueVector().weights,
        guardrail={"hard_veto": False},
        health="normal",
        chain_audit=chain,
    )
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(record)
    replayed = log.replay()
    assert AuditLog.verify(replayed)
    replayed[0]["chain_audit"]["relations"] = []
    assert not AuditLog.verify(replayed)


def test_audit_replay_and_tamper_detection(tmp_path):
    candidates = [
        {"id": "a", "similarity": 0.9, "value_domain": "task_fidelity", "safety_coeff": 1.0},
        {"id": "b", "similarity": 0.8, "value_domain": "human_comfort", "safety_coeff": 1.0},
    ]
    vv = ValueVector()
    scores = {c["id"]: vv.score(c) for c in candidates}
    decision = max(scores, key=scores.get)
    record = AuditRecord(
        observation={"scene": "x"},
        entities=["e1"],
        emotion={"valence": 0.5},
        motivation={"goal": "g"},
        candidates=candidates,
        decision=decision,
        scores={"final": scores[decision], "per_candidate": scores},
        value_vector=vv.weights,
        guardrail={"hard_veto": False},
        health="normal",
    )
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(record)
    replayed = log.replay()
    assert AuditLog.verify(replayed)
    replayed[0]["decision"] = "b" if decision == "a" else "a"
    assert not AuditLog.verify(replayed)
