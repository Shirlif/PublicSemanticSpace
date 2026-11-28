"""PerceptionService records admission failures as audit events.

The heavy loader is monkeypatched: these tests cover the failure path and the
event log, not the encoders themselves.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pss.runtime.perception as perception_mod
from pss.experts.contract import ExpertValidationError


def _models():
    return SimpleNamespace(weights=None, vision=SimpleNamespace(image_size=64))


def _rejecting_builder(*args, **kwargs):
    raise ExpertValidationError("bad dim: 5 != 6")


def test_rejection_is_recorded_and_falls_back(monkeypatch):
    monkeypatch.setattr(perception_mod, "build_vision_encoder", _rejecting_builder)
    monkeypatch.setattr(
        perception_mod, "load_perception_models", lambda *a, **k: _models()
    )

    service = perception_mod.PerceptionService(
        perception_mod.PerceptionConfig(allow_fallback=True)
    )

    assert service.fallback is True
    assert "bad dim" in service.error
    rejected = [
        e for e in service.admission_events if e["event"] == "expert_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0]["modality"] == "vision"
    assert "bad dim" in rejected[0]["reason"]
    assert any(e["event"] == "admission_fallback" for e in service.admission_events)

    described = service.describe()
    assert described["admission"]["rejected"] == rejected
    assert described["admission"]["admitted"] == []


def test_rejection_raises_when_fallback_disabled(monkeypatch):
    monkeypatch.setattr(perception_mod, "build_vision_encoder", _rejecting_builder)

    with pytest.raises(ExpertValidationError, match="bad dim"):
        perception_mod.PerceptionService(
            perception_mod.PerceptionConfig(allow_fallback=False)
        )


def test_successful_admission_is_recorded(monkeypatch, tmp_path):
    (tmp_path / "vision.pt").write_bytes(b"projection-stub")
    good = SimpleNamespace(out_dim=6, image_size=8)
    monkeypatch.setattr(perception_mod, "build_vision_encoder", lambda *a, **k: good)
    monkeypatch.setattr(perception_mod, "build_tactile_encoder", lambda *a, **k: good)
    monkeypatch.setattr(
        perception_mod, "load_perception_models", lambda *a, **k: _models()
    )

    service = perception_mod.PerceptionService(
        perception_mod.PerceptionConfig(
            allow_fallback=False, projection_dir=str(tmp_path)
        )
    )

    assert service.fallback is False
    admitted = [e for e in service.admission_events if e["event"] == "expert_admitted"]
    assert [e["modality"] for e in admitted] == ["vision", "tactile"]
    assert service.describe()["admission"]["rejected"] == []


def test_describe_tolerates_services_built_without_init():
    service = object.__new__(perception_mod.PerceptionService)
    assert service._admission_summary() == {"events": [], "rejected": [], "admitted": []}


def test_rejection_is_written_to_versioned_audit(monkeypatch):
    from pss.runtime.state import CognitiveState

    state = CognitiveState()
    monkeypatch.setattr(perception_mod, "build_vision_encoder", _rejecting_builder)
    monkeypatch.setattr(
        perception_mod, "load_perception_models", lambda *a, **k: _models()
    )

    service = perception_mod.PerceptionService(
        perception_mod.PerceptionConfig(allow_fallback=True), audit=state
    )

    actions = [entry.action for entry in state.audit]
    assert "expert_rejected" in actions
    assert "admission_fallback" in actions
    rejected = next(e for e in state.audit if e.action == "expert_rejected")
    assert rejected.target == "vision:dinov2"
    assert rejected.actor == "perception"
    assert "bad dim" in rejected.reason
    assert state.version == len(state.audit)
    assert service.admission_events


def test_attach_audit_can_replay_existing_events(monkeypatch):
    from pss.runtime.state import CognitiveState

    monkeypatch.setattr(perception_mod, "build_vision_encoder", _rejecting_builder)
    monkeypatch.setattr(
        perception_mod, "load_perception_models", lambda *a, **k: _models()
    )
    service = perception_mod.PerceptionService(
        perception_mod.PerceptionConfig(allow_fallback=True)
    )
    expected = len(service.admission_events)

    state = CognitiveState()
    service.attach_audit(state, replay=True)

    assert len(state.audit) == expected
    assert service.audit is state


def test_attach_audit_accepts_plain_list_and_callable():
    service = object.__new__(perception_mod.PerceptionService)
    service.admission_events = []
    service.audit = None

    events = []
    service.attach_audit(events.append)
    service._record_event({"event": "expert_admitted", "modality": "vision", "kind": "x"})
    assert events and events[0]["event"] == "expert_admitted"

    sink = []
    service.attach_audit(sink)
    service._record_event({"event": "expert_rejected", "modality": "vision", "kind": "x"})
    assert sink[-1]["event"] == "expert_rejected"

