"""Tests for the C5 contact-event -> PSS runtime bridge."""

from __future__ import annotations

import numpy as np
import pytest

from pss.events.layer import (
    apply_events,
    contact_relations,
    event_to_relation,
    relation_metadata,
)
from pss.events.model import ContactEvent, ContactEvidence
from pss.runtime import PermissionDenied, PSSRuntime


def _event(
    event_id="e1",
    subject="a",
    object="b",
    confidence=0.9,
    status="confirmed",
    sources=("force", "tactile"),
    provenance=None,
):
    evidence = [
        ContactEvidence(
            source=source,
            timestamp=0.1 + 0.01 * i,
            score=confidence,
            intensity=2.5,
            location=np.array([0.01, 0.02, 0.03]),
            reliability=0.9,
            metadata={"channel": source},
        )
        for i, source in enumerate(sources)
    ]
    return ContactEvent(
        event_id=event_id,
        subject=subject,
        object=object,
        onset=0.10,
        offset=0.20,
        location=np.array([1.0, 2.0, 3.0]),
        intensity=2.5,
        confidence=confidence,
        status=status,
        evidence=evidence,
        provenance={"channel": "force"} if provenance is None else provenance,
    )


def test_event_to_relation_carries_evidence_without_schema_change():
    event = _event()
    relation = event_to_relation(event)

    assert relation.source == "a"
    assert relation.target == "b"
    assert relation.predicate == "contact"
    assert relation.relation_type == "contact_event"
    assert relation.origin == "contact_event"
    assert relation.frame == "world"
    assert relation.probability == pytest.approx(0.9)

    metadata = relation_metadata(relation)
    assert metadata["event_id"] == "e1"
    assert metadata["status"] == "confirmed"
    assert metadata["onset"] == pytest.approx(0.10)
    assert metadata["offset"] == pytest.approx(0.20)
    assert metadata["location"] == [1.0, 2.0, 3.0]
    assert {row["source"] for row in metadata["evidence"]} == {"force", "tactile"}
    assert metadata["provenance"]["channel"] == "force"
    # The runtime's serialized view is unchanged (seven base fields only).
    assert set(relation.to_dict()) == {
        "source",
        "target",
        "type",
        "predicate",
        "probability",
        "origin",
        "frame",
    }


def test_apply_events_writes_relation_and_audit():
    runtime = PSSRuntime()
    summary = apply_events(runtime, [_event()])

    assert summary["events"] == 1
    assert summary["written"] == 1
    assert summary["decisions"] == {"accept": 1}
    assert summary["statuses"] == {"confirmed": 1}
    assert summary["sources"] == {"force": 1}
    assert summary["before_version"] == 0
    assert summary["version"] == 1
    assert summary["audit_action"] == "contact_event_ingest"

    contacts = contact_relations(runtime.state)
    assert len(contacts) == 1
    relation = contacts[0]
    assert relation.predicate == "contact"
    assert relation.origin == "contact_event"
    assert relation.probability == pytest.approx(0.9)
    assert relation_metadata(relation)["event_id"] == "e1"
    assert runtime.state.query_relations(predicate="contact") == contacts
    # The relation-fact layer mirrors the edge in its record cache.
    assert any(record.predicate == "contact" for record in runtime.relations.records())

    assert runtime.state.version == 1
    assert len(runtime.state.audit) == 1
    entry = runtime.state.audit[0]
    assert entry.action == "contact_event_ingest"
    assert entry.target == "contact_events:1"
    assert entry.actor == "contact_event"
    assert entry.after["events"] == 1
    assert entry.after["written"] == 1
    assert entry.after["statuses"] == {"confirmed": 1}
    assert entry.after["sources"] == {"force": 1}
    assert entry.after["recommendation"] == "accept"
    assert entry.after["entries"][0]["event_id"] == "e1"

    report = runtime.state.uncertainty
    assert report is not None
    assert report.recommendation == "accept"


def test_low_confidence_event_goes_to_review():
    runtime = PSSRuntime()
    summary = apply_events(runtime, [_event(confidence=0.3)])

    assert summary["decisions"] == {"review": 1}
    assert summary["written"] == 1
    contacts = contact_relations(runtime.state)
    assert len(contacts) == 1  # review is written with its low probability
    assert contacts[0].probability == pytest.approx(0.3)

    report = runtime.state.uncertainty
    assert report is not None
    assert report.recommendation == "review"
    assert "contact_event_review" in report.flags
    assert summary["recommendation"] == "review"


def test_very_low_confidence_event_abstains_without_asserting():
    runtime = PSSRuntime()
    summary = apply_events(runtime, [_event(confidence=0.1)])

    assert summary["decisions"] == {"abstain": 1}
    assert summary["written"] == 0
    assert contact_relations(runtime.state) == []

    report = runtime.state.uncertainty
    assert report is not None
    assert report.recommendation == "abstain"
    assert "contact_event_abstain" in report.flags
    assert summary["recommendation"] == "abstain"
    entry = runtime.state.audit[-1]
    assert entry.action == "contact_event_ingest"
    assert entry.after["events"] == 1
    assert entry.after["written"] == 0
    assert entry.after["entries"][0]["decision"] == "abstain"


def test_contradicted_and_expired_events_are_not_asserted():
    runtime = PSSRuntime()
    events = [
        _event("e-contra", confidence=0.99, status="contradicted"),
        _event("e-expired", confidence=0.99, status="expired"),
    ]
    summary = apply_events(runtime, events)

    assert summary["decisions"] == {"rejected": 2}
    assert summary["written"] == 0
    assert summary["statuses"] == {"contradicted": 1, "expired": 1}
    assert contact_relations(runtime.state) == []
    assert runtime.state.uncertainty.recommendation == "accept"


def test_mixed_events_counts_and_state():
    runtime = PSSRuntime()
    events = [
        _event("e-accept", confidence=0.9, status="confirmed"),
        _event("e-review", subject="b", object="c", confidence=0.4),
        _event("e-abstain", subject="c", object="d", confidence=0.05),
        _event("e-rejected", subject="d", object="e", confidence=0.9, status="expired"),
    ]
    summary = apply_events(runtime, events)

    assert summary["decisions"] == {
        "accept": 1,
        "review": 1,
        "abstain": 1,
        "rejected": 1,
    }
    assert summary["written"] == 2
    assert len(contact_relations(runtime.state)) == 2
    assert runtime.state.uncertainty.recommendation == "abstain"


def test_rollback_restores_pre_ingest_state_and_keeps_audit():
    runtime = PSSRuntime()
    runtime.assert_fact("cup", "location", "A")
    before = runtime.state.version
    assert before == 1

    apply_events(runtime, [_event()])
    assert runtime.state.version == 2
    assert len(contact_relations(runtime.state)) == 1

    rolled = runtime.rollback(before)
    assert rolled == before
    assert runtime.state.version == before
    assert contact_relations(runtime.state) == []
    assert runtime.state.get_fact("cup", "location").value == "A"
    # Audit is append-only: the ingest entry survives the rollback.
    assert any(e.action == "contact_event_ingest" for e in runtime.state.audit)
    assert runtime.state.audit[-1].action == "rollback"


def test_commit_false_writes_without_audit_or_version():
    runtime = PSSRuntime()
    summary = apply_events(runtime, [_event()], commit=False)

    assert summary["audit_action"] is None
    assert summary["version"] == 0
    assert runtime.state.version == 0
    assert runtime.state.audit == []
    assert len(contact_relations(runtime.state)) == 1


def test_empty_events_is_a_noop():
    runtime = PSSRuntime()
    summary = apply_events(runtime, [])

    assert summary["events"] == 0
    assert summary["written"] == 0
    assert summary["version"] == 0
    assert summary["audit_action"] is None
    assert runtime.state.relations == []
    assert runtime.state.audit == []


def test_invalid_thresholds_are_rejected_before_writing():
    runtime = PSSRuntime()
    with pytest.raises(ValueError):
        apply_events(runtime, [_event()], abstain_threshold=0.7, review_threshold=0.5)
    assert runtime.state.relations == []
    assert runtime.state.audit == []


def test_permission_denied_does_not_write_contact_relations():
    runtime = PSSRuntime(permissions="read_only")
    with pytest.raises(PermissionDenied):
        apply_events(runtime, [_event()])
    assert contact_relations(runtime.state) == []
    assert runtime.state.audit[-1].action == "permission_denied"


def test_audit_disabled_writes_relations_without_audit():
    runtime = PSSRuntime(audit=False)
    summary = apply_events(runtime, [_event()])

    assert summary["audit_action"] is None
    assert summary["version"] == 0
    assert runtime.state.version == 0
    assert runtime.state.audit == []
    assert len(contact_relations(runtime.state)) == 1


def test_runtime_defaults_are_unchanged_by_the_contact_layer():
    runtime = PSSRuntime()
    assert runtime.state.relations == []
    assert runtime.relations.records() == []
    assert not contact_relations(runtime.state)

    runtime.assert_fact("cup", "location", "A")
    assert runtime.state.version == 1
    assert runtime.state.get_fact("cup", "location").value == "A"
    assert contact_relations(runtime.state) == []
    assert runtime.relations.rule_set.axis_mode == "off"
