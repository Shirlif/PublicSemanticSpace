"""Property tests for the entity-state audit suite (claim-gap E1)."""

from __future__ import annotations

from pss.property.entity_audit import run_entity_audit


def test_entity_audit_properties_hold():
    metrics = run_entity_audit(seeds=3)
    assert metrics["all_pass"], metrics["summary"]
    for key in metrics["pass"]:
        assert metrics["summary"][key]["min"] == 1.0, (key, metrics["summary"][key])
    assert metrics["summary"]["stale_provenance_rate"]["mean"] == 0.0


def test_entity_audit_controls_are_sensitive():
    metrics = run_entity_audit(seeds=2)
    assert metrics["controls_valid"]
    assert metrics["controls"]["no_audit"]["mean"] < 1.0
    assert metrics["controls"]["no_chain"]["mean"] < 1.0
    assert metrics["controls"]["shallow_rollback"]["mean"] < 1.0
