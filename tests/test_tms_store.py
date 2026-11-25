"""Tests for the TMS-backed production fact store (claim-gap E10)."""

from __future__ import annotations

from pss.knowledge.tms_store import TMSFactStore


def test_assertion_and_retraction_propagate():
    store = TMSFactStore()
    store.assert_fact("cup", "material", "glass", source="vision")
    store.justify(
        ("cup", "fragile", "yes"),
        in_nodes=[("cup", "material", "glass")],
        source="rule:fragile",
        rule="fragile",
    )
    assert store.resolve("cup", "material") == "glass"
    assert store.label(("cup", "fragile", "yes"))

    store.retract("cup", "material", "vision")
    assert store.resolve("cup", "material") is None
    assert not store.label(("cup", "fragile", "yes"))


def test_conflicts_and_priority_resolution():
    store = TMSFactStore()
    store.assert_fact("cup", "color", "red", source="s1", priority=1)
    store.assert_fact("cup", "color", "blue", source="s2", priority=0)
    assert store.resolve("cup", "color") == "red"
    assert ("cup", "color") in store.conflicts()
    assert store.tms_conflicts()


def test_rollback_restores_assertions_and_labels():
    store = TMSFactStore()
    store.assert_fact("cup", "material", "glass", source="vision")
    store.justify(
        ("cup", "fragile", "yes"),
        in_nodes=[("cup", "material", "glass")],
        source="rule:fragile",
    )
    before = store.state_hash()
    version = store.version

    store.retract("cup", "material", "vision")
    assert store.state_hash() != before
    assert not store.label(("cup", "fragile", "yes"))

    store.rollback(version)
    assert store.state_hash() == before
    assert store.resolve("cup", "material") == "glass"
    assert store.label(("cup", "fragile", "yes"))


def test_retract_source_removes_all_assertions_from_that_source():
    store = TMSFactStore()
    store.assert_fact("cup", "material", "glass", source="vision")
    store.assert_fact("cup", "color", "red", source="vision")
    store.assert_fact("block", "color", "blue", source="slam")
    store.retract_source("vision")
    assert store.resolve("cup", "material") is None
    assert store.resolve("cup", "color") is None
    assert store.resolve("block", "color") == "blue"
    assert store.audit_count() >= 3
