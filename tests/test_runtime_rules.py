"""Tests for learnable/versioned relation rules (row 4)."""

from __future__ import annotations

import numpy as np

from pss.reasoning.relations.rules import RelationRuleSet
from pss.runtime import CognitiveState, RelationFactLayer


def _instances():
    return {
        "a": {"pos": np.array([0.0, 0.0, 0.0]), "radius": 0.025},
        "b": {"pos": np.array([0.05, 0.0, 0.0]), "radius": 0.025},
    }


def test_rules_generate_records_and_version_roundtrip():
    rules = RelationRuleSet(near_dist=0.2, contact_pad=0.01)
    records = rules.relations(_instances())
    predicates = {r.predicate for r in records}
    assert "near" in predicates
    assert "contact" in predicates

    snapshot = rules.snapshot()
    version = rules.version
    rules.update(near_dist=0.3, reason="test edit")
    assert rules.version == version + 1
    assert rules.near_dist == 0.3
    rules.rollback(snapshot)
    assert rules.version == version
    assert rules.near_dist == 0.2


def test_fit_learns_near_threshold_from_labeled_frame():
    sample = {
        "instance_ids": ["a", "b"],
        "categories": ["cup", "cup"],
        "poses": np.array(
            [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              [0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]]
        ),
        "relations": [[("a", "near", "b")]],
    }
    rules = RelationRuleSet.fit([(sample, 0)])
    assert np.isfinite(rules.near_dist)
    assert rules.near_dist > 0.0
    assert rules.source == "learned"


def test_relation_fact_layer_uses_rule_set_and_reports_version():
    rules = RelationRuleSet(near_dist=0.2, contact_pad=0.01)
    layer = RelationFactLayer(rule_set=rules)
    state = CognitiveState()
    added = layer.add_geometry(_instances(), state)
    assert added > 0
    assert state.relations
    assert layer.rule_set.version == rules.version
    assert layer.rule_set.source == rules.source
