"""Tests for the runtime axis-predicate integration (hard + Bayes)."""

from __future__ import annotations

import numpy as np

from pss.reasoning.relations.rules import RelationRuleSet
from pss.reasoning.relations.bayes_boundary import GridBayesBoundary
from pss.reasoning.relations.bayes_predicate import AxisPredicate
from pss.runtime import CognitiveState, RelationFactLayer


def _instances():
    return {
        "a": {"pos": np.array([0.10, 0.0, 0.0]), "radius": 0.05},
        "b": {"pos": np.array([0.0, 0.0, 0.0]), "radius": 0.05},
        "c": {"pos": np.array([0.0, 0.10, 0.0]), "radius": 0.05},
    }


def _fitted_boundary():
    rng = np.random.default_rng(0)
    dx = rng.normal(0, 0.2, 20000)
    dy = rng.normal(0, 0.2, 20000)
    return GridBayesBoundary(eps=0.02, bins=129).fit(dx, dy)


def test_hard_axis_predicate_adds_inverse():
    predicate = AxisPredicate(mode="hard", eps=0.02)
    records = predicate.records(_instances())
    pairs = {(r.subject, r.predicate, r.object) for r in records}
    assert ("a", "right", "b") in pairs
    assert ("b", "left", "a") in pairs
    assert ("c", "above", "b") in pairs
    assert ("b", "below", "c") in pairs


def test_rule_set_axis_mode_is_opt_in():
    instances = _instances()
    default = RelationRuleSet().relations(instances)
    assert not any(r.predicate in ("left", "right", "above", "below") for r in default)
    hard = RelationRuleSet(axis_mode="hard", axis_eps=0.02).relations(instances)
    assert any(r.predicate == "right" for r in hard)


def test_bayes_axis_predicate_uses_prior():
    boundary = _fitted_boundary()
    predicate = AxisPredicate(
        mode="bayes", boundary=boundary, sigma=0.0, eps=0.02
    )
    records = predicate.records(_instances())
    pairs = {(r.subject, r.predicate, r.object) for r in records}
    assert ("a", "right", "b") in pairs
    assert all(0.5 <= r.confidence <= 1.0 for r in records)


def test_relation_fact_layer_writes_axis_relations_with_origin():
    layer = RelationFactLayer(axis_mode="hard", axis_eps=0.02)
    state = CognitiveState()
    added = layer.add_geometry(_instances(), state)
    assert added > 0
    origins = {r.origin for r in state.relations}
    assert "hard_geometry" in origins
    assert any(r.predicate == "right" for r in state.relations)
