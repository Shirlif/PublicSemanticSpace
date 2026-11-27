"""Tests for relation-algebra consistency checks."""

from __future__ import annotations

import numpy as np
import pytest

from pss.reasoning.relations.algebra import (
    antisymmetry_penalty,
    closure,
    consistency,
    exclusivity_penalty,
    inverse_gap,
    violations,
)


def test_exclusivity_violation_is_reported():
    found = violations({("a", "left", "b"), ("a", "right", "b")})
    assert any(v.kind == "exclusivity" for v in found)


def test_antisymmetry_violation_is_reported():
    found = violations({("a", "left", "b"), ("b", "left", "a")})
    assert any(v.kind == "antisymmetry" for v in found)


def test_transitive_gap_is_reported():
    found = violations({("a", "left", "b"), ("b", "left", "c")})
    gap = [v for v in found if v.kind == "transitive_gap"]
    assert gap and gap[0].detail == "missing ('a', 'left', 'c')"


def test_closure_adds_inverse_and_transitive_consequences():
    closed = closure({("a", "left", "b"), ("b", "left", "c")})
    assert ("a", "left", "c") in closed
    assert ("b", "right", "a") in closed
    assert ("c", "right", "a") in closed


def test_inverse_gap_lists_unpaired_facts():
    assert inverse_gap({("a", "left", "b")}) == [("a", "left", "b")]
    assert inverse_gap({("a", "left", "b"), ("b", "right", "a")}) == []


def test_consistency_score_only_counts_contradictions():
    clean = consistency({("a", "left", "b"), ("b", "right", "a")})
    assert clean["exclusivity"] == 0
    assert clean["antisymmetry"] == 0
    assert clean["score"] == 1.0

    contradiction = consistency({("a", "left", "b"), ("a", "right", "b")})
    assert contradiction["exclusivity"] == 1
    assert contradiction["score"] < 1.0


def test_antisymmetry_penalty_zero_for_consistent_pair():
    probs = np.array([[0.9, 0.1, 0.5, 0.5]], dtype=float)
    reverse = np.array([[0.1, 0.9, 0.5, 0.5]], dtype=float)
    assert antisymmetry_penalty(probs, reverse) == pytest.approx(0.0)


def test_exclusivity_penalty_positive_when_both_fire():
    probs = np.array([[0.8, 0.8, 0.2, 0.2]], dtype=float)
    assert exclusivity_penalty(probs) == pytest.approx(0.6)
