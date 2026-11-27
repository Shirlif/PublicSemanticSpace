"""Tests for the BCD per-label diagnosis helpers."""

from __future__ import annotations

from pss.eval.scripts.run_relation_per_label import (
    _exact,
    _pair_type,
    _prf,
    _true_labels,
)


def test_pair_type_classifies_grid_pairs():
    assert _pair_type("A", "B") == "horizontal"
    assert _pair_type("B", "A") == "horizontal"
    assert _pair_type("C", "G") == "vertical"
    assert _pair_type("A", "C") == "diagonal"


def test_true_labels_match_geometry():
    assert _true_labels("A", "B") == [True, False, False, False]  # A is left of B
    assert _true_labels("B", "A") == [False, True, False, False]  # B is right of A
    assert _true_labels("C", "G") == [False, False, True, False]  # C is above G


def test_prf_counts_false_positives():
    rows = [
        {"true": [False, True, False, False], "pred": [False, True, True, False]},
        {"true": [False, True, False, False], "pred": [False, True, False, False]},
    ]
    above = _prf(rows, 2)
    assert above["tp"] == 0
    assert above["fp"] == 1
    assert above["precision"] == 0.0
    assert above["recall"] == 1.0  # convention when there are no positives

    right = _prf(rows, 1)
    assert right["f1"] == 1.0


def test_exact_requires_all_four_labels():
    good = {"true": [False, True, False, False], "pred": [False, True, False, False]}
    bad = {"true": [False, True, False, False], "pred": [False, True, True, False]}
    assert _exact([good]) == 1.0
    assert _exact([good, bad]) == 0.5
