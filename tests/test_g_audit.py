"""Tests for the g-factor audit helpers and pre-registered g arms."""

from __future__ import annotations

import numpy as np
import pytest

from pss.eval.g_audit import (
    cell_accuracy,
    coverage_cells,
    coverage_dashboard,
    paired_bootstrap_ci,
    worst_cell_exact4,
)
from pss.eval.readout_reasoning_preregistration import (
    check_readout_reasoning_preregistration,
)
from pss.eval.scripts.run_g_audit import (
    _audit_rows,
    _labels_g,
    _pair_feature_g,
    _split_held_pairs,
)


def test_coverage_decomposition_sums_to_total_error():
    items = [
        ("a|left", 1.0, 8),
        ("a|left", 0.0, 2),
        ("a|above", 0.0, 5),
        ("a|above", 1.0, 5),
    ]
    cells = coverage_cells(items)
    total_error = sum(row["contribution"] for row in cells)
    assert total_error == pytest.approx(0.35)  # 2/20 wrong in cell one, 5/20 in cell two
    by_cell = {row["cell"]: row for row in cells}
    assert by_cell["a|left"]["support"] == 10
    assert by_cell["a|left"]["error"] == pytest.approx(0.2)
    assert by_cell["a|above"]["error"] == pytest.approx(0.5)


def test_coverage_dashboard_marks_empty_and_worst_cells():
    cells = coverage_cells([("present", 0.0, 10)])
    dashboard = coverage_dashboard(
        cells, expected_cells=["present", "missing"], min_support=1
    )
    assert dashboard["empty_cells"] == ["missing"]
    assert dashboard["worst_cell"] == "present"
    assert dashboard["worst_contribution_share"] == 1.0


def test_worst_cell_exact4_respects_min_support():
    rows = [
        {"cell": "big", "exact": True},
        {"cell": "big", "exact": False},
        {"cell": "tiny", "exact": False},
    ]
    result = worst_cell_exact4(rows, lambda r: r["cell"], min_support=2)
    assert result["cell"] == "big"
    assert result["value"] == pytest.approx(0.5)


def test_cell_accuracy_sorted_worst_first():
    rows = [
        {"cell": "b", "exact": True},
        {"cell": "b", "exact": True},
        {"cell": "a", "exact": False},
    ]
    cells = cell_accuracy(rows, lambda r: r["cell"])
    assert [c["cell"] for c in cells] == ["a", "b"]


def test_paired_bootstrap_ci_is_centered_on_the_mean():
    values = [1.0] * 30 + [-1.0] * 10
    result = paired_bootstrap_ci(values, n_boot=500, seed=0)
    assert result["mean"] == pytest.approx(0.5)
    assert result["lo"] <= result["mean"] <= result["hi"]
    assert result["n"] == 40


def test_relative_features_are_translation_invariant():
    a = _pair_feature_g((-1.0, 0.5), (2.0, 0.5), "relative")
    b = _pair_feature_g((9.0, -4.0), (12.0, -4.0), "relative")
    assert a == b


def test_absolute_features_carry_position():
    a = _pair_feature_g((-1.0, 0.0), (1.0, 0.0), "absolute")
    b = _pair_feature_g((0.0, 0.0), (2.0, 0.0), "absolute")
    assert a != b


def test_scale_invariant_features_ignore_scale():
    small = _pair_feature_g((0.0, 0.0), (1.0, 2.0), "scale_invariant")
    large = _pair_feature_g((0.0, 0.0), (5.0, 10.0), "scale_invariant")
    assert np.allclose(small, large)


def test_mixed_labels_are_eight_bits():
    labels = _labels_g((-1.0, 0.0), (1.0, 0.0), "mixed")
    assert len(labels) == 8


def test_split_held_pairs_is_a_partition():
    pairs = [(i, i + 1) for i in range(10)]
    val, test = _split_held_pairs(pairs, seed=2026)
    assert len(val) + len(test) == len(pairs)
    assert set(val).isdisjoint(set(test))
    assert set(val) | set(test) == set(pairs)


def test_audit_rows_builds_dashboards():
    rows = [
        {
            "seed": seed,
            "pair": "A->B",
            "type": "horizontal",
            "dx": 2.0,
            "dy": 0.0,
            "correct": [True, False, True, True],
            "exact": False,
        }
        for seed in (0, 1)
    ]
    audit = _audit_rows(rows)
    assert audit["overall_exact4"]["mean"] == 0.0
    dashboard = audit["dashboards"]["by_type_predicate"]
    assert dashboard["worst_cell"] is not None
    assert dashboard["total_support"] == 8  # 2 seeds x 4 predicates


def test_g_combination_arm_passes():
    observed = {
        "g_response_combination": {"worst_cell_acc": 0.95, "delta_vs_full": -0.01}
    }
    result = check_readout_reasoning_preregistration(observed)
    assert result["ok"], result["mismatches"]


def test_g_combination_arm_fails_short_of_bar():
    observed = {
        "g_response_combination": {"worst_cell_acc": 0.80, "delta_vs_full": -0.10}
    }
    result = check_readout_reasoning_preregistration(observed)
    assert not result["ok"]
    assert any("worst_cell_acc" in m for m in result["mismatches"])
    assert any("delta_vs_full" in m for m in result["mismatches"])


def test_g_single_arm_must_trail_full():
    passing = {"g_response_single": {"worst_cell_acc": 0.30, "delta_vs_full": -0.60}}
    assert check_readout_reasoning_preregistration(passing)["ok"]
    failing = {"g_response_single": {"worst_cell_acc": 0.95, "delta_vs_full": -0.01}}
    assert not check_readout_reasoning_preregistration(failing)["ok"]


def test_g_invariance_bars():
    passing = {
        "g_invariance_matched": {"worst_cell_acc": 0.84},
        "g_invariance_mismatched": {"worst_cell_acc": 0.15},
    }
    assert check_readout_reasoning_preregistration(passing)["ok"]
    failing = {
        "g_invariance_matched": {"worst_cell_acc": 0.70},
        "g_invariance_mismatched": {"worst_cell_acc": 0.70},
    }
    assert not check_readout_reasoning_preregistration(failing)["ok"]
