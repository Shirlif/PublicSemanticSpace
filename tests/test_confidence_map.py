"""Tests for the separated confidence channel (monotone binning map)."""

from __future__ import annotations

import numpy as np
import pytest

from pss.projection.calibration import expected_calibration_error
from pss.projection.confidence_map import (
    ConfidenceMap,
    apply_confidence_map,
    fit_confidence_map,
)


def _toy(n: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.0, 1.0, n)
    correct = (rng.random(n) < scores).astype(np.float64)
    return scores, correct


def test_map_is_monotone() -> None:
    scores, correct = _toy()
    cmap = fit_confidence_map(scores, correct, n_bins=10)
    assert cmap.n_bins == 10
    assert np.all(np.diff(np.asarray(cmap.values)) >= -1e-12)


def test_map_calibrates_out_of_sample() -> None:
    scores, correct = _toy()
    cmap = fit_confidence_map(scores[:1500], correct[:1500], n_bins=10)
    mapped = apply_confidence_map(scores[1500:], cmap)
    ece = expected_calibration_error(mapped, correct[1500:])
    assert ece < 0.05


def test_binning_is_invariant_to_monotone_rescaling() -> None:
    scores, correct = _toy()
    a = fit_confidence_map(scores, correct, n_bins=8)
    b = fit_confidence_map(scores * 3.0 + 1.0, correct, n_bins=8)
    # Same partition in quantile space, so the fitted values must match.
    assert np.allclose(a.values, b.values, atol=1e-12)


def test_constant_scores_return_a_constant_map() -> None:
    scores = np.full(50, 0.5)
    correct = np.ones(50)
    cmap = fit_confidence_map(scores, correct)
    mapped = apply_confidence_map(np.array([0.1, 0.5, 0.9]), cmap)
    assert np.allclose(mapped, mapped[0])
    assert 0.0 < mapped[0] < 1.0


def test_pav_pools_a_non_monotone_pattern() -> None:
    # Bin accuracies deliberately non-monotone: 0.9, 0.2, 0.8.
    scores = np.concatenate(
        [np.full(20, 0.1), np.full(20, 0.5), np.full(20, 0.9)]
    )
    correct = np.concatenate(
        [np.ones(18), np.zeros(2), np.ones(4), np.zeros(16), np.ones(16), np.zeros(4)]
    )
    cmap = fit_confidence_map(scores, correct, n_bins=3, min_per_bin=2)
    assert np.all(np.diff(np.asarray(cmap.values)) >= -1e-12)


def test_underpopulated_bins_degrade_gracefully() -> None:
    scores = np.array([0.0, 0.0, 0.0, 1.0])
    correct = np.array([0.0, 0.0, 1.0, 1.0])
    cmap = fit_confidence_map(scores, correct, n_bins=6, min_per_bin=2)
    assert cmap.n_bins == 6
    assert np.all(np.isfinite(np.asarray(cmap.values)))


def test_empty_input_returns_a_safe_map() -> None:
    cmap = fit_confidence_map(np.array([]), np.array([]))
    assert isinstance(cmap, ConfidenceMap)
    assert 0.0 < cmap.values[0] < 1.0


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        fit_confidence_map(np.zeros(3), np.zeros(4))


def test_apply_clips_out_of_range_scores() -> None:
    scores, correct = _toy()
    cmap = fit_confidence_map(scores, correct, n_bins=5)
    mapped = apply_confidence_map(np.array([-1e9, 1e9]), cmap)
    assert mapped[0] == pytest.approx(cmap.values[0])
    assert mapped[1] == pytest.approx(cmap.values[-1])
