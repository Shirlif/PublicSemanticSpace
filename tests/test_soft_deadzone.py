"""Tests for the soft dead zone and the boundary benchmark helpers."""

from __future__ import annotations

import numpy as np

from pss.eval.scripts.run_boundary_bench import _features, _metrics, _noisy
from pss.reasoning.relations.soft_deadzone import (
    axis_probs,
    calibrate_abstain,
    calibrate_k,
    calibrate_tau,
    confidence,
    exact4,
    predict,
    predict_proba,
)


def test_axis_probs_sign_conventions():
    p_pos = axis_probs(np.array([1.0]), 0.02, 0.01)[0]
    p_neg = axis_probs(np.array([-1.0]), 0.02, 0.01)[0]
    p_zero = axis_probs(np.array([0.0]), 0.02, 0.001)[0]
    assert p_pos[0] > 0.99  # positive = left / below
    assert p_neg[1] > 0.99  # negative = right / above
    assert p_zero[2] > 0.99  # neither
    assert np.allclose(p_pos.sum(), 1.0)


def test_predict_proba_order():
    pa = np.array([[0.0, 0.0]])
    pb = np.array([[1.0, 0.0]])
    proba, state = predict_proba(pa, pb, eps=0.02, tau=0.01)
    assert proba.shape == (1, 4)
    assert state.shape == (1, 2, 3)
    assert proba[0, 0] > 0.99  # left
    assert proba[0, 1] < 0.01  # right
    bits, conf = predict(pa, pb, eps=0.02, tau=0.01)
    assert bits[0].tolist() == [True, False, False, False]
    assert 0.0 <= conf[0] <= 1.0


def test_calibration_returns_grid_values():
    rng = np.random.default_rng(0)
    pa = rng.normal(0, 0.2, size=(500, 2))
    pb = rng.normal(0, 0.2, size=(500, 2))
    y = (pb[:, 0] - pa[:, 0] > 0.02).astype(np.float32)
    # A four-bit label from the x axis only; enough to exercise calibration.
    y4 = np.stack([y, 1 - y, np.zeros_like(y), np.zeros_like(y)], axis=-1)
    tau, score = calibrate_tau(pa, pb, y4, eps=0.02)
    assert 0.002 <= tau <= 0.10
    assert 0.0 <= score <= 1.0
    k, k_score = calibrate_k(pa, pb, y4, eps=0.02, tau0=tau, sigma=0.01)
    assert k >= 0.0
    thr, selective, coverage = calibrate_abstain(
        pa, pb, y4, eps=0.02, tau=tau, target_coverage=0.8
    )
    assert 0.0 <= selective <= 1.0
    assert coverage >= 0.8


def test_exact4_and_confidence():
    y = np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=bool)
    p = y.copy()
    assert exact4(y, p) == 1.0
    p[1, 2] = True
    assert exact4(y, p) == 0.5
    state = np.array([[[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]]])
    assert np.isclose(confidence(state)[0], 0.8)


def test_boundary_helpers():
    pa = np.array([[0.0, 0.0], [0.0, 0.0]])
    pb = np.array([[1.0, 0.0], [0.0, 1.0]])
    X = _features(pa, pb)
    assert X.shape == (2, 4)
    qa, qb = _noisy(pa, pb, 0.0, np.random.default_rng(0))
    assert np.array_equal(qa, pa) and np.array_equal(qb, pb)
    y = np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=bool)
    types = np.array(["horizontal", "vertical"], dtype=object)
    m = _metrics(y, y, types)
    assert m["overall"] == 1.0
    assert m["near_worst"] == 1.0
    assert m["absence_error"] == 0.0
