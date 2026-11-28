"""Tests for the grid empirical-Bayes boundary."""

from __future__ import annotations

import numpy as np

from pss.reasoning.relations.bayes_boundary import GridBayesBoundary


def _prior(n: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.2, n), rng.normal(0, 0.2, n)


def test_clean_prediction_matches_hard_threshold():
    dx, dy = _prior()
    boundary = GridBayesBoundary(eps=0.02, bins=129).fit(dx, dy)
    bits, conf = boundary.predict(
        np.array([0.10, -0.10, 0.01, 0.10]),
        np.array([0.00, 0.00, 0.00, 0.10]),
        sigma=0.0,
    )
    assert bits[0].tolist() == [True, False, False, False]  # left
    assert bits[1].tolist() == [False, True, False, False]  # right
    assert bits[2].tolist() == [False, False, False, False]  # neither
    assert bits[3].tolist() == [True, False, False, True]  # left + below
    assert np.all(conf >= 0.5)


def test_vertical_sign_convention():
    dx, dy = _prior()
    boundary = GridBayesBoundary(eps=0.02, bins=129).fit(dx, dy)
    bits, _ = boundary.predict(
        np.array([0.0, 0.0]), np.array([0.10, -0.10]), sigma=0.0
    )
    assert bits[0].tolist() == [False, False, False, True]  # below
    assert bits[1].tolist() == [False, False, True, False]  # above


def test_tables_are_cached_per_sigma():
    dx, dy = _prior()
    boundary = GridBayesBoundary(eps=0.02, bins=65).fit(dx, dy)
    first = boundary._state_tables(0.01)
    second = boundary._state_tables(0.01)
    assert first is second
    assert 0.01 in boundary._cache


def test_predict_shape_and_confidence_range():
    dx, dy = _prior()
    boundary = GridBayesBoundary(eps=0.02, bins=65).fit(dx, dy)
    bits, conf = boundary.predict(
        np.array([0.2, -0.2, 0.0]), np.array([0.1, -0.1, 0.0]), sigma=0.01
    )
    assert bits.shape == (3, 4)
    assert conf.shape == (3,)
    assert np.all((conf >= 0.0) & (conf <= 1.0))
