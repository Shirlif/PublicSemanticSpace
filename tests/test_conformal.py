"""Tests for adaptive conformal inference."""

from __future__ import annotations

import numpy as np

from pss.projection.conformal import AdaptiveConformal, evaluate_conformal


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _sample(n: int, k: int = 5, sharpness: float = 2.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    logits = rng.normal(size=(n, k)) + sharpness * np.eye(k)[labels]
    return _softmax(logits), labels


def test_coverage_on_exchangeable_data() -> None:
    calib_p, calib_y = _sample(600, seed=1)
    test_p, test_y = _sample(600, seed=2)
    result = evaluate_conformal(calib_p, calib_y, test_p, test_y, alpha_target=0.1)
    # Exchangeable data: the guarantee should hold closely.
    assert abs(result.coverage - result.target_coverage) < 0.05


def test_sets_are_smaller_when_the_model_is_sharper() -> None:
    calib_p, calib_y = _sample(600, sharpness=3.0, seed=3)
    sharp_p, sharp_y = _sample(600, sharpness=3.0, seed=4)
    flat_p, flat_y = _sample(600, sharpness=0.5, seed=5)
    sharp = evaluate_conformal(calib_p, calib_y, sharp_p, sharp_y)
    flat = evaluate_conformal(calib_p, calib_y, flat_p, flat_y)
    assert sharp.mean_size < flat.mean_size


def test_untrusted_steps_do_not_move_the_threshold() -> None:
    predictor = AdaptiveConformal(alpha_target=0.1, gamma=0.1).calibrate(
        *_sample(200, seed=11)
    )
    before = predictor.q
    probs, _ = _sample(30, seed=12)
    for row in probs:
        predictor.step(row, 0, trusted=False)  # every one deliberately wrong
    assert predictor.q == before


def test_trust_mask_filters_rows() -> None:
    calib_p, calib_y = _sample(400, seed=13)
    test_p, test_y = _sample(200, seed=14)
    trust = np.zeros(len(test_y), dtype=bool)
    trust[:50] = True
    result = evaluate_conformal(
        calib_p, calib_y, test_p, test_y, alpha_target=0.1, trust=trust
    )
    assert result.n == 50


def test_threshold_reacts_to_miscoverage() -> None:
    predictor = AdaptiveConformal(alpha_target=0.1, gamma=0.05)
    predictor.calibrate(*_sample(200, seed=6))
    before = predictor.q
    # Feed deliberately wrong labels: every sample miscovered -> threshold grows.
    probs, _ = _sample(50, seed=7)
    for row, label in zip(probs, np.zeros(50, dtype=np.int64)):
        predictor.step(row, int(label) + 3)
    assert predictor.q > before


def _shifted(n: int, k: int = 5, seed: int = 0):
    """A stream where the true class is systematically suppressed.

    This is the failure mode that matters: the model becomes *less* confident on
    the answer that is actually correct, so a static prediction set under-covers.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    logits = rng.normal(size=(n, k)) + 2.0 * np.eye(k)[labels]
    logits[np.arange(n), labels] -= 4.0
    return _softmax(logits), labels


def test_adaptation_recovers_coverage_under_shift() -> None:
    calib_p, calib_y = _sample(800, sharpness=1.0, seed=8)
    shifted_p, shifted_y = _shifted(1500, seed=9)

    adaptive = evaluate_conformal(
        calib_p, calib_y, shifted_p, shifted_y, alpha_target=0.10, gamma=0.02
    )

    # Static prediction set from the calibration pool, applied unchanged.
    static = AdaptiveConformal(alpha_target=0.10, gamma=0.0).calibrate(
        calib_p, calib_y
    )
    q = static.quantile()
    covered = int(
        np.sum((1.0 - shifted_p[np.arange(len(shifted_y)), shifted_y]) <= q + 1e-12)
    )
    static_cov = covered / len(shifted_y)

    assert static_cov < 0.5  # the static set fails badly under the shift
    assert adaptive.coverage > static_cov
    assert adaptive.coverage >= 1 - 0.10 - 0.10
