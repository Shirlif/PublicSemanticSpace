"""Tests for the one-sided CUSUM used against slow targeted attacks."""

from __future__ import annotations

import numpy as np

from pss.projection.cusum import (
    CusumBank,
    CusumDetector,
    calibrate_cusum,
    calibrate_per_class,
)


def test_calibrate_returns_positive_scale() -> None:
    values = np.array([0.95, 0.96, 0.94, 0.95, 0.96, 0.95])
    reference, slack, threshold = calibrate_cusum(values)
    assert abs(reference - values.mean()) < 1e-9
    assert slack > 0
    assert threshold > slack


def test_cusum_ignores_pure_noise() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(0.95, 0.02, 200)
    reference, slack, threshold = calibrate_cusum(values)
    det = CusumDetector(reference, slack, threshold, direction=-1.0)
    alarms = sum(det.update(v) for v in values)
    # Calibrated on this very distribution, it should fire very rarely.
    assert alarms <= 2


def test_cusum_detects_a_sustained_small_shift() -> None:
    rng = np.random.default_rng(1)
    clean = rng.normal(0.95, 0.02, 200)
    reference, slack, threshold = calibrate_cusum(clean)
    det = CusumDetector(reference, slack, threshold, direction=-1.0)
    shifted = rng.normal(0.93, 0.02, 60)  # 1 sd shift per window
    first = None
    for i, v in enumerate(shifted, start=1):
        if det.update(v):
            first = i
            break
    assert first is not None and first <= 30


def test_cusum_does_not_fire_on_the_wrong_direction() -> None:
    det = CusumDetector(reference=0.95, slack=0.01, threshold=0.05, direction=-1.0)
    for _ in range(50):
        assert det.update(0.97) is False


def test_two_sided_bank_catches_both_signatures() -> None:
    down = CusumDetector(0.95, 0.005, 0.03, direction=-1.0)
    up = CusumDetector(0.95, 0.005, 0.03, direction=+1.0)
    bank = CusumBank(agreement_down=down, agreement_up=up)

    reasons_fall = []
    for _ in range(10):
        reasons_fall += bank.update(0.92)
    assert "agreement_down" in reasons_fall

    down2 = CusumDetector(0.95, 0.005, 0.03, direction=-1.0)
    up2 = CusumDetector(0.95, 0.005, 0.03, direction=+1.0)
    bank2 = CusumBank(agreement_down=down2, agreement_up=up2)
    reasons_rise = []
    for _ in range(10):
        reasons_rise += bank2.update(0.98)
    assert "agreement_up" in reasons_rise


def test_per_class_detects_a_concentrated_attack() -> None:
    rng = np.random.default_rng(0)
    clean = [
        {c: float(np.clip(rng.normal(0.04, 0.08), 0.0, 1.0)) for c in range(5)}
        for _ in range(200)
    ]
    detector = calibrate_per_class(clean)
    assert set(detector.detectors) == set(range(5))

    fired: list[int] = []
    for _ in range(20):
        values = {
            c: float(np.clip(rng.normal(0.04, 0.08), 0.0, 1.0)) for c in range(5)
        }
        values[2] = float(np.clip(rng.normal(0.20, 0.08), 0.0, 1.0))
        fired += detector.update(values)
    assert 2 in fired


def test_per_class_stays_quiet_on_clean() -> None:
    rng = np.random.default_rng(1)
    clean = [
        {c: float(np.clip(rng.normal(0.04, 0.08), 0.0, 1.0)) for c in range(5)}
        for _ in range(100)
    ]
    detector = calibrate_per_class(clean)
    alarms = 0
    for _ in range(200):
        values = {
            c: float(np.clip(rng.normal(0.04, 0.08), 0.0, 1.0)) for c in range(5)
        }
        alarms += len(detector.update(values))
    # 5 classes x 200 windows = 1000 updates; a family-wise rate around 1%
    # means roughly 10 alarms is the expected cost of per-class testing.
    assert alarms <= 15


def test_per_class_skips_classes_absent_from_the_window() -> None:
    rng = np.random.default_rng(2)
    clean = [
        {c: float(np.clip(rng.normal(0.04, 0.08), 0.0, 1.0)) for c in range(3)}
        for _ in range(50)
    ]
    detector = calibrate_per_class(clean)
    # Only class 0 present: others must not be updated or divide by nothing.
    assert detector.update({0: 0.05}) == []
    assert all(d.visits <= 1 for d in detector.detectors.values())


def test_cusum_ignores_nan() -> None:
    det = CusumDetector(0.95, 0.01, 0.05, direction=-1.0)
    assert det.update(float("nan")) is False
    assert det.state == 0.0
