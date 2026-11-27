"""Tests for source isolation: per-source CUSUM, quarantine, and the baseline."""

from __future__ import annotations

import numpy as np

from pss.projection.calibration_pipeline import CalibrationBatch, source_health
from pss.projection.source_registry import SourceRegistry

MODALITIES = ("vision", "geometry", "tactile")


def _clean_windows(n: int = 200, seed: int = 0, sources=("a", "b")):
    rng = np.random.default_rng(seed)
    return [
        {
            s: float(np.clip(rng.normal(0.04, 0.08), 0.0, 1.0))
            for s in sources
        }
        for _ in range(n)
    ]


def test_calibrate_builds_one_detector_per_source() -> None:
    registry = SourceRegistry.calibrate(_clean_windows(100))
    assert set(registry.detectors) == {"a", "b"}


def test_sustained_excess_quarantines_a_source() -> None:
    registry = SourceRegistry.calibrate(_clean_windows(200, seed=1))
    rng = np.random.default_rng(2)
    first = None
    for w in range(1, 30):
        mismatches = {
            "a": float(np.clip(rng.normal(0.04, 0.08), 0.0, 1.0)),
            "b": float(np.clip(rng.normal(0.30, 0.08), 0.0, 1.0)),
        }
        statuses = registry.update(mismatches)
        if statuses["b"] == "quarantined" and first is None:
            first = w
    assert first is not None and first <= 10
    assert "b" in registry.quarantined()


def test_quarantined_source_is_excluded_from_trust() -> None:
    registry = SourceRegistry.calibrate(_clean_windows(200, seed=3))
    registry.states["b"] = registry.states.get("b") or None  # type: ignore[assignment]
    from pss.projection.source_registry import QUARANTINED, SourceState

    registry.states["b"] = SourceState(name="b", status=QUARANTINED)
    adjusted = registry.apply({"a": 1.0, "b": 0.7})
    assert adjusted["a"] == 1.0
    assert adjusted["b"] == 0.0


def test_quarantine_releases_after_clean_windows() -> None:
    registry = SourceRegistry.calibrate(_clean_windows(200, seed=4), recover_windows=3)
    rng = np.random.default_rng(5)
    for _ in range(20):
        registry.update(
            {
                "a": 0.04,
                "b": float(np.clip(rng.normal(0.30, 0.08), 0.0, 1.0)),
            }
        )
    assert "b" in registry.quarantined()
    for _ in range(3):
        registry.update({"a": 0.04, "b": 0.04})
    assert "b" not in registry.quarantined()
    assert any(e["event"] == "release" for e in registry.events)


def test_clean_stream_stays_trusted() -> None:
    registry = SourceRegistry.calibrate(_clean_windows(200, seed=6))
    rng = np.random.default_rng(7)
    for _ in range(300):
        registry.update(
            {
                "a": float(np.clip(rng.normal(0.04, 0.08), 0.0, 1.0)),
                "b": float(np.clip(rng.normal(0.04, 0.08), 0.0, 1.0)),
            }
        )
    assert registry.quarantined() == []


def test_large_bad_source_is_detected_regardless_of_size() -> None:
    """Detection uses the absolute mismatch rate, so size cannot hide it."""
    registry = SourceRegistry.calibrate(
        _clean_windows(200, seed=8, sources=("big", "small"))
    )
    statuses = {}
    for _ in range(20):
        statuses = registry.update({"big": 0.32, "small": 0.04})
    assert statuses["big"] == "quarantined"
    assert statuses["small"] == "trusted"


def _source_batch(n_good: int = 80, n_bad: int = 120, bad_flip: float = 0.35, seed: int = 0):
    rng = np.random.default_rng(seed)
    centres = np.eye(3, 8)
    n = n_good + n_bad
    labels = rng.permutation(np.tile(np.arange(3), n // 3 + 1)[:n])
    embeddings = {
        m: centres[labels] + rng.normal(scale=0.15, size=(n, 8)) for m in MODALITIES
    }
    sources = ["good"] * n_good + ["bad"] * n_bad
    bad_idx = np.arange(n_good, n)
    flips = rng.choice(bad_idx, size=int(round(bad_flip * n_bad)), replace=False)
    labels = labels.copy()
    labels[flips] = (labels[flips] + 1) % 3
    return CalibrationBatch(
        embeddings=embeddings,
        prototypes={m: centres.copy() for m in MODALITIES},
        labels=labels,
        sources=sources,
        sample_ids=[f"s{i}" for i in range(n)],
    )


def test_baseline_uses_the_median_so_a_large_bad_source_cannot_mask_itself() -> None:
    """With the pooled mean the bad source drags the reference down."""
    batch = _source_batch()
    trust = source_health(batch)
    assert trust["good"] > 0.8
    assert trust["bad"] < 0.3
