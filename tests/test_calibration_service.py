"""Tests for the runtime-facing calibration adapter."""

from __future__ import annotations

import numpy as np

from pss.projection.calibration_pipeline import (
    CalibrationBatch,
    RobustCalibrationPipeline,
    bootstrap_params,
    fit_params,
    score_samples,
    source_health,
)
from pss.runtime.calibration_service import CalibrationService

MODALITIES = ("vision", "geometry", "tactile")


def _batch(n: int = 60, k: int = 3, d: int = 8, seed: int = 0, noise: float = 0.15):
    rng = np.random.default_rng(seed)
    centres = np.eye(k, d)
    labels = rng.permutation(np.tile(np.arange(k), n // k + 1)[:n])
    embeddings = {
        m: centres[labels] + rng.normal(scale=noise, size=(n, d)) for m in MODALITIES
    }
    return CalibrationBatch(
        embeddings=embeddings,
        prototypes={m: centres.copy() for m in MODALITIES},
        labels=labels,
        sources=["s0"] * n,
        sample_ids=[f"s{i}" for i in range(n)],
    )


def _service(batch) -> CalibrationService:
    seed = bootstrap_params(batch)
    params = fit_params(batch, score_samples(batch, seed, source_health(batch)))
    pipeline = RobustCalibrationPipeline(params=params, gate_set=batch)
    return CalibrationService(pipeline=pipeline)


def test_update_records_a_window_and_keeps_state() -> None:
    batch = _batch()
    service = _service(batch)
    record = service.update(batch)
    assert record["state"] == "normal"
    assert service.state() == "normal"
    assert service.describe()["windows"] == 1


def test_confidence_uses_fused_top1_scale() -> None:
    batch = _batch()
    service = _service(batch)
    service.fit_confidence(batch)
    value = service.confidence(5.0)
    assert 0.0 < value < 1.0


def test_conformal_channel_returns_a_set_and_updates() -> None:
    from pss.projection.calibration_pipeline import fused_probabilities

    batch = _batch()
    service = _service(batch)
    service.confidence_mode = "conformal"
    service.calibrate_conformal(batch, alpha_target=0.1, gamma=0.01)

    probs, labels = fused_probabilities(service.params(), batch)
    row, label = probs[0], int(labels[0])
    mask = service.prediction_set(row)
    assert mask.shape == (probs.shape[1],)
    assert mask[label]
    out = service.observe_outcome(row, label, trusted=True)
    assert out["covered"] is True and out["updated"] is True
    skipped = service.observe_outcome(row, label, trusted=False)
    assert skipped["updated"] is False


def test_describe_exposes_pinning() -> None:
    batch = _batch()
    service = _service(batch)
    described = service.describe()
    assert "pinned" in described
    assert isinstance(described["pinned"], list)
