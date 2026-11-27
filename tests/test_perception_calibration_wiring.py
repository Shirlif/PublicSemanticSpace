"""Wiring test: the runtime's recognition weights follow the streaming calibrator.

The heavy ``PerceptionService.__init__`` loads torch models, so these tests
construct the object without ``__init__`` and substitute a lightweight models
stub.  They check the wiring contract, not the encoders.
"""

from __future__ import annotations

from types import SimpleNamespace

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
from pss.runtime.perception import PerceptionConfig, PerceptionService

MODALITIES = ("vision", "geometry", "tactile")


def _batch(n: int = 60, k: int = 3, d: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    centres = np.eye(k, d)
    labels = rng.permutation(np.tile(np.arange(k), n // k + 1)[:n])
    embeddings = {
        m: centres[labels] + rng.normal(scale=0.15, size=(n, d)) for m in MODALITIES
    }
    return CalibrationBatch(
        embeddings=embeddings,
        prototypes={m: centres.copy() for m in MODALITIES},
        labels=labels,
        sources=["s0"] * n,
        sample_ids=[f"s{i}" for i in range(n)],
    )


def _calibration(batch) -> CalibrationService:
    seed = bootstrap_params(batch)
    params = fit_params(batch, score_samples(batch, seed, source_health(batch)))
    return CalibrationService(
        pipeline=RobustCalibrationPipeline(params=params, gate_set=batch)
    )


def _service(calibration=None) -> PerceptionService:
    service = object.__new__(PerceptionService)  # skip torch model loading
    service.config = PerceptionConfig()
    service.device = "cpu"
    service.noise = None
    service.fallback = False
    service.error = ""
    service.models = SimpleNamespace(
        weights={"vision": 0.1, "tactile": 0.1},
        vision=SimpleNamespace(image_size=64),
    )
    service.calibration = calibration
    return service


def test_without_calibration_nothing_changes() -> None:
    batch = _batch()
    service = _service()
    before = dict(service.models.weights)
    assert service.update_calibration(batch) is None
    assert service.models.weights == before
    assert np.isnan(service.confidence(1.0))


def test_update_replaces_recognition_weights() -> None:
    batch = _batch()
    service = _service(_calibration(batch))
    record = service.update_calibration(batch)
    assert record is not None
    assert service.models.weights == dict(service.calibration.params().reliability)
    # The runtime reads this through Observation.modality_reliability.
    assert set(service.models.weights) == set(MODALITIES)


def test_describe_exposes_calibration_state() -> None:
    batch = _batch()
    service = _service(_calibration(batch))
    service.update_calibration(batch)
    described = service.describe()
    assert described["calibration_service"] is not None
    assert described["calibration_service"]["windows"] == 1
    assert isinstance(described["calibration_service"]["pinned"], list)


def test_attach_calibration_is_optional() -> None:
    batch = _batch()
    service = _service()
    service.attach_calibration(_calibration(batch))
    assert service.update_calibration(batch) is not None
    service.attach_calibration(None)
    assert service.update_calibration(batch) is None
