"""Tests for the runtime perception service and calibration surfacing."""

from __future__ import annotations

from pathlib import Path

import pytest

from pss.encoders.data import load_index
from pss.runtime import CognitiveState, PSSRuntime, UncertaintyService
from pss.runtime.perception import PerceptionConfig, PerceptionService
from pss.sim.dataset import load_sample

DATA_DIR = Path("data/tracking_bench_v2")
EXPERT_DIR = Path("checkpoints/category_pretrain_experts")

requires_experts = pytest.mark.skipif(
    not all((EXPERT_DIR / name).exists() for name in ("vision.pt", "tactile.pt")),
    reason="category pretrain checkpoints not available",
)
requires_data = pytest.mark.skipif(
    not (DATA_DIR / "index.json").exists(),
    reason="tracking_bench_v2 dataset not available",
)


def _sample():
    index = load_index("data/tracking_bench_v2")
    entry = next(e for e in index["entries"] if e.get("split") == "test")
    return entry, load_sample(Path("data/tracking_bench_v2") / entry["path"])


@requires_experts
def test_perception_default_is_dinov2_with_calibration():
    service = PerceptionService(device="cpu")
    desc = service.describe()
    assert desc["vision"] == "dinov2"
    assert desc["fallback"] is False
    assert desc["calibration"]["whiten_dim"] is not None
    assert desc["calibration"]["residual"] is True
    assert desc["reliability"]


@requires_experts
def test_perception_fallback_when_projection_missing():
    service = PerceptionService(
        config=PerceptionConfig(
            vision="dinov2",
            projection_dir="checkpoints/does_not_exist",
            allow_fallback=True,
        ),
        device="cpu",
    )
    assert service.describe()["fallback"] is True
    assert service.describe()["vision"] == "smallcnn"


def test_uncertainty_flags_low_modality_reliability():
    state = CognitiveState()
    report = UncertaintyService().assess(
        state, modality_reliability={"tactile": 0.0, "vision": 0.9}
    )
    assert "low_modality_reliability" in report.flags
    assert report.recommendation == "review"
    assert report.modality_reliability["tactile"] == 0.0


@requires_experts
@requires_data
def test_runtime_ingest_sample_with_perception():
    _, sample = _sample()
    service = PerceptionService(device="cpu")
    runtime = PSSRuntime(perception=service, uncertainty_enabled=True)
    result = runtime.ingest_sample(sample, 0)
    assert "events" in result
    assert runtime.state.entities
    summary = runtime.summary()
    assert summary["perception"]["vision"] == "dinov2"
    assert summary["uncertainty"] is not None
