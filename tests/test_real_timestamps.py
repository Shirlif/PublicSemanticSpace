"""Producer-supplied timestamps must survive into the staleness contract.

``Observation.stamp_streams`` is the entry point a real driver uses after
filling ``raw``: it records capture/arrival seconds, computes skew against
``t_ref``, and excludes stale modalities from fusion.  ``extract_frame`` accepts
the same timings so simulated and replayed streams use one path.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pss.memory.perception import (
    ModalityObservation,
    Observation,
    PoseNoise,
    extract_frame,
)


def _observation(raw=None) -> Observation:
    raw = raw or {"vision": np.ones(4), "tactile": np.ones(4), "geometry": np.ones(4)}
    return Observation(
        gt_id="obj",
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        appearance=np.ones(4),
        raw=raw,
        confidence=1.0,
        category="cup",
    )


def test_stamp_streams_marks_a_late_modality_stale():
    obs = _observation()
    obs.stamp_streams(
        {"vision": (10.0, 10.0), "geometry": (10.0, 10.0), "tactile": (9.97, 10.0)},
        t_ref=10.0,
    )
    assert obs.stale_modalities == ["tactile"]
    assert "tactile" not in obs.valid_raw()
    assert set(obs.valid_raw()) == {"vision", "geometry"}
    assert obs.time_skew_ms["tactile"] == pytest.approx(-30.0)


def test_stamp_streams_defaults_missing_timings_to_t_ref():
    obs = _observation()
    obs.stamp_streams({"vision": (2.0, 2.0)}, t_ref=2.0)
    assert obs.stale_modalities == []
    assert set(obs.valid_raw()) == {"vision", "tactile", "geometry"}


def test_stamp_streams_rejects_clock_reversal():
    obs = _observation()
    with pytest.raises(ValueError, match="t_arrival < t_capture"):
        obs.stamp_streams({"vision": (2.0, 1.0)}, t_ref=2.0)


def test_explicit_stream_can_be_invalidated():
    obs = _observation()
    obs.add_stream(ModalityObservation("vision", np.ones(4), 5.0, 5.0), t_ref=5.0)
    obs.streams["vision"].valid = False
    obs.refresh_staleness()
    assert obs.stale_modalities == ["vision"]
    assert "vision" not in obs.valid_raw()


def _proj(x):
    return torch.ones((int(x.shape[0]), 128), dtype=torch.float32)


def _fake_models() -> SimpleNamespace:
    return SimpleNamespace(
        vision=SimpleNamespace(
            image_size=8,
            encode=lambda x: np.ones((int(x.shape[0]), 4), dtype=np.float32),
        ),
        tactile=SimpleNamespace(
            encode=lambda x: np.ones((int(x.shape[0]), 8), dtype=np.float32),
        ),
        vision_proj=_proj,
        tactile_proj=_proj,
        geometry_proj=_proj,
        device="cpu",
        image_size=8,
        bank=None,
        weights=None,
        threshold=None,
        preserve_aspect=True,
        dino=None,
        dino_image_size=224,
    )


def _sample() -> dict:
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(1, 8, 8, 3), dtype=np.uint8)
    depth = np.full((1, 8, 8), 0.5, dtype=np.float32)
    seg = np.zeros((1, 8, 8, 1), dtype=np.int32)
    tactile = rng.random((1, 1, 64), dtype=np.float32)
    poses = np.zeros((1, 1, 7), dtype=np.float32)
    poses[..., 3] = 1.0
    return {
        "rgb": rgb,
        "depth": depth,
        "segmentation": seg,
        "tactile": tactile,
        "poses": poses,
        "categories": ["cup"],
        "instance_ids": ["obj"],
        "geom_to_instance": {"0": "obj"},
    }


def test_extract_frame_uses_producer_timings():
    observations = extract_frame(
        _sample(),
        0,
        _fake_models(),
        PoseNoise(enabled=False),
        timings={"tactile": (0.07, 0.07)},
        t_ref=0.1,
    )
    assert len(observations) == 1
    obs = observations[0]
    assert obs.t_ref == pytest.approx(0.1)
    assert obs.stale_modalities == ["tactile"]
    assert "tactile" not in obs.valid_raw()


def test_extract_frame_without_timings_stays_synchronous():
    observations = extract_frame(
        _sample(), 0, _fake_models(), PoseNoise(enabled=False)
    )
    obs = observations[0]
    assert obs.t_ref == pytest.approx(0.0)
    assert obs.stale_modalities == []
    assert set(obs.valid_raw()) == {"vision", "tactile", "geometry"}
