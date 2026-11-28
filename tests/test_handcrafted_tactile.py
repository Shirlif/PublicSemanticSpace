import numpy as np
import pytest

pytest.importorskip("torch")

from pss.encoders.registry import build_tactile_encoder
from pss.encoders.tactile.handcrafted import (
    HandcraftedTactileEncoder,
    handcrafted_tactile_features,
)


def _grid(seed: int = 0, shape=(12, 32)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


def test_feature_shapes_for_grid_and_gelsight_clip():
    grid = handcrafted_tactile_features(_grid())
    assert grid.ndim == 1 and grid.size > 0
    assert np.isfinite(grid).all()

    rng = np.random.default_rng(0)
    clip = rng.random((5, 64, 64, 3), dtype=np.float32)
    feats = handcrafted_tactile_features(clip)
    assert feats.ndim == 1 and feats.size > 0
    assert np.isfinite(feats).all()


def test_deterministic_and_zero_params():
    first = handcrafted_tactile_features(_grid(0))
    second = handcrafted_tactile_features(_grid(0))
    other = handcrafted_tactile_features(_grid(1))
    assert np.array_equal(first, second)
    assert not np.allclose(first, other)

    encoder = HandcraftedTactileEncoder(out_shape=(12, 32))
    assert encoder.params() == 0
    assert encoder.out_dim == handcrafted_tactile_features(np.zeros((12, 32))).size


def test_encoder_accepts_batched_torch_inputs():
    import torch

    encoder = HandcraftedTactileEncoder(out_shape=(12, 32))
    out = encoder.encode(torch.rand(2, 12, 32))
    assert tuple(out.shape) == (2, encoder.out_dim)
    out = encoder.encode(torch.rand(2, 1, 12, 32))
    assert tuple(out.shape) == (2, encoder.out_dim)
    assert torch.isfinite(out).all()


def test_registry_builds_handcrafted_tactile():
    encoder = build_tactile_encoder("handcrafted", out_shape=(8, 8))
    assert encoder.input_dim == 64
    assert encoder.params() == 0
