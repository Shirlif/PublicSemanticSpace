import numpy as np
import pytest

pytest.importorskip("torch")

from pss.encoders.registry import build_vision_encoder
from pss.encoders.vision.handcrafted import (
    HandcraftedVisionEncoder,
    default_out_dim,
    handcrafted_features,
)


def _image(seed: int = 0, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def test_feature_shape_and_zero_params():
    feats = handcrafted_features(_image())
    assert feats.shape == (default_out_dim(),)
    assert np.isfinite(feats).all()
    encoder = HandcraftedVisionEncoder()
    assert encoder.params() == 0
    assert encoder.out_dim == default_out_dim()


def test_deterministic_and_image_dependent():
    first = handcrafted_features(_image(0))
    second = handcrafted_features(_image(0))
    other = handcrafted_features(_image(1))
    assert np.array_equal(first, second)
    assert not np.allclose(first, other)


def test_const_image_stays_finite():
    feats = handcrafted_features(np.full((64, 64, 3), 128, dtype=np.uint8))
    assert np.isfinite(feats).all()


def test_encoder_accepts_unit_range_tensor():
    import torch

    encoder = HandcraftedVisionEncoder()
    out = encoder.encode(torch.rand(2, 3, 64, 64))
    assert tuple(out.shape) == (2, encoder.out_dim)
    assert torch.isfinite(out).all()


def test_registry_builds_handcrafted():
    encoder = build_vision_encoder("handcrafted", image_size=64)
    assert encoder.image_size == 64
    assert encoder.params() == 0
    assert encoder.out_dim == default_out_dim()
