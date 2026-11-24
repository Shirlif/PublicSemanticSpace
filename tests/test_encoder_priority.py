"""Tests for the encoder-priority path: swappable encoders + calibration."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pss.encoders.registry import build_tactile_encoder, build_vision_encoder
from pss.encoders.tactile import load_tactile_expert
from pss.encoders.tactile.conv import (
    TactileConvEncoder,
    load_tactile_conv_expert,
    save_tactile_conv_expert,
)
from pss.encoders.vision.local import VLADEncoder, fit_vlad_codebook, load_vlad, save_vlad
from pss.projection.calibration import (
    calibrate_modality,
    calibrated_confidence,
    calibrated_scores,
    reliability_weights,
)
from pss.projection.heads import ProjectionHead, load_projection, save_projection


def test_projection_head_whitening_roundtrip(tmp_path):
    torch.manual_seed(0)
    X = np.random.default_rng(0).normal(size=(40, 12)).astype(np.float32)
    head = ProjectionHead(12, 8, whiten_dim=6, residual=True)
    head.fit_whitening(X)
    z = head(torch.as_tensor(X))
    assert z.shape == (40, 8)
    assert torch.allclose(z.norm(dim=-1), torch.ones(40), atol=1e-5)

    path = tmp_path / "vision.pt"
    save_projection(path, head)
    loaded = load_projection(path)
    assert loaded.whiten is not None
    assert loaded.whiten.out_dim == 6
    assert loaded.residual is True
    assert torch.allclose(loaded(torch.as_tensor(X)), z, atol=1e-6)


def test_projection_head_backward_compatible(tmp_path):
    head = ProjectionHead(10, 8)
    path = tmp_path / "plain.pt"
    save_projection(path, head)
    loaded = load_projection(path)
    assert loaded.whiten is None
    assert loaded.residual is False
    x = torch.randn(3, 10)
    assert loaded(x).shape == (3, 8)


def test_tactile_conv_shapes_and_loader_dispatch(tmp_path):
    enc = TactileConvEncoder(64, 32)
    assert enc(torch.randn(4, 64)).shape == (4, 32)
    assert enc(torch.randn(2, 384)).shape == (2, 32)
    assert enc(torch.randn(2, 8, 8)).shape == (2, 32)

    path = tmp_path / "tactile_conv.pt"
    save_tactile_conv_expert(path, enc, ["a", "b"], 64)
    direct = load_tactile_conv_expert(path)
    generic = load_tactile_expert(path)  # dispatch by arch marker
    x = torch.randn(3, 64)
    assert torch.allclose(direct.encode(x), generic.encode(x), atol=1e-6)  # type: ignore[arg-type]


class _FakeBackbone:
    device = "cpu"
    image_size = 8
    out_dim = 6

    def encode_patches(self, images):
        b = images.shape[0]
        return images.reshape(b, -1, 6)


def test_vlad_fit_encode_save(tmp_path):
    backbone = _FakeBackbone()
    images = torch.randn(12, 16, 6)
    vlad = fit_vlad_codebook(backbone, images, k=4, max_patches=500)
    assert vlad.out_dim == 24
    z = vlad.encode(images[:3])
    assert z.shape == (3, 24)

    path = tmp_path / "vlad.pt"
    save_vlad(path, vlad)
    loaded = load_vlad(path, backbone)
    assert loaded.out_dim == 24
    assert torch.allclose(loaded.encode(images[:3]), z, atol=1e-6)


def test_calibration_temperature_and_reliability():
    rng = np.random.default_rng(0)
    prototypes = rng.normal(size=(5, 16)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    labels = np.array([0, 1, 2, 3, 4] * 6, dtype=np.int64)
    query = prototypes[labels] + 0.01 * rng.normal(size=(len(labels), 16))
    calib = calibrate_modality(query, prototypes, labels, "vision")
    assert calib.temperature > 0
    assert calib.reliability == pytest.approx(1.0, abs=1e-6)
    weights = reliability_weights({"vision": calib})
    assert weights["vision"] > 0.9

    scores = calibrated_scores({"vision": query[0]}, {"vision": prototypes}, {"vision": calib})
    assert scores.shape == (5,)
    score = calibrated_confidence(
        {"vision": query[0]}, {"vision": prototypes}, {"vision": calib}, [str(i) for i in range(5)]
    )
    assert 0.0 <= score <= 1.0


def test_projection_head_residual_without_whitening():
    head = ProjectionHead(12, 8, residual=True)
    assert head.whiten is None
    assert head.skip is not None
    z = head(torch.randn(5, 12))
    assert z.shape == (5, 8)
    assert torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_registry_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_vision_encoder("not-a-backbone", device="cpu")
    with pytest.raises(ValueError):
        build_tactile_encoder("not-a-backbone", device="cpu")
