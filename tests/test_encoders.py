from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

import torch

from pss.encoders.data import build_split, crop_object, load_category_examples, normalize_image
from pss.encoders.geometry import GEOMETRY_DIM, geometry_features
from pss.encoders.tactile import TactileMLP, load_tactile_expert, save_tactile_expert
from pss.encoders.train import train_encoder
from pss.encoders.vision import (
    SmallCNNEncoder,
    load_vision_expert,
    save_vision_expert,
)

DATA = Path("data/expert_v1")


def test_geometry_features_shape_and_finite():
    depth = np.abs(np.random.default_rng(0).normal(1.0, 0.2, size=(32, 32))).astype(np.float32)
    seg = np.zeros((32, 32, 2), dtype=np.int32)
    seg[10:20, 12:22, 0] = 2
    seg[10:20, 12:22, 1] = 5
    feats = geometry_features(depth, seg, 2)
    assert feats.shape == (GEOMETRY_DIM,)
    assert np.all(np.isfinite(feats))
    assert feats[0] == 1.0


def test_geometry_features_empty_mask():
    depth = np.ones((16, 16), dtype=np.float32)
    seg = np.zeros((16, 16, 2), dtype=np.int32)
    feats = geometry_features(depth, seg, 7)
    assert feats.shape == (GEOMETRY_DIM,)
    assert feats[0] == 0.0


def test_crop_and_normalize():
    rgb = np.random.default_rng(0).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    seg = np.zeros((32, 32, 2), dtype=np.int32)
    seg[8:24, 8:24, 0] = 3
    crop = crop_object(rgb, seg, 3, size=16)
    assert crop.shape == (16, 16, 3)
    x = normalize_image(crop)
    assert x.shape == (3, 16, 16)
    assert x.dtype == np.float32


def test_vision_encoder_forward_and_checkpoint(tmp_path):
    model = SmallCNNEncoder(out_dim=32)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 32)

    path = tmp_path / "vision.pt"
    save_vision_expert(path, model, ["a", "b"], 32)
    expert = load_vision_expert(path)
    out = expert.encode(x)
    assert out.shape == (2, 32)
    assert expert.classes == ["a", "b"]


def test_tactile_encoder_forward_and_checkpoint(tmp_path):
    model = TactileMLP(in_dim=64, out_dim=16)
    x = torch.rand(3, 64)
    assert model(x).shape == (3, 16)

    path = tmp_path / "tactile.pt"
    save_tactile_expert(path, model, ["a", "b"], 64)
    expert = load_tactile_expert(path)
    assert expert.encode(x).shape == (3, 16)


def test_train_encoder_smoke():
    torch.manual_seed(0)
    X = np.random.default_rng(0).normal(size=(8, 3, 32, 32)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    encoder = SmallCNNEncoder(out_dim=16)
    encoder, head, info = train_encoder(
        encoder, X, y, X, y, n_classes=2, epochs=1, batch_size=4
    )
    assert head.out_features == 2
    assert 0.0 <= info["best_val"] <= 1.0


@pytest.mark.skipif(not DATA.exists(), reason="run pss.sim.scripts.generate_dataset first")
def test_expert_dataset_loads():
    examples = load_category_examples(DATA, size=32)
    assert len(examples) > 0
    assert examples[0].image.shape == (3, 32, 32)
    assert examples[0].tactile.shape == (64,)
    classes = sorted({e.category for e in examples if e.split == "train"})
    Xv, Xt, y, _ = build_split(examples, "train", classes)
    assert Xv.shape[1:] == (3, 32, 32)
    assert Xt.shape[1] == 64
    assert len(y) == len(Xv)
