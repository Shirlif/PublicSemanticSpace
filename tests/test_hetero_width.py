"""Tests for the heterogeneous-width expert adapter (F4)."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from pss.encoders.hetero import (
    HETERO_BY_NAME,
    REQUIRED_WIDTHS,
    build_hetero_expert,
    feature_dim,
    get_expert_spec,
)
from pss.encoders.registry import build_vision_encoder
from pss.projection.hetero import (
    CONTRACT_CASES,
    EXPECTED_CONTRACT,
    admission_contract,
    few_shot_prototype_curve,
    few_shot_ridge_curve,
    project_features,
    split_family_indices,
    train_expert_head,
)
from pss.projection.heads import PUBLIC_DIM, ProjectionHead
from pss.experts import CategoryCodebook


WIDTHS = (64, 384, 768)
CLASSES = ("a", "b", "c", "d")
TRAIN_ENTRIES = 12
VAL_ENTRIES = 3
TEST_ENTRIES = 3
VIEWS = 4


def _synthetic(width: int, seed: int = 0):
    """Linearly separable synthetic features with a train/val/test split."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(CLASSES), width))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rows, categories, splits, views = [], [], [], []
    for ci, category in enumerate(CLASSES):
        for entry in range(TRAIN_ENTRIES + VAL_ENTRIES + TEST_ENTRIES):
            split = (
                "train"
                if entry < TRAIN_ENTRIES
                else "val"
                if entry < TRAIN_ENTRIES + VAL_ENTRIES
                else "test"
            )
            for view in range(VIEWS):
                noise = rng.normal(scale=0.03, size=width)
                rows.append(centers[ci] + noise)
                categories.append(category)
                splits.append(split)
                views.append(view)
    X = np.stack(rows).astype(np.float32)
    feats = SimpleNamespace(
        vision=X,
        tactile=X,
        categories=categories,
        splits=splits,
        views=views,
    )
    return feats, X


def _labels(present):
    index = {c: i for i, c in enumerate(present)}
    return index


def test_required_widths_cover_three_distinct_dims():
    specs = [HETERO_BY_NAME[name] for name in REQUIRED_WIDTHS]
    assert {spec.in_dim for spec in specs} == {64, 384, 768}
    assert HETERO_BY_NAME["vision_clip_768"].model_id == "openai/clip-vit-base-patch16"


def test_get_expert_spec_rejects_unknown():
    with pytest.raises(ValueError, match="unknown heterogeneous expert"):
        get_expert_spec("not_an_expert")


def test_feature_dim_reads_wrapped_encoders():
    class Inner:
        out_dim = 384

    assert feature_dim(SimpleNamespace(encoder=Inner())) == 384
    assert feature_dim(SimpleNamespace(out_dim=768)) == 768
    assert feature_dim(SimpleNamespace(input_dim=64)) == 64


def test_projection_head_accepts_each_width():
    for width in WIDTHS:
        head = ProjectionHead(width, out_dim=PUBLIC_DIM)
        z = head(torch.randn(3, width))
        assert z.shape == (3, PUBLIC_DIM)
        assert torch.allclose(z.norm(dim=1), torch.ones(3), atol=1e-5)


def test_registry_passes_model_id_to_clip(monkeypatch):
    import pss.encoders.vision.clip as clip_mod

    captured = {}

    class _FakeClip:
        out_dim = 768

        def encode(self, images):
            return torch.zeros((len(images), 768))

    def fake_loader(**kwargs):
        captured.update(kwargs)
        return _FakeClip()

    monkeypatch.setattr(clip_mod, "load_clip_vision_encoder", fake_loader)
    build_vision_encoder("clip", model_id="openai/clip-vit-base-patch16")
    assert captured["model_id"] == "openai/clip-vit-base-patch16"


def test_build_hetero_expert_checks_declared_width(monkeypatch):
    spec = get_expert_spec("vision_dinov2_384")

    def fake_build(**kwargs):
        return SimpleNamespace(out_dim=384)

    monkeypatch.setattr(type(spec), "build", lambda self, **kwargs: fake_build(**kwargs))
    expert = build_hetero_expert(spec)
    assert feature_dim(expert) == 384

    def wrong_build(self, **kwargs):
        return SimpleNamespace(out_dim=999)

    monkeypatch.setattr(type(spec), "build", wrong_build)
    with pytest.raises(ValueError, match="declared in_dim"):
        build_hetero_expert(spec)


def test_split_uses_disjoint_head_and_calibration_views():
    feats, _ = _synthetic(64)
    split = split_family_indices(feats, CLASSES)
    assert split.head_views == [0, 1]
    assert split.calib_views == [2, 3]
    head_set = set(split.head_idx)
    calib_set = {i for idx in split.calib_by_class.values() for i in idx}
    assert head_set and calib_set
    assert not head_set & calib_set
    assert split.present == list(CLASSES)
    # 12 train entries x 2 head views x 4 classes
    assert len(split.head_idx) == TRAIN_ENTRIES * 2 * len(CLASSES)
    # val+test holdout per class
    assert all(
        len(split.holdout_by_class[c]) == (VAL_ENTRIES + TEST_ENTRIES) * VIEWS
        for c in CLASSES
    )


@pytest.mark.parametrize("width", WIDTHS)
def test_end_to_end_width_pipeline(width):
    feats, X = _synthetic(width)
    split = split_family_indices(feats, CLASSES)
    labels = np.array([_labels(split.present).get(c, -1) for c in feats.categories])
    head = train_expert_head(
        X, labels, split.head_idx, width, epochs=40, seed=0, device="cpu"
    )
    Z = project_features(head, X, device="cpu")
    assert Z.shape == (len(X), PUBLIC_DIM)

    gt = [feats.categories[i] for i in split.test_idx]
    proto = few_shot_prototype_curve(
        Z, split.calib_by_class, split.test_idx, gt, split.present, (0, 1, 5, 20)
    )
    assert proto[1] >= 0.90

    codebook = CategoryCodebook.from_vectors(
        split.present,
        np.random.default_rng(0).normal(size=(len(split.present), PUBLIC_DIM)),
    )
    ridge = few_shot_ridge_curve(
        Z, split.calib_by_class, split.test_idx, gt, codebook, split.present, (0, 1, 5, 20)
    )
    assert set(ridge) == {0, 1, 5, 20}
    assert all(np.isfinite(ridge[k]) for k in ridge)


@pytest.mark.parametrize("width", WIDTHS)
def test_admission_contract_same_decisions_at_every_width(width):
    feats, X = _synthetic(width)
    split = split_family_indices(feats, CLASSES)
    labels = np.array([_labels(split.present).get(c, -1) for c in feats.categories])
    head = train_expert_head(
        X, labels, split.head_idx, width, epochs=40, seed=0, device="cpu"
    )
    Z = project_features(head, X, device="cpu")
    result = admission_contract(
        Z,
        split.bank_idx,
        split.holdout_by_class,
        split.present,
        "vision",
        far_class="d",
        near_class="a",
    )
    assert result["consistent_with_expected"]
    for case in CONTRACT_CASES:
        assert result[case]["decision"] == EXPECTED_CONTRACT[case]


def test_admission_contract_requires_two_classes():
    feats, X = _synthetic(64)
    split = split_family_indices(feats, ("a",))
    with pytest.raises(ValueError):
        admission_contract(
            X,
            split.bank_idx,
            split.holdout_by_class,
            split.present,
            "vision",
            far_class="a",
            near_class="a",
        )
