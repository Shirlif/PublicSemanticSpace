"""Tests for end-to-end / bilevel joint training."""

from __future__ import annotations

import numpy as np
import torch

from pss.learning.bilevel import (
    EndToEndPSS,
    RawBatch,
    RawPairs,
    TrainConfig,
    consistency_metrics,
    episodic_prototype_loss,
    evaluate_recognition,
    load_joint,
    permutation_loss,
    save_joint,
    train_bilevel,
)


def _batch(b=2, n=3, vision=8, tactile=4, geometry=3, categories=4):
    return RawBatch(
        vision=torch.randn(b, n, vision),
        tactile=torch.randn(b, n, tactile),
        geometry=torch.randn(b, n, geometry),
        pose=torch.randn(b, n, 3),
        category=torch.eye(categories)[torch.randint(0, categories, (b, n))],
        mask=torch.ones(b, n),
    )


def _pairs(b=2, n=3, **kwargs):
    pre = _batch(b, n, **kwargs)
    post = _batch(b, n, **kwargs)
    labels = torch.zeros(b, n, n)
    for i in range(b):
        for j in range(min(n, 3)):
            labels[i, j, j] = 1.0
    targets = {
        "spatial": torch.zeros(b, n, n, 5),
        "semantic": torch.zeros(b, n, n, 1),
        "tactile": torch.zeros(b, n, n, 2),
    }
    valid = torch.zeros(b, n, n)
    valid[:, :, :n] = 1.0
    return RawPairs(
        pre=pre,
        post=post,
        labels=labels,
        pre_targets={k: v.clone() for k, v in targets.items()},
        post_targets={k: v.clone() for k, v in targets.items()},
        pre_valid=valid.clone(),
        post_valid=valid.clone(),
        subsets=["mixed"] * b,
    )


def test_bind_and_recognition_views():
    model = EndToEndPSS(
        vision_dim=8, tactile_dim=4, geometry_dim=3, n_categories=3,
        category_dim=4, token_dim=8, hidden=16,
    )
    batch = _batch()
    match_token = model.bind(batch, include_category=True)
    rec_token = model.bind_rec(batch)
    assert match_token.shape == (2, 3, 8)
    assert rec_token.shape == (2, 3, 8)
    assert not torch.allclose(match_token, rec_token)


def test_prototype_loss_finite():
    tokens = torch.randn(12, 8)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    loss = episodic_prototype_loss(tokens, labels, n_support=2, n_query=2)
    assert torch.isfinite(loss)


def test_permutation_loss_is_zero_for_pairwise_head():
    model = EndToEndPSS(
        vision_dim=8, tactile_dim=4, geometry_dim=3, n_categories=3,
        category_dim=4, token_dim=8, hidden=16,
    )
    pairs = _pairs()
    loss = permutation_loss(model, pairs.pre, pairs.post)
    assert float(loss.detach()) < 1e-5


def test_train_bilevel_smoke_and_eval():
    categories = 3
    n = 12
    labels = torch.tensor([i % categories for i in range(n)])
    data = {
        "vision": torch.randn(n, 8),
        "tactile": torch.randn(n, 4),
        "geometry": torch.randn(n, 3),
        "pose": torch.randn(n, 3),
        "category": torch.eye(4)[torch.randint(0, 4, (n,))],
        "labels": labels,
        "instances": torch.arange(n) % 6,
        "train_vision": torch.randn(n, 8),
        "train_tactile": torch.randn(n, 4),
        "train_geometry": torch.randn(n, 3),
        "train_pose": torch.randn(n, 3),
        "train_category": torch.eye(4)[torch.randint(0, 4, (n,))],
        "train_labels": labels,
        "test_vision": torch.randn(n, 8),
        "test_tactile": torch.randn(n, 4),
        "test_geometry": torch.randn(n, 3),
        "test_pose": torch.randn(n, 3),
        "test_category": torch.eye(4)[torch.randint(0, 4, (n,))],
        "test_labels": labels,
        "n_classes": categories,
    }
    pairs = _pairs()
    model, info = train_bilevel(
        data,
        pairs,
        TrainConfig(mode="joint", epochs=2, n_categories=categories, token_dim=8, hidden=16),
    )
    assert info["mode"] == "joint"
    acc = evaluate_recognition(model, data)
    assert 0.0 <= acc <= 1.0
    cons = consistency_metrics(model, pairs)
    assert all(np.isfinite(v) for v in cons.values())


def test_save_load_roundtrip(tmp_path):
    model = EndToEndPSS(
        vision_dim=8, tactile_dim=4, geometry_dim=3, n_categories=3,
        category_dim=4, token_dim=8, hidden=16,
    )
    path = tmp_path / "joint.pt"
    save_joint(model, path)
    loaded = load_joint(path)
    assert loaded.token_dim == 8
