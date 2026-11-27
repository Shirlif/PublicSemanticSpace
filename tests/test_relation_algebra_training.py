"""Tests for the optional relation-algebra training hook."""

from __future__ import annotations

import numpy as np
import torch

from pss.eval.scripts.run_typed_relations import (
    _algebra_penalty,
    _extra_axis_rows,
    _pair_feature,
    _p5b_relation_rows,
    _reverse_pair_features,
    _train_spatial_head,
)
from pss.memory.typed_relations import SpatialRelationHead


def test_reverse_pair_features_is_an_involution():
    x = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.2, 0.2]])
    reversed_x = _reverse_pair_features(x)
    assert torch.allclose(_reverse_pair_features(reversed_x), x)
    assert torch.allclose(reversed_x[..., 0], x[..., 2])
    assert torch.allclose(reversed_x[..., 4], -x[..., 4])
    assert torch.allclose(reversed_x[..., 6], x[..., 6])


def test_algebra_penalty_zero_for_consistent_reverse():
    left = torch.tensor([[8.0, -8.0, -8.0, -8.0]])
    right = torch.tensor([[-8.0, 8.0, -8.0, -8.0]])
    assert float(_algebra_penalty(left, right)) < 1e-4


def test_algebra_penalty_positive_when_reverse_agrees():
    left = torch.tensor([[8.0, -8.0, -8.0, -8.0]])
    also_left = torch.tensor([[8.0, -8.0, -8.0, -8.0]])
    assert float(_algebra_penalty(left, also_left)) > 0.5


def test_train_spatial_head_accepts_algebra_weight():
    head, pairs, exact, per = _train_spatial_head(
        "cpu",
        epochs=2,
        seed=0,
        train_regions=("A", "B", "C"),
        algebra_weight=0.1,
        n_per_pair=10,
    )
    assert pairs
    assert 0.0 <= exact <= 1.0
    assert 0.0 <= per <= 1.0
    assert head is not None


def test_extra_axis_rows_isolate_one_axis():
    rng = np.random.default_rng(0)
    horizontal = _extra_axis_rows(rng, "x", 8, 0.0)
    assert len(horizontal) == 8
    for features, _labels in horizontal:
        # feature order: [ax, ay, bx, by, dx, dy, |dx|, |dy|]
        assert abs(features[5]) < 1e-9
        assert abs(features[4]) >= 0.5

    vertical = _extra_axis_rows(rng, "y", 8, 0.0)
    assert len(vertical) == 8
    for features, _labels in vertical:
        assert abs(features[4]) < 1e-9
        assert abs(features[5]) >= 0.5


def test_p5b_relation_rows_appends_extra_pairs():
    X, _y, _Xte, _yte, _pairs = _p5b_relation_rows(
        train_regions=("B", "C", "D"), n_per_pair=1, extra_horizontal=5
    )
    assert X.shape[0] == 6 + 5


def test_relative_feature_mode_has_four_dims():
    features = _pair_feature((0.0, 0.0), (1.0, 0.0), "relative")
    assert len(features) == 4
    assert features[0] == 1.0
    assert features[1] == 0.0

    head = SpatialRelationHead(feature_mode="relative")
    out = head(torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 0.0]]))
    assert out.shape == (1, 4)


def test_p5b_relation_rows_relative_dim():
    X, _y, Xte, _yte, _pairs = _p5b_relation_rows(
        train_regions=("B", "C", "D"), n_per_pair=2, feature_mode="relative"
    )
    assert X.shape[1] == 4
    assert Xte.shape[1] == 4
