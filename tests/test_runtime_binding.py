"""Tests for object attention as a runtime binding option (row 3)."""

from __future__ import annotations

import numpy as np
import torch

from pss.memory.learned_metric import TACTILE_TOKEN_DIM
from pss.memory.object_attention import (
    ObjectAttentionMatcher,
    ObjectAttentionReID,
    load_object_attention,
    save_object_attention,
)
from pss.memory.perception import Observation
from pss.runtime import EntityLayer, PSSRuntime


def _obs(i, category="cup"):
    raw = {
        "vision": np.ones(128, dtype=np.float32) * (i + 1),
        "geometry": np.ones(128, dtype=np.float32) * (i + 1),
        "tactile": np.ones(128, dtype=np.float32) * (i + 1),
    }
    return Observation(
        gt_id=f"obj_{i}",
        pos=np.array([0.1 * i, 0.0, 0.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        appearance=np.eye(3)[i % 3],
        raw=raw,
        confidence=0.9,
        category=category,
        frame=0,
        gt_category=category,
    )


def test_object_attention_reid_cost_matrix():
    model = ObjectAttentionMatcher(token_dim=TACTILE_TOKEN_DIM, d_model=32, heads=2, layers=1)
    adapter = ObjectAttentionReID(model)
    tracks = [_obs(0), _obs(1)]
    observations = [_obs(2), _obs(3)]
    cost = adapter.cost_matrix(tracks, observations)
    assert cost.shape == (2, 2)
    assert np.isfinite(cost).all()
    assert 0.0 <= adapter.cost(tracks[0], observations[0]) <= 1.0


def test_entity_layer_and_runtime_report_associator():
    model = ObjectAttentionMatcher(token_dim=TACTILE_TOKEN_DIM, d_model=32, heads=2, layers=1)
    adapter = ObjectAttentionReID(model)
    layer = EntityLayer(associator=adapter)
    assert layer.associator_name == "ObjectAttentionReID"
    runtime = PSSRuntime(entity_layer=layer)
    assert runtime.summary()["components"]["associator"] == "ObjectAttentionReID"


def test_object_attention_save_load_roundtrip(tmp_path):
    model = ObjectAttentionMatcher(token_dim=TACTILE_TOKEN_DIM, d_model=32, heads=2, layers=1)
    path = tmp_path / "object_attention.pt"
    save_object_attention(model, path)
    loaded = load_object_attention(path)
    assert loaded.encoder.proj.in_features == TACTILE_TOKEN_DIM
