"""Tests for the learnable typed relation layer and scene-graph edges."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from pss.memory.learned_metric import TACTILE_TOKEN_DIM
from pss.memory.scene_graph import SceneGraph, Track
from pss.memory.typed_relations import (
    HybridRelationMatcher,
    RelationAwareGraphMatcher,
    RelationEdge,
    SpatialRelationHead,
    TypedRelationHead,
    frame_targets,
    write_relation_edges,
)


def _obs(pos, category, tactile_value):
    raw = {
        "vision": np.ones(128, dtype=np.float32),
        "geometry": np.ones(128, dtype=np.float32) * 2,
        "tactile": np.ones(128, dtype=np.float32) * tactile_value,
    }
    return SimpleNamespace(
        pos=np.asarray(pos, dtype=float),
        category=category,
        gt_category=category,
        raw=raw,
    )


def test_frame_targets_shapes_and_semantics():
    obs = [
        _obs([0.0, 0.0, 0.0], "cup", 1.0),
        _obs([0.2, 0.0, 0.0], "cup", 1.0),
        _obs([0.0, 0.2, 0.0], "bowl", 1.0),
    ]
    targets = frame_targets(obs)
    assert targets.spatial.shape == (3, 3, 5)
    assert targets.semantic.shape == (3, 3, 1)
    assert targets.tactile.shape == (3, 3, 2)
    # object 1 is right of object 0
    assert targets.spatial[0, 1, 1] == 1.0
    # object 2 is above object 0
    assert targets.spatial[0, 2, 2] == 1.0
    # 0 and 1 share a category, 0 and 2 do not
    assert targets.semantic[0, 1, 0] == 1.0
    assert targets.semantic[0, 2, 0] == 0.0
    assert targets.valid[0, 1] == 1.0
    assert targets.valid[0, 0] == 0.0


def test_typed_relation_head_shapes():
    head = TypedRelationHead(token_dim=32, hidden=16)
    a = torch.randn(2, 3, 32)
    b = torch.randn(2, 3, 32)
    logits = head(a, b)
    assert set(logits) == {"spatial", "semantic", "tactile"}
    assert logits["spatial"].shape == (2, 3, 3, 5)
    assert logits["semantic"].shape == (2, 3, 3, 1)
    assert logits["tactile"].shape == (2, 3, 3, 2)


def test_spatial_head_dead_zone_and_relation_order():
    labels = SpatialRelationHead.labels([0.0, 0.0], [0.0, 0.0])
    assert torch.equal(labels, torch.zeros(4))  # aligned: no relation fires
    labels = SpatialRelationHead.labels([0.0, 0.0], [1.0, 0.0])
    assert labels.tolist() == [1.0, 0.0, 0.0, 0.0]  # left
    labels = SpatialRelationHead.labels([0.0, 0.0], [0.0, 1.0])
    assert labels.tolist() == [0.0, 0.0, 0.0, 1.0]  # above


def _track(i, pos, category="cup"):
    return Track(
        id=f"obj_{i}",
        pos=np.asarray(pos, dtype=float),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        category=category,
        raw={
            "vision": np.ones(128, dtype=np.float32),
            "geometry": np.ones(128, dtype=np.float32),
            "tactile": np.ones(128, dtype=np.float32),
        },
    )


def test_scene_graph_relation_roundtrip(tmp_path):
    graph = SceneGraph()
    graph.add(_track(0, [0.0, 0.0, 0.0]))
    graph.add(_track(1, [0.2, 0.0, 0.0]))
    edge = RelationEdge("obj_0", "obj_1", "spatial", "right", 0.9)
    graph.set_relation(edge)
    assert len(graph.get_relations(source="obj_0")) == 1
    assert graph.relation_logits("obj_0", "obj_1")["spatial:right"]["probability"] == 0.9

    path = tmp_path / "graph.json"
    graph.save(path)
    loaded = SceneGraph.load(path)
    assert len(loaded.get_relations()) == 1
    assert loaded.relation_logits("obj_0", "obj_1")["spatial:right"]["probability"] == 0.9


def test_write_relation_edges_from_tracks():
    graph = SceneGraph()
    graph.add(_track(0, [0.0, 0.0, 0.0], "cup"))
    graph.add(_track(1, [0.2, 0.0, 0.0], "cup"))
    head = TypedRelationHead(token_dim=TACTILE_TOKEN_DIM, hidden=16)
    edges = write_relation_edges(graph, head, threshold=0.0)
    assert len(edges) > 0
    assert len(graph.relations) > 0
    for edge in edges:
        record = graph.relation_logits(edge.source, edge.target)[
            f"{edge.relation_type}:{edge.predicate}"
        ]
        assert record["predicate"] == edge.predicate


def test_relation_matchers_forward_shapes():
    head = TypedRelationHead(token_dim=TACTILE_TOKEN_DIM, hidden=16)
    tokens = torch.randn(1, 3, TACTILE_TOKEN_DIM)
    mask = torch.ones(1, 3)
    for model in (
        RelationAwareGraphMatcher(TACTILE_TOKEN_DIM, rel_head=head),
        HybridRelationMatcher(TACTILE_TOKEN_DIM, rel_head=head),
    ):
        out = model(tokens, tokens, mask, mask)
        assert out.shape == (1, 3, 3)
