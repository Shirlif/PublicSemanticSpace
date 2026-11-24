from types import SimpleNamespace

import numpy as np

from pss.memory.graph_matcher import GraphMatcher, PrunedGraphMatcherReID
from pss.memory.learned_metric import TOKEN_DIM


def _entity(pos, category="block"):
    v = np.ones(128, dtype=np.float32) / np.sqrt(128)
    g = np.ones(128, dtype=np.float32) / np.sqrt(128)
    return SimpleNamespace(
        pos=np.asarray(pos, dtype=float),
        category=category,
        appearance=v,
        raw={"vision": v, "geometry": g},
    )


def test_pruned_cost_and_match():
    tracks = [_entity([0, 0, 0]), _entity([1, 0, 0]), _entity([5, 0, 0])]
    observations = [
        _entity([0.01, 0, 0]),
        _entity([1.01, 0, 0]),
        _entity([5.01, 0, 0]),
    ]
    model = GraphMatcher(token_dim=TOKEN_DIM, d_model=32, layers=1)
    adapter = PrunedGraphMatcherReID(
        model, pose_gate=0.1, top_k=4, max_cost=10.0
    )
    cost = adapter.cost_matrix(tracks, observations)
    assert cost.shape == (3, 3)
    matches = adapter.match(tracks, observations)
    assert 0 < len(matches) <= 3
    assert all(0 <= i < 3 and 0 <= j < 3 for i, j, _ in matches)


def test_pruned_gate_drops_far_pairs():
    tracks = [_entity([0, 0, 0])]
    observations = [_entity([5, 0, 0])]
    model = GraphMatcher(token_dim=TOKEN_DIM, d_model=32, layers=1)
    adapter = PrunedGraphMatcherReID(model, pose_gate=0.1, top_k=4, max_cost=10.0)
    assert adapter.match(tracks, observations) == []
    assert np.all(adapter.cost_matrix(tracks, observations) >= 1e6)
