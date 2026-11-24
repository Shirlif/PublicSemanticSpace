import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pss.eval.baselines.entity_transformer import TOKEN_DIM, TokenPair
from pss.memory import Track
from pss.memory.graph_matcher import GraphMatcher, GraphMatcherReID, evaluate_graph, train_graph


def _pair(seed):
    rng = np.random.default_rng(seed)
    return TokenPair(
        pre=rng.normal(size=(3, TOKEN_DIM)).astype(np.float32),
        post=rng.normal(size=(3, TOKEN_DIM)).astype(np.float32),
        pre_mask=np.ones(3, dtype=np.float32),
        post_mask=np.ones(3, dtype=np.float32),
        labels=np.eye(3, dtype=np.float32),
        subset="mixed",
        sample_id=str(seed),
    )


def _track(i, gt_id):
    rng = np.random.default_rng(i)
    return Track(
        id=gt_id,
        pos=rng.normal(size=3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        raw={
            "vision": rng.normal(size=128).astype(np.float32),
            "geometry": rng.normal(size=128).astype(np.float32),
        },
        category="cup",
    )


def test_graph_matcher_forward_and_train_smoke():
    pairs = [_pair(i) for i in range(6)]
    model = GraphMatcher(d_model=32, layers=1)
    scores = model(
        torch.zeros(2, 3, TOKEN_DIM),
        torch.zeros(2, 3, TOKEN_DIM),
        torch.ones(2, 3),
        torch.ones(2, 3),
    )
    assert scores.shape == (2, 3, 3)
    trained, info = train_graph(pairs, pairs, epochs=1, batch_size=3, seed=0)
    result = evaluate_graph(trained, pairs)
    assert 0.0 <= result["reid_accuracy"] <= 1.0


def test_graph_matcher_reid_cost_matrix():
    model, _ = train_graph([_pair(0)], [_pair(0)], epochs=1, seed=0)
    adapter = GraphMatcherReID(model)
    tracks = [_track(0, "a"), _track(1, "b")]
    obs = [_track(2, "a"), _track(3, "b")]
    cost = adapter.cost_matrix(tracks, obs)
    assert cost.shape == (2, 2)
    assert np.isfinite(cost).all()
