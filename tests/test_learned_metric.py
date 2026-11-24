import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pss.eval.baselines.entity_transformer import TOKEN_DIM, TokenPair
from pss.memory.learned_metric import (
    LearnedMetricReID,
    evaluate_metric,
    train_metric,
)
from pss.memory.perception import Observation


def _pair(seed: int) -> TokenPair:
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


def _obs(seed: int, gt_id: str):
    rng = np.random.default_rng(seed)
    return Observation(
        gt_id=gt_id,
        pos=rng.normal(size=3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        appearance=rng.normal(size=128).astype(np.float32),
        raw={
            "vision": rng.normal(size=128).astype(np.float32),
            "geometry": rng.normal(size=128).astype(np.float32),
        },
        confidence=1.0,
        category="cup",
    )


def test_metric_training_and_evaluation_smoke():
    pairs = [_pair(i) for i in range(8)]
    model, info = train_metric(pairs, pairs, epochs=2, batch_size=4, seed=0)
    assert "best_val_reid" in info
    result = evaluate_metric(model, pairs, "cpu")
    assert 0.0 <= result["reid_accuracy"] <= 1.0
    assert "mixed" in result["per_subset"]


def test_learned_metric_reid_cost_matrix():
    model, _ = train_metric([_pair(0)], [_pair(0)], epochs=1, seed=0)
    adapter = LearnedMetricReID(model)
    pre = [_obs(1, "a"), _obs(2, "b")]
    post = [_obs(3, "a"), _obs(4, "b")]
    cost = adapter.cost_matrix(pre, post)
    assert cost.shape == (2, 2)
    assert np.isfinite(cost).all()
