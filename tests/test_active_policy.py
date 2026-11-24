"""Tests for learned active-set and write policies."""

from __future__ import annotations

import numpy as np
import torch

from pss.memory.active_policy import (
    ActiveSetPolicy,
    WritePolicy,
    calibrate_margin,
    select_active,
    train_active_policy,
    train_write_policy,
    tracking_pair_features,
)
from pss.memory.learned_metric import TACTILE_TOKEN_DIM
from pss.memory.typed_relations import RelationPair, frame_targets


def _pair(seed=0):
    rng = np.random.default_rng(seed)
    pre = rng.normal(size=(3, TACTILE_TOKEN_DIM)).astype(np.float32)
    post = rng.normal(size=(3, TACTILE_TOKEN_DIM)).astype(np.float32)
    labels = np.eye(3, dtype=np.float32)
    return RelationPair(
        pre=pre,
        post=post,
        pre_mask=np.ones(3, dtype=np.float32),
        post_mask=np.ones(3, dtype=np.float32),
        pre_targets=frame_targets([]),
        post_targets=frame_targets([]),
        labels=labels,
        subset="mixed",
        sample_id=str(seed),
    )


def test_tracking_pair_features_shape():
    pair = _pair()
    features = tracking_pair_features(pair.pre, pair.post)
    assert features.shape == (3, 3, 8)
    assert np.isfinite(features).all()


def test_select_active_topk_and_fallback():
    scores = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.1]], dtype=np.float32)
    sel = select_active(scores, k=1)
    assert sel.mask.tolist() == [[True, False, False], [False, True, False]]
    assert sel.mean_active == 1.0
    assert not sel.fallback_rows.any()

    # A large threshold forces every row to keep all candidates.
    sel = select_active(scores, k=1, margin_threshold=10.0)
    assert sel.mask.all()
    assert sel.fallback_rows.all()
    assert sel.mean_active == 3.0


def test_train_active_policy_and_calibrate(tmp_path):
    pairs = [_pair(i) for i in range(4)]
    policy = train_active_policy(pairs, epochs=2, seed=0)
    scores = policy.scores(tracking_pair_features(pairs[0].pre, pairs[0].post))
    assert scores.shape == (3, 3)
    threshold = calibrate_margin(policy, pairs, k=1, target_fallback=0.2)
    assert np.isfinite(threshold)


def test_write_policy_bands_and_training():
    policy = WritePolicy()
    features = np.zeros(8, dtype=np.float32)
    decision = policy.decide(features, write_threshold=0.6, review_low=0.4, review_high=0.6)
    assert decision in ("write", "review", "skip")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 8)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    trained = train_write_policy(X, y, epochs=10, seed=0)
    high = np.zeros(8, dtype=np.float32)
    high[0] = 3.0
    low = np.zeros(8, dtype=np.float32)
    low[0] = -3.0
    assert trained.probability(high) > trained.probability(low)
