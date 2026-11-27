"""Tests for the open-category stream protocol and preregistration."""

from __future__ import annotations

import numpy as np
import pytest

from pss.eval.open_stream import (
    calibrate_threshold,
    cluster_features,
    select_query,
)
from pss.eval.open_stream_preregistration import (
    check_open_stream_preregistration,
)
from pss.eval.scripts.run_open_stream import _category_of, _instance_labels
from pss.prototypes.scorers import MahalanobisScorer

MODALITIES = ("vision", "tactile", "geometry")


def _rep(n: int = 40, dim: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {m: rng.normal(size=(n, dim)) for m in MODALITIES}


def test_novelty_query_picks_lowest_scores():
    rep = _rep()
    rows = np.arange(40)
    scores = np.linspace(0.0, 1.0, 40)
    query = select_query(
        rep, rows, scores, 5, "novelty", np.random.default_rng(0), 0.5
    )
    assert len(query) == 5
    assert np.all(scores[query] <= scores[np.argsort(scores)[4]])


def test_random_query_returns_the_budget():
    rep = _rep()
    rows = np.arange(40)
    scores = np.linspace(0.0, 1.0, 40)
    query = select_query(
        rep, rows, scores, 7, "random", np.random.default_rng(0), 0.5
    )
    assert len(query) == 7
    assert len(set(query.tolist())) == 7


def test_active_cluster_returns_unique_rows():
    rep = _rep()
    rows = np.arange(40)
    scores = np.linspace(0.0, 1.0, 40)
    query = select_query(
        rep, rows, scores, 5, "active_cluster", np.random.default_rng(0), 0.8
    )
    assert len(query) == 5
    assert len(set(query.tolist())) == 5


def test_cluster_features_separates_two_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack(
        [rng.normal(0.0, 0.05, size=(12, 4)), rng.normal(5.0, 0.05, size=(12, 4))]
    )
    labels = cluster_features(X, min_cluster_size=2)
    clusters = {int(x) for x in labels if int(x) >= 0}
    assert len(clusters) >= 2


def test_calibrate_threshold_is_finite_and_between_scores():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.0, 0.1, size=(10, 4)), rng.normal(1.0, 0.1, size=(10, 4))])
    y = np.array([0] * 10 + [1] * 10)
    scorer = MahalanobisScorer().fit(X, y)
    threshold = calibrate_threshold(scorer, X, acceptance=0.95)
    assert np.isfinite(threshold)
    scores = scorer.scores(X).max(axis=1)
    assert scores.min() <= threshold <= scores.max()


def test_category_of_strips_instance_suffix():
    assert _category_of("cup_v012") == "cup"
    assert _category_of("cup") == "cup"


def test_instance_labels_reads_train_split(tmp_path):
    import json

    index = {
        "entries": [
            {"split": "train", "instance": "cup_v000", "category": "cup"},
            {"split": "train", "instance": "cup_v001", "category": "cup"},
            {"split": "test", "instance": "cup_v002", "category": "cup"},
            {"split": "train", "instance": "mug_v000", "category": "mug"},
        ]
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    seen, novel, categories = _instance_labels(str(path), 2, start=100)
    assert seen == ["cup_v000", "cup_v001", "mug_v000"]
    assert novel == ["cup_v100", "cup_v101", "mug_v100", "mug_v101"]
    assert categories == ["cup", "mug"]


def test_prereg_passing_arms():
    observed = {
        "active_cluster": {"novel_acc": 0.85, "old_forgetting": 0.01},
        "random_query": {"novel_acc": 0.70},
        "frozen": {"novel_acc": 0.05},
        "tier_c": {"novel_acc": 0.65, "unknown_auroc": 0.85},
    }
    relative = {"random_query_vs_active": -0.15}
    result = check_open_stream_preregistration(observed, relative)
    assert result["ok"], result["mismatches"]


def test_prereg_random_must_trail():
    observed = {
        "active_cluster": {"novel_acc": 0.85, "old_forgetting": 0.01},
        "random_query": {"novel_acc": 0.83},
    }
    result = check_open_stream_preregistration(
        observed, {"random_query_vs_active": -0.02}
    )
    assert not result["ok"]
    assert any("random_query_vs_active" in m for m in result["mismatches"])


def test_prereg_self_training_must_fail():
    passing = check_open_stream_preregistration(
        {}, {"self_train_old_forgetting": 0.15}
    )
    assert passing["ok"]
    failing = check_open_stream_preregistration(
        {}, {"self_train_old_forgetting": 0.0, "self_train_novel_delta": 0.0}
    )
    assert not failing["ok"]


def test_prereg_no_replay_must_forget_more():
    observed = {
        "active_no_replay": {"old_forgetting": 0.12},
        "active_replay": {"old_forgetting": 0.02},
    }
    result = check_open_stream_preregistration(
        observed, {"no_replay_vs_replay": 0.10}
    )
    assert result["ok"], result["mismatches"]


def test_prereg_tier_c_unknown_bar():
    observed = {"tier_c": {"novel_acc": 0.65, "unknown_auroc": 0.70}}
    result = check_open_stream_preregistration(observed)
    assert not result["ok"]
    assert any("unknown_auroc" in m for m in result["mismatches"])


def test_prereg_model_cells_either_metric():
    result = check_open_stream_preregistration(
        {}, {"model_cells_top1_match": 0.70}
    )
    assert result["ok"]
    result = check_open_stream_preregistration(
        {}, {"model_cells_spearman": 0.75}
    )
    assert result["ok"]
    result = check_open_stream_preregistration(
        {}, {"model_cells_top1_match": 0.2, "model_cells_spearman": 0.2}
    )
    assert not result["ok"]
