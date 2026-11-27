"""Tests for the A0 gate-recall diagnostic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pss.eval.scripts.run_attribution_grid import recall_profile, union_recall


def test_recall_profile_ranks_and_coverage():
    scores = np.array(
        [
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
        ]
    )
    y = np.array([0, 2])
    profile = recall_profile(scores, y, ks=(1, 2, 3))

    assert profile["recall@1"] == 0.5
    assert profile["recall@2"] == 1.0
    assert profile["recall@3"] == 1.0
    assert profile["k_at_95"] == 2
    assert profile["k_at_99"] == 2
    assert profile["n"] == 2
    # ranks are [0, 1] -> MRR = (1 + 0.5) / 2
    assert abs(profile["mrr"] - 0.75) < 1e-9


def test_recall_profile_when_true_class_is_never_top1():
    # The true class is not in the score columns at all (y out of range would
    # raise, so simulate a class whose score never wins by using a large bank).
    scores = np.array([[5.0, 1.0], [5.0, 1.0]])
    y = np.array([1, 1])
    profile = recall_profile(scores, y, ks=(1, 2))
    assert profile["recall@1"] == 0.0
    assert profile["recall@2"] == 1.0


def test_union_recall_recovers_a_class_the_fused_score_misses():
    # Two classes in 2-D. Fused prefers class 0, but tactile alone prefers
    # class 1, so the union of top-1 sets contains the true class.
    rep = {
        "vision": np.array([[1.0, 0.0]]),
        "tactile": np.array([[0.0, 1.0]]),
        "geometry": np.array([[1.0, 0.0]]),
    }
    protos = {
        "vision": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "tactile": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "geometry": np.array([[1.0, 0.0], [0.0, 1.0]]),
    }
    bank = SimpleNamespace(protos=protos, categories=["a", "b"])

    assert union_recall(rep, bank, rows=[0], y=np.array([1]), k=1) == 1.0
    assert union_recall(rep, bank, rows=[0], y=np.array([0]), k=1) == 1.0


def test_union_recall_misses_when_no_modality_proposes_the_class():
    rep = {
        "vision": np.array([[1.0, 0.0]]),
        "tactile": np.array([[1.0, 0.0]]),
        "geometry": np.array([[1.0, 0.0]]),
    }
    protos = {
        "vision": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "tactile": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "geometry": np.array([[1.0, 0.0], [0.0, 1.0]]),
    }
    bank = SimpleNamespace(protos=protos, categories=["a", "b"])
    assert union_recall(rep, bank, rows=[0], y=np.array([1]), k=1) == 0.0
