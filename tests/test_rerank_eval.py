"""Tests for the A2 rerank components."""

from __future__ import annotations

import numpy as np
import torch

from pss.eval.scripts.run_rerank_eval import (
    BilinearMetric,
    candidate_mask,
    topk_candidates,
)
from pss.prototypes.scorers import MahalanobisScorer


def _three_clusters(n=20, seed=0):
    rng = np.random.default_rng(seed)
    centers = [(-3.0, 0.0), (3.0, 0.0), (0.0, 3.0)]
    Z = np.vstack([rng.normal(loc=c, scale=0.4, size=(n, 2)) for c in centers])
    y = np.repeat(np.arange(3), n)
    return Z, y


def test_topk_candidates_orders_by_score():
    scores = np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])
    cand = topk_candidates(scores, 2)
    assert cand.shape == (2, 2)
    assert cand[0].tolist() == [0, 2]
    assert cand[1].tolist() == [1, 2]


def test_candidate_mask_marks_only_candidates():
    mask = candidate_mask(np.array([[0, 2], [1, 2]]), 3)
    assert mask.tolist() == [[True, False, True], [False, True, True]]


def test_bilinear_initialises_to_lda_ranking():
    Z, y = _three_clusters()
    lda = MahalanobisScorer().fit(Z, y)
    model = BilinearMetric(Z.shape[1], lda.precision_)
    model.set_means(lda.means_)
    with torch.no_grad():
        bilinear = model(torch.as_tensor(Z, dtype=torch.float32)).numpy()
    lda_scores = lda.scores(Z)

    # The two differ by 0.5 * z^T P z, which is constant across classes and so
    # cannot change the ranking.
    zPz = np.einsum("ij,ij->i", Z @ lda.precision_, Z)
    assert np.allclose(bilinear - lda_scores, 0.5 * zPz[:, None], atol=1e-4)
    assert (bilinear.argmax(1) == lda_scores.argmax(1)).all()


def test_bilinear_update_moves_one_mean():
    Z, y = _three_clusters()
    lda = MahalanobisScorer().fit(Z, y)
    model = BilinearMetric(Z.shape[1], lda.precision_)
    model.set_means(lda.means_)
    model.update_class(0, np.array([[-9.0, -9.0]]))
    with torch.no_grad():
        scores = model(torch.as_tensor([[-9.0, -9.0]], dtype=torch.float32)).numpy()
    assert scores.argmax(1)[0] == 0
