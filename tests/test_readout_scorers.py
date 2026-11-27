"""Tests for the distance-based class scorers."""

from __future__ import annotations

import numpy as np

from pss.prototypes.scorers import (
    CosineProtoScorer,
    KNNScorer,
    MahalanobisScorer,
    fit_novelty_map,
    fuse_scores,
    predict_classes,
)


def _two_clusters(n=20, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=(-2.0, -2.0), scale=0.3, size=(n, 2))
    b = rng.normal(loc=(2.0, 2.0), scale=0.3, size=(n, 2))
    Z = np.vstack([a, b])
    y = np.array([0] * n + [1] * n)
    return Z, y


def test_cosine_proto_scorer_separates_clusters():
    Z, y = _two_clusters()
    scorer = CosineProtoScorer().fit(Z, y)
    assert (scorer.scores(Z).argmax(axis=1) == y).all()
    # The centroid of a cluster points roughly along its mean direction.
    assert scorer.scores(np.array([[3.0, 3.0]])).argmax(axis=1)[0] == 1


def test_cosine_update_moves_the_centroid():
    Z, y = _two_clusters()
    scorer = CosineProtoScorer().fit(Z, y)
    scorer.update_class(0, np.array([[5.0, -5.0]]))
    assert scorer.scores(np.array([[5.0, -5.0]])).argmax(axis=1)[0] == 0


def test_mahalanobis_separates_clusters_and_updates_mean():
    Z, y = _two_clusters()
    scorer = MahalanobisScorer().fit(Z, y)
    assert (scorer.scores(Z).argmax(axis=1) == y).all()
    assert scorer.scores(np.array([[9.0, 9.0]])).argmax(axis=1)[0] == 1

    scorer.update_class(0, np.array([[-9.0, -9.0]]))
    assert scorer.scores(np.array([[-9.0, -9.0]])).argmax(axis=1)[0] == 0
    # The pooled precision is frozen by design.
    assert scorer.covariance_frozen


def test_knn_scores_and_updates():
    Z, y = _two_clusters()
    scorer = KNNScorer(k=1).fit(Z, y)
    assert (scorer.scores(Z).argmax(axis=1) == y).all()

    scorer.update_class(0, np.array([[0.0, 0.0]]))
    assert scorer.scores(np.array([[0.0, 0.0]])).argmax(axis=1)[0] == 0


def test_fuse_scores_is_invariant_to_positive_rescaling():
    m1 = np.array([[1.0, 0.0, -1.0]])
    m2 = np.array([[0.5, 0.2, -0.1]])
    weights = {"m1": 1.0, "m2": 1.0}
    base = fuse_scores({"m1": m1, "m2": m2}, weights)
    scaled = fuse_scores({"m1": 10.0 * m1, "m2": m2}, weights)
    assert np.allclose(base, scaled)


def test_novelty_map_standardises_train_scores():
    Z, y = _two_clusters()
    scorer = CosineProtoScorer().fit(Z, y)
    novelty = fit_novelty_map(scorer, Z)
    mapped = novelty(scorer.scores(Z).max(axis=1))
    assert abs(float(mapped.mean())) < 1e-9
    assert abs(float(mapped.std()) - 1.0) < 1e-6


def test_novelty_map_ranks_outliers_lower():
    Z, y = _two_clusters()
    scorer = CosineProtoScorer().fit(Z, y)
    novelty = fit_novelty_map(scorer, Z)
    in_dist = novelty(scorer.scores(Z).max(axis=1)).mean()
    # Orthogonal to both centroids: max cosine ~= 0.
    far = novelty(scorer.scores(np.array([[1.0, -1.0]])).max(axis=1)).mean()
    assert far < in_dist


def test_predict_classes_maps_back_to_label_space():
    Z, y = _two_clusters()
    scorer = CosineProtoScorer().fit(Z, y * 10 + 10)
    assert set(scorer.classes_) == {10, 20}
    assert predict_classes(scorer, np.array([[3.0, 3.0]]))[0] == 20


def test_add_class_grows_the_label_space():
    Z, y = _two_clusters()
    scorer = CosineProtoScorer().fit(Z, y)
    scorer.add_class(2, np.array([[0.0, 1.0]]))
    assert list(scorer.classes_) == [0, 1, 2]
    assert predict_classes(scorer, np.array([[0.0, 1.0]]))[0] == 2


def test_mahalanobis_add_class_keeps_precision_frozen():
    Z, y = _two_clusters()
    scorer = MahalanobisScorer().fit(Z, y)
    precision = scorer.precision_.copy()
    scorer.add_class(2, np.array([[5.0, 5.0], [5.1, 5.1]]))
    assert np.allclose(scorer.precision_, precision)
    assert predict_classes(scorer, np.array([[5.0, 5.0]]))[0] == 2


def test_mahalanobis_scores_match_the_quadratic_form():
    Z, y = _two_clusters()
    scorer = MahalanobisScorer().fit(Z, y)
    query = np.array([[0.3, -0.2], [1.0, 1.0], [-3.0, -2.5]])
    expected = np.stack(
        [
            -0.5 * np.einsum("ij,jk,ik->i", query - mu, scorer.precision_, query - mu)
            for mu in scorer.means_
        ],
        axis=1,
    )
    assert np.allclose(scorer.scores(query), expected)


def test_knn_add_class():
    Z, y = _two_clusters()
    scorer = KNNScorer(k=1).fit(Z, y)
    scorer.add_class(2, np.array([[0.0, -3.0]]))
    assert list(scorer.classes_) == [0, 1, 2]
    assert predict_classes(scorer, np.array([[0.0, -3.0]]))[0] == 2
