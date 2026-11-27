"""Tests for the incremental LDA bank."""

from __future__ import annotations

import numpy as np

from pss.prototypes.incremental import IncrementalLDABank
from pss.prototypes.scorers import MahalanobisScorer


def _three_clusters(n=20, seed=0):
    rng = np.random.default_rng(seed)
    centers = [(-3.0, 0.0), (3.0, 0.0), (0.0, 3.0)]
    Z = np.vstack([rng.normal(loc=c, scale=0.4, size=(n, 2)) for c in centers])
    y = np.repeat(np.arange(3), n)
    return Z, y


def test_incremental_matches_batch_mahalanobis():
    Z, y = _three_clusters()
    bank = IncrementalLDABank().fit(Z, y)
    reference = MahalanobisScorer().fit(Z, y)
    assert np.allclose(bank.scores(Z), reference.scores(Z))
    assert (bank.predict(Z) == y).all()


def test_add_class_predicts_the_new_class():
    Z, y = _three_clusters()
    bank = IncrementalLDABank().fit(Z, y)
    bank.add_class(3, np.array([[10.0, 10.0]]))
    assert bank.n_classes == 4
    assert bank.predict(np.array([[10.0, 10.0]]))[0] == 3


def test_refit_policy_counts_refits():
    Z, y = _three_clusters()
    bank = IncrementalLDABank(refit_every=2).fit(Z, y)
    assert bank.n_refits == 1
    bank.add_class(3, np.array([[10.0, 10.0]]))
    assert bank.n_refits == 1
    bank.add_class(4, np.array([[-10.0, -10.0]]))
    assert bank.n_refits == 2


def test_update_class_replaces_its_samples():
    Z, y = _three_clusters()
    bank = IncrementalLDABank().fit(Z, y)
    bank.update_class(0, np.array([[9.0, 9.0]]))
    assert int((bank.y_ == 0).sum()) == 1
    assert bank.predict(np.array([[9.0, 9.0]]))[0] == 0


def test_add_existing_class_delegates_to_update():
    Z, y = _three_clusters()
    bank = IncrementalLDABank().fit(Z, y)
    before = bank.n_classes
    bank.add_class(0, np.array([[4.0, -4.0]]))
    assert bank.n_classes == before
    assert int((bank.y_ == 0).sum()) == 1
