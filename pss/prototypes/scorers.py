"""Distance-based class scorers for the public semantic space.

``recognize`` ranks classes with a weighted cosine vote over per-modality
prototypes.  That readout is cheap and hot-swappable, but the max fused cosine
is not the best novelty statistic: ``eval/results/openset_scoring_diag``
measured AUROC 0.935 for max-softmax vs 0.999 for a pooled-covariance
Mahalanobis score on one fixed representation.

A scorer maps a representation ``(n, d)`` to class scores ``(n, C)`` with
higher = more likely.  It knows nothing about modalities; fusion stays a
caller decision.  Two rules matter:

- **Ranking and novelty use different quantities.**  ``fuse_scores``
  standardises each modality's class scores per query so that a cosine in
  [-1, 1] and a negative Mahalanobis distance can be summed.  That transform
  destroys the absolute scale, so rejection must use the *raw* max score
  through ``fit_novelty_map`` instead.
- **Hot update is exact only where the statistic allows it.**  The cosine and
  kNN scorers update one class in O(1); the Mahalanobis scorer updates the
  class mean but keeps the pooled covariance frozen.  That is an
  approximation and is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

__all__ = [
    "CosineProtoScorer",
    "KNNScorer",
    "MahalanobisScorer",
    "NoveltyMap",
    "Scorer",
    "fit_novelty_map",
    "fuse_scores",
    "predict_classes",
]


class Scorer(Protocol):
    """Class scorer: ``fit`` once, then ``scores`` on new rows."""

    name: str
    classes_: np.ndarray

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "Scorer": ...

    def scores(self, Z: np.ndarray) -> np.ndarray: ...

    def update_class(self, label: int, Z_c: np.ndarray) -> None: ...

    def add_class(self, label: int, Z_c: np.ndarray) -> None: ...


def predict_classes(scorer: Scorer, Z: np.ndarray) -> np.ndarray:
    """Argmax score mapped back to the scorer's own label space.

    Subset models do not see every class, so comparing ``argmax`` directly to
    global label indices would be wrong.
    """

    return np.asarray(scorer.classes_)[np.asarray(scorer.scores(Z)).argmax(axis=1)]


def _l2_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norm, 1e-12, None)


def _l2_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _standardize_rows(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    mean = S.mean(axis=1, keepdims=True)
    std = S.std(axis=1, keepdims=True)
    return (S - mean) / np.clip(std, 1e-12, None)


class CosineProtoScorer:
    """The deployed readout: cosine to the per-class centroid."""

    name = "cosine_proto"

    def __init__(self) -> None:
        self.classes_: np.ndarray = np.empty(0, dtype=np.int64)
        self.protos_: np.ndarray = np.empty((0, 0), dtype=np.float64)

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "CosineProtoScorer":
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        self.protos_ = np.stack(
            [_l2_vec(Z[y == c].mean(0)) for c in self.classes_]
        ).astype(np.float64)
        return self

    def scores(self, Z: np.ndarray) -> np.ndarray:
        return _l2_rows(Z) @ self.protos_.T

    def update_class(self, label: int, Z_c: np.ndarray) -> None:
        k = int(np.searchsorted(self.classes_, label))
        self.protos_[k] = _l2_vec(np.asarray(Z_c, dtype=np.float64).mean(0))

    def add_class(self, label: int, Z_c: np.ndarray) -> None:
        k = int(np.searchsorted(self.classes_, label))
        self.classes_ = np.insert(self.classes_, k, label)
        self.protos_ = np.insert(
            self.protos_, k, _l2_vec(np.asarray(Z_c, dtype=np.float64).mean(0)), axis=0
        )

    def storage_bytes(self) -> int:
        return int(self.protos_.nbytes)


class MahalanobisScorer:
    """Pooled-covariance Mahalanobis: class means + one shared precision.

    Per-class covariance is singular in the few-shot regime (384 classes x 6
    shots), so the covariance is pooled across classes, which is the setting
    §5.5 measured.  ``update_class`` moves the mean and leaves the pooled
    precision frozen; that is what keeps the update O(d) instead of O(d^2).
    """

    name = "mahalanobis"

    def __init__(self, shrinkage: float = 1e-3) -> None:
        self.shrinkage = float(shrinkage)
        self.classes_: np.ndarray = np.empty(0, dtype=np.int64)
        self.means_: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.precision_: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.covariance_frozen = False

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "MahalanobisScorer":
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        inverse = np.searchsorted(self.classes_, y)
        self.means_ = np.stack(
            [Z[inverse == k].mean(0) for k in range(len(self.classes_))]
        ).astype(np.float64)
        residual = Z - self.means_[inverse]
        dof = max(len(Z) - len(self.classes_), 1)
        cov = residual.T @ residual / dof
        scale = float(np.trace(cov) / max(cov.shape[0], 1))
        cov = cov + self.shrinkage * max(scale, 1e-12) * np.eye(cov.shape[0])
        self.precision_ = np.linalg.pinv(cov)
        self.covariance_frozen = True
        return self

    def scores(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        # (z - mu)^T P (z - mu) = z^T P z - 2 z^T P mu + mu^T P mu.
        # Vectorised over classes: the per-class loop this replaces did a full
        # quadratic form per class and dominated the A4 weight grid.
        ZP = Z @ self.precision_
        zPz = np.einsum("ij,ij->i", ZP, Z)
        muPmu = np.einsum("ij,jk,ik->i", self.means_, self.precision_, self.means_)
        d2 = zPz[:, None] - 2.0 * (ZP @ self.means_.T) + muPmu[None, :]
        np.maximum(d2, 0.0, out=d2)
        return -0.5 * d2

    def update_class(self, label: int, Z_c: np.ndarray) -> None:
        k = int(np.searchsorted(self.classes_, label))
        self.means_[k] = np.asarray(Z_c, dtype=np.float64).mean(0)

    def add_class(self, label: int, Z_c: np.ndarray) -> None:
        # Additive only: the pooled precision is frozen, so a new class costs
        # one mean and no covariance re-estimation.
        k = int(np.searchsorted(self.classes_, label))
        self.classes_ = np.insert(self.classes_, k, label)
        self.means_ = np.insert(
            self.means_, k, np.asarray(Z_c, dtype=np.float64).mean(0), axis=0
        )

    def storage_bytes(self) -> int:
        return int(self.means_.nbytes + self.precision_.nbytes)


class KNNScorer:
    """Class-conditional kNN: score = negative mean distance to k neighbours."""

    name = "knn"

    def __init__(self, k: int = 5) -> None:
        self.k = int(k)
        self.classes_: np.ndarray = np.empty(0, dtype=np.int64)
        self.Z_: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.y_: np.ndarray = np.empty(0, dtype=np.int64)
        self._by_class: list[np.ndarray] = []

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "KNNScorer":
        self.Z_ = np.asarray(Z, dtype=np.float64)
        self.y_ = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(self.y_)
        self._by_class = [np.flatnonzero(self.y_ == c) for c in self.classes_]
        return self

    def scores(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        out = np.empty((len(Z), len(self.classes_)), dtype=np.float64)
        query_norm = (Z**2).sum(axis=1)[:, None]
        for k, idx in enumerate(self._by_class):
            ref = self.Z_[idx]
            d2 = query_norm + (ref**2).sum(axis=1)[None, :] - 2.0 * Z @ ref.T
            np.maximum(d2, 0.0, out=d2)
            kk = min(self.k, ref.shape[0])
            part = np.partition(d2, kk - 1, axis=1)[:, :kk]
            out[:, k] = -np.sqrt(part).mean(axis=1)
        return out

    def update_class(self, label: int, Z_c: np.ndarray) -> None:
        k = int(np.searchsorted(self.classes_, label))
        Z_c = np.asarray(Z_c, dtype=np.float64)
        start = len(self.Z_)
        self.Z_ = np.concatenate([self.Z_, Z_c], axis=0)
        self.y_ = np.concatenate([self.y_, np.full(len(Z_c), label, dtype=np.int64)])
        self._by_class[k] = np.concatenate(
            [self._by_class[k], np.arange(start, start + len(Z_c))]
        )

    def add_class(self, label: int, Z_c: np.ndarray) -> None:
        k = int(np.searchsorted(self.classes_, label))
        Z_c = np.asarray(Z_c, dtype=np.float64)
        start = len(self.Z_)
        self.Z_ = np.concatenate([self.Z_, Z_c], axis=0)
        self.y_ = np.concatenate([self.y_, np.full(len(Z_c), label, dtype=np.int64)])
        self.classes_ = np.insert(self.classes_, k, label)
        self._by_class.insert(k, np.arange(start, start + len(Z_c)))

    def storage_bytes(self) -> int:
        return int(self.Z_.nbytes)


@dataclass(frozen=True)
class NoveltyMap:
    """Global affine map from a raw max score to a novelty score.

    Fitted on the training split (in-distribution).  Unlike ``fuse_scores``
    this is *not* per-query, so the absolute distance information survives and
    the mapped score can be thresholded.
    """

    center: float
    scale: float

    def __call__(self, max_scores: np.ndarray) -> np.ndarray:
        return (np.asarray(max_scores, dtype=np.float64) - self.center) / max(
            self.scale, 1e-12
        )


def fit_novelty_map(scorer: Scorer, Z_train: np.ndarray) -> NoveltyMap:
    max_scores = np.asarray(scorer.scores(Z_train)).max(axis=1)
    return NoveltyMap(float(max_scores.mean()), float(max_scores.std()))


def fuse_scores(
    per_modality_scores: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Weighted sum of per-query standardised class scores.

    For ranking only: the per-query standardisation removes the absolute
    novelty signal.  Use ``fit_novelty_map`` on the raw scores for rejection.
    """

    total = sum(weights[m] for m in per_modality_scores)
    fused = None
    for modality, scores in per_modality_scores.items():
        term = weights[modality] * _standardize_rows(scores)
        fused = term if fused is None else fused + term
    if fused is None:
        raise ValueError("no modality scores to fuse")
    return fused / max(total, 1e-12)
