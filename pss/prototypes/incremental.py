"""Incrementally grown pooled-covariance LDA.

A4's deployed candidate keeps class means plus one pooled precision.  Adding a
class is then O(d) (a new mean) with a frozen precision, which is cheap but
forgets: A4 measured −1.6 pt on old classes after 50 additions.

This bank makes the refresh policy explicit.  ``refit_every=0`` freezes the
precision (A4 behaviour); ``refit_every=k`` recomputes it from a stored feature
reservoir after every k additions.  The reservoir is what makes the refit exact
without storing a d x d scatter per class.

Cost model
----------
- mean update: O(d) per class
- refit: O(N d^2 + d^3) over the reservoir

The bank records both so a caller can price a policy.
"""

from __future__ import annotations

import time

import numpy as np

__all__ = ["IncrementalLDABank"]


class IncrementalLDABank:
    def __init__(self, shrinkage: float = 1e-3, refit_every: int = 0):
        self.shrinkage = float(shrinkage)
        self.refit_every = int(refit_every)
        self.classes_: np.ndarray = np.empty(0, dtype=np.int64)
        self.means_: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.precision_: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.Z_: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.y_: np.ndarray = np.empty(0, dtype=np.int64)
        self._since_refit = 0
        self.n_refits = 0
        self.refit_seconds = 0.0
        self.mean_update_seconds = 0.0

    # -- fitting ----------------------------------------------------------- #
    def fit(self, Z: np.ndarray, y: np.ndarray) -> "IncrementalLDABank":
        self.Z_ = np.asarray(Z, dtype=np.float64)
        self.y_ = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(self.y_)
        self._refit()
        return self

    def _refit(self) -> None:
        start = time.perf_counter()
        self.classes_ = np.unique(self.y_)
        inverse = np.searchsorted(self.classes_, self.y_)
        self.means_ = np.stack(
            [self.Z_[inverse == k].mean(0) for k in range(len(self.classes_))]
        ).astype(np.float64)
        residual = self.Z_ - self.means_[inverse]
        dof = max(len(self.Z_) - len(self.classes_), 1)
        cov = residual.T @ residual / dof
        scale = float(np.trace(cov) / max(cov.shape[0], 1))
        cov = cov + self.shrinkage * max(scale, 1e-12) * np.eye(cov.shape[0])
        self.precision_ = np.linalg.pinv(cov)
        self.n_refits += 1
        self.refit_seconds += time.perf_counter() - start
        self._since_refit = 0

    # -- updates ----------------------------------------------------------- #
    def _maybe_refit(self) -> None:
        if self.refit_every > 0 and self._since_refit >= self.refit_every:
            self._refit()

    def add_class(self, label: int, Z_c: np.ndarray) -> None:
        label = int(label)
        if label in self.classes_:
            self.update_class(label, Z_c)
            return
        Z_c = np.asarray(Z_c, dtype=np.float64)
        start = time.perf_counter()
        self.Z_ = np.concatenate([self.Z_, Z_c], axis=0)
        self.y_ = np.concatenate([self.y_, np.full(len(Z_c), label, dtype=np.int64)])
        k = int(np.searchsorted(self.classes_, label))
        self.classes_ = np.insert(self.classes_, k, label)
        self.means_ = np.insert(self.means_, k, Z_c.mean(0), axis=0)
        self.mean_update_seconds += time.perf_counter() - start
        self._since_refit += 1
        self._maybe_refit()

    def update_class(self, label: int, Z_c: np.ndarray) -> None:
        label = int(label)
        Z_c = np.asarray(Z_c, dtype=np.float64)
        start = time.perf_counter()
        mask = self.y_ != label
        self.Z_ = np.concatenate([self.Z_[mask], Z_c], axis=0)
        self.y_ = np.concatenate(
            [self.y_[mask], np.full(len(Z_c), label, dtype=np.int64)]
        )
        k = int(np.searchsorted(self.classes_, label))
        self.means_[k] = Z_c.mean(0)
        self.mean_update_seconds += time.perf_counter() - start
        self._since_refit += 1
        self._maybe_refit()

    # -- deletion / merge --------------------------------------------------- #
    def remove_class(self, label: int) -> None:
        """Drop one class and its reservoir rows; refresh policy still applies.

        Deletion is the maintenance case A5 never tested: it changes the
        between-class scatter, so a frozen precision is an approximation.  The
        caller can force a refit by constructing the bank with
        ``refit_every=1`` or by calling :meth:`refit_now`.
        """

        label = int(label)
        if label not in self.classes_:
            raise KeyError(f"class {label} not in bank")
        start = time.perf_counter()
        mask = self.y_ != label
        self.Z_ = self.Z_[mask]
        self.y_ = self.y_[mask]
        k = int(np.searchsorted(self.classes_, label))
        self.classes_ = np.delete(self.classes_, k)
        self.means_ = np.delete(self.means_, k, axis=0)
        self.mean_update_seconds += time.perf_counter() - start
        self._since_refit += 1
        self._maybe_refit()

    def merge_classes(self, label_from: int, label_into: int) -> None:
        """Merge ``label_from`` into ``label_into`` (e.g. duplicate entities)."""

        label_from = int(label_from)
        label_into = int(label_into)
        if label_from == label_into:
            return
        for label in (label_from, label_into):
            if label not in self.classes_:
                raise KeyError(f"class {label} not in bank")
        start = time.perf_counter()
        self.y_[self.y_ == label_from] = label_into
        k_from = int(np.searchsorted(self.classes_, label_from))
        k_into = int(np.searchsorted(self.classes_, label_into))
        merged_mean = self.Z_[self.y_ == label_into].mean(0)
        self.classes_ = np.delete(self.classes_, k_from)
        self.means_ = np.delete(self.means_, k_from, axis=0)
        # ``label_into`` keeps its rank because its index was below/above the
        # deleted one; recompute by search after the deletion.
        k_into = int(np.searchsorted(self.classes_, label_into))
        self.means_[k_into] = merged_mean
        self.mean_update_seconds += time.perf_counter() - start
        self._since_refit += 1
        self._maybe_refit()

    def refit_now(self) -> None:
        """Force a precision refresh (used after delete/merge audits)."""

        self._refit()

    # -- rollback ----------------------------------------------------------- #
    def snapshot(self) -> tuple:
        return (
            self.Z_.copy(),
            self.y_.copy(),
            self.classes_.copy(),
            self.means_.copy(),
            self.precision_.copy(),
            int(self.n_refits),
            float(self.refit_seconds),
            int(self._since_refit),
        )

    def restore(self, snap: tuple) -> None:
        (
            self.Z_, self.y_, self.classes_, self.means_, self.precision_,
            self.n_refits, self.refit_seconds, self._since_refit,
        ) = snap

    # -- scoring ----------------------------------------------------------- #
    def scores(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        ZP = Z @ self.precision_
        zPz = np.einsum("ij,ij->i", ZP, Z)
        muPmu = np.einsum("ij,jk,ik->i", self.means_, self.precision_, self.means_)
        d2 = zPz[:, None] - 2.0 * (ZP @ self.means_.T) + muPmu[None, :]
        np.maximum(d2, 0.0, out=d2)
        return -0.5 * d2

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(self.classes_)[self.scores(Z).argmax(axis=1)]

    @property
    def n_classes(self) -> int:
        return int(len(self.classes_))
