"""Empirical-Bayes boundary layer for noisy coordinates.

The boundary benchmark's global-temperature soft dead zone failed (NO-GO), but
the information ceiling showed the contract is reachable: at sigma=0.01 the
Bayes exact-4 is 0.971 while the hard/soft arms sit at 0.68.  The gap is the
*decision rule*, not the information.

This module approximates the Bayes decision with a local Gaussian posterior:

    p(true_pair | observed) ∝ prior(true_pair) * N(observed; true_pair, sigma)

For every observed grid cell it computes the posterior mean and variance of the
true coordinate from the empirical prior, then turns them into the three
state probabilities ``positive / negative / neither`` with a probit.  The
global-temperature layer used one scalar for all cells; this layer uses the
local prior, which is what the ceiling says is needed.

A lookup table makes the per-pair decision O(1); the tables are rebuilt per
noise level.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

EPS_DEFAULT = 0.02
BINS_DEFAULT = 257
LO_DEFAULT = -1.0
HI_DEFAULT = 1.0


class GridBayesBoundary:
    def __init__(
        self,
        eps: float = EPS_DEFAULT,
        bins: int = BINS_DEFAULT,
        lo: float = LO_DEFAULT,
        hi: float = HI_DEFAULT,
    ):
        self.eps = float(eps)
        self.bins = int(bins)
        self.lo = float(lo)
        self.hi = float(hi)
        edges = np.linspace(self.lo, self.hi, self.bins + 1)
        self.centers = 0.5 * (edges[:-1] + edges[1:])
        self.bin_width = (self.hi - self.lo) / self.bins
        self.prior: np.ndarray | None = None
        self._cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def _bin(self, values: np.ndarray) -> np.ndarray:
        idx = np.floor((values - self.lo) / (self.hi - self.lo) * self.bins)
        return np.clip(idx.astype(np.int64), 0, self.bins - 1)

    def fit(
        self, dx: np.ndarray, dy: np.ndarray, *, weights: np.ndarray | None = None
    ) -> "GridBayesBoundary":
        hist, _, _ = np.histogram2d(
            np.asarray(dx, dtype=np.float64),
            np.asarray(dy, dtype=np.float64),
            bins=self.bins,
            range=[[self.lo, self.hi], [self.lo, self.hi]],
            weights=None if weights is None else np.asarray(weights, dtype=np.float64),
        )
        self.prior = hist + 1e-9
        self.prior /= self.prior.sum()
        self._cache.clear()
        return self

    def _state_tables(
        self, sigma: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``state_x, state_y, conf_x, conf_y, probs_x, probs_y``."""

        if self.prior is None:
            raise RuntimeError("call fit() first")
        key = float(sigma)
        if key in self._cache:
            return self._cache[key]
        sigma_bins = max(float(sigma) / self.bin_width, 1e-3)
        prior = self.prior
        x = np.broadcast_to(self.centers[:, None], prior.shape)
        y = np.broadcast_to(self.centers[None, :], prior.shape)

        den = gaussian_filter(prior, sigma=sigma_bins, mode="constant")
        # Sparse prior cells have a tiny but non-zero baseline; clamping the
        # denominator too high would shrink the posterior mean toward zero.
        den = np.maximum(den, 1e-30)
        mean_x = gaussian_filter(prior * x, sigma=sigma_bins, mode="constant") / den
        mean_y = gaussian_filter(prior * y, sigma=sigma_bins, mode="constant") / den
        var_x = (
            gaussian_filter(prior * x * x, sigma=sigma_bins, mode="constant") / den
            - mean_x * mean_x
        )
        var_y = (
            gaussian_filter(prior * y * y, sigma=sigma_bins, mode="constant") / den
            - mean_y * mean_y
        )
        sd_x = np.sqrt(np.maximum(var_x, 1e-12))
        sd_y = np.sqrt(np.maximum(var_y, 1e-12))

        def _probs(mean, sd):
            p_pos = norm.sf((self.eps - mean) / sd)
            p_neg = norm.cdf((-self.eps - mean) / sd)
            p_neither = np.clip(1.0 - p_pos - p_neg, 0.0, 1.0)
            return np.stack([p_pos, p_neg, p_neither], axis=0)

        probs_x = _probs(mean_x, sd_x)
        probs_y = _probs(mean_y, sd_y)
        state_x = probs_x.argmax(axis=0)
        state_y = probs_y.argmax(axis=0)
        conf_x = probs_x.max(axis=0)
        conf_y = probs_y.max(axis=0)
        out = (state_x, state_y, conf_x, conf_y, probs_x, probs_y)
        self._cache[key] = out
        return out

    def predict_proba(
        self, dx: np.ndarray, dy: np.ndarray, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(N, 4)`` predicate probabilities and ``(N, 2)`` axis confidences.

        Columns are ``left, right, above, below``.  The two-axis confidence is
        ``[conf_x, conf_y]``; the minimum is the row confidence used by
        abstention.
        """

        _, _, conf_x, conf_y, probs_x, probs_y = self._state_tables(sigma)
        ix = self._bin(np.asarray(dx, dtype=np.float64))
        iy = self._bin(np.asarray(dy, dtype=np.float64))
        # probs_x: [positive=left, negative=right, neither]
        # probs_y: [positive=below, negative=above, neither]
        proba = np.stack(
            [
                probs_x[0, ix, iy],
                probs_x[1, ix, iy],
                probs_y[1, ix, iy],
                probs_y[0, ix, iy],
            ],
            axis=1,
        )
        return proba, np.stack([conf_x[ix, iy], conf_y[ix, iy]], axis=1)

    def predict(
        self, dx: np.ndarray, dy: np.ndarray, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        state_x, state_y, conf_x, conf_y, _, _ = self._state_tables(sigma)
        ix = self._bin(np.asarray(dx, dtype=np.float64))
        iy = self._bin(np.asarray(dy, dtype=np.float64))
        sx = state_x[ix, iy]
        sy = state_y[ix, iy]
        bits = np.stack([sx == 0, sx == 1, sy == 1, sy == 0], axis=1)
        return bits, np.minimum(conf_x[ix, iy], conf_y[ix, iy])


__all__ = ["GridBayesBoundary"]
