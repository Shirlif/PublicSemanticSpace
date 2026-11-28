"""KDE / mixture-posterior boundary layer for noisy coordinates.

The grid Bayes boundary (``bayes_boundary.GridBayesBoundary``) approximates the
posterior with a per-cell Gaussian mean/variance and a probit.  The earlier
boundary report attributed the stress failure (near-worst ~0.11 at
``sigma=0.04``) to that approximation and pointed at the information ceiling
(overall ~0.92 at ``sigma=0.04``).  Evaluating the exact rule on genuinely
noisy observations shows something else: the exact KDE matches the grid's
overall *and* its near cells, so the approximation was not the cause.  The
ceiling's overall is high because diagonal pairs dominate; its near cells
collapse for the same reason the raw KDE's do: near-aligned pairs are only
~3.2% (horizontal) and ~1.6% (vertical) of the train prior, so the
overall-optimal vote never predicts "neither" once the noise is 2x the dead
zone.

This module implements the exact empirical-Bayes rule directly, as a kernel
density / mixture posterior over the train-pair empirical prior:

    p(k | observed) ∝ exp(-||observed - prior_k||^2 / (2 sigma^2))

and the three-way state per axis is the posterior-weighted vote of the prior
samples' states (``positive`` / ``negative`` / ``neither``).  It is the same
decision rule used by ``run_boundary_ceiling._bayes_exact4``; the difference is
that this module returns per-pair bits and confidence and is chunked so that a
10k x 30k evaluation does not materialise the full distance matrix.

``balance_types=True`` is a diagnostic, not the default rule: it equalises
prior mass per pair type so the worst cell is not neglected by the overall
prior.  It shows that the axis information is present in the coordinates; the
binding constraint is the prior/cost balance.

Complexity is O(N * K) per pair, which is fine for a diagnostic ceiling but
not a deployment path; the grid table is the deployable form.
"""

from __future__ import annotations

import numpy as np

EPS_DEFAULT = 0.02
PRIOR_SAMPLES_DEFAULT = 3000
CHUNK_DEFAULT = 500
SEED_DEFAULT = 0


def _axis_states(values: np.ndarray, eps: float) -> np.ndarray:
    """0 = positive, 1 = negative, 2 = neither."""

    out = np.full(values.shape, 2, dtype=np.int8)
    out[values > eps] = 0
    out[values < -eps] = 1
    return out


def _pair_types(
    dx: np.ndarray, dy: np.ndarray, eps: float
) -> np.ndarray:
    """0 = horizontal, 1 = vertical, 2 = diagonal (same rule as the eval)."""

    out = np.full(dx.shape, 2, dtype=np.int8)
    out[np.abs(dy) <= eps] = 0
    out[(np.abs(dx) <= eps) & (np.abs(dy) > eps)] = 1
    return out


class KDEBayesBoundary:
    """Weighted-vote boundary using the empirical pair prior as particles.

    Parameters
    ----------
    eps:
        Dead-zone half width; the same ``eps`` used to label the data.
    prior_samples:
        Number of train pairs kept as prior particles.  2000-5000 keeps the
        O(N*K) pass cheap, but at ``sigma=0.01`` that leaves only ~4 effective
        particles per observation and degrades the near cells; the main
        evaluation uses the full train prior (~34 effective particles).
    chunk:
        Observation chunk size; bounds the ``chunk x prior_samples`` memory.
    seed:
        Seed for the prior subsample, so the run is reproducible.
    balance_types:
        Diagnostic option.  When true, each prior particle is weighted by the
        inverse frequency of its pair type, so horizontal / vertical /
        diagonal carry equal prior mass.  The default (false) is the plain
        empirical train prior, which is ~95% diagonal and therefore makes the
        overall-optimal vote neglect the near-aligned cells.
    """

    def __init__(
        self,
        eps: float = EPS_DEFAULT,
        prior_samples: int = PRIOR_SAMPLES_DEFAULT,
        chunk: int = CHUNK_DEFAULT,
        seed: int = SEED_DEFAULT,
        balance_types: bool = False,
    ):
        self.eps = float(eps)
        self.prior_samples = int(prior_samples)
        self.chunk = max(int(chunk), 1)
        self.seed = int(seed)
        self.balance_types = bool(balance_types)
        self.prior: np.ndarray | None = None
        self.prior_states: np.ndarray | None = None
        self.prior_weights: np.ndarray | None = None
        self._onehot: np.ndarray | None = None

    def fit(self, dx: np.ndarray, dy: np.ndarray) -> "KDEBayesBoundary":
        prior = np.stack(
            [
                np.asarray(dx, dtype=np.float64),
                np.asarray(dy, dtype=np.float64),
            ],
            axis=1,
        )
        if len(prior) > self.prior_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(
                len(prior), size=self.prior_samples, replace=False
            )
            prior = prior[idx]
        self.prior = prior
        self.prior_states = np.stack(
            [
                _axis_states(prior[:, 0], self.eps),
                _axis_states(prior[:, 1], self.eps),
            ],
            axis=1,
        )
        if self.balance_types:
            kinds = _pair_types(prior[:, 0], prior[:, 1], self.eps)
            counts = np.bincount(kinds, minlength=3).astype(np.float64)
            freq = counts / counts.sum()
            weights = (1.0 / np.maximum(freq[kinds], 1e-12)).astype(np.float64)
            weights /= weights.mean()
        else:
            weights = np.ones(len(prior), dtype=np.float64)
        self.prior_weights = weights
        # (K, 6): columns are axis0 x {pos,neg,neither}, then axis1.
        onehot = np.zeros((len(prior), 6), dtype=np.float64)
        for axis in range(2):
            for state in range(3):
                onehot[:, 3 * axis + state] = (
                    self.prior_states[:, axis] == state
                )
        self._onehot = onehot
        return self

    def _predict_chunk(
        self, obs: np.ndarray, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        prior = self.prior
        onehot = self._onehot
        two_sigma2 = 2.0 * float(sigma) ** 2
        d2 = (
            (obs[:, None, 0] - prior[None, :, 0]) ** 2
            + (obs[:, None, 1] - prior[None, :, 1]) ** 2
        )
        # log-sum-exp stabilisation: subtract the per-row max before exp so a
        # tiny sigma cannot underflow the whole row to zero.
        logw = -d2 / two_sigma2
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw) * self.prior_weights[None, :]
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
        votes = (w @ onehot).reshape(len(obs), 2, 3)
        decided = votes.argmax(axis=2)
        conf = np.minimum(votes[:, 0, :].max(axis=1), votes[:, 1, :].max(axis=1))
        return decided, conf

    def predict(
        self, dx: np.ndarray, dy: np.ndarray, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(bits, confidence)`` for noisy observations.

        ``bits`` is ``(N, 4)`` bool in ``[left, right, above, below]`` order,
        matching ``pss.eval.upper_bound.pair_labels``.  ``confidence`` is the
        minimum over the two axes of the winning state's posterior weight.
        """

        if self.prior is None:
            raise RuntimeError("call fit() first")
        if float(sigma) <= 0.0:
            raise ValueError("sigma must be positive")
        obs = np.stack(
            [
                np.asarray(dx, dtype=np.float64),
                np.asarray(dy, dtype=np.float64),
            ],
            axis=1,
        )
        n = len(obs)
        bits = np.empty((n, 4), dtype=bool)
        conf = np.empty(n, dtype=np.float64)
        for start in range(0, n, self.chunk):
            stop = min(start + self.chunk, n)
            decided, chunk_conf = self._predict_chunk(obs[start:stop], sigma)
            sx, sy = decided[:, 0], decided[:, 1]
            bits[start:stop, 0] = sx == 0  # left
            bits[start:stop, 1] = sx == 1  # right
            bits[start:stop, 2] = sy == 1  # above
            bits[start:stop, 3] = sy == 0  # below
            conf[start:stop] = chunk_conf
        return bits, conf


__all__ = ["KDEBayesBoundary"]
