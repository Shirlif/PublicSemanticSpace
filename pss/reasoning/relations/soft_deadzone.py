"""Soft / probabilistic dead-zone predicates with abstention.

Hard predicates (``1[dx > eps]``) are correct only when the coordinate is
noiseless.  Under measurement noise the decision boundary is crossed by true
near-aligned pairs, which is the U6 failure: absence error grows with
``sigma_rel / eps``.  This module replaces the hard boundary with a 3-way
softmax over ``{negative, positive, neither}`` per axis and lets the
temperature scale with the estimated noise.

The module is deliberately parameter-light (a temperature and a confidence
threshold) so the intervention tests *uncertainty handling*, not capacity.

Sign conventions match ``pss.eval.scripts.run_typed_relations._pair_labels``:

    left  = dx > eps
    right = dx < -eps
    above = dy < -eps
    below = dy > eps
"""

from __future__ import annotations

import numpy as np

EPS_DEFAULT = 0.02
PREDICATES = ("left", "right", "above", "below")


def axis_probs(delta: np.ndarray, eps: float, tau: float) -> np.ndarray:
    """Return ``(N, 3)`` probabilities for ``{positive, negative, neither}``.

    ``delta`` is the signed coordinate difference on one axis.  At
    ``delta = +eps`` the positive and negative logits are both 0; at
    ``delta = 0`` both are negative and ``neither`` dominates.
    """

    delta = np.asarray(delta, dtype=np.float64)
    tau = max(float(tau), 1e-9)
    logits = np.stack(
        [
            (delta - eps) / tau,
            (-delta - eps) / tau,
            np.zeros_like(delta),
        ],
        axis=-1,
    )
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=-1, keepdims=True)


def predict_proba(
    pa: np.ndarray, pb: np.ndarray, eps: float = EPS_DEFAULT, tau: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(N, 4)`` predicate probabilities and ``(N, 2, 3)`` axis states.

    The four columns are ``left, right, above, below``.  The axis-state tensor
    keeps the ``neither`` probability, which is what abstention and confidence
    read.
    """

    pa = np.asarray(pa, dtype=np.float64)
    pb = np.asarray(pb, dtype=np.float64)
    dx = pb[:, 0] - pa[:, 0]
    dy = pb[:, 1] - pa[:, 1]
    px = axis_probs(dx, eps, tau)  # [positive=left, negative=right, neither]
    py = axis_probs(dy, eps, tau)  # [positive=below, negative=above, neither]
    proba = np.stack([px[:, 0], px[:, 1], py[:, 1], py[:, 0]], axis=-1)
    return proba, np.stack([px, py], axis=1)


def confidence(axis_state: np.ndarray) -> np.ndarray:
    """Per-row confidence: the weaker axis's most likely state."""

    return np.min(np.max(axis_state, axis=-1), axis=-1)


def predict(
    pa: np.ndarray,
    pb: np.ndarray,
    eps: float = EPS_DEFAULT,
    tau: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bits, confidence)``; bits are thresholded at 0.5."""

    proba, state = predict_proba(pa, pb, eps=eps, tau=tau)
    return proba > 0.5, confidence(state)


def exact4(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    p = np.asarray(p).astype(bool)
    return float(np.all(y == p, axis=1).mean())


def calibrate_tau(
    pa: np.ndarray,
    pb: np.ndarray,
    y: np.ndarray,
    eps: float = EPS_DEFAULT,
    taus: np.ndarray | None = None,
) -> tuple[float, float]:
    """Choose the temperature that maximises exact-4 on a calibration set."""

    if taus is None:
        taus = np.geomspace(0.002, 0.10, 30)
    best_tau, best_score = float(taus[0]), -1.0
    for tau in taus:
        bits, _ = predict(pa, pb, eps=eps, tau=float(tau))
        score = exact4(y, bits)
        if score > best_score:
            best_tau, best_score = float(tau), score
    return best_tau, best_score


def calibrate_k(
    pa: np.ndarray,
    pb: np.ndarray,
    y: np.ndarray,
    eps: float,
    tau0: float,
    sigma: float,
    ks: np.ndarray | None = None,
) -> tuple[float, float]:
    """Choose ``k`` in ``tau = max(tau0, k * sigma)`` on a noisy calibration set."""

    if ks is None:
        ks = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    best_k, best_score = float(ks[0]), -1.0
    for k in ks:
        tau = max(float(tau0), float(k) * float(sigma))
        bits, _ = predict(pa, pb, eps=eps, tau=tau)
        score = exact4(y, bits)
        if score > best_score:
            best_k, best_score = float(k), score
    return best_k, best_score


def calibrate_abstain(
    pa: np.ndarray,
    pb: np.ndarray,
    y: np.ndarray,
    eps: float,
    tau: float,
    thresholds: np.ndarray | None = None,
    target_coverage: float = 0.90,
) -> tuple[float, float, float]:
    """Choose a confidence threshold for selective exact-4 at a coverage floor."""

    if thresholds is None:
        thresholds = np.linspace(0.30, 0.99, 40)
    bits, conf = predict(pa, pb, eps=eps, tau=tau)
    correct = np.all(bits == np.asarray(y).astype(bool), axis=1)
    best = (float(thresholds[0]), -1.0, 0.0)
    for thr in thresholds:
        keep = conf >= float(thr)
        coverage = float(keep.mean())
        if coverage < target_coverage or coverage == 0.0:
            continue
        selective = float(correct[keep].mean())
        if selective > best[1]:
            best = (float(thr), selective, coverage)
    return best


__all__ = [
    "EPS_DEFAULT",
    "PREDICATES",
    "axis_probs",
    "calibrate_abstain",
    "calibrate_k",
    "calibrate_tau",
    "confidence",
    "exact4",
    "predict",
    "predict_proba",
]
