"""Calibration for the public semantic space.

The projection head maps each expert into a unit hypersphere, but raw cosine
scores are not comparable across modalities: a 384-d DINO projection and a
64-d tactile projection have different concentration and different reliability.
This module fits, per modality:

- a temperature that makes softmax probabilities match observed accuracy;
- a reliability weight derived from validation accuracy above chance.

The output is a small, auditable calibration record. PSS consumers can keep
using the same ``recognize(..., weights=...)`` call; the weights are simply
better justified than hand-tuned constants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class ModalityCalibration:
    modality: str
    temperature: float
    reliability: float
    val_accuracy: float
    val_nll: float
    n_val: int

    def to_dict(self) -> dict:
        return asdict(self)


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = _softmax(logits)
    picked = probs[np.arange(len(labels)), labels].clip(1e-12)
    return float(-np.log(picked).mean())


#: Search range for log-temperature.  Kept at the historical value for the
#: legacy streaming calibrator (``online.py``); the robustness pipeline passes
#: :data:`RANKING_LOG_BOUNDS` explicitly.
TEMPERATURE_LOG_BOUNDS: tuple[float, float] = (-3.0, 3.0)

#: Wider range for the **ranking** channel.  With the confidence channel
#: separated, temperature no longer has to carry calibration quality — it only
#: sets cross-modality comparability for fusion.  The old (-3, 3) box pinned
#: geometry and vision at ``exp(-3) = 0.0498`` while their optimum was ~0.027,
#: so the fusion weights were silently wrong.
RANKING_LOG_BOUNDS: tuple[float, float] = (-8.0, 3.0)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    log_bounds: tuple[float, float] = TEMPERATURE_LOG_BOUNDS,
) -> float:
    """Fit a positive scalar temperature by minimizing validation NLL."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0:
        return 1.0

    def objective(log_t: float) -> float:
        return _nll(logits / np.exp(log_t), labels)

    result = minimize_scalar(objective, bounds=log_bounds, method="bounded")
    return float(np.exp(result.x))


def _weighted_nll(
    logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted mean negative log-likelihood.

    Returns ``inf`` when the effective sample mass is zero, so a caller that
    accidentally passes all-zero weights fails loudly instead of silently
    fitting to nothing.
    """
    probs = _softmax(logits)
    picked = probs[np.arange(len(labels)), labels].clip(1e-12)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(w.sum())
    if total <= 0.0:
        return float("inf")
    return float(-(w * np.log(picked)).sum() / total)


def fit_temperature_weighted(
    logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    min_effective: float = 1.0,
    fallback: float = 1.0,
    log_bounds: tuple[float, float] = TEMPERATURE_LOG_BOUNDS,
) -> float:
    """Fit a temperature with per-sample weights.

    Used by the robustness pipeline: suspect samples carry ``w < 1`` and
    trimmed samples carry ``w = 0``.  Trimming is therefore the special case
    ``w = 0`` rather than a second, competing selection rule.

    Returns ``fallback`` when the effective mass is below ``min_effective``;
    callers must treat that as an untrusted fit and let the gate reject it.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    w = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
    if len(labels) == 0 or w.sum() < float(min_effective):
        return float(fallback)

    def objective(log_t: float) -> float:
        return _weighted_nll(logits / np.exp(log_t), labels, w)

    result = minimize_scalar(objective, bounds=log_bounds, method="bounded")
    if not np.isfinite(result.fun):
        return float(fallback)
    return float(np.exp(result.x))


def temperature_at_bound(
    temperature: float,
    tol: float = 1e-5,
    *,
    log_bounds: tuple[float, float] = TEMPERATURE_LOG_BOUNDS,
) -> bool:
    """True when a fitted temperature sits on the search bound.

    The old failure was silent: a pinned fit looks like "nothing changed".  This
    makes it visible so it can be reported and, when appropriate, the bound can
    be widened via ``TEMPERATURE_LOG_BOUNDS``.
    """
    if temperature <= 0:
        return False
    log_t = float(np.log(temperature))
    lo, hi = log_bounds
    return abs(log_t - lo) < tol or abs(log_t - hi) < tol


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Equal-width ECE from per-sample confidence and correctness.

    ``confidences`` are the probabilities assigned to the predicted class and
    ``correct`` is 1 for a correct prediction, 0 otherwise.  The estimator is
    the standard weighted absolute gap between mean confidence and mean
    accuracy in each confidence bin.
    """
    conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
    ok = np.asarray(correct, dtype=np.float64).reshape(-1)
    if len(conf) == 0 or len(conf) != len(ok):
        return float("nan")
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for b in range(int(n_bins)):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf >= lo) & ((conf < hi) if b < int(n_bins) - 1 else (conf <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(ok[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def calibrate_modality(
    query: np.ndarray,
    prototypes: np.ndarray,
    labels: np.ndarray,
    name: str,
) -> ModalityCalibration:
    """Fit temperature + reliability for one modality on a validation split."""
    query = _unit(query)
    prototypes = _unit(prototypes)
    logits = query @ prototypes.T
    labels = np.asarray(labels, dtype=np.int64)
    top1 = logits.argmax(axis=1)
    accuracy = float((top1 == labels).mean()) if len(labels) else 0.0
    n_classes = int(prototypes.shape[0])
    chance = 1.0 / max(1, n_classes)
    reliability = float(np.clip((accuracy - chance) / max(1e-9, 1.0 - chance), 0.0, 1.0))
    temperature = fit_temperature(logits, labels)
    return ModalityCalibration(
        modality=name,
        temperature=temperature,
        reliability=reliability,
        val_accuracy=accuracy,
        val_nll=_nll(logits / temperature, labels),
        n_val=int(len(labels)),
    )


def calibrate_fusion(
    embeddings: dict[str, np.ndarray],
    prototypes: dict[str, np.ndarray],
    labels: np.ndarray,
    modalities: tuple[str, ...] | None = None,
) -> dict[str, ModalityCalibration]:
    """Calibrate every modality against the same validation labels."""
    modalities = modalities or tuple(embeddings)
    return {
        m: calibrate_modality(embeddings[m], prototypes[m], labels, m)
        for m in modalities
        if m in embeddings and m in prototypes
    }


def reliability_weights(
    calibration: dict[str, ModalityCalibration], floor: float = 0.0
) -> dict[str, float]:
    """Weights suitable for ``recognize(..., weights=...)``."""
    return {m: max(float(floor), c.reliability) for m, c in calibration.items()}


def calibrated_scores(
    query: dict[str, np.ndarray],
    prototypes: dict[str, np.ndarray],
    calibration: dict[str, ModalityCalibration],
) -> np.ndarray:
    """Reliability-weighted, temperature-normalized class scores.

    Each modality is divided by its fitted temperature before fusion, so a
    noisy modality's score scale is comparable with a clean one. The result is
    an unbounded score vector suitable for open-set ranking.
    """
    fused = None
    total_weight = 0.0
    for modality, calib in calibration.items():
        if modality not in query or modality not in prototypes:
            continue
        logits = _unit(query[modality].reshape(1, -1)) @ _unit(prototypes[modality]).T
        weight = max(calib.reliability, 0.0)
        term = weight * (logits[0] / max(calib.temperature, 1e-3))
        fused = term if fused is None else fused + term
        total_weight += weight
    if fused is None or total_weight <= 0:
        return np.zeros(0, dtype=np.float64)
    return fused / total_weight


def calibrated_confidence(
    query: dict[str, np.ndarray],
    prototypes: dict[str, np.ndarray],
    calibration: dict[str, ModalityCalibration],
    categories: list[str],
) -> float:
    """Probability-like confidence in [0, 1] for one query."""
    scores = calibrated_scores(query, prototypes, calibration)
    if len(scores) == 0:
        return 0.0
    return float(_softmax(scores.reshape(1, -1))[0].max())
