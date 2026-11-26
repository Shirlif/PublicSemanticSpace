"""Stateful calibration, drift detection and calibration-set audit (F6).

The one-shot calibration in :mod:`pss.projection.calibration` is a good
offline default, but a deployed expert sees a stream: the validation
distribution can move, a calibration batch can be corrupted, and a full
re-calibration may be too expensive.  This module adds three things without
changing the one-shot API:

1. :class:`StreamingModalityCalibration` keeps a bounded rolling window of
   ``(logit, label)`` observations and refits temperature + reliability from
   that window.  Feeding the whole window in one batch reproduces
   :func:`pss.projection.calibration.calibrate_modality`.
2. :class:`StreamingCalibrator` groups one state per modality, records drift
   metrics (rolling accuracy, ECE, temperature, reliability, class coverage)
   and turns them into a reproducible ``continue / recalibrate / abstain``
   decision.
3. :func:`audit_calibration_set` is a negative control for poisoned
   calibration data.  It flags label mismatches, margin violations,
   near-duplicate embeddings and class-centroid outliers, and compares the
   rates against a clean baseline so a weak modality is not flagged merely for
   being weak.

Everything here is offline/numpy and does not touch runtime code.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .calibration import (
    ModalityCalibration,
    _nll,
    _softmax,
    _unit,
    calibrated_scores,
    expected_calibration_error,
    fit_temperature,
)

# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


@dataclass
class DriftThresholds:
    """Thresholds for the normalised drift score.

    Each raw deviation is divided by its threshold; the score is the maximum
    ratio.  A score of 1.0 means "at least one signal reached its threshold".
    The defaults are conservative floors; :func:`calibrate_drift_thresholds`
    can raise them from a clean stream.
    """

    accuracy_drop: float = 0.10
    ece_increase: float = 0.10
    reliability_drop: float = 0.10
    coverage_drop: float = 0.50
    score: float = 1.0
    abstain_score: float = 2.0
    min_window: int = 8

    def to_dict(self) -> dict:
        return {
            "accuracy_drop": float(self.accuracy_drop),
            "ece_increase": float(self.ece_increase),
            "reliability_drop": float(self.reliability_drop),
            "coverage_drop": float(self.coverage_drop),
            "score": float(self.score),
            "abstain_score": float(self.abstain_score),
            "min_window": int(self.min_window),
        }


@dataclass
class DriftReport:
    drift: bool
    abstain: bool
    action: str  # continue | recalibrate | abstain
    score: float
    accuracy_drop: float
    ece_increase: float
    reliability_drop: float
    coverage_drop: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "drift": bool(self.drift),
            "abstain": bool(self.abstain),
            "action": self.action,
            "score": float(self.score),
            "accuracy_drop": float(self.accuracy_drop),
            "ece_increase": float(self.ece_increase),
            "reliability_drop": float(self.reliability_drop),
            "coverage_drop": float(self.coverage_drop),
            "reasons": list(self.reasons),
        }


def calibrate_drift_thresholds(
    deviations: list[dict] | dict[str, list[float]],
    *,
    quantile: float = 0.99,
    floors: DriftThresholds | None = None,
) -> DriftThresholds:
    """Raise drift thresholds from clean-stream deviations.

    ``deviations`` is either a list of per-step dicts with keys
    ``accuracy_drop``, ``ece_increase``, ``reliability_drop`` and
    ``coverage_drop``, or a dict of lists.  The threshold for each signal is
    the ``quantile`` of its absolute value, floored by the default thresholds
    so a degenerate clean stream cannot make the detector hair-trigger.
    """
    floors = floors or DriftThresholds()
    if isinstance(deviations, dict):
        keys = ("accuracy_drop", "ece_increase", "reliability_drop", "coverage_drop")
        values = {k: [float(x) for x in deviations.get(k, [])] for k in keys}
    else:
        keys = ("accuracy_drop", "ece_increase", "reliability_drop", "coverage_drop")
        values = {
            k: [float(d.get(k, 0.0)) for d in deviations] for k in keys
        }

    def _q(vals: list[float], floor: float) -> float:
        if not vals:
            return float(floor)
        return float(max(floor, np.quantile(np.abs(vals), float(quantile))))

    return DriftThresholds(
        accuracy_drop=_q(values["accuracy_drop"], floors.accuracy_drop),
        ece_increase=_q(values["ece_increase"], floors.ece_increase),
        reliability_drop=_q(values["reliability_drop"], floors.reliability_drop),
        coverage_drop=_q(values["coverage_drop"], floors.coverage_drop),
        score=float(floors.score),
        abstain_score=float(floors.abstain_score),
        min_window=int(floors.min_window),
    )


# ---------------------------------------------------------------------------
# Streaming calibration
# ---------------------------------------------------------------------------


class StreamingModalityCalibration:
    """Bounded-window temperature/reliability calibration for one modality.

    The state is deliberately small and auditable: the rolling window stores
    the last ``window_size`` logits and labels, and a refit recomputes the same
    quantities as :func:`calibrate_modality`.  Labels ``< 0`` are treated as
    "unknown": they count as errors for accuracy but are excluded from the
    temperature fit and ECE.
    """

    def __init__(
        self,
        name: str,
        n_classes: int | None = None,
        *,
        window_size: int = 256,
        refit_every: int = 1,
        min_samples: int = 2,
        ece_bins: int = 10,
    ):
        self.name = str(name)
        self.n_classes = int(n_classes) if n_classes is not None else None
        self.window_size = max(1, int(window_size))
        self.refit_every = max(1, int(refit_every))
        self.min_samples = max(1, int(min_samples))
        self.ece_bins = int(ece_bins)
        self._logits: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._labels: deque[int] = deque(maxlen=self.window_size)
        self.total_seen = 0
        self.updates = 0
        self.temperature = 1.0
        self.reliability = 0.0
        self.accuracy = 0.0
        self.ece = float("nan")
        self.nll = float("nan")
        self.class_set: frozenset[int] = frozenset()
        self.ref: dict | None = None
        self.history: list[dict] = []

    # -- state ---------------------------------------------------------
    @property
    def window_n(self) -> int:
        return len(self._logits)

    def state(self) -> ModalityCalibration:
        return ModalityCalibration(
            modality=self.name,
            temperature=float(self.temperature),
            reliability=float(self.reliability),
            val_accuracy=float(self.accuracy),
            val_nll=float(self.nll) if np.isfinite(self.nll) else float("nan"),
            n_val=int(self.window_n),
        )

    def to_dict(self) -> dict:
        return {
            "modality": self.name,
            "n_classes": self.n_classes,
            "window_size": self.window_size,
            "refit_every": self.refit_every,
            "total_seen": int(self.total_seen),
            "window": int(self.window_n),
            "temperature": float(self.temperature),
            "reliability": float(self.reliability),
            "accuracy": float(self.accuracy),
            "ece": float(self.ece),
            "nll": float(self.nll),
            "coverage": int(len(self.class_set)),
            "reference": dict(self.ref) if self.ref else None,
        }

    # -- updates -------------------------------------------------------
    def _refit(self) -> None:
        if self.window_n == 0:
            return
        logits = np.stack(list(self._logits), axis=0).astype(np.float64)
        labels = np.asarray(list(self._labels), dtype=np.int64)
        if self.n_classes is None:
            self.n_classes = int(logits.shape[1])
        valid = (labels >= 0) & (labels < int(logits.shape[1]))
        self.class_set = frozenset(int(x) for x in labels[valid].tolist())

        pred = logits.argmax(axis=1)
        # Unknown labels are counted as errors.
        self.accuracy = float(np.mean(pred == labels)) if len(labels) else 0.0
        chance = 1.0 / max(1, int(self.n_classes))
        self.reliability = float(
            np.clip((self.accuracy - chance) / max(1e-9, 1.0 - chance), 0.0, 1.0)
        )

        if valid.sum() >= self.min_samples:
            self.temperature = float(fit_temperature(logits[valid], labels[valid]))
        # With too few valid observations, keep the previous temperature.

        if valid.any():
            probs = _softmax(logits[valid] / max(self.temperature, 1e-3))
            conf = probs.max(axis=1)
            ok = (probs.argmax(axis=1) == labels[valid]).astype(np.float64)
            self.ece = float(expected_calibration_error(conf, ok, n_bins=self.ece_bins))
            self.nll = float(_nll(logits[valid] / max(self.temperature, 1e-3), labels[valid]))
        else:
            self.ece = float("nan")
            self.nll = float("nan")

    def update(self, logits: np.ndarray, labels: np.ndarray) -> dict:
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{self.name}: {logits.shape[0]} logits vs {labels.shape[0]} labels"
            )
        if self.n_classes is None:
            self.n_classes = int(logits.shape[1])
        prev_t = float(self.temperature)
        prev_rel = float(self.reliability)
        for row, label in zip(logits, labels):
            self._logits.append(np.asarray(row, dtype=np.float64))
            self._labels.append(int(label))
        self.total_seen += int(len(labels))
        self.updates += 1
        if self.window_n >= self.min_samples and (
            self.updates % self.refit_every == 0 or self.window_n == self.min_samples
        ):
            self._refit()
        record = {
            "modality": self.name,
            "n_seen": int(self.total_seen),
            "window": int(self.window_n),
            "temperature": float(self.temperature),
            "reliability": float(self.reliability),
            "accuracy": float(self.accuracy),
            "ece": float(self.ece),
            "nll": float(self.nll),
            "d_temperature": float(self.temperature - prev_t),
            "d_reliability": float(self.reliability - prev_rel),
        }
        self.history.append(record)
        return record

    def update_queries(
        self,
        query: np.ndarray,
        prototypes: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        query = _unit(np.asarray(query, dtype=np.float64))
        if query.ndim == 1:
            query = query.reshape(1, -1)
        prototypes = _unit(np.asarray(prototypes, dtype=np.float64))
        return self.update(query @ prototypes.T, labels)

    # -- reference and drift ------------------------------------------
    def set_reference(self) -> None:
        """Snapshot the current window as the clean reference."""
        self.ref = {
            "accuracy": float(self.accuracy),
            "ece": float(self.ece),
            "reliability": float(self.reliability),
            "temperature": float(self.temperature),
            "window": int(self.window_n),
            "classes": sorted(int(c) for c in self.class_set),
        }

    def drift_report(self, thresholds: DriftThresholds | None = None) -> DriftReport:
        thresholds = thresholds or DriftThresholds()
        if self.ref is None or self.window_n < thresholds.min_window:
            return DriftReport(
                drift=False,
                abstain=False,
                action="continue",
                score=0.0,
                accuracy_drop=0.0,
                ece_increase=0.0,
                reliability_drop=0.0,
                coverage_drop=0.0,
                reasons=["insufficient_window"],
            )

        acc_drop = float(self.ref["accuracy"] - self.accuracy)
        ece_inc = (
            float(self.ece - self.ref["ece"])
            if np.isfinite(self.ece) and np.isfinite(self.ref["ece"])
            else 0.0
        )
        rel_drop = float(self.ref["reliability"] - self.reliability)
        ref_classes = set(self.ref.get("classes", []))
        if ref_classes:
            cov_drop = 1.0 - len(self.class_set & ref_classes) / float(len(ref_classes))
        else:
            cov_drop = 0.0

        terms = {
            "accuracy_drop": max(0.0, acc_drop) / max(1e-9, thresholds.accuracy_drop),
            "ece_increase": max(0.0, ece_inc) / max(1e-9, thresholds.ece_increase),
            "reliability_drop": max(0.0, rel_drop) / max(1e-9, thresholds.reliability_drop),
            "coverage_drop": max(0.0, cov_drop) / max(1e-9, thresholds.coverage_drop),
        }
        score = float(max(terms.values()))
        reasons = [k for k, v in terms.items() if v >= thresholds.score]
        drift = score >= float(thresholds.score)
        abstain = score >= float(thresholds.abstain_score) or (
            self.ref["accuracy"] > 0.2 and self.accuracy < 0.5 * self.ref["accuracy"]
        )
        action = "abstain" if abstain else ("recalibrate" if drift else "continue")
        return DriftReport(
            drift=bool(drift),
            abstain=bool(abstain),
            action=action,
            score=score,
            accuracy_drop=acc_drop,
            ece_increase=ece_inc,
            reliability_drop=rel_drop,
            coverage_drop=float(cov_drop),
            reasons=reasons,
        )

    def recalibrate(self) -> None:
        """Refit from the current window and accept it as the new reference."""
        self._refit()
        self.set_reference()


class StreamingCalibrator:
    """One :class:`StreamingModalityCalibration` per modality."""

    def __init__(
        self,
        modalities,
        *,
        window_size: int = 256,
        refit_every: int = 1,
        min_samples: int = 2,
        ece_bins: int = 10,
        thresholds: DriftThresholds | None = None,
    ):
        self.modalities = tuple(modalities)
        self.window_size = int(window_size)
        self.refit_every = int(refit_every)
        self.min_samples = int(min_samples)
        self.ece_bins = int(ece_bins)
        self.thresholds = thresholds or DriftThresholds()
        self.states: dict[str, StreamingModalityCalibration] = {
            m: StreamingModalityCalibration(
                m,
                window_size=window_size,
                refit_every=refit_every,
                min_samples=min_samples,
                ece_bins=ece_bins,
            )
            for m in self.modalities
        }
        self.history: list[dict] = []
        self.accepted: dict[str, ModalityCalibration] | None = None

    def update(
        self,
        embeddings: dict,
        prototypes: dict,
        labels: np.ndarray,
        *,
        record: bool = True,
    ) -> dict:
        out = {}
        for m, state in self.states.items():
            if m not in embeddings or m not in prototypes:
                continue
            out[m] = state.update_queries(embeddings[m], prototypes[m], labels)
        if record:
            self.history.append(self.summary())
        return out

    def set_reference(self) -> None:
        for state in self.states.values():
            state.set_reference()
        self.accepted = self.calibration()

    def calibration(self) -> dict[str, ModalityCalibration]:
        return {m: s.state() for m, s in self.states.items()}

    def accepted_calibration(self) -> dict[str, ModalityCalibration]:
        """Last calibration accepted as the clean reference."""
        return self.accepted or self.calibration()

    def drift_report(self) -> dict:
        reports = {m: s.drift_report(self.thresholds) for m, s in self.states.items()}
        actions = {r.action for r in reports.values()}
        action = (
            "abstain"
            if "abstain" in actions
            else "recalibrate"
            if "recalibrate" in actions
            else "continue"
        )
        score = max((r.score for r in reports.values()), default=0.0)
        reasons = sorted({x for r in reports.values() for x in r.reasons})
        return {
            "action": action,
            "drift": any(r.drift for r in reports.values()),
            "abstain": any(r.abstain for r in reports.values()),
            "score": float(score),
            "reasons": reasons,
            "per_modality": {m: r.to_dict() for m, r in reports.items()},
        }

    def summary(self) -> dict:
        return {
            "drift": self.drift_report(),
            "modalities": {m: s.to_dict() for m, s in self.states.items()},
        }

    def recalibrate(self) -> None:
        for state in self.states.values():
            state.recalibrate()


# ---------------------------------------------------------------------------
# Calibration-set audit (poisoned negative control)
# ---------------------------------------------------------------------------


@dataclass
class PoisonThresholds:
    """Thresholds for calibration-set audit.

    Rates are compared against a clean baseline when one is supplied; the
    thresholds below therefore describe *excess* rates, which is what makes
    the audit usable for a weak modality whose absolute error rate is high.
    """

    label_mismatch: float = 0.05
    margin_violation: float = 0.05
    duplicate_rate: float = 0.05
    centroid_outlier: float = 0.05
    flag_rate: float = 0.05
    abstain_flag_rate: float = 0.30
    abstain_label_mismatch: float = 0.30
    dup_cosine: float = 0.999
    margin: float = 0.0
    outlier_z: float = 3.5

    def to_dict(self) -> dict:
        return {
            "label_mismatch": float(self.label_mismatch),
            "margin_violation": float(self.margin_violation),
            "duplicate_rate": float(self.duplicate_rate),
            "centroid_outlier": float(self.centroid_outlier),
            "flag_rate": float(self.flag_rate),
            "abstain_flag_rate": float(self.abstain_flag_rate),
            "abstain_label_mismatch": float(self.abstain_label_mismatch),
            "dup_cosine": float(self.dup_cosine),
            "margin": float(self.margin),
            "outlier_z": float(self.outlier_z),
        }


@dataclass
class PoisonAudit:
    n: int
    rates: dict
    excess: dict
    detected: bool
    decision: str  # clean | recalibrate | abstain
    reasons: list[str]
    flags: np.ndarray
    baseline_rates: dict | None = None
    signal_flags: dict | None = None

    def to_dict(self) -> dict:
        return {
            "n": int(self.n),
            "rates": {k: float(v) for k, v in self.rates.items()},
            "excess": {k: float(v) for k, v in self.excess.items()},
            "detected": bool(self.detected),
            "decision": self.decision,
            "reasons": list(self.reasons),
            "flag_rate": float(self.flags.mean()) if len(self.flags) else 0.0,
            "baseline_rates": (
                {k: float(v) for k, v in self.baseline_rates.items()}
                if self.baseline_rates is not None
                else None
            ),
        }


def _audit_flags(
    query: np.ndarray,
    prototypes: np.ndarray,
    labels: np.ndarray,
    thresholds: PoisonThresholds,
    temperature: float | None = None,
) -> tuple[np.ndarray, dict, dict]:
    q = _unit(np.asarray(query, dtype=np.float64))
    if q.ndim == 1:
        q = q.reshape(1, -1)
    p = _unit(np.asarray(prototypes, dtype=np.float64))
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = len(q)
    if n == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, {
            "label_mismatch": 0.0,
            "margin_violation": 0.0,
            "duplicate": 0.0,
            "centroid_outlier": 0.0,
            "flag_rate": 0.0,
        }, {
            "label_mismatch": empty,
            "margin_violation": empty,
            "duplicate": empty,
            "centroid_outlier": empty,
        }

    logits = q @ p.T
    if temperature is not None and np.isfinite(temperature) and temperature > 0:
        logits = logits / float(temperature)
    pred = logits.argmax(axis=1)
    valid = (labels >= 0) & (labels < p.shape[0])

    mismatch = np.ones(n, dtype=bool)
    mismatch[valid] = pred[valid] != labels[valid]

    margin = np.full(n, -np.inf, dtype=np.float64)
    if valid.any():
        rows = np.arange(n)[valid]
        own = logits[rows, labels[valid]]
        other = logits[valid].copy()
        other[np.arange(len(rows)), labels[valid]] = -np.inf
        margin[valid] = own - other.max(axis=1)
    margin_violation = valid & (margin < float(thresholds.margin))

    if n > 1:
        sims = q @ q.T
        np.fill_diagonal(sims, -1.0)
        duplicate = sims.max(axis=1) >= float(thresholds.dup_cosine)
    else:
        duplicate = np.zeros(n, dtype=bool)

    outlier = np.zeros(n, dtype=bool)
    for c in np.unique(labels[valid]):
        idx = np.where(labels == c)[0]
        if len(idx) < 4:
            continue
        centroid = _unit(q[idx].mean(axis=0))
        dist = 1.0 - q[idx] @ centroid
        med = float(np.median(dist))
        mad = float(np.median(np.abs(dist - med)))
        z = (dist - med) / (1.4826 * mad + 1e-9)
        outlier[idx] = z > float(thresholds.outlier_z)

    flags = mismatch | margin_violation | duplicate | outlier
    rates = {
        "label_mismatch": float(mismatch.mean()),
        "margin_violation": float(margin_violation.mean()),
        "duplicate": float(duplicate.mean()),
        "centroid_outlier": float(outlier.mean()),
        "flag_rate": float(flags.mean()),
    }
    signal_flags = {
        "label_mismatch": mismatch,
        "margin_violation": margin_violation,
        "duplicate": duplicate,
        "centroid_outlier": outlier,
    }
    return flags, rates, signal_flags


def audit_calibration_set(
    query: np.ndarray,
    prototypes: np.ndarray,
    labels: np.ndarray,
    *,
    baseline: PoisonAudit | dict | None = None,
    thresholds: PoisonThresholds | None = None,
    temperature: float | None = None,
) -> PoisonAudit:
    """Flag poisoned calibration samples and decide what to do with them.

    ``baseline`` is a clean audit (or its ``rates`` dict) from the same
    modality.  When supplied, each rate is reduced by the baseline rate before
    thresholding; this keeps a weak modality (tactile is at ~0.3 accuracy) from
    being flagged for being weak rather than poisoned.
    """
    thresholds = thresholds or PoisonThresholds()
    flags, rates, signal_flags = _audit_flags(
        query, prototypes, labels, thresholds, temperature=temperature
    )
    if isinstance(baseline, PoisonAudit):
        base = {k: float(v) for k, v in baseline.rates.items()}
    elif isinstance(baseline, dict):
        base = {k: float(v) for k, v in baseline.items()}
    else:
        base = None
    if base is None:
        excess = dict(rates)
    else:
        excess = {k: max(0.0, rates[k] - base.get(k, 0.0)) for k in rates}

    checks = {
        "label_mismatch": thresholds.label_mismatch,
        "margin_violation": thresholds.margin_violation,
        "duplicate": thresholds.duplicate_rate,
        "centroid_outlier": thresholds.centroid_outlier,
        "flag_rate": thresholds.flag_rate,
    }
    reasons = [k for k, lim in checks.items() if excess[k] > float(lim)]
    detected = bool(reasons)
    abstain = (
        excess.get("label_mismatch", 0.0) > float(thresholds.abstain_label_mismatch)
        or excess.get("flag_rate", 0.0) > float(thresholds.abstain_flag_rate)
    )
    decision = "abstain" if abstain else ("recalibrate" if detected else "clean")
    return PoisonAudit(
        n=int(len(flags)),
        rates=rates,
        excess=excess,
        detected=detected,
        decision=decision,
        reasons=reasons,
        flags=flags,
        baseline_rates=base,
        signal_flags=signal_flags,
    )


# ---------------------------------------------------------------------------
# Poisoning helpers and evaluation utilities
# ---------------------------------------------------------------------------


def poison_labels(
    labels: np.ndarray,
    fraction: float,
    n_classes: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Flip ``fraction`` of labels to a different class."""
    rng = rng or np.random.default_rng(0)
    labels = np.asarray(labels, dtype=np.int64).copy()
    n = len(labels)
    k = int(round(float(fraction) * n))
    if k <= 0 or n_classes < 2:
        return labels
    idx = rng.choice(n, size=min(k, n), replace=False)
    for i in idx:
        choices = np.delete(np.arange(int(n_classes)), labels[i])
        labels[i] = int(rng.choice(choices))
    return labels


def poison_embeddings_replace(
    embeddings: dict[str, np.ndarray],
    labels: np.ndarray,
    fraction: float,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Replace ``fraction`` of samples with a different-class donor embedding."""
    rng = rng or np.random.default_rng(0)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    k = int(round(float(fraction) * n))
    out = {m: np.asarray(v, dtype=np.float64).copy() for m, v in embeddings.items()}
    if k <= 0:
        return out
    idx = rng.choice(n, size=min(k, n), replace=False)
    for i in idx:
        donors = np.where(labels != labels[i])[0]
        if len(donors) == 0:
            continue
        j = int(rng.choice(donors))
        for m in out:
            out[m][i] = out[m][j]
    return out


def poison_duplicate_inject(
    embeddings: dict[str, np.ndarray],
    labels: np.ndarray,
    fraction: float,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Append ``fraction * n`` duplicates of randomly chosen samples."""
    rng = rng or np.random.default_rng(0)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    k = int(round(float(fraction) * n))
    out = {m: np.asarray(v, dtype=np.float64).copy() for m, v in embeddings.items()}
    if k <= 0:
        return out, labels.copy()
    idx = rng.choice(n, size=min(k, n), replace=False)
    for m in out:
        out[m] = np.concatenate([out[m], out[m][idx]], axis=0)
    return out, np.concatenate([labels, labels[idx]], axis=0)


def select_balanced_subset(
    labels: np.ndarray,
    n: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Pick ``n`` indices with as many classes represented as possible.

    For ``n < n_classes`` one sample is drawn from ``n`` randomly chosen
    classes.  For larger ``n`` the budget is split evenly and the remainder is
    assigned to a seeded random subset of classes.  The returned order is
    shuffled so streaming does not see class blocks.
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = int(n)
    if n <= 0 or len(labels) == 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    classes = np.array(sorted(set(int(x) for x in labels.tolist())))
    if n < len(classes):
        chosen_classes = rng.choice(classes, size=n, replace=False)
        idx = [int(rng.choice(np.where(labels == c)[0])) for c in chosen_classes]
    else:
        base = n // len(classes)
        rem = n % len(classes)
        order = rng.permutation(classes)
        idx = []
        for pos, c in enumerate(order):
            want = base + (1 if pos < rem else 0)
            pool = np.where(labels == c)[0]
            take = min(want, len(pool))
            if take:
                idx.extend(int(x) for x in rng.choice(pool, size=take, replace=False))
        if len(idx) < n:
            remaining = np.setdiff1d(np.arange(len(labels)), np.array(idx, dtype=int))
            if len(remaining):
                extra = rng.choice(remaining, size=min(n - len(idx), len(remaining)), replace=False)
                idx.extend(int(x) for x in extra)
    idx_arr = np.array(idx, dtype=np.int64)
    rng.shuffle(idx_arr)
    return idx_arr


def _metrics_from_probs(probs: np.ndarray, labels: np.ndarray, ece_bins: int = 10) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(labels) == 0 or probs.ndim != 2 or probs.shape[1] == 0:
        return {
            "accuracy": float("nan"),
            "nll": float("nan"),
            "ece": float("nan"),
            "mean_confidence": float("nan"),
            "n": int(len(labels)),
        }
    pred = probs.argmax(axis=1)
    valid = (labels >= 0) & (labels < probs.shape[1])
    accuracy = float(np.mean(pred == labels))
    if valid.any():
        picked = probs[np.arange(len(labels))[valid], labels[valid]].clip(1e-12)
        nll = float(-np.log(picked).mean())
        conf = probs[valid].max(axis=1)
        ok = (pred[valid] == labels[valid]).astype(np.float64)
        ece = float(expected_calibration_error(conf, ok, n_bins=ece_bins))
    else:
        nll = float("nan")
        ece = float("nan")
    return {
        "accuracy": accuracy,
        "nll": nll,
        "ece": ece,
        "mean_confidence": float(probs.max(axis=1).mean()),
        "n": int(len(labels)),
    }


def calibrated_test_metrics(
    embeddings: dict[str, np.ndarray],
    prototypes: dict[str, np.ndarray],
    labels: np.ndarray,
    calibration: dict[str, ModalityCalibration],
    *,
    modalities: tuple[str, ...] | None = None,
    ece_bins: int = 10,
) -> dict:
    """Evaluate a calibration record on a labelled set.

    Returns per-modality metrics plus the reliability-weighted fused metrics,
    matching the scoring used by :func:`calibrated_scores`.
    """
    modalities = tuple(modalities or calibration.keys())
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    per: dict[str, dict] = {}
    for m in modalities:
        if m not in calibration or m not in embeddings or m not in prototypes:
            continue
        cal = calibration[m]
        logits = _unit(np.asarray(embeddings[m], dtype=np.float64)) @ _unit(
            np.asarray(prototypes[m], dtype=np.float64)
        ).T
        probs = _softmax(logits / max(float(cal.temperature), 1e-3))
        per[m] = _metrics_from_probs(probs, labels, ece_bins=ece_bins)

    scores = []
    for i in range(len(labels)):
        q = {m: embeddings[m][i] for m in modalities if m in embeddings}
        scores.append(calibrated_scores(q, prototypes, calibration))
    if scores and len(scores[0]) > 0:
        stacked = np.stack(scores, axis=0)
        fused = _metrics_from_probs(_softmax(stacked), labels, ece_bins=ece_bins)
    else:
        fused = _metrics_from_probs(np.zeros((len(labels), 0)), labels, ece_bins=ece_bins)
    return {"per_modality": per, "fused": fused}


__all__ = [
    "DriftReport",
    "DriftThresholds",
    "PoisonAudit",
    "PoisonThresholds",
    "StreamingCalibrator",
    "StreamingModalityCalibration",
    "audit_calibration_set",
    "calibrate_drift_thresholds",
    "calibrated_test_metrics",
    "poison_duplicate_inject",
    "poison_embeddings_replace",
    "poison_labels",
    "select_balanced_subset",
]
