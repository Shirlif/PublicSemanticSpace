"""Adaptive conformal inference for the confidence channel.

The binned map calibrates on the current window and is re-fitted every window.
That works, but it is still a *fit*: there is no statement about what happens if
the next window is different.  Adaptive conformal inference (Gibbs & Candès,
2021) replaces the fit with a **coverage guarantee**:

    err_t = 1 if the true label is outside the prediction set
    alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)

with prediction set ``{y : 1 - p(y) <= q_t}`` where ``q_t`` is the
``1 - alpha_t`` empirical quantile of the stored nonconformity scores.  The
long-run miscoverage rate stays at ``alpha_target`` under arbitrary distribution
shift -- including drift -- with no temperature, no bins and no hard bound.

Trade-off: the output is a *set*, not a probability, so its size is the relevant
efficiency measure.  Under drift the sets grow; that is the correct behaviour,
not a failure.

``gamma`` is the only knob and its trade-off is explicit: larger adapts faster
but makes coverage noisier.  Unlike a temperature bound it does not have to be
tuned per distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "AdaptiveConformal",
    "ConformalResult",
    "evaluate_conformal",
]


@dataclass
class AdaptiveConformal:
    """Streaming adaptive conformal predictor (threshold form).

    The threshold is updated directly, ``q_{t+1} = q_t + gamma (err_t - alpha)``,
    rather than by re-quantiling a fixed calibration pool.  The distinction
    matters: a fixed pool cannot express a threshold larger than its own maximum,
    so under a strong shift coverage would saturate below target.  Updating the
    threshold lets the sets grow past the calibration range.
    """

    alpha_target: float = 0.10
    gamma: float = 0.01
    q: float = field(init=False, default=1.0)
    scores: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.alpha_target = float(np.clip(self.alpha_target, 1e-3, 1 - 1e-3))

    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> "AdaptiveConformal":
        """Initialise the threshold from a labelled calibration window."""
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if probs.ndim != 2 or len(probs) != len(labels):
            raise ValueError("probs must be (n, n_classes) aligned with labels")
        valid = (labels >= 0) & (labels < probs.shape[1])
        if not valid.any():
            raise ValueError("no valid labels for calibration")
        picked = probs[np.arange(len(labels)), np.clip(labels, 0, probs.shape[1] - 1)]
        self.scores = [float(1.0 - p) for p in picked[valid]]
        level = float(np.clip(1.0 - self.alpha_target, 0.0, 1.0))
        self.q = float(np.quantile(np.asarray(self.scores), level, method="higher"))
        return self

    def quantile(self) -> float:
        return float(self.q)

    def predict_set(self, probs_row: np.ndarray) -> np.ndarray:
        return (1.0 - np.asarray(probs_row, dtype=np.float64)) <= self.q + 1e-12

    def step(
        self,
        probs_row: np.ndarray,
        label: int,
        *,
        trusted: bool = True,
    ) -> dict:
        """Predict for one sample, then update the threshold from the outcome.

        ``trusted=False`` predicts but does **not** update the threshold.  The
        guarantee assumes the observed label is the truth; a poisoned label would
        steer ``q`` in the wrong direction while coverage, measured against the
        same poisoned labels, still looks correct.  Quarantined windows must be
        passed with ``trusted=False``.
        """
        mask = self.predict_set(probs_row)
        label = int(label)
        covered = bool(0 <= label < len(mask) and mask[label])
        if trusted:
            err = 0.0 if covered else 1.0
            self.q = float(
                np.clip(self.q + self.gamma * (err - self.alpha_target), 0.0, 1.0)
            )
        return {
            "covered": covered,
            "size": int(mask.sum()),
            "q": self.q,
            "updated": bool(trusted),
        }


@dataclass(frozen=True)
class ConformalResult:
    """Aggregate conformal behaviour over a stream."""

    coverage: float
    target_coverage: float
    mean_size: float
    max_size: int
    n: int
    q_final: float

    def to_dict(self) -> dict:
        return {
            "coverage": float(self.coverage),
            "target_coverage": float(self.target_coverage),
            "mean_size": float(self.mean_size),
            "max_size": int(self.max_size),
            "n": int(self.n),
            "q_final": float(self.q_final),
        }


def evaluate_conformal(
    calib_probs: np.ndarray,
    calib_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    *,
    alpha_target: float = 0.10,
    gamma: float = 0.01,
    trust: np.ndarray | None = None,
) -> ConformalResult:
    """Run ACI: calibrate on one window, stream the test set.

    ``trust`` is an optional boolean mask over the test rows.  Untrusted rows
    are predicted but neither counted nor used to update the threshold — see
    :meth:`AdaptiveConformal.step`.
    """
    test_probs = np.asarray(test_probs, dtype=np.float64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    if test_probs.ndim != 2 or len(test_probs) != len(test_labels):
        raise ValueError("test_probs must be (n, n_classes) aligned with labels")
    valid = (test_labels >= 0) & (test_labels < test_probs.shape[1])
    if trust is not None:
        trust = np.asarray(trust, dtype=bool).reshape(-1)
        if len(trust) != len(test_labels):
            raise ValueError("trust must align with test labels")
        valid = valid & trust
    if not valid.any():
        raise ValueError("no valid test labels")

    predictor = AdaptiveConformal(alpha_target=alpha_target, gamma=gamma)
    predictor.calibrate(calib_probs, calib_labels)

    covered = 0
    sizes: list[int] = []
    for row, label in zip(test_probs[valid], test_labels[valid]):
        out = predictor.step(row, int(label), trusted=True)
        covered += int(out["covered"])
        sizes.append(int(out["size"]))

    n = len(sizes)
    return ConformalResult(
        coverage=covered / n if n else float("nan"),
        target_coverage=1.0 - alpha_target,
        mean_size=float(np.mean(sizes)) if sizes else float("nan"),
        max_size=int(np.max(sizes)) if sizes else 0,
        n=n,
        q_final=float(predictor.q),
    )
