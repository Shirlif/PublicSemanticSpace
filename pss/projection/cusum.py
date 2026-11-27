"""One-sided CUSUM detectors for slow-drift attacks.

Rationale: a single window carries ~96 samples, so a targeted attack that moves
the global label/consensus agreement by ~1.5 percentage points sits inside the
window noise (sd ~2 points).  No single-window threshold can see it without a
false-alarm rate that makes the system unusable.

Streaming is the right frame: the effect is small *per window* but consistent
*across windows*.  A one-sided CUSUM accumulates evidence and fires when the
cumulative excess is large enough.

    S_t = max(0, S_{t-1} + direction * (x_t - reference) - slack)
    alarm when S_t > threshold

``slack`` is the per-window deviation treated as noise; ``threshold`` is how much
consistent excess we require.  Both are calibrated from clean bootstrap samples
so the clean false-alarm rate is controlled rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "CusumBank",
    "CusumDetector",
    "PerClassCusum",
    "calibrate_cusum",
    "calibrate_per_class",
]


@dataclass
class CusumDetector:
    """One-sided CUSUM on a scalar signal."""

    reference: float
    slack: float
    threshold: float
    direction: float = 1.0
    state: float = 0.0
    alarms: int = 0
    visits: int = 0

    def update(self, value: float) -> bool:
        if not np.isfinite(value):
            return False
        self.visits += 1
        deviation = self.direction * (float(value) - self.reference)
        self.state = max(0.0, self.state + deviation - self.slack)
        if self.state > self.threshold:
            self.alarms += 1
            self.state = 0.0
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "reference": float(self.reference),
            "slack": float(self.slack),
            "threshold": float(self.threshold),
            "direction": float(self.direction),
            "state": float(self.state),
            "alarms": int(self.alarms),
            "visits": int(self.visits),
        }


@dataclass
class CusumBank:
    """Two detectors on label/consensus agreement, in opposite directions.

    - agreement falling  -> labels disagree more than usual (label poisoning)
    - agreement rising   -> labels agree more than usual ("agree with the model")

    Both are poison signatures; drift moves the input distribution, not the
    label/consensus relation, so it should not trip either.
    """

    agreement_down: CusumDetector
    agreement_up: CusumDetector

    def update(self, label_consensus_agreement: float) -> list[str]:
        reasons: list[str] = []
        if self.agreement_down.update(label_consensus_agreement):
            reasons.append("agreement_down")
        if self.agreement_up.update(label_consensus_agreement):
            reasons.append("agreement_up")
        return reasons

    @property
    def alarmed(self) -> bool:
        return self.agreement_down.alarms > 0 or self.agreement_up.alarms > 0

    def to_dict(self) -> dict:
        return {
            "agreement_down": self.agreement_down.to_dict(),
            "agreement_up": self.agreement_up.to_dict(),
        }


@dataclass
class PerClassCusum:
    """One CUSUM per class, on that class's label/consensus mismatch rate.

    A single-class attack barely moves the global agreement, but it moves the
    attacked class's mismatch rate a lot (~20 points).  Per-class detectors see
    the strong local signal instead of the diluted global one.

    Detectors are keyed by **consensus** class: the attack relabels victims, so
    grouping by label would hide them inside the classes they spill into.
    """

    detectors: dict[int, CusumDetector] = field(default_factory=dict)
    alarms: dict[int, int] = field(default_factory=dict)

    def update(self, mixups: Mapping[int, float]) -> list[int]:
        fired: list[int] = []
        for cls, detector in self.detectors.items():
            value = mixups.get(int(cls))
            if value is None:
                continue
            if detector.update(value):
                fired.append(int(cls))
                self.alarms[int(cls)] = self.alarms.get(int(cls), 0) + 1
        return fired

    @property
    def alarmed(self) -> bool:
        return bool(self.alarms)

    def to_dict(self) -> dict:
        return {
            "classes": sorted(self.detectors),
            "alarms": {int(k): int(v) for k, v in self.alarms.items()},
        }


def calibrate_per_class(
    clean_windows: Sequence[Mapping[int, float]],
    *,
    slack_sd: float = 0.5,
    threshold_sd: float = 5.0,
    min_windows: int = 5,
) -> PerClassCusum:
    """Build one detector per class from clean per-class mismatch rates.

    ``threshold_sd`` defaults higher than the global detector (5 vs 4) because
    per-class testing multiplies the false-alarm rate by the number of classes:
    at 1% per class, 16 classes give roughly one spurious alarm every few
    windows.  The higher threshold buys back the family-wise rate at the cost
    of a couple of extra windows to detect.
    """
    values: dict[int, list[float]] = {}
    for window in clean_windows:
        for cls, value in window.items():
            if np.isfinite(value):
                values.setdefault(int(cls), []).append(float(value))

    detectors: dict[int, CusumDetector] = {}
    for cls, vals in values.items():
        if len(vals) < int(min_windows):
            continue
        reference = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        sd = max(sd, 1e-3)
        detectors[cls] = CusumDetector(
            reference=reference,
            slack=slack_sd * sd,
            threshold=threshold_sd * sd,
            direction=1.0,
        )
    return PerClassCusum(detectors=detectors)


def calibrate_cusum(
    clean_values: np.ndarray,
    *,
    slack_sd: float = 0.4,
    threshold_sd: float = 4.0,
) -> tuple[float, float, float]:
    """Return ``(reference, slack, threshold)`` calibrated on clean windows.

    ``clean_values`` must be the per-window signal on *resampled* clean windows;
    with no sampling noise the standard deviation collapses to zero and the
    calibration degenerates.

    Defaults come from a real-data grid: slack 0.4 sd / threshold 4 sd gave
    ~1% clean false alarms over 300 windows, detected a 1.15-sigma agreement
    shift in 5 windows, and a 0.39-sigma shift in 8.  Slack of a full standard
    deviation (the textbook-ish default) is too conservative here: it exceeds
    the targeted-attack effect and the CUSUM never accumulates.
    """
    values = np.asarray(clean_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.01, 0.05
    reference = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    sd = max(sd, 1e-3)
    return reference, slack_sd * sd, threshold_sd * sd
