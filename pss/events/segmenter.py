"""Single-channel onset/offset segmentation with hysteresis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def segment_signal(
    times: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
    hysteresis: float = 0.0,
    min_duration: float = 0.02,
    max_gap: float = 0.05,
) -> list[Interval]:
    """Threshold a score sequence into intervals.

    ``hysteresis`` is the lower threshold used to continue an active interval;
    ``max_gap`` merges intervals separated by a short dropout (contact chatter).
    """

    times = np.asarray(times, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(times) == 0:
        return []
    high = float(threshold)
    low = float(threshold) - max(0.0, float(hysteresis))
    intervals: list[list[float]] = []
    active = False
    start = 0.0
    last_on = 0.0
    for t, s in zip(times, scores):
        if not active and s >= high:
            active = True
            start = float(t)
            last_on = float(t)
        elif active and s >= low:
            last_on = float(t)
        elif active:
            intervals.append([start, last_on])
            active = False
    if active:
        intervals.append([start, last_on])

    merged: list[list[float]] = []
    for a, b in intervals:
        if merged and a - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [
        Interval(float(a), float(b))
        for a, b in merged
        if (b - a) >= float(min_duration)
    ]


def temporal_iou(a: Interval, b: Interval) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


__all__ = ["Interval", "segment_signal", "temporal_iou"]
