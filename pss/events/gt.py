"""Ground-truth contact events from MuJoCo contact forces."""

from __future__ import annotations

import numpy as np

from .model import ContactEvent, ContactFrame
from .segmenter import segment_signal


def events_from_contacts(
    frames: list[ContactFrame],
    *,
    threshold: float = 0.5,
    min_duration: float = 0.02,
    max_gap: float = 0.05,
) -> list[ContactEvent]:
    """Build GT events from per-frame contact forces.

    Contacts with the table/floor or unknown instances are ignored.  Each
    object pair gets a force time series over the same frame grid; intervals
    above ``threshold`` become events, with peak force and the contact
    location at the peak.
    """

    if not frames:
        return []
    known = set()
    for frame in frames:
        known.update(frame.poses)
    times = np.asarray([float(f.t) for f in frames], dtype=np.float64)

    def pair_of(c: dict) -> tuple[str, str] | None:
        a, b = str(c.get("instance1")), str(c.get("instance2"))
        if a not in known or b not in known or a == b:
            return None
        return tuple(sorted((a, b)))

    pairs: set[tuple[str, str]] = set()
    for frame in frames:
        for c in frame.contacts:
            pair = pair_of(c)
            if pair is not None:
                pairs.add(pair)

    events: list[ContactEvent] = []
    for pair in sorted(pairs):
        scores = np.zeros(len(frames), dtype=np.float64)
        peaks: list[dict] = []
        for i, frame in enumerate(frames):
            best = 0.0
            for c in frame.contacts:
                if pair_of(c) != pair:
                    continue
                force = float(c.get("normal_force", 0.0))
                if force > best:
                    best = force
                    peaks.append(
                        {
                            "t": float(frame.t),
                            "force": force,
                            "pos": np.asarray(
                                c.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64
                            ),
                        }
                    )
            scores[i] = best
        intervals = segment_signal(
            times, scores,
            threshold=threshold,
            min_duration=min_duration,
            max_gap=max_gap,
        )
        for k, interval in enumerate(intervals):
            window = [
                p for p in peaks
                if interval.start - 1e-9 <= p["t"] <= interval.end + 1e-9
            ]
            peak = max(window, key=lambda p: p["force"]) if window else None
            events.append(
                ContactEvent(
                    event_id=f"{pair[0]}-{pair[1]}-{interval.start:.4f}-{k}",
                    subject=pair[0],
                    object=pair[1],
                    onset=float(interval.start),
                    offset=float(interval.end),
                    location=None if peak is None else peak["pos"],
                    intensity=None if peak is None else float(peak["force"]),
                    confidence=1.0,
                    status="confirmed",
                    provenance={
                        "source": "mujoco_contact",
                        "threshold": float(threshold),
                    },
                )
            )
    return sorted(events, key=lambda e: (e.onset, e.subject, e.object))


__all__ = ["events_from_contacts"]
