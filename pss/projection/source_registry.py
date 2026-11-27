"""Per-source isolation for the calibration pipeline.

The first pass only down-weighted a source's samples by its current-window
agreement excess, computed against a global average that **included the source
itself**.  A large bad source therefore dragged the reference down with it and
masked its own excess: the bigger the poison, the harder it was to see.

This module makes a source a first-class detection axis:

- one CUSUM per source on that source's **absolute** label/consensus mismatch
  rate, calibrated on clean windows.  Absolute, so no cross-source reference can
  be gamed;
- two states, ``trusted`` and ``quarantined``.  Quarantine zeroes the source's
  sample weights and excludes it from the aggregation; it releases only after a
  run of normal windows;
- every transition is recorded for the audit trail.

Sources stay a *detection* axis, not a parameter axis: there is still exactly
one aggregated calibration, so the step bound remains well defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .cusum import CusumDetector

__all__ = ["SourceRegistry", "SourceState"]

TRUSTED = "trusted"
QUARANTINED = "quarantined"


@dataclass
class SourceState:
    name: str
    status: str = TRUSTED
    recover: int = 0
    quarantines: int = 0
    releases: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "recover": int(self.recover),
            "quarantines": int(self.quarantines),
            "releases": int(self.releases),
        }


@dataclass
class SourceRegistry:
    """Per-source CUSUM with quarantine and audited release."""

    detectors: dict[str, CusumDetector] = field(default_factory=dict)
    recover_windows: int = 3
    states: dict[str, SourceState] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    @classmethod
    def calibrate(
        cls,
        clean_windows: Sequence[Mapping[str, float]],
        *,
        slack_sd: float = 0.5,
        threshold_sd: float = 5.0,
        recover_windows: int = 3,
        min_windows: int = 5,
    ) -> "SourceRegistry":
        """Build one detector per source from clean mismatch-rate windows.

        ``threshold_sd`` matches the per-class detector: testing many sources
        multiplies the false-alarm rate, so the bar is 5 sd rather than the
        global detector's 4.
        """
        values: dict[str, list[float]] = {}
        for window in clean_windows:
            for source, value in window.items():
                if np.isfinite(value):
                    values.setdefault(str(source), []).append(float(value))

        detectors: dict[str, CusumDetector] = {}
        for source, vals in values.items():
            if len(vals) < int(min_windows):
                continue
            reference = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            sd = max(sd, 1e-3)
            detectors[source] = CusumDetector(
                reference=reference,
                slack=slack_sd * sd,
                threshold=threshold_sd * sd,
                direction=1.0,
            )
        return cls(detectors=detectors, recover_windows=int(recover_windows))

    def update(self, mismatches: Mapping[str, float]) -> dict[str, str]:
        """Update every known source; return its status after this window."""
        statuses: dict[str, str] = {}
        for source, detector in self.detectors.items():
            state = self.states.setdefault(source, SourceState(name=source))
            value = mismatches.get(source)

            if state.status == QUARANTINED:
                recovered = value is None or value <= detector.reference + detector.slack
                state.recover = state.recover + 1 if recovered else 0
                if state.recover >= self.recover_windows:
                    state.status = TRUSTED
                    state.releases += 1
                    self.events.append(
                        {
                            "event": "release",
                            "source": source,
                            "after_windows": int(state.recover),
                        }
                    )
                    state.recover = 0
                    detector.state = 0.0
                statuses[source] = state.status
                continue

            if value is not None and detector.update(value):
                state.status = QUARANTINED
                state.quarantines += 1
                state.recover = 0
                detector.state = 0.0
                self.events.append({"event": "quarantine", "source": source})
            statuses[source] = state.status
        return statuses

    def apply(self, trust: Mapping[str, float]) -> dict[str, float]:
        """Zero the trust of quarantined sources; leave the rest untouched."""
        out: dict[str, float] = {}
        for source, value in trust.items():
            state = self.states.get(source)
            out[source] = 0.0 if state is not None and state.status == QUARANTINED else float(value)
        return out

    def quarantined(self) -> list[str]:
        return sorted(s for s, st in self.states.items() if st.status == QUARANTINED)

    def describe(self) -> dict:
        return {
            "sources": {s: st.to_dict() for s, st in sorted(self.states.items())},
            "quarantined": self.quarantined(),
            "events": list(self.events),
        }
