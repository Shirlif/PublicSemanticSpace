"""Runtime-facing adapter for the robust calibration pipeline.

This is the seam the runtime needs, not the wiring itself.  ``PerceptionService``
currently reads a static ``reliability`` dict from a checkpoint; this service
holds the streaming pipeline and exposes the three things the runtime would call:

    service.update(window, holdout)   # once per window
    service.params()                  # temperature / reliability for fusion
    service.confidence(probs_row)     # calibrated confidence or conformal set

Deliberately not wired into ``PerceptionService`` yet: that path decides how the
recognition loop consumes temperature and reliability, and changing it silently
would alter every L2 number.  The wiring is a separate, reviewable change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pss.projection.calibration_pipeline import (
    CalibrationBatch,
    Params,
    RobustCalibrationPipeline,
    fused_probabilities,
)
from pss.projection.confidence_map import ConfidenceMap, apply_confidence_map
from pss.projection.conformal import AdaptiveConformal

__all__ = ["CalibrationService"]


@dataclass
class CalibrationService:
    """Windowed calibration with a separated confidence channel.

    ``confidence_mode`` selects the output contract:

    - ``"map"``      probability from the monotone map (default)
    - ``"conformal"`` prediction set with an adaptive coverage guarantee

    The two are not interchangeable downstream: ``"map"`` answers "how sure",
    ``"conformal"`` answers "which labels to keep".
    """

    pipeline: RobustCalibrationPipeline
    confidence_mode: str = "map"
    confidence_map: ConfidenceMap | None = None
    conformal: AdaptiveConformal | None = None
    history: list[dict] = field(default_factory=list)

    def update(
        self,
        window: CalibrationBatch,
        holdout: CalibrationBatch | None = None,
        *,
        force_gate: bool = False,
    ) -> dict:
        record = self.pipeline.step(
            window, window_holdout=holdout, force_gate=force_gate
        )
        self.history.append(record)
        return record

    def params(self) -> Params:
        return self.pipeline.params

    def state(self) -> str:
        return self.pipeline.state.value

    def fit_confidence(self, batch: CalibrationBatch) -> ConfidenceMap:
        """Fit the monotone map on a window the audit accepted."""
        from pss.projection.calibration_pipeline import fit_confidence_maps

        maps = fit_confidence_maps(self.pipeline.params, batch)
        self.confidence_map = maps.get("fused")
        return self.confidence_map

    def calibrate_conformal(
        self,
        batch: CalibrationBatch,
        *,
        alpha_target: float = 0.10,
        gamma: float = 0.01,
    ) -> AdaptiveConformal:
        probs, labels = fused_probabilities(self.pipeline.params, batch)
        self.conformal = AdaptiveConformal(
            alpha_target=alpha_target, gamma=gamma
        ).calibrate(probs, labels)
        return self.conformal

    def confidence(self, fused_top1: float) -> float:
        """Map mode: probability for a **fused top-1 score**.

        The map is fitted on fused top-1 scores, so the caller must pass that
        statistic (not a probability) — mixing the two scales silently produces
        wrong numbers.
        """
        if self.confidence_map is None:
            return float("nan")
        return float(
            apply_confidence_map(np.asarray([float(fused_top1)]), self.confidence_map)[0]
        )

    def prediction_set(self, probs_row: np.ndarray) -> np.ndarray:
        """Conformal mode: boolean mask of labels to keep."""
        if self.conformal is None:
            raise RuntimeError("conformal channel is not calibrated")
        return self.conformal.predict_set(probs_row)

    def observe_outcome(self, probs_row: np.ndarray, label: int, *, trusted: bool = True) -> dict:
        """Feed the realised label back to the conformal channel."""
        if self.conformal is None:
            raise RuntimeError("conformal channel is not calibrated")
        return self.conformal.step(probs_row, label, trusted=trusted)

    def describe(self) -> dict:
        return {
            "state": self.state(),
            "confidence_mode": self.confidence_mode,
            "temperature": dict(self.pipeline.params.temperature),
            "reliability": dict(self.pipeline.params.reliability),
            "pinned": sorted(self.pipeline.params.pinned),
            "windows": len(self.history),
        }
