"""Axis predicates for the runtime geometry path.

``RelationRuleSet`` historically produced ``in/contact/near`` only.  This
module adds ``left/right/above/below`` with two modes:

  hard    fixed dead zone ``eps`` (the auditable fallback)
  bayes   empirical-Bayes boundary from ``GridBayesBoundary``

The Bayes mode expects the boundary to be fitted on clean coordinate pairs at
the same scale as the runtime poses (metres or normalised units, but the same
for fit and inference).  The hard mode is the fallback when no prior/sigma is
available; it is also what the runtime uses when ``axis_mode="off"`` (the
default), so existing behaviour is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import RelationRecord
from .bayes_boundary import GridBayesBoundary

PREDICATES = ("left", "right", "above", "below")
INVERSE = {"left": "right", "right": "left", "above": "below", "below": "above"}


@dataclass
class AxisPredicate:
    boundary: GridBayesBoundary | None = None
    sigma: float = 0.0
    eps: float = 0.02
    abstain: float = 0.0
    mode: str = "hard"  # "hard" | "bayes"

    def predict_pair(self, pa, pb) -> tuple[np.ndarray, np.ndarray]:
        pa = np.asarray(pa, dtype=np.float64)
        pb = np.asarray(pb, dtype=np.float64)
        dx = float(pb[0] - pa[0])
        dy = float(pb[1] - pa[1])
        if self.mode == "bayes" and self.boundary is not None:
            proba, _ = self.boundary.predict_proba(
                np.asarray([dx]), np.asarray([dy]), sigma=self.sigma
            )
            p = proba[0]
            return p > 0.5, p
        # Hard fallback; order matches pair_labels: left, right, above, below.
        bits = np.asarray(
            [dx > self.eps, dx < -self.eps, dy < -self.eps, dy > self.eps],
            dtype=bool,
        )
        probs = bits.astype(np.float64)
        return bits, probs

    def records(self, instances: dict[str, dict]) -> list[RelationRecord]:
        source = "bayes_geometry" if self.mode == "bayes" else "hard_geometry"
        ids = list(instances)
        out: list[RelationRecord] = []
        for i, a in enumerate(ids):
            pa = np.asarray(instances[a]["pos"], dtype=np.float64)
            for b in ids[i + 1 :]:
                pb = np.asarray(instances[b]["pos"], dtype=np.float64)
                bits, probs = self.predict_pair(pa, pb)
                for predicate, bit, prob in zip(PREDICATES, bits, probs):
                    if not bool(bit) or float(prob) < float(self.abstain):
                        continue
                    conf = float(np.clip(prob, 0.0, 1.0))
                    out.append(RelationRecord(a, predicate, b, conf, source))
                    out.append(
                        RelationRecord(b, INVERSE[predicate], a, conf, source)
                    )
        return out


__all__ = ["AxisPredicate", "INVERSE", "PREDICATES"]
