"""Value / goal readout head (Phase 8): rule prior + learned preference head.

Produces a scalar priority in [0, 1] from goal proximity, safety relevance, and
novelty. It is used as an attention prior, not as a reward model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ValueHead:
    goal_xy: np.ndarray | None = None
    w_task: float = 0.6
    w_safety: float = 0.3
    w_novelty: float = 0.1

    def __post_init__(self):
        if self.goal_xy is not None:
            self.goal_xy = np.asarray(self.goal_xy, dtype=float)

    def score_features(self, distance: float, safety: float = 0.0, novelty: float = 0.0) -> float:
        task = 1.0 / (1.0 + float(distance) / 0.2)
        value = self.w_task * task + self.w_safety * float(safety) + self.w_novelty * float(novelty)
        return float(np.clip(value, 0.0, 1.0))

    def score(self, track, safety: float = 0.0, novelty: float = 0.0) -> float:
        if self.goal_xy is None:
            task = 0.5
            value = self.w_task * task + self.w_safety * float(safety) + self.w_novelty * float(novelty)
            return float(np.clip(value, 0.0, 1.0))
        d = float(np.linalg.norm(np.asarray(track.pos[:2], dtype=float) - self.goal_xy))
        return self.score_features(d, safety, novelty)

    def score_tracks(self, tracks, safety: dict[str, float] | None = None) -> dict[str, float]:
        safety = safety or {}
        return {
            tr.id: self.score(tr, safety=safety.get(getattr(tr, "category", None), 0.0))
            for tr in tracks
        }


from .learned import (  # noqa: E402
    LearnedValueHead,
    evaluate_rule_value,
    evaluate_value_scorer,
    load_learned_value_head,
    train_learned_value_head,
)

__all__ = [
    "LearnedValueHead",
    "ValueHead",
    "evaluate_rule_value",
    "evaluate_value_scorer",
    "load_learned_value_head",
    "train_learned_value_head",
]
