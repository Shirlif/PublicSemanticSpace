"""Action readout head (Phase 8): rule controller + learned BC head.

Maps an entity state plus a goal into a primitive proposal. Both heads expose
the same ``propose_push_to`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActionProposal:
    kind: str
    target: int
    params: dict
    confidence: float = 1.0


class ActionHead:
    def __init__(
        self,
        gain: float = 4.0,
        max_force: float = 5.0,
        min_force: float = 0.5,
        tolerance: float = 0.03,
    ):
        self.gain = float(gain)
        self.max_force = float(max_force)
        self.min_force = float(min_force)
        self.tolerance = float(tolerance)

    def propose_push_to(self, pos_xy, goal_xy, target: int = 0) -> ActionProposal | None:
        pos = np.asarray(pos_xy, dtype=float)
        goal = np.asarray(goal_xy, dtype=float)
        delta = goal - pos
        dist = float(np.linalg.norm(delta))
        if dist < self.tolerance:
            return None
        direction = delta / max(dist, 1e-9)
        force = float(np.clip(self.gain * dist, self.min_force, self.max_force))
        return ActionProposal(
            kind="push",
            target=int(target),
            params={"force": force, "direction": direction.tolist()},
            confidence=float(min(1.0, dist / 0.2)),
        )

    def propose_place_at(self, target: int, goal_xy) -> ActionProposal:
        return ActionProposal(
            kind="place",
            target=int(target),
            params={"target_xy": [float(goal_xy[0]), float(goal_xy[1])]},
            confidence=1.0,
        )


from .learned import (  # noqa: E402  (import after the interface definitions)
    LearnedActionHead,
    evaluate_head,
    load_learned_action_head,
    train_learned_action_head,
)

__all__ = [
    "ActionHead",
    "ActionProposal",
    "LearnedActionHead",
    "evaluate_head",
    "load_learned_action_head",
    "train_learned_action_head",
]
