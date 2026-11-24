"""A* planner over relation facts (Phase 7).

State is a frozenset of facts; operators come from an OperatorLibrary. The
heuristic is the number of missing goal facts (not admissible when one action
satisfies several goals, but adequate for this v1).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from pss.world_model.actions import OperatorLibrary


@dataclass
class PlanResult:
    success: bool
    actions: list[str]
    states: list[frozenset]
    nodes: int
    cost: float
    reason: str = ""


class AStarPlanner:
    def __init__(
        self,
        library: OperatorLibrary,
        max_expansions: int = 20000,
        heuristic=None,
        heuristic_weight: float = 1.0,
    ):
        self.library = library
        self.max_expansions = int(max_expansions)
        self.heuristic = heuristic
        self.heuristic_weight = float(heuristic_weight)

    def _h(self, state, goal) -> float:
        if self.heuristic is None:
            base = float(len(goal - state))
        else:
            try:
                base = float(self.heuristic(state, goal))
            except Exception:
                base = float(len(goal - state))
        if not np.isfinite(base):
            base = float(len(goal - state))
        return max(0.0, base) * self.heuristic_weight

    def plan(
        self,
        initial,
        goal,
        confidence: dict[tuple, float] | None = None,
        threshold: float = 0.5,
        confidence_weight: float = 0.5,
    ) -> PlanResult:
        start = frozenset(initial)
        goal = frozenset(goal)
        if goal <= start:
            return PlanResult(True, [], [start], 0, 0.0, "already satisfied")

        counter = 0
        frontier = [(self._h(start, goal), 0.0, counter, start)]
        best = {start: 0.0}
        parent: dict[frozenset, tuple[frozenset, str] | None] = {start: None}
        nodes = 0
        low_confidence_blocked = False

        while frontier:
            _f, g, _c, state = heapq.heappop(frontier)
            if g > best.get(state, float("inf")):
                continue
            nodes += 1
            if nodes > self.max_expansions:
                return PlanResult(False, [], [], nodes, float("inf"), "max expansions")

            if goal <= state:
                actions: list[str] = []
                cur = state
                while parent[cur] is not None:
                    prev, op_name = parent[cur]
                    actions.append(op_name)
                    cur = prev
                actions.reverse()
                states = [start]
                cur = start
                for name in actions:
                    cur = self.library.apply(self.library.get(name), cur)
                    states.append(cur)
                return PlanResult(True, actions, states, nodes, g, "")

            applicable = self.library.applicable(state, confidence, threshold)
            if confidence:
                hard = [
                    op
                    for op in self.library.operators.values()
                    if op.preconditions <= state
                ]
                if len(applicable) < len(hard):
                    low_confidence_blocked = True
            for op, score in applicable:
                new_state = self.library.apply(op, state)
                penalty = 1.0 + float(confidence_weight) * (1.0 - float(score))
                ng = g + op.cost * penalty
                if ng < best.get(new_state, float("inf")):
                    best[new_state] = ng
                    parent[new_state] = (state, op.name)
                    counter += 1
                    heapq.heappush(
                        frontier, (ng + self._h(new_state, goal), ng, counter, new_state)
                    )

        reason = "low confidence" if low_confidence_blocked else "no path"
        return PlanResult(False, [], [], nodes, float("inf"), reason)

    def replan(
        self,
        current,
        goal,
        confidence=None,
        threshold: float = 0.5,
        confidence_weight: float = 0.5,
    ) -> PlanResult:
        return self.plan(current, goal, confidence, threshold, confidence_weight)
