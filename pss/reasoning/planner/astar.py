"""Placement-space A* for the P5b composition task (B3).

``pss.reasoning.planner.AStarPlanner`` searches over *facts*: the state is a
frozenset of relation facts and operators add/remove them.  The P5b task lives
in a different space -- three objects occupy distinct regions, and one action
moves one object to one free region -- so this module adds A* for that space.

The heuristic is the number of unsatisfied goal constraints.  One placement can
satisfy several constraints, so the heuristic can overestimate and the search
is not guaranteed optimal; that is fine here, because the reference planner is
BFS and A* is only adopted if it matches or beats it on success, validity and
steps.  Weighted A* (``weight > 1``) trades steps for expansions.

The module is task-agnostic: ``plan_placements`` takes the region names and a
predicate function, exactly like ``pss.eval.relational_bench.goal_ok``.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "PlacementPlan",
    "apply_actions",
    "plan_placements",
    "search",
    "unmet_goal_count",
]

Action = Hashable
State = Hashable


@dataclass(frozen=True)
class PlacementPlan:
    success: bool
    actions: tuple = ()
    nodes: int = 0
    cost: float = 0.0
    reason: str = ""


def search(
    start: State,
    is_goal: Callable[[State], bool],
    successors: Callable[[State], Iterable[tuple[Action, State, float]]],
    heuristic: Callable[[State], float] | None = None,
    max_expansions: int = 20000,
    weight: float = 1.0,
) -> PlacementPlan:
    """Generic best-first search; A* when ``heuristic`` is given.

    ``successors(state)`` yields ``(action, next_state, step_cost)``.  Ties are
    broken by insertion order, so states never need to be comparable.
    """

    def h(state: State) -> float:
        if heuristic is None:
            return 0.0
        value = float(heuristic(state))
        if not math.isfinite(value):
            return 0.0
        return max(0.0, value) * float(weight)

    counter = 0
    frontier = [(h(start), 0.0, counter, start)]
    best = {start: 0.0}
    parent: dict[State, tuple[State, Action] | None] = {start: None}
    nodes = 0

    while frontier:
        _f, g, _c, state = heapq.heappop(frontier)
        if g > best.get(state, float("inf")):
            continue
        nodes += 1
        if nodes > max_expansions:
            return PlacementPlan(False, (), nodes, float("inf"), "max expansions")
        if is_goal(state):
            actions = []
            cur = state
            while parent[cur] is not None:
                prev, action = parent[cur]
                actions.append(action)
                cur = prev
            actions.reverse()
            return PlacementPlan(True, tuple(actions), nodes, g, "")
        for action, next_state, step_cost in successors(state):
            ng = g + float(step_cost)
            if ng < best.get(next_state, float("inf")):
                best[next_state] = ng
                parent[next_state] = (state, action)
                counter += 1
                heapq.heappush(
                    frontier, (ng + h(next_state), ng, counter, next_state)
                )

    return PlacementPlan(False, (), nodes, float("inf"), "no path")


def unmet_goal_count(
    state: Sequence[int],
    rel1: str,
    rel2: str,
    regions: Sequence[str],
    rel_fn: Callable[[str, str, str], bool],
    goal_region: str = "G",
) -> int:
    """Number of unsatisfied constraints: object 0 at goal, then two relations."""

    p0, p1, p2 = (regions[int(i)] for i in state[:3])
    return (
        int(p0 != goal_region)
        + int(not rel_fn(p1, p0, rel1))
        + int(not rel_fn(p2, p1, rel2))
    )


def apply_actions(start: Sequence[int], actions: Iterable[tuple[int, int]]) -> tuple[int, ...]:
    """Apply ``(object, region)`` actions left to right."""

    state = [int(x) for x in start]
    for obj, region in actions:
        state[int(obj)] = int(region)
    return tuple(state)


def plan_placements(
    start: Sequence[int],
    rel1: str,
    rel2: str,
    regions: Sequence[str],
    rel_fn: Callable[[str, str, str], bool],
    goal_region: str = "G",
    max_expansions: int = 20000,
    weight: float = 1.0,
    heuristic: Callable[[Sequence[int]], float] | None = None,
) -> PlacementPlan:
    """A* over placements for the three-object P5b goal.

    A state is ``(region_index_object0, region_index_object1, region_index_object2)``.
    An action moves one object to a region that is currently free.  This is the
    same successor relation as ``pss.eval.relational_bench.plan`` (BFS).
    """

    regions = tuple(regions)
    start = tuple(int(x) for x in start)
    if len(start) < 3:
        raise ValueError(f"expected at least three objects, got {len(start)}")
    if goal_region not in regions:
        raise ValueError(f"goal region {goal_region!r} is not in {regions}")
    goal_index = regions.index(goal_region)

    def is_goal(state: Sequence[int]) -> bool:
        p0, p1, p2 = (regions[int(i)] for i in state[:3])
        return (
            int(state[0]) == goal_index
            and rel_fn(p1, p0, rel1)
            and rel_fn(p2, p1, rel2)
        )

    def successors(state: Sequence[int]):
        occupied = set(state)
        for obj in range(len(state)):
            for region in range(len(regions)):
                if region == state[obj] or region in occupied:
                    continue
                nxt = list(state)
                nxt[obj] = region
                yield (obj, region), tuple(nxt), 1.0

    h = heuristic
    if h is None:
        def h(state: Sequence[int]) -> float:
            return float(
                unmet_goal_count(state, rel1, rel2, regions, rel_fn, goal_region)
            )

    return search(
        start,
        is_goal,
        successors,
        heuristic=h,
        max_expansions=max_expansions,
        weight=weight,
    )
