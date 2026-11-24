"""Ground scene-graph state into planner facts (Phase 7 closed loop).

The planner works on discrete relation facts. This module turns live entity
positions into tabletop facts:

- ``at(target, region)``: nearest waypoint within a tolerance;
- clear edges: a waypoint-to-waypoint segment is usable when no obstacle is
  within ``clearance`` of it.

Actions are ``move_<src>_<dst>`` operators. No text is involved.
"""

from __future__ import annotations

import numpy as np

from pss.world_model.actions import Operator, OperatorLibrary

REGIONS = {
    "A": np.array([-0.22, 0.00]),
    "B": np.array([0.00, 0.16]),
    "C": np.array([0.00, -0.16]),
    "D": np.array([0.10, -0.10]),
    "G": np.array([0.22, 0.04]),
}


def point_segment_distance(p, a, b) -> float:
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def segment_clear(a, b, obstacles, clearance: float = 0.09) -> bool:
    return all(
        point_segment_distance(np.asarray(obs, dtype=float)[:2], a, b) > clearance
        for obs in obstacles
    )


def nearest_region(
    pos_xy,
    regions: dict[str, np.ndarray] | None = None,
    max_dist: float = 0.18,
) -> tuple[str | None, float]:
    regions = REGIONS if regions is None else regions
    pos = np.asarray(pos_xy, dtype=float)[:2]
    best, best_d = None, float("inf")
    for name, center in regions.items():
        d = float(np.linalg.norm(pos - center))
        if d < best_d:
            best, best_d = name, d
    if best_d > max_dist:
        return None, best_d
    return best, best_d


def tabletop_facts(
    target_id: str,
    pos_xy,
    regions: dict[str, np.ndarray] | None = None,
    max_dist: float = 0.20,
    goal_region: str = "G",
    goal_max_dist: float = 0.06,
) -> frozenset:
    """Facts from live position. The goal needs a tighter tolerance.

    A loose goal tolerance lets the planner declare "already satisfied" while
    the block is still 10+ cm away. Non-goal waypoints keep the loose bound so
    the target is not left without a region between waypoints.
    """
    regions = REGIONS if regions is None else regions
    pos = np.asarray(pos_xy, dtype=float)[:2]
    ranked = sorted(
        (float(np.linalg.norm(pos - center)), name)
        for name, center in regions.items()
    )
    for distance, name in ranked:
        if name == goal_region and distance > goal_max_dist:
            continue
        if distance <= max_dist:
            return frozenset({(target_id, "at", name)})
    return frozenset()


def tabletop_goal(target_id: str, goal_region: str = "G") -> frozenset:
    return frozenset({(target_id, "at", goal_region)})


def build_tabletop_library(
    target_id: str,
    obstacle_positions,
    regions: dict[str, np.ndarray] | None = None,
    clearance: float = 0.09,
) -> OperatorLibrary:
    regions = REGIONS if regions is None else regions
    operators = []
    names = list(regions)
    for src in names:
        for dst in names:
            if src == dst:
                continue
            if not segment_clear(regions[src], regions[dst], obstacle_positions, clearance):
                continue
            cost = float(np.linalg.norm(regions[src] - regions[dst]))
            operators.append(
                Operator(
                    f"move_{src}_{dst}",
                    frozenset({(target_id, "at", src)}),
                    frozenset({(target_id, "at", dst)}),
                    frozenset({(target_id, "at", src)}),
                    cost=cost,
                )
            )
    return OperatorLibrary(operators)


def parse_move_action(name: str) -> tuple[str, str]:
    parts = name.split("_")
    if len(parts) != 3 or parts[0] != "move":
        raise ValueError(f"not a move action: {name}")
    return parts[1], parts[2]
