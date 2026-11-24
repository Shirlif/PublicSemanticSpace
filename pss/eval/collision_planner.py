"""Collision-aware relational planner.

The symbolic planner moves objects between regions without checking the
continuous path. This planner adds a segment-clearance test: an edge is usable
only when the straight path from the object's current region to the target
region stays clear of the other objects' current regions. BFS then finds a
detour through intermediate regions.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from pss.eval.relational_bench import goal_ok

try:
    from pss.eval.relational_perception_bench import REGION_COORDS
except Exception:  # pragma: no cover - fallback for import-only contexts
    REGION_COORDS = np.array(
        [
            [-0.24, 0.00],
            [0.24, 0.00],
            [0.00, 0.18],
            [0.00, -0.18],
            [0.00, 0.00],
        ],
        dtype=float,
    )

N_REGIONS = len(REGION_COORDS)
N_OBJECTS = 3


def point_segment_distance(p, a, b) -> float:
    p = np.asarray(p, dtype=float)[:2]
    a = np.asarray(a, dtype=float)[:2]
    b = np.asarray(b, dtype=float)[:2]
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def edge_clear(state, obj: int, dst: int, clearance: float = 0.12) -> bool:
    a = REGION_COORDS[state[obj]]
    b = REGION_COORDS[dst]
    for other, region in enumerate(state):
        if other == obj:
            continue
        if point_segment_distance(REGION_COORDS[region], a, b) < clearance:
            return False
    return True


def plan_collision_aware(
    start,
    rel1: str,
    rel2: str,
    clearance: float = 0.12,
    max_nodes: int = 10000,
):
    """BFS over region assignments with segment-clearance constraints."""
    start = tuple(int(x) for x in start)
    if goal_ok(start, rel1, rel2):
        return []
    frontier = deque([start])
    parent = {start: None}
    while frontier and len(parent) < max_nodes:
        state = frontier.popleft()
        for obj in range(N_OBJECTS):
            for region in range(N_REGIONS):
                if state[obj] == region or region in state:
                    continue
                if not edge_clear(state, obj, region, clearance):
                    continue
                nxt = list(state)
                nxt[obj] = region
                nxt = tuple(nxt)
                if nxt in parent:
                    continue
                parent[nxt] = (state, (obj, region))
                if goal_ok(nxt, rel1, rel2):
                    actions = []
                    cur = nxt
                    while parent[cur] is not None:
                        prev, act = parent[cur]
                        actions.append(act)
                        cur = prev
                    return list(reversed(actions))
                frontier.append(nxt)
    return None


def make_episode_collision_aware(rng, rel1, rel2, clearance=0.12):
    """Sample a start from the four outer regions and solve it."""
    for _ in range(200):
        state = tuple(rng.choice([0, 1, 2, 3], size=N_OBJECTS, replace=False))
        solution = plan_collision_aware(state, rel1, rel2, clearance)
        if solution is not None:
            return state, solution
    raise RuntimeError("could not sample a solvable collision-aware episode")


def rows_collision_aware(rel_pairs, n_per_pair, rng, clearance=0.12):
    rows = []
    for rel1, rel2 in rel_pairs:
        for _ in range(n_per_pair):
            state, solution = make_episode_collision_aware(
                rng, rel1, rel2, clearance
            )
            cur = state
            for obj, region in solution:
                rows.append(
                    {
                        "state": cur,
                        "rel1": ("left", "right", "above", "below").index(rel1),
                        "rel2": ("left", "right", "above", "below").index(rel2),
                        "action": obj * N_REGIONS + region,
                    }
                )
                nxt = list(cur)
                nxt[obj] = region
                cur = tuple(nxt)
    return rows
