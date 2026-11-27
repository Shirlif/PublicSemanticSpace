"""Tests for placement-space A*."""

from __future__ import annotations

import numpy as np

from pss.eval.relational_bench import (
    NAMES,
    TEST_PAIRS,
    _rel,
    goal_ok,
    make_episode,
)
from pss.reasoning.planner.astar import (
    apply_actions,
    plan_placements,
    search,
    unmet_goal_count,
)


def test_search_returns_empty_plan_when_start_is_goal():
    result = search(0, lambda s: s == 0, lambda s: ())
    assert result.success
    assert result.actions == ()
    assert result.nodes == 1
    assert result.cost == 0.0


def test_search_respects_max_expansions():
    result = search(
        0,
        lambda s: False,
        lambda s: [(a, s + 1, 1.0) for a in range(2)],
        max_expansions=1,
    )
    assert not result.success
    assert result.reason == "max expansions"


def test_search_finds_shortest_path_in_a_chain():
    result = search(
        0,
        lambda s: s == 3,
        lambda s: [(s, s + 1, 1.0)] if s < 3 else (),
        heuristic=lambda s: float(3 - s),
    )
    assert result.success
    assert result.actions == (0, 1, 2)
    assert result.cost == 3.0


def test_unmet_goal_count_orders_constraints():
    # Object 0 at G, object 1 left of object 0, object 2 below object 1.
    state = (NAMES.index("G"), NAMES.index("A"), NAMES.index("D"))
    assert unmet_goal_count(state, "left", "below", NAMES, _rel) == 0
    assert unmet_goal_count((NAMES.index("A"), 1, 2), "left", "below", NAMES, _rel) >= 1


def test_apply_actions_moves_objects():
    assert apply_actions((0, 1, 2), [(0, 3), (1, 4)]) == (3, 4, 2)


def test_placement_astar_matches_bfs_on_held_out_pairs():
    rng = np.random.default_rng(0)
    for rel1, rel2 in TEST_PAIRS:
        for _ in range(5):
            state, _bfs = make_episode(rng, rel1, rel2)
            result = plan_placements(state, rel1, rel2, NAMES, _rel)
            assert result.success, (rel1, rel2, state)
            final = apply_actions(state, result.actions)
            assert goal_ok(final, rel1, rel2)
            assert len(set(final)) == len(final)


def test_placement_astar_reports_no_path_for_impossible_goal():
    result = plan_placements(
        (0, 1, 2),
        "left",
        "below",
        NAMES,
        lambda a, b, relation: False,
    )
    assert not result.success
    assert result.reason == "no path"
    assert 0 < result.nodes <= 60


def test_bfs_reference_matches_relational_bench_plan():
    from pss.eval.scripts.run_search_comparison import bfs_plan

    rng = np.random.default_rng(1)
    for rel1, rel2 in TEST_PAIRS[:2]:
        for _ in range(3):
            state, solution = make_episode(rng, rel1, rel2)
            actions, nodes = bfs_plan(state, rel1, rel2)
            assert actions is not None
            assert len(actions) == len(solution)
            assert nodes > 0
