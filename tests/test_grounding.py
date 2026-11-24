import numpy as np

from pss.reasoning.grounding import (
    build_tabletop_library,
    parse_move_action,
    segment_clear,
    tabletop_facts,
    tabletop_goal,
)
from pss.reasoning.planner import AStarPlanner


def test_segment_clear_rejects_obstacle_on_path():
    assert not segment_clear((0.0, 0.0), (0.0, 0.2), [np.array([0.0, 0.1, 0.0])], 0.05)
    assert segment_clear((0.0, 0.0), (0.0, 0.2), [np.array([0.2, 0.1, 0.0])], 0.05)


def test_tabletop_facts_require_tight_goal_tolerance():
    near_goal_but_not_enough = tabletop_facts("t", [0.12, 0.0])
    assert ("t", "at", "G") not in near_goal_but_not_enough
    at_goal = tabletop_facts("t", [0.21, 0.04])
    assert ("t", "at", "G") in at_goal


def test_initial_library_blocks_direct_and_b_c_routes():
    library = build_tabletop_library("t", [np.array([0.0, 0.0, 0.0])])
    names = set(library.operators)
    assert "move_A_B" in names
    assert "move_B_G" in names
    assert "move_A_G" not in names
    assert "move_B_C" not in names


def test_anomaly_reroutes_via_c():
    library = build_tabletop_library("t", [np.array([0.10, 0.105, 0.0])])
    plan = AStarPlanner(library).plan(
        frozenset({("t", "at", "B")}),
        tabletop_goal("t"),
    )
    assert plan.success
    assert plan.actions[0] == "move_B_C"
    assert "move_C_G" in plan.actions


def test_parse_move_action():
    assert parse_move_action("move_A_B") == ("A", "B")
