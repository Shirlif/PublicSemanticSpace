from pss.eval.collision_planner import (
    edge_clear,
    goal_ok,
    plan_collision_aware,
)


def test_edge_clear_detects_blocker():
    # moving object 0 from A to B; object 1 sits at G on the segment
    state = (0, 4, 3)
    assert not edge_clear(state, 0, 1, clearance=0.12)


def test_collision_aware_plan_reaches_goal():
    start = (0, 2, 3)
    solution = plan_collision_aware(start, "left", "above", clearance=0.12)
    assert solution
    state = list(start)
    for obj, region in solution:
        state[obj] = region
        assert edge_clear(tuple(state), obj, region, clearance=0.0)
    assert goal_ok(tuple(state), "left", "above")
