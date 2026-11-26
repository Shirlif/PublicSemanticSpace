"""Tests for goal decomposition and the composition-aware planner."""

from __future__ import annotations

from pss.reasoning.composition import Signature
from pss.reasoning.planner import CompositionPlanner, decompose, simulate
from pss.world_model.actions import (
    Operator,
    initial_laundry_facts,
    laundry_goal,
    laundry_library,
)


def test_decompose_reaches_laundry_goal():
    library = laundry_library()
    initial = initial_laundry_facts()
    goal = laundry_goal()

    result = decompose(goal, library.operators.values(), initial)
    assert result.success
    assert result.actions

    states, ok = simulate(initial, result.actions, library)
    assert ok
    assert goal <= states[-1]


def test_decompose_returns_unresolved_for_impossible_goal():
    library = laundry_library()
    result = decompose(
        {("unicorn", "exists")},
        library.operators.values(),
        initial_laundry_facts(),
    )
    assert not result.success
    assert result.unresolved == frozenset({("unicorn", "exists")})


def test_decompose_terminates_on_self_supporting_operator():
    loop = Operator(
        "loop",
        frozenset({("x", "on")}),
        frozenset({("x", "on")}),
        frozenset(),
    )
    result = decompose({("x", "on")}, [loop], frozenset())
    assert not result.success


def test_composition_planner_returns_a_valid_plan():
    library = laundry_library()
    initial = initial_laundry_facts()
    goal = laundry_goal()
    planner = CompositionPlanner(library)

    plan = planner.plan(initial, goal)
    assert plan.success

    states, ok = simulate(initial, plan.actions, library)
    assert ok
    assert goal <= states[-1]


def test_composition_planner_prefers_the_cheaper_route():
    library = laundry_library()
    initial = initial_laundry_facts()
    goal = laundry_goal()
    planner = CompositionPlanner(library)

    plan = planner.plan(initial, goal)
    astar = planner.astar.plan(initial, goal)
    assert plan.cost <= astar.cost + 1e-9


def test_plan_typed_returns_plan_result():
    planner = CompositionPlanner(laundry_library())

    class Step:
        name = "step"
        signature = Signature("start", "done")

        def applicable(self, payload):
            return True

        def apply(self, payload):
            out = dict(payload)
            out["type"] = "done"
            return out

    result = planner.plan_typed(
        {"type": "start"},
        [Step()],
        lambda payload: payload.get("type") == "done",
        max_steps=2,
    )
    assert result.success
    assert result.actions == ["step"]
