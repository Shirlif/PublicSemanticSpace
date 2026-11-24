"""Tests for the learned A* heuristic (row 5)."""

from __future__ import annotations

import numpy as np

from pss.reasoning.planner import AStarPlanner
from pss.reasoning.planner.learned import (
    DEFAULT_ENTITIES,
    DEFAULT_REGIONS,
    LearnedHeuristic,
    encode_state,
    load_heuristic,
    make_dataset,
    save_heuristic,
    train_heuristic,
)
from pss.runtime import CognitiveState, Entity, Fact, Goal, PSSRuntime, RuntimePlanner


def test_encode_state_shape_and_finite():
    state = frozenset(
        (e, "location", r) for e, r in zip(DEFAULT_ENTITIES, ("A", "B", "C"))
    )
    goal = frozenset({("e0", "location", "D")})
    features = encode_state(state, goal)
    assert features.ndim == 1
    assert np.isfinite(features).all()
    assert len(features) == len(DEFAULT_ENTITIES) * (len(DEFAULT_REGIONS) * 2 + 1) + 1


def test_astar_accepts_heuristic_and_preserves_success():
    states, goals, costs, library = make_dataset(40, seed=0)
    hand = AStarPlanner(library)
    zero = AStarPlanner(library, heuristic=lambda s, g: 0.0)
    for state, goal in zip(states[:5], goals[:5]):
        assert hand.plan(state, goal).success
        assert zero.plan(state, goal).success


def test_learned_heuristic_train_save_load(tmp_path):
    states, goals, costs, library = make_dataset(80, seed=0)
    model = train_heuristic(states, goals, costs, epochs=5, seed=0)
    value = model.predict(states[0], goals[0])
    assert np.isfinite(value)
    assert value >= 0.0

    path = tmp_path / "heuristic.pt"
    save_heuristic(path, model)
    loaded = load_heuristic(path)
    assert loaded.feature_dim == model.feature_dim


def test_runtime_planner_uses_learned_heuristic():
    states, goals, costs, library = make_dataset(80, seed=0)
    model = train_heuristic(states, goals, costs, epochs=5, seed=0)
    planner = RuntimePlanner(library, heuristic=model.predict)
    assert planner.heuristic_name == "learned"

    state = CognitiveState()
    for entity, _attribute, region in states[0]:
        state.add_entity(
            Entity(entity, entity, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        state.set_fact(Fact(entity, "location", region))
    runtime = PSSRuntime(planner=planner, state=state)
    goal = Goal("g", frozenset(goals[0]), "test")
    runtime.set_goal(goal)
    plan = runtime.plan("g")
    assert plan.confidence > 0
    assert runtime.summary()["planning"]["heuristic"] == "learned"
