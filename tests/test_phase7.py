from pathlib import Path

import numpy as np
import pytest

from pss.reasoning.conflict import detect_conflict, should_replan
from pss.reasoning.planner import AStarPlanner
from pss.reasoning.relations import (
    RelationRecord,
    RelationStore,
    geometric_relations,
)
from pss.world_model.actions import (
    hand_written_plan,
    initial_laundry_facts,
    laundry_goal,
    laundry_library,
    simulate_plan,
    spill_anomaly,
)

DATA = Path("data/v1")


def test_geometric_relations_contact_near():
    instances = {
        "a": {"pos": np.array([0.0, 0.0, 0.05]), "radius": 0.05},
        "b": {"pos": np.array([0.11, 0.0, 0.05]), "radius": 0.05},
        "c": {"pos": np.array([0.5, 0.0, 0.05]), "radius": 0.05},
    }
    facts = {r.as_tuple() for r in geometric_relations(instances, near_dist=0.12)}
    assert ("a", "contact", "b") in facts
    assert ("a", "near", "b") in facts
    assert ("a", "near", "c") not in facts


def test_geometric_relations_in_bin():
    instances = {"a": {"pos": np.array([0.38, 0.20, 0.05]), "radius": 0.05}}
    facts = {r.as_tuple() for r in geometric_relations(instances)}
    assert ("a", "in", "bin0") in facts


def test_relation_store_encode_and_vector_query():
    store = RelationStore(dim=128, seed=0)
    rng = np.random.default_rng(0)

    def unit():
        v = rng.normal(size=128)
        return v / np.linalg.norm(v)

    near = RelationRecord("a", "near", "b")
    contact = RelationRecord("a", "contact", "b")
    encoded_near = store.encode(near, unit(), unit())
    store.encode(contact, unit(), unit())
    top = store.vector_query(encoded_near.fact, top_k=1)[0]
    assert top[1] == ("a", "near", "b")


def test_operator_preconditions_and_effects():
    library = laundry_library()
    facts = initial_laundry_facts()
    op = library.get("pick_clothes")
    assert op.applicable(facts)
    new_facts = op.apply(facts)
    assert ("clothes", "held") in new_facts
    assert ("clothes", "in", "basket") not in new_facts
    assert not library.get("pour_detergent").applicable(new_facts)


def test_planner_finds_goal_and_reports_blocked():
    library = laundry_library()
    planner = AStarPlanner(library)
    result = planner.plan(initial_laundry_facts(), laundry_goal())
    assert result.success
    facts, _log, ok = simulate_plan(result.actions, initial_laundry_facts(), library)
    assert ok and laundry_goal() <= facts

    blocked_initial = frozenset(
        initial_laundry_facts() - {("detergent", "on", "counter")}
    )
    blocked = planner.plan(blocked_initial, laundry_goal())
    assert not blocked.success


def test_planner_replans_after_spill():
    library = laundry_library()
    planner = AStarPlanner(library)
    facts, _log, _ok = simulate_plan(
        hand_written_plan(),
        initial_laundry_facts(),
        library,
        anomaly_step=4,
        anomaly_fn=spill_anomaly,
    )
    assert ("detergent", "on", "floor") in facts
    result = planner.plan(facts, laundry_goal())
    assert result.success
    final_facts, _log2, ok2 = simulate_plan(result.actions, facts, library)
    assert ok2 and laundry_goal() <= final_facts


def test_planner_abstains_on_low_confidence_precondition():
    library = laundry_library()
    planner = AStarPlanner(library)
    initial = initial_laundry_facts()
    goal = laundry_goal()

    high = {fact: 0.95 for fact in initial}
    assert planner.plan(initial, goal, confidence=high, threshold=0.5).success

    low = dict(high)
    low[("detergent", "on", "counter")] = 0.2
    result = planner.plan(initial, goal, confidence=low, threshold=0.5)
    assert not result.success
    assert result.reason == "low confidence"


def test_confidence_weight_prefers_high_confidence_actions():
    from pss.world_model.actions import Operator, OperatorLibrary

    library = OperatorLibrary(
        [
            Operator(
                "low",
                frozenset({("a", "start")}),
                frozenset({("a", "goal")}),
                frozenset({("a", "start")}),
                cost=1.0,
            ),
            Operator(
                "high",
                frozenset({("b", "start")}),
                frozenset({("a", "goal")}),
                frozenset({("b", "start")}),
                cost=1.0,
            ),
        ]
    )
    planner = AStarPlanner(library)
    result = planner.plan(
        frozenset({("a", "start"), ("b", "start")}),
        frozenset({("a", "goal")}),
        confidence={("a", "start"): 0.55, ("b", "start"): 0.99},
        threshold=0.5,
        confidence_weight=2.0,
    )
    assert result.success
    assert result.actions == ["high"]


def test_conflict_detection_and_replan_gate():
    predicted = {("detergent", "in", "washer")}
    observed = {("detergent", "on", "floor")}
    critical = {("detergent", "in", "washer")}
    result = detect_conflict(observed, predicted, critical)
    assert result.conflict
    assert result.missing == [("detergent", "in", "washer")]
    assert result.unexpected == [("detergent", "on", "floor")]
    assert should_replan(result, critical)


@pytest.mark.skipif(not DATA.exists(), reason="run dataset generation first")
def test_world_model_training_smoke(tmp_path):
    pytest.importorskip("torch")
    from pss.world_model.model import train_world_model

    metrics = train_world_model(DATA, tmp_path, epochs=2, hidden=16, seed=0)
    assert metrics["n_train"] > 0
    assert metrics["n_val"] > 0
    assert "learned" in metrics["one_step_pos_error"]
    assert (tmp_path / "world_model.pt").exists()
