"""Negative controls for the property test suite (N1-N8).

Each control deliberately violates a contract.  The test suite is only credible
if these controls fail in the expected direction; a control that "passes" means
the metric is not sensitive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pss.experts import (
    CategoryCodebook,
    OracleExpert,
    RandomExpert,
    WrongDimExpert,
    manipulation_check,
    validate_expert,
)
from pss.observations import make_scene
from pss.property.compute import Budget, BudgetController
from pss.property.kb import VersionedKnowledgeBase
from pss.property.swap import ModuleRegistry
from pss.reasoning.planner import AStarPlanner
from pss.world_model.actions import initial_laundry_facts, laundry_library


@dataclass
class ControlResult:
    id: str
    name: str
    expected: str
    passed: bool
    details: dict

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "expected": self.expected,
            "passed": bool(self.passed),
            "details": self.details,
        }


class FlatKB:
    """Intentionally naive KB: no history, no justifications, no propagation."""

    def __init__(self):
        self.data: dict[tuple[str, str], object] = {}

    def set(self, entity, attribute, value):
        self.data[(entity, attribute)] = value

    def get(self, entity, attribute):
        return self.data.get((entity, attribute))

    def update(self, entity, attribute, value):
        self.set(entity, attribute, value)


class NaiveRegistry:
    """Intentionally naive registry: no contract validation."""

    def __init__(self):
        self.modules = {}

    def swap(self, expert):
        self.modules[expert.modality] = expert
        return True

    def get(self, modality):
        return self.modules.get(modality)


def _flat_kb_control() -> ControlResult:
    kb = FlatKB()
    kb.set("a", "v", "1")
    kb.set("b", "v", "2")
    kb.update("a", "v", "3")
    has_rollback = hasattr(kb, "rollback") and callable(getattr(kb, "rollback"))
    has_propagation = hasattr(kb, "tms") or hasattr(kb, "dependents")
    return ControlResult(
        "N1",
        "flat KB",
        "fail locality/rollback",
        passed=not has_rollback and not has_propagation,
        details={"rollback": has_rollback, "propagation": has_propagation},
    )


def _unbounded_planner_control() -> ControlResult:
    library = laundry_library()
    initial = initial_laundry_facts()
    goal = frozenset({("never", "happens", "yes")})
    planner = AStarPlanner(library, max_expansions=200)
    result = planner.plan(initial, goal)
    return ControlResult(
        "N2",
        "unbounded planner",
        "fail budget adherence",
        passed=int(result.nodes) > 20,
        details={"nodes": int(result.nodes), "success": bool(result.success)},
    )


def _naive_swap_control(seed: int = 0) -> ControlResult:
    scene = make_scene(8, dim=128, seed=seed)
    registry = NaiveRegistry()
    bad = WrongDimExpert(scene.codebook)
    accepted = registry.swap(bad)
    active = registry.get("vision")
    return ControlResult(
        "N3",
        "naive swap",
        "accept invalid module (should fail)",
        passed=bool(accepted and active is bad and int(active.dim) != 128),
        details={"accepted": bool(accepted), "active_dim": int(getattr(active, "dim", -1))},
    )


def _random_capability_control(seed: int = 0) -> ControlResult:
    scene = make_scene(200, dim=128, seed=seed)
    obs = scene.observations()
    expert = RandomExpert(scene.codebook, seed=seed)
    result = manipulation_check(expert, obs, scene.codebook)
    chance = 1.0 / len(scene.codebook.categories)
    return ControlResult(
        "N4",
        "random expert capability",
        "fail capability (near chance)",
        passed=float(result["accuracy"]) <= 2.0 * chance,
        details={"accuracy": result["accuracy"], "chance": chance},
    )


def _random_property_control(seed: int = 0) -> ControlResult:
    scene = make_scene(20, dim=128, seed=seed)
    obs = scene.observations()
    expert = RandomExpert(scene.codebook, seed=seed)
    valid = True
    try:
        validate_expert(expert, 128, probe=obs[:4])
    except Exception:
        valid = False
    registry = ModuleRegistry(public_dim=128, probe=obs[:4])
    registry.register(OracleExpert(scene.codebook, seed=seed), probe=obs[:4])
    swap = registry.swap(expert, probe=obs[:4])
    return ControlResult(
        "N5",
        "random expert property",
        "pass property contract",
        passed=bool(valid and swap.success),
        details={"valid": valid, "swap_success": bool(swap.success)},
    )


def _noop_update_control() -> ControlResult:
    kb = VersionedKnowledgeBase()
    kb.assert_fact("cup", "material", "glass", source="s1")
    before = kb.state_hash()
    kb.assert_fact("cup", "material", "glass", source="s1")
    after = kb.state_hash()
    return ControlResult(
        "N6",
        "no-op update",
        "identity",
        passed=before == after,
        details={"before": before, "after": after},
    )


def _shuffled_embedding_control(seed: int = 0) -> ControlResult:
    scene = make_scene(200, dim=128, seed=seed)
    obs = scene.observations()
    rng = np.random.default_rng(seed)
    shuffled = CategoryCodebook.from_vectors(
        scene.codebook.categories, scene.codebook.vectors[rng.permutation(len(scene.codebook.categories))]
    )
    expert = OracleExpert(shuffled, seed=seed)
    result = manipulation_check(expert, obs, scene.codebook)
    chance = 1.0 / len(scene.codebook.categories)
    return ControlResult(
        "N7",
        "shuffled embeddings",
        "capability fails, contract holds",
        passed=float(result["accuracy"]) <= 2.0 * chance,
        details={"accuracy": result["accuracy"], "chance": chance},
    )


def _overbudget_control() -> ControlResult:
    controller = BudgetController(Budget(max_entities=10, max_relations=10))
    report = controller.check(entities=100, relations=100)
    return ControlResult(
        "N8",
        "over-budget input",
        "budget metric catches violation",
        passed=not report.within_budget and set(report.violations) >= {"entities", "relations"},
        details={"violations": list(report.violations)},
    )


def run_negative_controls(seed: int = 0) -> dict:
    controls = [
        _flat_kb_control(),
        _unbounded_planner_control(),
        _naive_swap_control(seed=seed),
        _random_capability_control(seed=seed),
        _random_property_control(seed=seed),
        _noop_update_control(),
        _shuffled_embedding_control(seed=seed),
        _overbudget_control(),
    ]
    return {
        "seed": int(seed),
        "controls": [c.to_dict() for c in controls],
        "all_passed": all(c.passed for c in controls),
    }
