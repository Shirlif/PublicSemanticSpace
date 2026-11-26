"""Tests for pss.reasoning.constraints (hand-built, no scipy/checkpoints)."""

from __future__ import annotations

import numpy as np
import pytest

from pss.reasoning.constraints import (
    AttributeConstraint,
    AttributeInterval,
    PrototypeView,
    TypeConstraint,
    TypeNode,
    compatibility,
)


def _unit(*components: float) -> np.ndarray:
    vec = np.asarray(components, dtype=np.float64)
    return vec / np.linalg.norm(vec)


ANIMAL = _unit(1.0, 0.0, 0.0, 0.0)
VEHICLE = _unit(0.0, 1.0, 0.0, 0.0)
FOOD = _unit(0.0, 0.0, 1.0, 0.0)
TEXT_FOOD = _unit(0.0, 0.0, 0.0, 1.0)


class _Bank:
    """Minimal PrototypeBank stand-in: same .categories/.protos contract."""

    def __init__(self) -> None:
        self.categories = ["animal", "vehicle", "food"]
        self.protos = {
            "vision": np.stack([ANIMAL, VEHICLE, FOOD]),
            "text": np.stack([ANIMAL, VEHICLE, TEXT_FOOD]),
        }


@pytest.fixture
def view() -> PrototypeView:
    return PrototypeView(_Bank())


def _constraint(node: TypeNode, view: PrototypeView, name: str = "c") -> TypeConstraint:
    return TypeConstraint(name=name, node=node, view=view)


def test_prototype_view_from_bank_like_normalizes(view: PrototypeView) -> None:
    assert view.categories == ["animal", "vehicle", "food"]
    assert view.modalities() == {"vision", "text"}
    animal = view.protos_for("animal")
    assert np.isclose(np.linalg.norm(animal["vision"]), 1.0)
    assert np.isclose(np.linalg.norm(animal["text"]), 1.0)
    assert view.protos_for("missing") == {}


def test_prototype_view_from_mapping_normalizes() -> None:
    view = PrototypeView({"animal": {"vision": np.array([2.0, 0.0, 0.0, 0.0])}})
    assert view.categories == ["animal"]
    assert view.modalities() == {"vision"}
    vector = view.protos_for("animal")["vision"]
    assert np.allclose(vector, ANIMAL)


def test_membership_is_one_for_prototype_inside_node(view: PrototypeView) -> None:
    node = TypeNode.leaf("leaf_animal", "animal")
    constraint = _constraint(node, view)
    score = constraint.membership({"vision": ANIMAL})
    assert score == pytest.approx(1.0, abs=1e-9)


def test_membership_lower_for_unrelated_and_minus_one_without_shared(view: PrototypeView) -> None:
    node = TypeNode.leaf("leaf_animal", "animal")
    constraint = _constraint(node, view)
    unrelated = constraint.membership({"vision": VEHICLE})
    assert unrelated < 0.5
    assert constraint.membership({"audio": ANIMAL}) == -1.0
    empty = _constraint(TypeNode("empty", frozenset()), view)
    assert empty.membership({"vision": ANIMAL}) == -1.0


def test_membership_uses_only_shared_modalities(view: PrototypeView) -> None:
    node = TypeNode.leaf("leaf_animal", "animal")
    constraint = _constraint(node, view)
    score = constraint.membership({"vision": ANIMAL, "audio": np.ones(4)})
    assert np.isfinite(score)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_apply_and_project_alpha_one_return_unit_representatives(view: PrototypeView) -> None:
    node = TypeNode("animals", frozenset({"animal", "vehicle"}))
    constraint = _constraint(node, view)
    query = {"vision": _unit(0.9, 0.1, 0.0, 0.0), "text": _unit(0.1, 0.9, 0.0, 0.0)}
    projected = constraint.project(query, alpha=1.0)
    assert set(projected) == {"vision", "text"}
    for vector in projected.values():
        assert np.isclose(np.linalg.norm(vector), 1.0)
    # Each modality resolves to its own nearest category.
    assert np.allclose(projected["vision"], ANIMAL)
    assert np.allclose(projected["text"], VEHICLE)
    representatives = constraint.apply(query)
    assert np.allclose(representatives["vision"], ANIMAL)
    assert np.allclose(representatives["text"], VEHICLE)


def test_project_preserves_missing_modalities(view: PrototypeView) -> None:
    node = TypeNode.leaf("leaf_animal", "animal")
    constraint = _constraint(node, view)
    audio = np.array([1.0, 2.0, 3.0, 4.0])
    projected = constraint.project({"vision": VEHICLE, "audio": audio}, alpha=1.0)
    assert np.array_equal(projected["audio"], audio)
    assert np.allclose(projected["vision"], ANIMAL)


def test_residual_shared_and_disjoint(view: PrototypeView) -> None:
    constraint = _constraint(TypeNode.leaf("leaf", "animal"), view)
    assert constraint.residual({"vision": ANIMAL}, {"vision": ANIMAL}) == pytest.approx(0.0)
    assert constraint.residual({"vision": ANIMAL}, {"text": ANIMAL}) == 1.0


def test_compose_disjoint_returns_none_and_intersection_otherwise(view: PrototypeView) -> None:
    first = _constraint(TypeNode("a", frozenset({"animal", "vehicle"})), view, name="animalish")
    disjoint = _constraint(TypeNode("b", frozenset({"food"})), view, name="food")
    assert first.compatible(disjoint) is False
    assert first.compose(disjoint) is None

    overlapping = _constraint(TypeNode("c", frozenset({"animal", "food"})), view, name="edible")
    merged = first.compose(overlapping)
    assert merged is not None
    assert merged.node.categories == frozenset({"animal"})
    assert merged.name == "animalish&edible"
    assert merged.compatible(overlapping)


def test_type_node_leaf_and_contains() -> None:
    node = TypeNode.leaf("leaf_dog", "dog")
    assert "dog" in node
    assert "cat" not in node
    other = TypeNode("other", frozenset({"dog", "cat"}))
    assert node.intersects(other)
    assert node.intersection(other).categories == frozenset({"dog"})
    assert node.intersection(TypeNode("cat", frozenset({"cat"}))) is None


def test_attribute_interval_overlap_intersection_and_residual() -> None:
    interval = AttributeInterval(1.0, 3.0)
    assert interval.contains(2.0)
    assert not interval.contains(4.0)
    assert interval.overlaps(AttributeInterval(3.0, 5.0))
    assert not interval.overlaps(AttributeInterval(4.0, 5.0))
    assert interval.intersection(AttributeInterval(2.0, 5.0)) == AttributeInterval(2.0, 3.0)
    assert interval.intersection(AttributeInterval(4.0, 5.0)) is None
    assert interval.residual(1.0) == 0.0
    assert interval.residual(5.0) == 2.0
    with pytest.raises(ValueError):
        AttributeInterval(5.0, 1.0)


def test_attribute_constraint_clipping_residual_and_compose() -> None:
    constraint = AttributeConstraint("size", AttributeInterval(1.0, 2.0))
    assert constraint.apply(5.0) == 2.0
    assert constraint.apply(-3.0) == 1.0
    assert constraint.residual(5.0, 1.5) == pytest.approx(0.5)

    overlapping = AttributeConstraint("range", AttributeInterval(1.5, 4.0))
    assert constraint.compatible(overlapping)
    merged = constraint.compose(overlapping)
    assert merged is not None
    assert merged.interval == AttributeInterval(1.5, 2.0)
    assert merged.name == "size&range"

    disjoint = AttributeConstraint("wide", AttributeInterval(5.0, 6.0))
    assert not constraint.compatible(disjoint)
    assert constraint.compose(disjoint) is None


def test_compatibility_reports_clearance() -> None:
    incompatible = compatibility(AttributeInterval(2.0, 3.0), AttributeInterval(5.0, 6.0))
    assert incompatible["compatible"] is False
    assert incompatible["clearance"] == pytest.approx(2.0)
    assert incompatible["low"] > incompatible["high"]

    fitting = compatibility(AttributeInterval(2.0, 5.0), AttributeInterval(4.0, 6.0))
    assert fitting["compatible"] is True
    assert fitting["clearance"] == 0.0
    assert fitting["low"] == 4.0
    assert fitting["high"] == 5.0

    # Touching endpoints still count as compatible (zero clearance).
    touching = compatibility(AttributeInterval(2.0, 3.0), AttributeInterval(3.0, 4.0))
    assert touching["compatible"] is True
    assert touching["clearance"] == 0.0


def test_two_modality_query_with_one_modality_in_node_is_finite() -> None:
    # The node supports only vision, while the query carries vision and text.
    view = PrototypeView({"animal": {"vision": ANIMAL}})
    node = TypeNode("animal_only_vision", frozenset({"animal"}))
    constraint = _constraint(node, view)
    score = constraint.membership({"vision": ANIMAL, "text": TEXT_FOOD})
    assert np.isfinite(score)
    assert score == pytest.approx(1.0, abs=1e-9)
