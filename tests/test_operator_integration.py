"""End-to-end checks across the four operator families (P0-P3).

These tests are the contract between the modules: if one family changes its
semantics, the corresponding assertion here should fail.
"""

from __future__ import annotations

import numpy as np

from pss.binding.frames import FrameOp, relation_in_frame
from pss.binding.operators import SignedPermutationOperator
from pss.reasoning.constraints import (
    AttributeInterval,
    PrototypeView,
    TypeConstraint,
    TypeNode,
)
from pss.reasoning.queries import TemporalHistory, all_of, count_of
from pss.reasoning.relations import default_closure, derive
from pss.world_model.conditional import laundry_conditional_library


def test_left_chain_derives_transitive_fact():
    records = [("a", "left_of", "b"), ("b", "left_of", "c"), ("c", "left_of", "d")]
    result = derive(records, max_rounds=2, min_conf=0.5)
    facts = result.facts()
    assert ("a", "left_of", "c") in facts
    assert ("b", "left_of", "d") in facts
    assert ("a", "left_of", "d") in facts
    # Every derived fact keeps the two premises it came from.
    three_hop = next(f for f in result.derived if f.as_tuple() == ("a", "left_of", "d"))
    assert len(three_hop.premises) == 2
    assert three_hop.as_record().source == "inference"


def test_opposite_directions_license_nothing():
    records = [("a", "left_of", "b"), ("b", "right_of", "c")]
    result = derive(records, max_rounds=3)
    assert result.facts() == set()
    assert result.rejected > 0


def test_self_loops_are_suppressed():
    records = [("a", "left_of", "b"), ("b", "left_of", "a")]
    result = derive(records, max_rounds=3)
    assert ("a", "left_of", "a") not in result.facts()
    assert ("b", "left_of", "b") not in result.facts()


def test_type_check_can_veto_a_derivation():
    records = [("a", "left_of", "b"), ("b", "left_of", "c")]
    result = derive(records, type_check=lambda s, p, o: s != "a")
    assert ("a", "left_of", "c") not in result.facts()


def test_rotation_flips_directional_predicate():
    left = np.array([-1.0, 0.0, 0.0])
    right = np.array([1.0, 0.0, 0.0])
    identity = FrameOp.identity()
    flipped = FrameOp.from_axes(
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.zeros(3),
    )
    assert relation_in_frame(left, right, identity) == ["left_of"]
    assert relation_in_frame(left, right, flipped) == ["right_of"]


def test_frame_composition_is_non_commutative():
    yaw90 = FrameOp.from_axes(
        np.array([0.0, 1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    yaw180 = FrameOp.from_axes(
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    a = yaw90.compose(yaw180).matrix()
    b = yaw180.compose(yaw90).matrix()
    assert not np.allclose(a, b)


def test_scene_positions_feed_the_inference_engine():
    frame = FrameOp.identity()
    positions = {"a": (-0.3, 0.0, 0.0), "b": (0.0, 0.0, 0.0), "c": (0.3, 0.0, 0.0)}
    records = []
    for left, right in (("a", "b"), ("b", "c")):
        if "left_of" in relation_in_frame(
            np.asarray(positions[left]), np.asarray(positions[right]), frame
        ):
            records.append((left, "left_of", right))
    result = derive(records)
    assert ("a", "left_of", "c") in result.facts()


def test_type_constraint_composition():
    bank = {
        "cup": {"vision": np.array([1.0, 0.0, 0.0, 0.0])},
        "mug": {"vision": np.array([0.9, 0.1, 0.0, 0.0])},
        "block": {"vision": np.array([0.0, 1.0, 0.0, 0.0])},
    }
    view = PrototypeView(bank)
    vessels = TypeConstraint("vessels", TypeNode("v", frozenset({"cup", "mug"})), view)
    cups = TypeConstraint("cups", TypeNode("c", frozenset({"cup"})), view)
    blocks = TypeConstraint("blocks", TypeNode("b", frozenset({"block"})), view)

    assert vessels.compose(blocks) is None
    composed = vessels.compose(cups)
    assert composed is not None
    assert composed.node.categories == frozenset({"cup"})

    query = {"vision": np.array([1.0, 0.0, 0.0, 0.0])}
    assert vessels.membership(query) > 0.99
    assert blocks.membership(query) < 0.1


def test_signed_permutation_composition_is_non_commutative():
    p = SignedPermutationOperator.random(16, seed=1)
    q = SignedPermutationOperator.random(16, seed=2)
    left = p.compose(q)
    right = q.compose(p)
    assert left is not None and right is not None
    assert not np.allclose(left.matrix(), right.matrix())
    assert isinstance(left, SignedPermutationOperator)


def test_conditional_spill_depends_on_water():
    library = laundry_conditional_library()
    dry = frozenset({("door", "closed"), ("machine", "stopped")})
    wet = frozenset({("door", "closed"), ("machine", "stopped"), ("water", "in", "washer")})
    assert ("floor", "wet") not in library.apply("open_washer", dry)
    assert ("floor", "wet") in library.apply("open_washer", wet)


def test_closure_table_is_self_consistent():
    assert default_closure().validate() == []


def test_temporal_query_returns_the_past_location():
    history = TemporalHistory()
    history.record(1.0, [("key", "in", "drawer")])
    history.record(2.0, [("key", "on", "table")], removed=[("key", "in", "drawer")])

    assert history.where_was("key", 1.0) == {"drawer"}
    assert history.where_was("key", 2.0) == {"table"}
    assert ("key", "in", "drawer") in history.removed_between(1.0, 2.0)


def test_quantifiers_over_derived_facts():
    records = [("cup1", "left_of", "cup2"), ("cup2", "left_of", "cup3")]
    result = derive(records)
    facts = result.facts() | set(records)
    assert all_of(facts, ["cup1", "cup2"], "left_of", "cup3", min_conf=0.0).value is True
    counted = count_of(facts, ["cup1", "cup2"], "left_of", "cup3", min_conf=0.0)
    assert counted.value == 2


def test_attribute_compatibility_for_insertion():
    screw = AttributeInterval(0.10, 0.12)
    tight_hole = AttributeInterval(0.08, 0.11)
    loose_hole = AttributeInterval(0.20, 0.25)
    assert screw.overlaps(tight_hole)
    assert not screw.overlaps(loose_hole)
    assert screw.intersection(loose_hole) is None
