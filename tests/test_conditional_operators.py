"""Tests for the conditional action operators and the laundry library."""

from __future__ import annotations

import pytest

from pss.world_model.conditional import (
    ConditionalOperator,
    Effect,
    Resource,
    ResourcePool,
    laundry_conditional_library,
    laundry_resource_pool,
)


def _f(*facts: tuple[str, ...]) -> frozenset[tuple[str, ...]]:
    return frozenset(facts)


def _initial_facts() -> frozenset[tuple[str, ...]]:
    return _f(
        ("clothes", "in", "basket"),
        ("detergent", "on", "counter"),
        ("door", "closed"),
        ("machine", "stopped"),
        ("water", "drained"),
        ("drying_rack", "empty"),
    )


FULL_PLAN = [
    "pick_clothes",
    "open_washer",
    "put_in_clothes",
    "pick_detergent",
    "pour_detergent",
    "close_washer",
    "fill_water",
    "press_start",
    "wait_done",
    "open_washer",
    "take_out",
    "hang",
]


def test_conditional_effect_fires_only_when_condition_holds():
    lib = laundry_conditional_library()
    op = lib.get("open_washer")

    dry = _f(("door", "closed"), ("machine", "stopped"))
    dry_result = op.apply(dry)
    assert ("door", "open") in dry_result
    assert ("floor", "wet") not in dry_result

    wet = _f(("door", "closed"), ("machine", "stopped"), ("water", "in", "washer"))
    wet_result = op.apply(wet)
    assert ("door", "open") in wet_result
    assert ("floor", "wet") in wet_result
    # The conditional effect only adds the adverse fact.
    assert ("water", "in", "washer") in wet_result


def test_effects_apply_sequentially():
    op = ConditionalOperator(
        name="probe",
        effects=(
            Effect(add=_f(("first", "done"))),
            Effect(conditions=_f(("first", "done")), add=_f(("second", "done"))),
        ),
    )
    before = frozenset()
    after = op.apply(before)
    assert after == _f(("first", "done"), ("second", "done"))
    # The caller's state is untouched.
    assert before == frozenset()


def test_effect_apply_is_conditional_and_remove_before_add():
    effect = Effect(
        conditions=_f(("need",)),
        add=_f(("new",)),
        remove=_f(("need",)),
    )
    blocked = _f(("other",))
    assert effect.applicable(blocked) is False
    assert effect.apply(blocked) == blocked
    assert effect.apply(_f(("need",))) == _f(("new",))


def test_effect_inverted_swaps_add_and_remove():
    effect = Effect(conditions=_f(("cond",)), add=_f(("a",)), remove=_f(("b",)))
    inverted = effect.inverted()
    assert inverted.conditions == effect.conditions
    assert inverted.add == _f(("b",))
    assert inverted.remove == _f(("a",))


def test_resource_capacity_blocks_second_pick_and_release_restores():
    lib = laundry_conditional_library()
    pool = laundry_resource_pool()
    facts = lib.apply("pick_clothes", _initial_facts(), pool=pool)

    assert ("clothes", "held") in facts
    assert pool.used("hand") == 1
    assert pool.available("hand") == 0

    # Holding one item fills the only hand: the library hides the second pick
    # and apply() refuses it without changing the pool.
    ranked = [op.name for op, _ in lib.applicable(facts, pool=pool)]
    assert "pick_detergent" not in ranked
    with pytest.raises(ValueError):
        lib.apply("pick_detergent", facts, pool=pool)
    assert pool.used("hand") == 1
    assert ("detergent", "held") not in facts

    pool.release("hand")
    assert pool.available("hand") == 1
    facts = lib.apply("pick_detergent", facts, pool=pool)
    assert ("detergent", "held") in facts
    assert pool.used("hand") == 1


def test_resource_pool_consume_is_atomic_and_validates():
    pool = ResourcePool([Resource("hand", 1)])
    assert pool.consume("hand") is True
    assert pool.consume("hand") is False
    assert pool.used("hand") == 1
    assert pool.can_consume("hand") is False

    pool.release("hand", 5)
    assert pool.used("hand") == 0
    assert pool.available("hand") == 1

    with pytest.raises(KeyError):
        pool.available("missing")
    with pytest.raises(ValueError):
        Resource("hand", -1)


def test_resource_pool_snapshot_roundtrip():
    pool = ResourcePool({"hand": 1, "slot": 2})
    assert pool.consume("hand") is True
    assert pool.consume("slot", 2) is True

    snapshot = pool.snapshot()
    assert snapshot == {"hand": 1, "slot": 2}

    pool.release("hand")
    assert pool.used("hand") == 0
    pool.restore(snapshot)
    assert pool.used("hand") == 1
    assert pool.used("slot") == 2


def test_resource_helpers_return_plain_dicts():
    op = laundry_conditional_library().get("pick_clothes")
    assert op.consumes_dict() == {"hand": 1}
    assert op.produces_dict() == {}


def test_inverse_is_none_where_unspecified():
    lib = laundry_conditional_library()
    for name in ("fill_water", "press_start", "wait_done", "close_washer", "hang"):
        assert lib.get(name).inverse() is None


def test_inverse_undoes_pick_clothes():
    lib = laundry_conditional_library()
    pick = lib.get("pick_clothes")
    undo = pick.inverse()
    assert isinstance(undo, ConditionalOperator)
    assert undo.name == "undo_pick_clothes"
    assert undo.preconditions == frozenset()

    after = pick.apply(_f(("clothes", "in", "basket")))
    assert ("clothes", "held") in after
    restored = undo.apply(after)
    assert ("clothes", "in", "basket") in restored
    assert ("clothes", "held") not in restored


def test_inverse_of_open_washer_closes_door_but_keeps_adverse_effect():
    lib = laundry_conditional_library()
    op = lib.get("open_washer")
    undo = op.inverse()
    assert undo is not None
    assert undo.name == "undo_open_washer"
    assert undo.preconditions == frozenset()

    after = op.apply(
        _f(("door", "closed"), ("machine", "stopped"), ("water", "in", "washer"))
    )
    assert ("floor", "wet") in after
    restored = undo.apply(after)
    assert ("door", "closed") in restored
    assert ("door", "open") not in restored
    # The authored inverse only covers the door; adverse facts persist.
    assert ("floor", "wet") in restored


def test_applicable_respects_confidence_threshold():
    lib = laundry_conditional_library()
    op = lib.get("pick_clothes")
    facts = _f(("clothes", "in", "basket"))
    low = {("clothes", "in", "basket"): 0.4}

    assert op.applicable(facts, confidence=low, threshold=0.5) is False
    assert op.applicable(facts, confidence=low, threshold=0.3) is True
    # Missing confidence entries default to 1.0.
    assert op.applicable(facts, confidence={("other",): 0.0}, threshold=0.5) is True
    assert op.applicable(facts, confidence=None, threshold=1.0) is True
    # A missing hard precondition still fails.
    assert op.applicable(frozenset(), confidence=None) is False


def test_library_ranks_by_mean_confidence_then_name():
    lib = laundry_conditional_library()
    confidence = {
        ("clothes", "in", "basket"): 0.5,
        ("detergent", "on", "counter"): 0.5,
    }
    ranked = lib.applicable(_initial_facts(), confidence=confidence, threshold=0.4)
    names = [op.name for op, _ in ranked]
    scores = {op.name: score for op, score in ranked}

    assert scores["pick_clothes"] == pytest.approx(0.5)
    assert scores["pick_detergent"] == pytest.approx(0.5)
    assert scores["open_washer"] == pytest.approx(1.0)
    assert names.index("pick_clothes") < names.index("pick_detergent")


def test_residual_is_jaccard_distance():
    op = laundry_conditional_library().get("fill_water")
    one = _f(("x",))

    assert op.residual(one, one) == 0.0
    assert op.residual(one, _f(("y",))) == 1.0
    assert op.residual(frozenset(), frozenset()) == 0.0
    assert op.residual(one, _f(("x",), ("z",))) == pytest.approx(0.5)


def test_actions_do_not_compose():
    lib = laundry_conditional_library()
    assert lib.get("pick_clothes").compose(lib.get("open_washer")) is None


def test_full_laundry_sequence_reaches_washer_then_clean():
    lib = laundry_conditional_library()
    pool = laundry_resource_pool()
    facts = _initial_facts()

    for step, name in enumerate(FULL_PLAN):
        applicable = [op.name for op, _ in lib.applicable(facts, pool=pool)]
        assert name in applicable, f"step {step} {name} is not applicable"

        facts = lib.apply(name, facts, pool=pool)

        if name == "put_in_clothes":
            assert ("clothes", "in", "washer") in facts
        if name == "wait_done":
            assert ("clothes", "clean") in facts

    assert ("clothes", "clean") in facts
    assert ("clothes", "on", "drying_rack") in facts
    # No water was present on either door opening, so nothing spilled.
    assert ("floor", "wet") not in facts
    # Every hand use was balanced by a release.
    assert pool.used("hand") == 0
