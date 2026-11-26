"""F7 regression tests: incremental TMS propagation and snapshot modes."""

from __future__ import annotations

import random

import pytest

from pss.knowledge import MultiSourceFactStore, TMSFactStore
from pss.knowledge.tms import TruthMaintenanceSystem
from pss.runtime import RelationFactLayer


def _labels_equal(incremental: dict, full: dict) -> bool:
    for node in set(incremental) | set(full):
        if bool(incremental.get(node, False)) != bool(full.get(node, False)):
            return False
    return True


def test_incremental_matches_full_after_random_operations():
    rng = random.Random(0)
    nodes = [(f"e{i}", "v", "1") for i in range(8)]
    for _ in range(25):
        tms = TruthMaintenanceSystem(snapshot_mode="delta", checkpoint_interval=4)
        for _ in range(40):
            operation = rng.choice(["assume", "unassume", "justify", "remove"])
            node = rng.choice(nodes)
            if operation == "assume":
                tms.assume(node)
            elif operation == "unassume":
                tms.unassume(node)
            elif operation == "justify":
                antecedent = rng.choice(nodes)
                tms.justify(
                    node,
                    in_nodes=[antecedent],
                    source=f"s{rng.randrange(3)}",
                )
            else:
                tms.remove_source(f"s{rng.randrange(3)}")
            assert _labels_equal(tms.propagate(), tms.propagate_full())


def test_dependents_use_reverse_graph():
    tms = TruthMaintenanceSystem(snapshot_mode="delta")
    tms.assume(("a", "v", "1"))
    tms.justify(("b", "v", "1"), in_nodes=[("a", "v", "1")], source="r")
    tms.justify(("c", "v", "1"), in_nodes=[("b", "v", "1")], source="r")
    assert tms.dependents(("a", "v", "1")) == {
        ("b", "v", "1"),
        ("c", "v", "1"),
    }
    assert tms.dependents(("a", "v", "1"), transitive=False) == {("b", "v", "1")}


def test_new_antecedent_is_labelled():
    tms = TruthMaintenanceSystem(snapshot_mode="delta")
    tms.justify(("b", "v", "1"), in_nodes=[("a", "v", "1")], source="r")
    assert tms.label(("a", "v", "1")) is False
    assert tms.label(("b", "v", "1")) is False


@pytest.mark.parametrize("mode", ["full", "cow", "delta"])
def test_snapshot_modes_rollback_exact(mode):
    store = TMSFactStore(snapshot_mode=mode, checkpoint_interval=4)
    store.assert_fact("a", "x", "1", source="s1")
    store.justify(("b", "y", "1"), in_nodes=[("a", "x", "1")], source="r")
    before_hash = store.state_hash()
    before_version = store.version

    for i in range(12):
        store.update(f"e{i}", "x", str(i), source="s")
    store.rollback(before_version)

    assert store.state_hash() == before_hash
    assert store.label(("b", "y", "1")) is True
    assert store.resolve("e0", "x") is None


def test_relation_layer_defaults_to_tms_and_accepts_injection():
    layer = RelationFactLayer(use_default_write=False)
    assert isinstance(layer.facts, TMSFactStore)

    legacy = MultiSourceFactStore()
    injected = RelationFactLayer(facts=legacy, use_default_write=False)
    assert injected.facts is legacy
    injected.assert_fact("cup", "color", "red", source="s")
    assert injected.resolve("cup", "color") == "red"


def test_inner_multisource_history_can_be_disabled():
    quiet = MultiSourceFactStore(record_history=False)
    quiet.assert_fact("a", "x", "1", source="s")
    assert quiet.history == []

    default = MultiSourceFactStore()
    default.assert_fact("a", "x", "1", source="s")
    assert len(default.history) == 2


def test_tms_rollback_then_mutate_uses_new_branch():
    tms = TruthMaintenanceSystem(snapshot_mode="delta", checkpoint_interval=4)
    tms.assume(("a", "v", "1"))
    token = tms.snapshot()

    tms.assume(("b", "v", "1"))
    tms.restore(token)
    tms.assume(("c", "v", "1"))

    assert tms.label(("a", "v", "1")) is True
    assert tms.label(("c", "v", "1")) is True
    # The abandoned branch must not be replayed.
    assert tms.label(("b", "v", "1")) is False


def test_fact_store_rollback_then_mutate_uses_new_branch():
    store = TMSFactStore(snapshot_mode="delta", checkpoint_interval=4)
    store.assert_fact("a", "x", "1", source="s")
    store.assert_fact("b", "x", "1", source="s")
    token = store.snapshot()

    store.assert_fact("c", "x", "1", source="s")
    store.restore(token)
    assert store.resolve("c", "x") is None

    store.assert_fact("d", "x", "1", source="s")
    assert store.resolve("a", "x") == "1"
    assert store.resolve("d", "x") == "1"
    # The abandoned branch must not be replayed.
    assert store.resolve("c", "x") is None
