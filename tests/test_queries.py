from __future__ import annotations

from pss.reasoning.queries import (
    FactSet,
    TemporalHistory,
    all_of,
    any_of,
    count_of,
    none_of,
    resolve_location_reference,
    select_entities,
    unique_object,
)
from pss.reasoning.relations import RelationRecord


def test_factset_accepts_relation_records_and_tuples_and_dedupes():
    record = RelationRecord("cup", "on", "shelf", 0.8)
    facts = FactSet([record, ("cup", "on", "shelf", 0.9), ("cup", "on", "shelf")])

    assert len(facts) == 1
    assert ("cup", "on", "shelf") in facts
    assert ("cup", "on", "table") not in facts
    # A later entry for the same triple replaces the earlier confidence.
    assert facts.confidence("cup", "on", "shelf") == 1.0
    assert list(facts) == [("cup", "on", "shelf", 1.0)]


def test_factset_add_remove_select_and_views():
    facts = FactSet([("a", "in", "box"), ("b", "in", "box"), ("c", "on", "box")])
    facts.add("d", "near", "box", 0.4)

    assert len(facts) == 4
    assert facts.select(predicate="in") == {("a", "in", "box"), ("b", "in", "box")}
    assert facts.subjects("in", object="box") == {"a", "b"}
    assert facts.objects("in", subject="a") == {"box"}
    assert select_entities(facts, predicate="in", object="box") == {"a", "b"}
    assert facts.facts(min_conf=0.5) == {
        ("a", "in", "box"),
        ("b", "in", "box"),
        ("c", "on", "box"),
    }

    facts.remove("c", "on", "box")
    assert ("c", "on", "box") not in facts

    clone = facts.copy()
    clone.add("e", "in", "box")
    assert ("e", "in", "box") not in facts
    assert facts.to_dict()["count"] == len(facts)


def test_all_of_true_and_false_with_counterexamples():
    facts = FactSet(
        [
            ("cup", "on", "shelf", 0.9),
            ("plate", "on", "shelf", 0.6),
            ("bowl", "on", "table", 1.0),
        ]
    )

    ok = all_of(facts, ["cup", "plate"], "on", "shelf")
    assert ok.kind == "all"
    assert ok.value is True
    assert ok.total == 2
    assert ok.counterexamples == ()
    assert ok.confidence == 0.6

    bad = all_of(facts, ["cup", "bowl"], "on", "shelf")
    assert bad.value is False
    assert bad.total == 2
    assert bad.counterexamples == ("bowl",)
    assert bad.confidence == 0.9
    assert bad.to_dict() == {
        "kind": "all",
        "value": False,
        "total": 2,
        "confidence": 0.9,
        "counterexamples": ["bowl"],
    }


def test_any_none_and_count_quantifiers():
    facts = FactSet(
        [
            ("cup", "on", "shelf", 0.9),
            ("plate", "on", "shelf", 0.6),
            ("bowl", "on", "table", 1.0),
        ]
    )

    any_result = any_of(facts, ["bowl", "cup"], "on", "shelf")
    assert any_result.kind == "any"
    assert any_result.value is True
    assert any_result.total == 2
    assert any_result.counterexamples == ("bowl",)
    assert any_result.confidence == 0.9

    none_result = none_of(facts, ["bowl"], "on", "shelf")
    assert none_result.kind == "none"
    assert none_result.value is True
    assert none_result.confidence == 1.0

    violation = none_of(facts, ["cup"], "on", "shelf")
    assert violation.value is False
    assert violation.counterexamples == ("cup",)
    assert violation.confidence == 1.0 - 0.9

    counted = count_of(facts, ["cup", "plate", "bowl"], "on", "shelf")
    assert counted.kind == "count"
    assert counted.value == 2
    assert counted.total == 3
    assert counted.counterexamples == ("bowl",)


def test_quantifiers_dedupe_entities_preserving_order():
    facts = FactSet([("cup", "on", "shelf")])

    result = all_of(facts, ["cup", "cup", "bowl", "bowl"], "on", "shelf")
    assert result.total == 2
    assert result.counterexamples == ("bowl",)


def test_unique_object_only_and_ambiguous():
    facts = FactSet([("key", "in", "drawer", 0.9)])
    assert unique_object(facts, "key", "in") == "drawer"

    facts.add("key", "in", "pocket", 0.8)
    assert unique_object(facts, "key", "in") is None
    assert unique_object(facts, "key", "under") is None


def test_min_conf_filters_facts():
    facts = FactSet([("a", "p", "t", 0.9), ("b", "p", "t", 0.2)])

    assert facts.facts(min_conf=0.5) == {("a", "p", "t")}

    result = all_of(facts, ["a", "b"], "p", "t", min_conf=0.5)
    assert result.value is False
    assert result.counterexamples == ("b",)

    assert unique_object(facts, "b", "p") is None
    assert unique_object(facts, "b", "p", min_conf=0.1) == "t"


def _movement_history() -> TemporalHistory:
    history = TemporalHistory()
    history.record(0.0, [("key", "in", "drawer")])
    history.record(5.0, [("key", "in", "pocket")], removed=[("key", "in", "drawer")])
    return history


def test_at_returns_latest_state_not_later():
    history = _movement_history()

    assert history.times() == [0.0, 5.0]
    assert history.at(2.0).facts() == {("key", "in", "drawer")}
    assert history.at(5.0).facts() == {("key", "in", "pocket")}
    assert history.at(99.0).facts() == {("key", "in", "pocket")}
    assert history.at(-1.0).facts() == set()


def test_added_and_removed_between_detect_explicit_change():
    history = _movement_history()

    assert history.added_between(0.0, 5.0) == {("key", "in", "pocket")}
    assert history.removed_between(0.0, 5.0) == {("key", "in", "drawer")}
    assert history.between(1.0, 5.0).facts() == {("key", "in", "pocket")}
    assert history.added_between(5.0, 5.0) == set()


def test_where_was_uses_the_earlier_snapshot():
    history = _movement_history()

    assert history.where_was("key", 1.0) == {"drawer"}
    assert history.where_was("key", 6.0) == {"pocket"}
    assert history.where_was("key", 6.0, location_predicates=("inside",)) == set()


def test_first_and_last_seen():
    history = _movement_history()

    assert history.first_seen(("key", "in", "drawer")) == 0.0
    assert history.last_seen(("key", "in", "drawer")) == 0.0
    assert history.first_seen(("key", "in", "pocket")) == 5.0
    assert history.last_seen(("key", "in", "pocket")) == 5.0
    assert history.first_seen(("ghost", "in", "void")) is None
    assert history.last_seen(("ghost", "in", "void")) is None


def test_record_keeps_times_sorted_and_replaces_same_time():
    history = TemporalHistory()
    history.record(5.0, [("key", "in", "pocket")])
    history.record(0.0, [("key", "in", "drawer")])
    assert history.times() == [0.0, 5.0]

    history.record(5.0, [("key", "on", "shelf")], removed=[("key", "in", "pocket")])
    state = history.at(5.0)
    assert ("key", "on", "shelf") in state
    assert ("key", "in", "pocket") not in state
    # The replacement also discarded the old removal list.
    assert history.removals[5.0] == {("key", "in", "pocket")}


def test_history_of_returns_entity_facts_per_time():
    history = _movement_history()

    entries = history.history_of("key")
    assert [time for time, _ in entries] == [0.0, 5.0]
    assert entries[0][1] == {("key", "in", "drawer")}
    assert entries[1][1] == {("key", "in", "pocket")}


def test_resolve_location_reference():
    history = _movement_history()

    known = resolve_location_reference(history, "key", 1.0)
    assert known == {"locations": ["drawer"], "at": 1.0, "found": True}

    unknown = resolve_location_reference(history, "ghost", 1.0)
    assert unknown == {"locations": [], "at": 1.0, "found": False}
