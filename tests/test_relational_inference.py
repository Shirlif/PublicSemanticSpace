"""Tests for forward/backward relational inference over closure tables."""

from __future__ import annotations

from pss.reasoning.relations import RelationRecord
from pss.reasoning.relations.inference import (
    DerivedFact,
    InferenceResult,
    backward_chain,
    derive,
    derive_records,
)


def _r(subject, predicate, obj, confidence=1.0):
    return RelationRecord(subject, predicate, obj, confidence)


def _triples(result: InferenceResult):
    return [fact.as_tuple() for fact in result.derived]


def test_transitivity_derives_new_left_of():
    records = [_r("a", "left_of", "b"), _r("b", "left_of", "c")]
    result = derive(records)

    assert ("a", "left_of", "c") in result.facts()
    fact = next(f for f in result.derived if f.as_tuple() == ("a", "left_of", "c"))
    assert 0.0 < fact.confidence < 1.0
    assert fact.premises == (("a", "left_of", "b"), ("b", "left_of", "c"))
    assert fact.rule == "closure"
    assert fact.as_record().source == "inference"
    assert result.rounds == 1


def test_left_of_composed_with_right_of_is_not_licensed():
    records = [_r("a", "left_of", "b"), _r("b", "right_of", "c")]
    result = derive(records)

    assert result.facts() == set()
    assert result.considered >= 1
    assert result.rejected >= 1


def test_three_hop_chain_uses_second_round_without_duplicates():
    records = [
        _r("a", "left_of", "b"),
        _r("b", "left_of", "c"),
        _r("c", "left_of", "d"),
    ]
    result = derive(records, max_rounds=2)

    triples = _triples(result)
    assert ("a", "left_of", "c") in triples
    assert triples.count(("a", "left_of", "c")) == 1
    # Round 2 composes the round-1 fact ``a left_of c`` with ``c left_of d``.
    assert ("a", "left_of", "d") in result.facts()
    assert result.rounds == 2


def test_self_loops_are_suppressed():
    records = [_r("a", "left_of", "b"), _r("b", "left_of", "a")]
    result = derive(records)

    assert result.facts() == set()
    assert all(fact.subject != fact.object for fact in result.derived)


def test_min_conf_prunes_low_confidence_chains():
    records = [_r("a", "left_of", "b", 0.5), _r("b", "left_of", "c", 0.5)]
    # 0.95 * sqrt(0.5 * 0.5) == 0.475, below the default 0.5 threshold.
    assert derive(records).facts() == set()

    relaxed = derive(records, min_conf=0.4)
    assert ("a", "left_of", "c") in relaxed.facts()


def test_identical_derived_fact_is_deduped_across_paths():
    records = [
        _r("a", "left_of", "b"),
        _r("b", "left_of", "c"),
        _r("a", "left_of", "x"),
        _r("x", "left_of", "c"),
    ]
    result = derive(records)

    assert _triples(result).count(("a", "left_of", "c")) == 1


def test_asserted_facts_are_not_re_emitted_as_derived():
    records = [
        _r("a", "left_of", "b"),
        _r("b", "left_of", "c"),
        _r("a", "left_of", "c"),
    ]
    result = derive(records)

    assert ("a", "left_of", "c") not in result.facts()
    assert result.rejected >= 1


def test_type_check_can_veto_a_derivation():
    records = [_r("a", "left_of", "b"), _r("b", "left_of", "c")]

    vetoed = derive(records, type_check=lambda subject, predicate, obj: False)
    assert vetoed.facts() == set()
    assert vetoed.rejected >= 1

    allowed = derive(records, type_check=lambda subject, predicate, obj: predicate == "left_of")
    assert ("a", "left_of", "c") in allowed.facts()


def test_backward_chain_finds_left_of_via_middle():
    records = [_r("a", "left_of", "b"), _r("b", "left_of", "c")]
    candidates = backward_chain(records, "a", "c")

    assert candidates
    predicate, confidence, path = candidates[0]
    assert predicate == "left_of"
    assert path == (("a", "left_of", "b"), ("b", "left_of", "c"))
    assert 0.0 < confidence < 1.0


def test_backward_chain_respects_max_depth_and_confidence():
    records = [_r("a", "left_of", "b"), _r("b", "left_of", "c")]
    assert backward_chain(records, "a", "c", max_depth=1) == []
    assert backward_chain(records, "a", "c", min_conf=0.99) == []


def test_input_records_are_not_mutated():
    records = [
        _r("a", "left_of", "b", 0.9),
        _r("b", "left_of", "c", 0.8),
    ]
    snapshot = [
        (record.subject, record.predicate, record.object, record.confidence, record.source)
        for record in records
    ]

    derive(records)

    after = [
        (record.subject, record.predicate, record.object, record.confidence, record.source)
        for record in records
    ]
    assert after == snapshot
    assert len(records) == 2


def test_derive_records_wrapper_and_result_helpers():
    records = [("a", "left_of", "b"), ("b", "left_of", "c", 1.0)]
    output = derive_records(records)

    assert len(output) == 1
    assert isinstance(output[0], RelationRecord)
    assert output[0].as_tuple() == ("a", "left_of", "c")
    assert output[0].source == "inference"

    result = derive(records)
    assert isinstance(result.derived[0], DerivedFact)
    payload = result.to_dict()
    assert payload["rounds"] == 1
    assert payload["derived"][0]["predicate"] == "left_of"
    assert result.records()[0].source == "inference"


def test_max_derived_caps_output():
    records = [
        _r("a", "left_of", "b"),
        _r("b", "left_of", "c"),
        _r("c", "left_of", "d"),
    ]
    result = derive(records, max_derived=1)

    assert len(result.derived) <= 1
