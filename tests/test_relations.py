from __future__ import annotations

from pss.relations import RelObject, compute_relations, relation_holds
from pss.metrics.relation import consistency_violations, per_predicate_prf

PREDICATES = ["left_of", "right_of", "above", "below", "near", "touching", "on_top_of", "inside"]


def obj(name, x, y, z, size=0.04):
    return RelObject(name, [x, y, z, 1, 0, 0, 0], [size, size, size])


def test_basic_predicates():
    a = obj("a", -0.1, 0.0, 0.79)
    b = obj("b", 0.1, 0.0, 0.79)
    assert relation_holds(a, "left_of", b)
    assert relation_holds(b, "right_of", a)
    assert not relation_holds(a, "right_of", b)
    assert relation_holds(a, "near", b, near_m=0.25)
    assert not relation_holds(a, "near", b)
    assert not relation_holds(a, "touching", b)


def test_touching_and_stacked():
    base = obj("base", 0.0, 0.0, 0.79)
    top = obj("top", 0.0, 0.0, 0.83)
    assert relation_holds(top, "above", base)
    assert relation_holds(top, "on_top_of", base, tolerance_m=0.01)
    side = obj("side", 0.04, 0.0, 0.79)
    assert relation_holds(side, "touching", base, tolerance_m=0.001)
    assert relation_holds(obj("small", 0, 0, 0.79, 0.02), "inside", base)


def test_compute_relations_is_ordered_and_complete():
    objects = [obj("a", -0.1, 0.0, 0.79), obj("b", 0.1, 0.0, 0.79)]
    triples = compute_relations(objects, PREDICATES)
    assert ("a", "left_of", "b") in triples
    assert ("b", "right_of", "a") in triples
    assert all(s != o for s, _, o in triples)


def test_consistency_violations_detect_contradiction():
    triples = [("a", "left_of", "b"), ("b", "left_of", "a")]
    out = consistency_violations(triples)
    assert out["violation_count"] >= 1


def test_per_predicate_prf_perfect_and_missing():
    truth = [("a", "left_of", "b")]
    assert per_predicate_prf(truth, truth)["per_predicate"]["left_of"]["f1"] == 1.0
    out = per_predicate_prf([], truth)
    assert out["per_predicate"]["left_of"]["f1"] == 0.0
