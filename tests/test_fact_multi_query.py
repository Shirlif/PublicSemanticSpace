"""Tests for multi-query fact selection (F9)."""

from __future__ import annotations

import numpy as np

from pss.eval.scripts.run_gap_fact_multi_query import (
    METHODS,
    SPEC_BY_NAME,
    coverage,
    make_queries,
    make_stream,
    redundancy,
    select,
)
from pss.memory.active_policy import allocate_budget, greedy_max_coverage


def test_allocate_budget_even_and_remainder():
    assert allocate_budget(4, np.zeros(4)).tolist() == [1, 1, 1, 1]
    assert allocate_budget(2, np.zeros(4)).tolist() == [1, 1, 0, 0]
    # Remainder goes to the highest-priority item.
    assert allocate_budget(7, np.array([0.0, 1.0, 2.0])).tolist() == [2, 2, 3]
    assert allocate_budget(0, np.zeros(3)).tolist() == [0, 0, 0]
    assert allocate_budget(5, np.zeros(0)).tolist() == []


def test_greedy_prefers_shared_fact():
    # Fact 0 covers queries 0 and 1; fact 1 covers only query 0; fact 2 covers 2.
    support = np.array(
        [
            [True, True, False],
            [True, False, False],
            [False, False, True],
        ]
    )
    selected = greedy_max_coverage(support, k=2)
    assert set(selected.tolist()) == {0, 2}
    covered = support[selected].any(axis=0)
    assert covered.sum() == 3


def test_greedy_budget_and_bounds():
    support = np.zeros((10, 3), dtype=bool)
    support[3, 0] = True
    support[7, 1] = True
    support[9, 2] = True
    selected = greedy_max_coverage(support, k=20)
    assert len(selected) == 10
    assert len(set(selected.tolist())) == 10
    assert greedy_max_coverage(support, k=0).tolist() == []


def test_stream_and_queries_are_deterministic():
    spec = SPEC_BY_NAME["hard"]
    a = make_stream(spec, seed=3)
    b = make_stream(spec, seed=3)
    assert np.array_equal(a.keys, b.keys)
    assert np.array_equal(a.values, b.values)
    assert np.allclose(a.reliability, b.reliability)

    qa = make_queries(a, Q=4, seed=11, spec=spec)
    qb = make_queries(b, Q=4, seed=11, spec=spec)
    assert np.array_equal(qa.keys, qb.keys)
    assert np.array_equal(qa.as_of, qb.as_of)
    assert np.array_equal(qa.support, qb.support)
    assert (qa.support_sizes > 0).all()


def test_easy_stream_reproduces_query_recency_saturation():
    spec = SPEC_BY_NAME["easy"]
    stream = make_stream(spec, seed=0)
    qs = make_queries(stream, Q=4, seed=4, spec=spec)
    selected = select("query_recency", stream, qs, k=4, seed=0)
    assert coverage(selected, qs) == 1.0


def test_drift_breaks_query_recency_but_not_greedy():
    spec = SPEC_BY_NAME["drift"]
    stream = make_stream(spec, seed=0)
    qs = make_queries(stream, Q=4, seed=4, spec=spec)
    rec = select("query_recency", stream, qs, k=4, seed=0)
    greedy = select("greedy_coverage", stream, qs, k=4, seed=0)
    assert coverage(rec, qs) < 1.0
    assert coverage(greedy, qs) == 1.0


def test_greedy_beats_union_on_conflict_hard_setting():
    spec = SPEC_BY_NAME["conflict"]
    stream = make_stream(spec, seed=0)
    qs = make_queries(stream, Q=4, seed=4, spec=spec)
    union = select("per_query_union", stream, qs, k=4, seed=0)
    greedy = select("greedy_coverage", stream, qs, k=4, seed=0)
    assert coverage(greedy, qs) > coverage(union, qs)
    assert coverage(greedy, qs) == 1.0


def test_greedy_beats_union_on_hotkey_budget_constraint():
    spec = SPEC_BY_NAME["hotkey"]
    stream = make_stream(spec, seed=0)
    qs = make_queries(stream, Q=8, seed=8, spec=spec)
    union = select("per_query_union", stream, qs, k=4, seed=0)
    greedy = select("greedy_coverage", stream, qs, k=4, seed=0)
    assert coverage(greedy, qs) > coverage(union, qs)


def test_all_methods_respect_budget_and_report_redundancy():
    spec = SPEC_BY_NAME["hard"]
    stream = make_stream(spec, seed=0)
    qs = make_queries(stream, Q=4, seed=4, spec=spec)
    for method in METHODS:
        selected = select(method, stream, qs, k=8, seed=0)
        if method != "full":
            assert len(np.unique(selected)) <= 8
        cov = coverage(selected, qs)
        assert 0.0 <= cov <= 1.0
        red = redundancy(selected, stream, qs)
        assert set(red) == {
            "n_selected",
            "wasted_frac",
            "duplicate_frac",
            "facts_per_covered_query",
            "coverage_per_fact",
            "mean_queries_per_fact",
        }
