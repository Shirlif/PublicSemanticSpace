from __future__ import annotations

import pandas as pd
import pytest

from pss.metrics.entity import (
    compute_entity_metrics,
    duplicate_rate,
    id_retention,
    id_switches,
    idf1,
    reappearance_rate,
    slot_purity,
)


def trace(rows):
    return pd.DataFrame(rows, columns=["frame", "gt_id", "pred_id"])


def gt(rows):
    return pd.DataFrame(rows, columns=["frame", "instance_id"])


def test_perfect_tracking_scores_one():
    rows = [(f, g, g) for f in range(3) for g in ("a", "b")]
    t = trace(rows)
    g = gt([(f, gg) for f, gg, _ in rows])
    assert idf1(g, t) == pytest.approx(1.0)
    assert id_retention(t) == pytest.approx(1.0)
    assert id_switches(t) == 0.0
    assert duplicate_rate(t) == 0.0
    assert slot_purity(t) == pytest.approx(1.0)


def test_identity_switch_lowers_idf1_and_retention():
    rows = [
        (0, "a", "p1"),
        (1, "a", "p1"),
        (2, "a", "p2"),
        (3, "a", "p2"),
    ]
    t = trace(rows)
    g = gt([(f, "a") for f, _, _ in rows])
    assert id_switches(t) == 1.0
    assert id_retention(t) == pytest.approx(0.5)
    assert idf1(g, t) == pytest.approx(0.5)


def test_duplicate_tracks_are_penalised():
    rows = [(0, "a", "p1"), (1, "a", "p2"), (2, "a", "p2")]
    t = trace(rows)
    assert duplicate_rate(t) == pytest.approx(0.5)


def test_reappearance_after_gap():
    rows = [
        (0, "a", "p1"),
        (1, "a", "p1"),
        (4, "a", "p1"),
        (5, "a", "p1"),
    ]
    assert reappearance_rate(trace(rows)) == pytest.approx(1.0)
    rows[-2] = (4, "a", "p9")
    assert reappearance_rate(trace(rows)) == pytest.approx(0.0)


def test_empty_trace_is_perfect_only_if_gt_is_empty():
    empty = trace([])
    assert idf1(gt([]), empty) == 1.0
    assert idf1(gt([(0, "a")]), empty) == 0.0


def test_metrics_dict_contains_counts():
    rows = [(0, "a", "p1"), (0, "b", "p2")]
    out = compute_entity_metrics(gt([(0, "a"), (0, "b")]), trace(rows))
    assert out["gt_instances"] == 2
    assert out["pred_identities"] == 2
    assert out["idf1"] == pytest.approx(1.0)
