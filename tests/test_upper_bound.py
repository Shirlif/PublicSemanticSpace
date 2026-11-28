"""Tests for the architecture upper-bound review (U0-U4)."""

from __future__ import annotations

import numpy as np

from pss.eval.readout_reasoning_preregistration import (
    check_readout_reasoning_preregistration,
)
from pss.eval.upper_bound import (
    absence_cell_error,
    cell_accuracy,
    delta_vs_full,
    exact4,
    pair_labels,
    pair_types,
    snap_pairs,
    worst_cell_accuracy,
)
from pss.prototypes.incremental import IncrementalLDABank


def test_pair_types_and_labels():
    pa = np.array([0.0, 0.0], dtype=np.float32)
    horizontal = pair_types(pa, np.array([1.0, 0.0], dtype=np.float32))
    vertical = pair_types(pa, np.array([0.0, 1.0], dtype=np.float32))
    diagonal = pair_types(pa, np.array([1.0, 1.0], dtype=np.float32))
    assert horizontal == "horizontal"
    assert vertical == "vertical"
    assert diagonal == "diagonal"

    bits = pair_labels(pa, np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    assert bits.tolist() == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    ]


def test_cell_metrics_and_absence():
    types = np.array(["horizontal", "vertical", "diagonal"], dtype=object)
    y = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 1]], dtype=np.float32)
    p = y.copy()
    p[1, 2] = 1.0  # vertical cell: above false positive
    assert exact4(y, p) == 2 / 3
    cells = cell_accuracy(types, y, p)
    assert cells["horizontal"] == 1.0
    assert cells["vertical"] == 0.0
    assert cells["diagonal"] == 1.0
    assert worst_cell_accuracy(types, y, p) == 0.0
    assert absence_cell_error(types, y, p, "vertical") == 1.0
    assert delta_vs_full(0.5, 1.0) == -0.5


def test_snap_pairs_preserves_other_axis():
    pa = np.array([0.2, 0.3], dtype=np.float32)
    pb = np.array([0.8, 0.7], dtype=np.float32)
    h_a, h_b = snap_pairs(pa, pb, "horizontal")
    assert h_b[1] == h_a[1]
    assert h_b[0] == pb[0]
    v_a, v_b = snap_pairs(pa, pb, "vertical")
    assert v_b[0] == v_a[0]
    assert v_b[1] == pb[1]


def test_incremental_delete_merge_and_rollback():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(120, 6))
    y = np.repeat(np.arange(4), 30)
    bank = IncrementalLDABank(refit_every=0).fit(Z, y)
    before = bank.predict(Z)

    bank.remove_class(1)
    assert 1 not in bank.classes_
    assert 1 not in set(bank.predict(Z).tolist())

    bank.merge_classes(2, 0)
    assert 2 not in bank.classes_
    assert 0 in bank.classes_
    assert bank.means_.shape[0] == len(bank.classes_)

    snap = bank.snapshot()
    bank.remove_class(3)
    assert 3 not in bank.classes_
    bank.restore(snap)
    assert 3 in bank.classes_
    assert np.array_equal(bank.predict(Z), np.asarray(bank.classes_)[bank.scores(Z).argmax(1)])
    assert before.shape == bank.predict(Z).shape


def test_upper_bound_arms_pass_and_fail_as_registered():
    good = {
        "u_real_proto": {"test_acc": 0.50},
        "u_real_lda_whiten": {
            "accuracy_delta_vs_proto": 0.10,
            "auroc": 0.995,
            "unseen_reject_at_95": 0.96,
        },
        "u_real_rerank": {"accuracy_delta_vs_gate": 0.0},
        "u_real_dual_head": {
            "accuracy_delta_vs_proto": 0.10,
            "auroc": 0.995,
            "unseen_reject_at_95": 0.96,
        },
        "u_geo_combination": {"worst_cell_acc": 0.95, "delta_vs_full": -0.01},
        "u_geo_single": {"worst_cell_acc": 0.30, "delta_vs_full": -0.65},
        "u_geo_noop": {"worst_cell_acc": 0.10, "delta_vs_full": -0.90},
        "u_geo_full": {"worst_cell_acc": 1.0, "delta_vs_full": 0.0},
        "u_comp_hybrid": {
            "composition_gap": 0.0,
            "absence_cell_error": 0.05,
            "test_acc": 0.99,
        },
        "u_incr_delete_merge": {"forgetting": 0.01, "added_acc": 0.95},
    }
    check = check_readout_reasoning_preregistration(good)
    assert check["ok"], check["mismatches"]

    bad = {
        **good,
        "u_real_lda_whiten": {
            "accuracy_delta_vs_proto": 0.01,
            "auroc": 0.995,
            "unseen_reject_at_95": 0.96,
        },
        "u_geo_single": {"worst_cell_acc": 0.95, "delta_vs_full": -0.01},
    }
    check = check_readout_reasoning_preregistration(bad)
    assert not check["ok"]
    assert any("u_real_lda_whiten.accuracy_delta_vs_proto" in m for m in check["mismatches"])
    assert any("u_geo_single.delta_vs_full" in m for m in check["mismatches"])


def test_unmeasured_declared_bar_is_a_failure():
    check = check_readout_reasoning_preregistration(
        {"u_real_lda_whiten": {"test_acc": 0.90}}
    )
    assert not check["ok"]
    assert any("not measured" in m for m in check["mismatches"])
