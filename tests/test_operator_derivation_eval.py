"""Smoke test for the operator derivation evaluation (P0-P3)."""

from __future__ import annotations

from pss.eval.scripts.run_operator_derivation import run


def test_run_smoke():
    metrics = run(n_per_family=3, seed=0)

    transitive = metrics["transitive_answer"]["overall"]
    assert transitive["n"] > 0
    assert metrics["transitive_answer"]["direct_only"]["n"] == transitive["n"]

    chain = metrics["transitive_sparse_chain"]
    assert chain["scenes"] > 0
    assert 0.0 <= chain["precision"] <= 1.0

    for family in ("frame_scoping", "perspective_scoping"):
        row = metrics[family]
        assert row["frame_op"]["n"] > 0
        assert row["frame_op"]["accuracy"] == 1.0
        assert 0.0 <= row["frame_sensitive_rate"] <= 1.0

    holdout = metrics["holdout_derivation"]
    assert holdout["scenes"] > 0
    assert 0.0 <= holdout["precision"] <= 1.0

    for name, row in metrics["cross_frame"].items():
        rate = row["contradiction_rate"]
        assert rate is None or 0.0 <= rate <= 1.0, name
