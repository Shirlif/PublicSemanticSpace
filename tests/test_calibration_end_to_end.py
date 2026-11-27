"""End-to-end regression for the calibration harness.

The rest of the calibration tests are unit-level on synthetic batches.  This one
runs the whole harness so a refactor that silently changes the numbers fails
loudly, and it checks determinism across two runs.
"""

from __future__ import annotations

from argparse import Namespace

from pss.eval.scripts.run_gap_calibration_robust import run


def _args() -> Namespace:
    return Namespace(
        source="synthetic",
        seed=0,
        data=None,
        experts=None,
        projections=None,
        device=None,
        max_per_class=None,
    )


def test_harness_invariants() -> None:
    metrics = run(_args())
    rows = {r["scenario"]: r for r in metrics["scenarios"]}

    assert rows["clean"]["state"] == "normal"
    assert rows["clean"]["ece_map_fused"] < 0.05
    assert rows["clean"]["accept_update"] is True

    # Drift must be recognised, and the confidence channel must stay calibrated
    # without a temperature bound.
    for sigma in ("0.10", "0.20", "0.30"):
        row = rows[f"drift_sigma@{sigma}"]
        assert row["state"] == "recalibrate"
        assert row["ece_map_fused"] < 0.15

    # Quarantined poison must freeze the parameters.
    assert rows["random_label_flip@0.40"]["state"] == "quarantine"
    assert rows["random_label_flip@0.40"]["accept_update"] is False

    # Only the known-open targeted-attack detections may fail.
    mismatches = metrics["preregistration"]["mismatches"]
    assert len(mismatches) <= 2
    assert all("targeted" in m for m in mismatches)


def test_harness_is_deterministic() -> None:
    first = run(_args())
    second = run(_args())
    first.pop("generated", None)
    second.pop("generated", None)
    assert first == second
