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

    # Strong drift must be recognised, the confidence channel must stay
    # calibrated without a temperature bound, and the latency must actually be
    # measured (no vacuous None).
    for sigma in ("0.20", "0.30"):
        row = rows[f"drift_sigma@{sigma}"]
        assert row["state"] == "recalibrate"
        assert row["ece_map_fused"] < 0.15
        assert row["recovery_samples"] is not None

    # Sigma=0.10 on the synthetic stream sits inside the calibrated clean tail;
    # the real-data report is the claim holder for that scenario.  The latency
    # is still measured, so a missing measurement fails loudly.
    junior = rows["drift_sigma@0.10"]
    assert junior["recovery_detail"] is not None

    # Quarantined poison must freeze the parameters.
    assert rows["random_label_flip@0.40"]["state"] == "quarantine"
    assert rows["random_label_flip@0.40"]["accept_update"] is False

    # Clean-stream false alarms are measured, not assumed.
    clean = metrics["clean_stream"]
    assert clean["windows"] > 0
    assert clean["reject_rate"] < 0.05

    # Thresholds come from the clean stream, not the dataclass defaults alone.
    assert metrics["thresholds"]["theta_label"] > 0.0
    assert metrics["thresholds"]["theta_drift"] > 0.0

    # Pre-registration failures stay visible.  The synthetic stream is not the
    # real-data claim holder, so its own gaps are allowed; what must never
    # happen is a silent "ok".  An unmeasured recovery is covered separately by
    # the contract test.
    mismatches = metrics["preregistration"]["mismatches"]
    assert mismatches
    assert any("targeted" in item for item in mismatches)


def test_harness_is_deterministic() -> None:
    first = run(_args())
    second = run(_args())
    first.pop("generated", None)
    second.pop("generated", None)
    assert first == second
