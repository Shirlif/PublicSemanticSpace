"""Pre-registered expectations for the external baselines.

These expectations are the executable form of
``docs/pss_eval/DESIGN_AND_PREREGISTRATION.md``.  They are fixed before the
adapters are run, and a mismatch is a harness bug or a false pre-registration --
not something to "fix" by relaxing the expectation.

The contract-free E5 baseline must *fail* the contract property: it accepts
invalid modules, cannot roll back, and keeps no audit.  The transformer-state
baseline must be *skipped* there: it does not implement the optional
snapshot/swap/audit capabilities at all.  Both are expected to pass the three
protocol smoke properties.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "EXPECTED_CONTRACT_METRICS",
    "EXPECTED_PROTOCOL_STATUS",
    "EXPECTED_STATUS",
    "check_preregistration",
    "protocol_expectations",
]

PASS = "pass"
FAIL = "fail"
SKIP = "skip"

EXPECTED_STATUS: dict[str, dict[str, str]] = {
    "e5-flat": {
        "determinism": PASS,
        "reset_exactness": PASS,
        "state_hygiene": PASS,
        "contract": FAIL,
    },
    "transformer-state": {
        "determinism": PASS,
        "reset_exactness": PASS,
        "state_hygiene": PASS,
        "contract": SKIP,
    },
}

# Pre-registered expectations for the full (extended) protocol suite.  The
# PSS runtime must pass every contract property; the contract-free E5 baseline
# must fail the swap/rollback ones; the transformer baseline is skipped on all
# capabilities it does not implement.
EXPECTED_PROTOCOL_STATUS: dict[str, dict[str, str]] = {
    "e5-flat": {
        "determinism": PASS,
        "reset_exactness": PASS,
        "state_hygiene": PASS,
        "contract": FAIL,
        "swap_rollback": FAIL,
        "identity_retention": SKIP,
        "budget": SKIP,
        "latency": PASS,
    },
    "transformer-state": {
        "determinism": PASS,
        "reset_exactness": PASS,
        "state_hygiene": PASS,
        "contract": SKIP,
        "swap_rollback": SKIP,
        "identity_retention": SKIP,
        "budget": SKIP,
        "latency": PASS,
    },
    "pss-runtime": {
        "determinism": PASS,
        "reset_exactness": PASS,
        "state_hygiene": PASS,
        "contract": PASS,
        "swap_rollback": PASS,
        "budget": PASS,
        "latency": PASS,
    },
}

# Exact values the E5 adapter must produce on the contract probe.  These mirror
# the E5 scorecard: SRR=0 (invalid modules accepted), RBE=0, AUD=0.
EXPECTED_CONTRACT_METRICS: dict[str, dict[str, float]] = {
    "e5-flat": {
        "swap_accepted": 1.0,
        "invalid_rejected": 0.0,
        "invalid_keeps_state": 0.0,
        "rollback_exact": 0.0,
        "audit_present": 0.0,
    },
    # The PSS runtime validates modules before mutating, keeps the old module
    # on rejection, rolls back exactly, and records an audit trail.
    "pss-runtime": {
        "swap_accepted": 1.0,
        "invalid_rejected": 1.0,
        "invalid_keeps_state": 1.0,
        "rollback_exact": 1.0,
        "audit_present": 1.0,
    },
}


def protocol_expectations(
    *,
    bundle_stream: bool,
    budget_enabled: bool = True,
) -> dict[str, dict[str, str]]:
    """Return the expected status table for a protocol-suite run.

    ``identity_retention`` is only expected to pass when the stream carries
    harness-side identity labels (real bundles); on synthetic streams it is
    skipped.  ``label_accuracy`` is not pre-registered for the external
    baselines: it depends on the synthetic stream's separability, so it stays
    a reported measurement rather than a fixed expectation.
    """

    expectations = {
        system: dict(statuses)
        for system, statuses in EXPECTED_PROTOCOL_STATUS.items()
    }
    expectations["pss-runtime"]["budget"] = PASS if budget_enabled else SKIP
    if bundle_stream:
        expectations["pss-runtime"]["identity_retention"] = PASS
        expectations["pss-runtime"]["label_accuracy"] = SKIP
        expectations["e5-flat"]["label_accuracy"] = SKIP
        expectations["transformer-state"]["label_accuracy"] = SKIP
    else:
        expectations["pss-runtime"]["identity_retention"] = SKIP
        expectations["pss-runtime"]["label_accuracy"] = SKIP
        expectations["transformer-state"]["label_accuracy"] = SKIP
        # ``e5-flat`` label accuracy on the synthetic stream is left
        # unregistered: it depends on the synthetic class separation and the
        # per-seed codebook, so it is reported as a measurement instead.
    return expectations


def _systems_with_metrics(report: dict) -> dict[str, dict]:
    """Normalise aggregate and single-run reports to {system: statuses, metrics}."""

    summary = report.get("summary", {}).get("systems", {})
    normalised: dict[str, dict] = {}
    for name, system_summary in summary.items():
        statuses = dict(system_summary.get("statuses", {}))
        metrics = system_summary.get("metrics")
        if metrics is None:
            rows = [
                row
                for row in report.get("results", [])
                if row.get("system") == name
            ]
            metrics = {
                row["property"]: dict(row.get("metrics", {}))
                for row in rows
            }
        normalised[name] = {"statuses": statuses, "metrics": metrics}
    return normalised


def check_preregistration(
    report: dict,
    *,
    expectations: Mapping[str, Mapping[str, str]] | None = None,
) -> dict:
    """Compare an observed report against the fixed expectations.

    ``expectations`` defaults to :data:`EXPECTED_STATUS` (the original external
    suite).  Systems or properties not mentioned in the expectation tables are
    ignored, so the runner remains usable for ad-hoc suites.
    """

    expected_status = EXPECTED_STATUS if expectations is None else expectations
    observed = _systems_with_metrics(report)
    mismatches: list[str] = []

    for system, expected_properties in expected_status.items():
        if system not in observed:
            continue
        for prop, expected_status in expected_properties.items():
            actual = observed[system]["statuses"].get(prop)
            if actual is None:
                mismatches.append(
                    f"{system}/{prop}: expected {expected_status}, property was not run"
                )
            elif actual != expected_status:
                mismatches.append(
                    f"{system}/{prop}: expected {expected_status}, got {actual}"
                )

    for system, expected_metrics in EXPECTED_CONTRACT_METRICS.items():
        if system not in observed:
            continue
        actual_metrics = observed[system]["metrics"].get("contract", {})
        for metric, expected_value in expected_metrics.items():
            actual_value = actual_metrics.get(metric)
            if actual_value is None:
                mismatches.append(
                    f"{system}/contract.{metric}: expected {expected_value:.3f}, missing"
                )
            elif abs(float(actual_value) - float(expected_value)) > 1e-9:
                mismatches.append(
                    f"{system}/contract.{metric}: expected {expected_value:.3f}, "
                    f"got {float(actual_value):.3f}"
                )

    return {"ok": not mismatches, "mismatches": mismatches}
