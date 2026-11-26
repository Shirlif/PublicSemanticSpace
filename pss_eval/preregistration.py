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

__all__ = [
    "EXPECTED_CONTRACT_METRICS",
    "EXPECTED_STATUS",
    "check_preregistration",
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
}


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


def check_preregistration(report: dict) -> dict:
    """Compare an observed report against the fixed expectations.

    Returns ``{"ok": bool, "mismatches": [str, ...]}``.  Systems or properties
    not mentioned in the expectation tables are ignored, so the runner remains
    usable for ad-hoc suites.
    """

    observed = _systems_with_metrics(report)
    mismatches: list[str] = []

    for system, expected_properties in EXPECTED_STATUS.items():
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
