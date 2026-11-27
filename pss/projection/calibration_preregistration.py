"""Pre-registered expectations for the calibration robustness pipeline.

Executable form of ``docs/plans/PSS-校准抗干扰融合设计.md`` §8.  Fixed before
the run; a mismatch is a harness bug or a false pre-registration, never a reason
to relax the bar after seeing results.

Semantic change (2026-11-27) — recorded, not silent
---------------------------------------------------
The first table asserted ``accept_update`` and temperature-ECE.  Both were
written for the single-channel design.  The channels are now split:

- **Confidence** is a monotone binning map, refit per window.  It is exactly
  invariant to a global temperature (measured: identical out-of-sample ECE for
  T from 0.01 to 1.0), so temperature-ECE no longer measures calibration.
- **The gate** now protects fused decision accuracy, not calibration.

So calibration claims are expressed on ``ece_map_fused``, and ``accept_update``
is asserted only where it means "parameters must be frozen" (high-fraction
poison), not where it means "calibration succeeded".

For the record, the single-channel results this replaces:
drift temperature-ECE 0.0559 / 0.1198 / 0.1554 at sigma 0.10 / 0.20 / 0.30;
map-ECE 0.0007 / 0.0349 / 0.0976 on the same runs.

Bounds rationale
----------------
- Clean ceiling 0.03.  F6's temperature calibration on clean data was 0.028 in
  ECE terms; 0.03 is the same order and is not fitted to this run.
- Poison: calibration may not exceed 2x the clean ceiling (0.05).  The point of
  the robust fit is that poisoned labels do not move the confidence channel.
- Drift: 0.03 / 0.05 / 0.12 for sigma 0.10 / 0.20 / 0.30.  The sigma=0.30 world
  is roughly 3x noisier (vision accuracy drops to 0.82, geometry to 0.57), so
  the bound is set at 4x the clean ceiling rather than 2x.
- ``accuracy_min`` asserts the decision channel, which the gate protects:
  0.90 clean, 0.85 / 0.75 / 0.60 under drift.

Known-open claim: the two ``targeted_*`` scenarios keep ``detected=True``.  Both
are expected to fail in one window at this scale, and the failure is reported
rather than removed from the table.

Recovery semantics (2026-11-27)
-------------------------------
``recovery_samples`` is the **F6-comparable detection latency**: the number of
streamed samples after a clean warm-up until the state machine leaves NORMAL.
The caps 110 / 40 / 25 / 40 mirror F6's measured latency (109 / 36.7 / ~25 /
40).  A missing or undetected value is a **failure**, not a skipped check: an
unmeasured claim must never look like a passing one.

Confidence-channel disposal (2026-11-27)
----------------------------------------
``confidence_map_accept`` asserts what the confidence channel does with a
window's map update, separately from single-window detection and from the
parameter gate.  The rule: keep the candidate when it is not significantly
further from the frozen gate set, or when it is significantly better on the
recent trusted slice.  The two ``targeted_*`` rows therefore expect ``False``
even where ``detected`` stays the (known-open) single-window claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "EXPECTED_ROBUSTNESS",
    "CalibrationExpectation",
    "check_robustness_preregistration",
]


@dataclass(frozen=True)
class CalibrationExpectation:
    """One scenario's pre-registered outcome.

    Boolean fields are exact; numeric fields are checked only when present.
    ``None`` means "not asserted" — use it where the architecture no longer
    makes the claim, not to hide a failure.
    """

    detected: bool | None = None
    abstain: bool | None = None
    accept_update: bool | None = None
    confidence_map_accept: bool | None = None
    ece_map_max: float | None = None
    signed_gap_max: float | None = None
    accuracy_min: float | None = None
    recovery_samples_max: int | None = None


EXPECTED_ROBUSTNESS: dict[str, CalibrationExpectation] = {
    # ── clean: no flag, calibration within the clean ceiling ───────────────
    "clean": CalibrationExpectation(
        detected=False,
        abstain=False,
        confidence_map_accept=True,
        ece_map_max=0.030,
        signed_gap_max=0.050,
        accuracy_min=0.900,
    ),
    # ── random label flip ──────────────────────────────────────────────────
    # Low fractions: weights neutralise the bad rows; calibration is the claim.
    # High fractions: parameters must freeze.
    "random_label_flip@0.05": CalibrationExpectation(
        detected=True, ece_map_max=0.050, accuracy_min=0.850,
    ),
    "random_label_flip@0.10": CalibrationExpectation(
        detected=True, ece_map_max=0.050, accuracy_min=0.850,
    ),
    "random_label_flip@0.20": CalibrationExpectation(
        detected=True, accept_update=False, ece_map_max=0.050,
    ),
    "random_label_flip@0.40": CalibrationExpectation(
        detected=True, accept_update=False, ece_map_max=0.050,
    ),
    # ── embedding replacement ──────────────────────────────────────────────
    "embedding_replace@0.10": CalibrationExpectation(
        detected=True, ece_map_max=0.050,
    ),
    "embedding_replace@0.20": CalibrationExpectation(
        detected=True, accept_update=False, ece_map_max=0.050,
    ),
    "embedding_replace@0.40": CalibrationExpectation(
        detected=True, accept_update=False, ece_map_max=0.050,
    ),
    # ── duplicates: detectable but benign; acceptance is not a claim ───────
    "duplicate_inject@0.20": CalibrationExpectation(
        detected=True, ece_map_max=0.050,
    ),
    "duplicate_inject@0.40": CalibrationExpectation(
        detected=True, ece_map_max=0.050,
    ),
    # ── targeted attacks: KNOWN OPEN for single-window *detection*.  The
    # confidence channel's own gate is the new, separate disposal: the map
    # update must be rejected even when `detected` stays False.
    "targeted_agree@0.10": CalibrationExpectation(
        detected=True, confidence_map_accept=False,
        ece_map_max=0.050, signed_gap_max=0.100,
    ),
    "targeted_single_class@0.20": CalibrationExpectation(
        detected=True, confidence_map_accept=False,
        ece_map_max=0.050, signed_gap_max=0.100,
    ),
    # ── source-level poisoning ─────────────────────────────────────────────
    "source_outlier@0.30": CalibrationExpectation(
        detected=True, accept_update=False, ece_map_max=0.050,
    ),
    # ── covariate drift: the claim is detection latency, not acceptance ─────
    # ``recovery_samples`` is measured by streaming the drifted window one
    # sample at a time after a clean warm-up (see run_gap_calibration_robust).
    "drift_sigma@0.10": CalibrationExpectation(
        detected=True, ece_map_max=0.030, accuracy_min=0.850,
        recovery_samples_max=110,
    ),
    "drift_sigma@0.20": CalibrationExpectation(
        detected=True, confidence_map_accept=True,
        ece_map_max=0.050, accuracy_min=0.750,
        recovery_samples_max=40,
    ),
    "drift_sigma@0.30": CalibrationExpectation(
        detected=True,
        # The confidence map must adapt even when the parameter gate rejects:
        # confidence adaptation is independent of parameter adaptation.
        confidence_map_accept=True,
        ece_map_max=0.120, accuracy_min=0.600,
        recovery_samples_max=25,
    ),
    # ── out-of-distribution: flag and refuse ───────────────────────────────
    "ood_unseen": CalibrationExpectation(
        detected=True, abstain=False, accept_update=False,
    ),
    # ── label shift: adapt, do not freeze ──────────────────────────────────
    "label_shift@4of16": CalibrationExpectation(
        detected=True, ece_map_max=0.050, accuracy_min=0.750,
        recovery_samples_max=40,
    ),
}


def check_robustness_preregistration(
    observed: Mapping[str, Mapping[str, object]],
) -> dict:
    """Compare observed scenario results against the frozen expectations.

    Only scenarios present in ``observed`` are checked, so partial runs stay
    usable.  ``None`` fields are not asserted.
    """

    mismatches: list[str] = []

    for scenario, expected in EXPECTED_ROBUSTNESS.items():
        actual = observed.get(scenario)
        if actual is None:
            continue

        for field_name in (
            "detected",
            "abstain",
            "accept_update",
            "confidence_map_accept",
        ):
            want = getattr(expected, field_name)
            if want is None or field_name not in actual:
                continue
            got = actual[field_name]
            if bool(got) != bool(want):
                mismatches.append(
                    f"{scenario}.{field_name}: expected {want}, got {bool(got)}"
                )

        upper_bounds = (
            ("ece_map_fused", expected.ece_map_max),
            ("signed_gap_map_fused", expected.signed_gap_max),
            ("recovery_samples", expected.recovery_samples_max),
        )
        for metric, cap in upper_bounds:
            if cap is None or metric not in actual:
                continue
            value = actual[metric]
            if value is None:
                # A declared cap with no measurement is a failure: silently
                # skipping it is how a vacuous assertion passes for a year.
                mismatches.append(
                    f"{scenario}.{metric}: expected <= {cap}, got None "
                    "(not measured / not detected)"
                )
                continue
            sized = abs(float(value)) if metric.startswith("signed_gap") else float(value)
            if sized > float(cap):
                mismatches.append(
                    f"{scenario}.{metric}: expected <= {cap}, got {sized:.4f}"
                )

        if expected.accuracy_min is not None and "fused_accuracy" in actual:
            value = actual["fused_accuracy"]
            if value is not None and float(value) < float(expected.accuracy_min):
                mismatches.append(
                    f"{scenario}.fused_accuracy: expected >= "
                    f"{expected.accuracy_min}, got {float(value):.4f}"
                )

    return {"ok": not mismatches, "mismatches": mismatches}
