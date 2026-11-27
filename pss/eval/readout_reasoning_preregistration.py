"""Pre-registered expectations for the readout/reasoning optimisation plan.

Executable form of ``docs/plans/PSS-读出与推理优化执行方案.md`` §2 and §7.
Fixed before the run; a mismatch is a harness bug or a false pre-registration,
never a reason to relax the bar after seeing results.

Why deltas, not absolute accuracies, for readout arms
-----------------------------------------------------
Absolute accuracy depends on the split (views 0-5 / 6 / 7) and on which
checkpoint produced the projection.  An absolute ``accuracy_min`` for a new
readout arm would partly measure the split, not the arm.  Readout arms are
therefore asserted on the **delta vs the matched MLP readout on the same
representation**, which is what §5.6 defines as the reasoning loss.  The
primary value is still recorded for every arm; it is just not asserted.

Composition arms keep absolute bars: the P5b task has a known ceiling of 1.000
and a known floor near 0.39, so the scale is fixed.

Operating point, not AUROC alone
--------------------------------
§5.5 showed AUROC can be inflated by rejecting everything.  Every open-set
claim is therefore paired with ``unseen_reject_at_95`` (unseen reject rate at a
95% seen-accept operating point), mirroring
``pss/eval/scripts/run_openset_scoring_diag.py``.

Missing measurements
--------------------
An arm that is present in the observed table but did not measure one of its
declared bars is a **failure**, not a skipped check.  Partial runs are still
usable: an arm that is absent entirely is not checked.  This split is what
lets P0/P1 stage results be checked before the later arms exist, without
letting an unmeasured claim pass.

Threshold calibration note (2026-11-27, recorded after A0)
----------------------------------------------------------
``gate_decision`` uses a 0.97 gate-recall threshold that meant "the gate is
nearly lossless".  That threshold was **not** derived from the P0 target
(closed-set >= MLP - 0.5 pt).  A0 measured production-gate recall at 0.953
(top-3) and 0.971 (top-5), against a prototype readout of 0.833 and an MLP of
0.866 -- so even the top-3 ceiling clears the target by 8.7 points.

The threshold and its label are kept unchanged (a bar is never relaxed after
seeing results).  The consequence is recorded here instead: A2 runs k=5 as the
pre-registered arm and k=3 as an exploratory arm; neither is allowed to call
itself a win without the open-set bars above.

A4 arms (2026-11-27, registered *before* the fusion run)
--------------------------------------------------------
P1 found that per-modality score fusion destroys the Mahalanobis gain:
``concat_mahalanobis`` 0.930 closed-set, ``proto_mahalanobis`` 0.763 on the
same scorer.  A4 therefore tests "mix before discriminate" directly.  These
arms had never been run when the bars below were written.

``lda_per_modality_fused`` is a **negative control** with an upper bound: the
claim is that score-level fusion stays far behind, so if it unexpectedly
reaches the MLP it invalidates the P1 explanation rather than passing.

A2 arm (2026-11-27, registered *before* the rerank run)
-------------------------------------------------------
``rerank_bilinear`` is a learned metric over the A4 gate's candidates.  Its
absolute bar is set from the P2 A4 result: ``lda_concat_whiten`` reached 0.948,
so the reranker must reach 0.955 to count as a gain.  The gate's top-k recall
is reported alongside; a reranker cannot exceed it.  ``mlp_whiten_concat`` has
no bar — it is a reference for "what a full learned head gets", not a claim.

A5 arms (2026-11-27, registered *before* the incremental run)
-------------------------------------------------------------
A5 tests whether refreshing the pooled covariance removes the forgetting A4
measured (−1.6 pt over 50 added classes with a frozen precision).  The claim is
about **forgetting**, not absolute accuracy, so these arms use
``forgetting_max``: ``incr_frozen`` may forget up to 5 pt (it is the baseline),
``incr_refit_25`` and ``incr_refit_all`` must forget at most 1 pt.  The
``batch_joint`` row is the joint-training reference and carries no bar.

Calibration note (recorded after the run, bar unchanged)
--------------------------------------------------------
``incr_refit_25`` / ``incr_refit_all`` missed the 1 pt bar at 0.0182 / 0.0130.
``incr_refit_all`` is the same model as ``batch_joint`` (same data, same
covariance), and ``batch_joint`` also loses 0.0130 on the base classes, so
0.0130 is the **class-dilution floor**, not a refresh error.  The 1 pt bar was
tighter than the achievable floor and is left unchanged; the near-miss is
reported as such.

g arms (2026-11-27, registered *before* the g audit run)
--------------------------------------------------------
``docs/plans/PSS-g因子优化实验计划.md`` §3.1 defines four asserted arms and one
reference.  The claim is about the **worst cell**, not the mean, so the metric
is ``worst_cell_acc``: the minimum exact-4 over the factor cells of the audit
(pair type for the B line, the pre-registered invariance-sensitive split for
the C line).

- ``g_response_combination`` must reach ``worst_cell_acc >= 0.90`` and be
  within 0.03 of full coverage.
- ``g_response_single`` is a **negative control with an upper bound**: if
  single-axis augmentation reaches full coverage, H2 (single axis is not
  enough) is false and the "pair-type combination" rule must be withdrawn.
- ``g_invariance_matched`` / ``g_invariance_mismatched`` are read on the
  absolute-position split, the one where the earlier run measured 0.856 vs
  0.367.  The relative-position split is not invariance-sensitive and is
  reported but never fed to these two bars.
- ``g_a_full`` is the full-coverage reference and carries no bar.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "BARS",
    "DECISION_RULES",
    "EXPECTED",
    "Expectation",
    "check_readout_reasoning_preregistration",
    "coverage_decision",
    "gate_decision",
]


# Global bars from the plan, kept separate from per-arm predictions so a later
# stage can quote one number without reconstructing it from an arm row.
BARS = {
    "readout_closed_minus_mlp": 0.005,   # may trail the MLP by at most 0.5 pt
    "readout_auroc_min": 0.990,
    "readout_unseen_reject_at_95": 0.950,
    "hot_update_ms_max": 1.0,
    "seed_std_max": 0.010,
    "composition_ood_min": 0.950,
    "relation_exact4_min": 0.850,
}


# §7 kill rules, in one place.  The two decisions that can be computed from P0
# diagnostics are executable below; the rest are applied when their stage runs.
DECISION_RULES = {
    "RERANK_CEILING_LOW": (
        "gate recall@3 < 0.97 and recall@5 < 0.97 -> A2 cannot beat its gate; "
        "skip rerank and go to A3 distillation"
    ),
    "SCORER_NO_GAIN": (
        "proto_mahalanobis.auroc < 0.95 -> the scorer is not the bottleneck; "
        "report the negative result and go to A3"
    ),
    "DISTILL_NO_GAIN": (
        "distill trails the MLP and misses the AUROC bar -> co-adaptation is "
        "not recoverable by distillation on this task"
    ),
    "COVERAGE_SATURATED": (
        "OOD < 0.80 at full region coverage -> the bottleneck is predicate "
        "quality / search, not data"
    ),
    "PREDICATE_NO_GAIN": (
        "relation-algebra terms do not lift held-out exact-4 -> the relation "
        "head is memory, not algebra"
    ),
    "ASTAR_NO_GAIN": (
        "A* does not beat BFS on steps/invalid -> keep BFS, do not add complexity"
    ),
}


@dataclass(frozen=True)
class Expectation:
    """One arm's pre-registered outcome.

    Lower bounds and upper bounds are checked only when declared.  ``None``
    means "not asserted" — use it where the plan makes no claim, never to hide
    a failure.
    """

    # -- readout arms -------------------------------------------------------
    accuracy_min: float | None = None
    accuracy_delta_vs_mlp_min: float | None = None
    accuracy_delta_vs_mlp_max: float | None = None
    auroc_min: float | None = None
    unseen_reject_at_95_min: float | None = None
    hot_update_ms_max: float | None = None
    seed_std_max: float | None = None
    # -- composition arms ---------------------------------------------------
    id_success_min: float | None = None
    ood_success_min: float | None = None
    ood_delta_vs_probe_mlp_min: float | None = None
    relation_exact4_min: float | None = None
    invalid_max: float | None = None
    # -- incremental A5 arms ------------------------------------------------
    forgetting_max: float | None = None
    added_acc_min: float | None = None
    # -- g-factor arms (coverage / invariance) ------------------------------
    worst_cell_acc_min: float | None = None
    worst_cell_acc_max: float | None = None
    delta_vs_full_min: float | None = None
    delta_vs_full_max: float | None = None


EXPECTED: dict[str, Expectation] = {
    # ── readout: the baseline is allowed to trail the MLP by ~2-3 pt ────────
    "proto": Expectation(
        accuracy_delta_vs_mlp_min=-0.030,
        auroc_min=0.990,
        unseen_reject_at_95_min=0.950,
        hot_update_ms_max=0.5,
        seed_std_max=0.010,
    ),
    # §5.5: same representation, Mahalanobis 0.999 vs max-softmax 0.935.
    # Closed-set may move a little; the claim is the open-set ordering.
    "proto_mahalanobis": Expectation(
        accuracy_delta_vs_mlp_min=-0.010,
        auroc_min=0.995,
        unseen_reject_at_95_min=0.980,
        hot_update_ms_max=2.0,
    ),
    "proto_knn": Expectation(
        accuracy_delta_vs_mlp_min=-0.010,
        auroc_min=0.990,
        unseen_reject_at_95_min=0.950,
        hot_update_ms_max=5.0,
    ),
    # A reranker may only reorder the gate's candidates, so it must at least
    # match the MLP; its ceiling is the gate recall measured by A0.
    "proto_rerank": Expectation(
        accuracy_delta_vs_mlp_min=0.000,
        auroc_min=0.990,
        unseen_reject_at_95_min=0.950,
        hot_update_ms_max=1.0,
        seed_std_max=0.010,
    ),
    # §5.6: naive joint finetuning was harmful (0.848 vs 0.876).  Distillation
    # must not repeat that: the bar is near-parity, not a win.
    "distill": Expectation(
        accuracy_delta_vs_mlp_min=-0.005,
        auroc_min=0.995,
        unseen_reject_at_95_min=0.950,
        hot_update_ms_max=1.0,
        seed_std_max=0.010,
    ),
    # Constrained reliability weights only; §5.3's free gate collapsed.
    "proto_calibrated": Expectation(
        accuracy_delta_vs_mlp_min=-0.015,
        auroc_min=0.990,
        unseen_reject_at_95_min=0.950,
        hot_update_ms_max=1.0,
    ),
    # ── A4: mix across modalities before discriminating ─────────────────────
    # LDA = pooled-covariance Mahalanobis.  The bar is to beat the MLP, not
    # merely to match it: P1's exploratory arm was +5.2.
    "lda_concat": Expectation(
        accuracy_delta_vs_mlp_min=0.000,
        accuracy_delta_vs_mlp_max=0.100,
        auroc_min=0.995,
        unseen_reject_at_95_min=0.980,
        hot_update_ms_max=2.0,
        seed_std_max=0.010,
    ),
    "lda_concat_whiten": Expectation(
        accuracy_delta_vs_mlp_min=0.000,
        auroc_min=0.995,
        unseen_reject_at_95_min=0.980,
        hot_update_ms_max=2.0,
    ),
    "lda_concat_weighted": Expectation(
        accuracy_delta_vs_mlp_min=0.000,
        auroc_min=0.995,
        unseen_reject_at_95_min=0.980,
        hot_update_ms_max=2.0,
    ),
    "lda_subset_dropout": Expectation(
        accuracy_delta_vs_mlp_min=0.000,
        auroc_min=0.995,
        unseen_reject_at_95_min=0.980,
        hot_update_ms_max=2.0,
    ),
    # Negative control: score-level fusion must stay well behind the MLP.
    "lda_per_modality_fused": Expectation(
        accuracy_delta_vs_mlp_max=-0.050,
        auroc_min=0.990,
        hot_update_ms_max=2.0,
    ),
    # ── A2: learned rerank over the A4 gate ────────────────────────────────
    "rerank_bilinear": Expectation(
        accuracy_min=0.955,
        auroc_min=0.995,
        hot_update_ms_max=2.0,
    ),
    # Reference only: no bar.
    "mlp_whiten_concat": Expectation(),
    # ── A5: incremental class addition under covariance policies ───────────
    "incr_frozen": Expectation(
        forgetting_max=0.050,
        added_acc_min=0.850,
    ),
    "incr_refit_25": Expectation(
        forgetting_max=0.010,
        added_acc_min=0.850,
    ),
    "incr_refit_all": Expectation(
        forgetting_max=0.010,
        added_acc_min=0.850,
    ),
    # Reference only: no bar.
    "batch_joint": Expectation(),
    "cosine_whiten": Expectation(),
    # ── g: coverage response and invariance matching ───────────────────────
    # Bars from `PSS-g因子优化实验计划.md` §3.1, fixed before the audit run.
    "g_response_combination": Expectation(
        worst_cell_acc_min=0.90,
        delta_vs_full_min=-0.03,
    ),
    # Negative control: a single missing axis must stay well behind full
    # coverage.  If it does not, H2 is false and the combination rule is void.
    "g_response_single": Expectation(delta_vs_full_max=-0.10),
    "g_invariance_matched": Expectation(worst_cell_acc_min=0.80),
    "g_invariance_mismatched": Expectation(worst_cell_acc_max=0.60),
    # Reference only: no bar.
    "g_a_full": Expectation(),
    # ── composition: absolute bars on the fixed P5b scale ───────────────────
    "pss_learned": Expectation(
        id_success_min=0.950,
        ood_success_min=0.950,
        relation_exact4_min=0.950,
    ),
    # A/B/C head: measured 0.571 exact-4 -> 0.682 OOD.  These bars assert the
    # known baseline, not an improvement; B1 must lift the first number.
    "pss_learned_heldout": Expectation(
        id_success_min=0.800,
        ood_success_min=0.600,
        relation_exact4_min=0.550,
    ),
    "probe_linear": Expectation(invalid_max=0.050),
    "probe_mlp": Expectation(invalid_max=0.050),
    "transformer": Expectation(id_success_min=0.950, invalid_max=0.050),
    "pss_astar": Expectation(
        id_success_min=0.950,
        ood_success_min=0.950,
    ),
}


_UPPER_BOUNDS = {
    "hot_update_ms_max": "hot_update_ms",
    "invalid_max": "invalid_rate",
    "seed_std_max": "seed_std",
    "accuracy_delta_vs_mlp_max": "accuracy_delta_vs_mlp",
    "forgetting_max": "forgetting",
    "worst_cell_acc_max": "worst_cell_acc",
    "delta_vs_full_max": "delta_vs_full",
}

_LOWER_BOUNDS = {
    "accuracy_min": "test_acc",
    "accuracy_delta_vs_mlp_min": "accuracy_delta_vs_mlp",
    "added_acc_min": "added_acc",
    "auroc_min": "auroc",
    "unseen_reject_at_95_min": "unseen_reject_at_95",
    "id_success_min": "id_success",
    "ood_success_min": "ood_success",
    "ood_delta_vs_probe_mlp_min": "ood_delta_vs_probe_mlp",
    "relation_exact4_min": "relation_exact4",
    "worst_cell_acc_min": "worst_cell_acc",
    "delta_vs_full_min": "delta_vs_full",
}


def check_readout_reasoning_preregistration(
    observed: Mapping[str, Mapping[str, object]],
) -> dict:
    """Compare observed arm results against the frozen expectations.

    ``observed`` maps arm name -> metric dict.  Arms absent from ``observed``
    are not checked, so stage-by-stage runs stay usable.  An arm that is
    present but missing a declared metric is a mismatch: an unmeasured claim
    must never look like a passing one.
    """

    mismatches: list[str] = []
    checked = 0

    for arm, expected in EXPECTED.items():
        actual = observed.get(arm)
        if actual is None:
            continue
        checked += 1

        for field_name, metric in _UPPER_BOUNDS.items():
            cap = getattr(expected, field_name)
            if cap is None:
                continue
            if metric not in actual or actual[metric] is None:
                mismatches.append(
                    f"{arm}.{metric}: expected <= {cap}, got none "
                    "(not measured)"
                )
                continue
            value = float(actual[metric])
            if value > float(cap):
                mismatches.append(
                    f"{arm}.{metric}: expected <= {cap}, got {value:.4f}"
                )

        for field_name, metric in _LOWER_BOUNDS.items():
            floor = getattr(expected, field_name)
            if floor is None:
                continue
            if metric not in actual or actual[metric] is None:
                mismatches.append(
                    f"{arm}.{metric}: expected >= {floor}, got none "
                    "(not measured)"
                )
                continue
            value = float(actual[metric])
            if value < float(floor):
                mismatches.append(
                    f"{arm}.{metric}: expected >= {floor}, got {value:.4f}"
                )

    return {"ok": not mismatches, "mismatches": mismatches, "checked": checked}


def gate_decision(recall3: float, recall5: float) -> str:
    """Pre-registered A2 branch from A0's measured gate recall.

    A reranker cannot recover a class the gate never proposes, so the decision
    is made on the gate's ceiling, not on the reranker's own accuracy.
    """

    if float(recall3) >= 0.97:
        return "rerank_viable"
    if float(recall5) >= 0.97:
        return "rerank_viable_at_k5"
    return "skip_rerank_go_distill"


def coverage_decision(ood_at_full_coverage: float) -> str:
    """Pre-registered branch from B0's coverage curve at 100% coverage."""

    return (
        "coverage_binding"
        if float(ood_at_full_coverage) < 0.80
        else "coverage_sufficient"
    )
