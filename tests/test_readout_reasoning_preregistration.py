"""Tests for the readout/reasoning pre-registration checker."""

from __future__ import annotations

from pss.eval.readout_reasoning_preregistration import (
    check_readout_reasoning_preregistration,
    coverage_decision,
    gate_decision,
)


def test_absent_arms_are_not_checked():
    result = check_readout_reasoning_preregistration({})
    assert result["ok"]
    assert result["checked"] == 0


def test_passing_readout_arm():
    observed = {
        "proto_mahalanobis": {
            "accuracy_delta_vs_mlp": -0.004,
            "auroc": 0.997,
            "unseen_reject_at_95": 0.990,
            "hot_update_ms": 1.4,
        }
    }
    result = check_readout_reasoning_preregistration(observed)
    assert result["ok"], result["mismatches"]
    assert result["checked"] == 1


def test_delta_below_bar_fails():
    observed = {
        "proto_rerank": {
            "accuracy_delta_vs_mlp": -0.012,
            "auroc": 0.995,
            "unseen_reject_at_95": 0.980,
            "hot_update_ms": 0.8,
            "seed_std": 0.002,
        }
    }
    result = check_readout_reasoning_preregistration(observed)
    assert not result["ok"]
    assert any("accuracy_delta_vs_mlp" in m for m in result["mismatches"])


def test_upper_bound_violation_fails():
    observed = {
        "proto_knn": {
            "accuracy_delta_vs_mlp": -0.002,
            "auroc": 0.995,
            "unseen_reject_at_95": 0.980,
            "hot_update_ms": 9.0,
        }
    }
    result = check_readout_reasoning_preregistration(observed)
    assert not result["ok"]
    assert any("hot_update_ms" in m for m in result["mismatches"])


def test_declared_bar_without_measurement_is_a_failure():
    observed = {
        "proto_rerank": {
            "accuracy_delta_vs_mlp": 0.002,
            "auroc": 0.995,
            # unseen_reject_at_95, hot_update_ms, seed_std are declared bars
        }
    }
    result = check_readout_reasoning_preregistration(observed)
    assert not result["ok"]
    missing = [m for m in result["mismatches"] if "not measured" in m]
    assert len(missing) == 3


def test_passing_composition_arm():
    observed = {
        "pss_learned": {
            "id_success": 1.000,
            "ood_success": 1.000,
            "relation_exact4": 1.000,
        }
    }
    result = check_readout_reasoning_preregistration(observed)
    assert result["ok"], result["mismatches"]


def test_composition_ood_below_bar_fails():
    observed = {
        "pss_learned": {
            "id_success": 0.990,
            "ood_success": 0.900,
            "relation_exact4": 1.000,
        }
    }
    result = check_readout_reasoning_preregistration(observed)
    assert not result["ok"]
    assert any("ood_success" in m for m in result["mismatches"])


def test_gate_decision_branches():
    assert gate_decision(0.980, 0.995) == "rerank_viable"
    assert gate_decision(0.960, 0.980) == "rerank_viable_at_k5"
    assert gate_decision(0.900, 0.950) == "skip_rerank_go_distill"


def test_coverage_decision_branches():
    assert coverage_decision(0.79) == "coverage_binding"
    assert coverage_decision(0.95) == "coverage_sufficient"


def test_a4_main_arm_passes():
    observed = {
        "lda_concat": {
            "accuracy_delta_vs_mlp": 0.052,
            "auroc": 0.997,
            "unseen_reject_at_95": 0.985,
            "hot_update_ms": 0.005,
            "seed_std": 0.0,
        }
    }
    result = check_readout_reasoning_preregistration(observed)
    assert result["ok"], result["mismatches"]


def test_negative_control_must_stay_behind():
    observed = {
        "lda_per_modality_fused": {
            "accuracy_delta_vs_mlp": -0.115,
            "auroc": 0.991,
            "hot_update_ms": 0.017,
        }
    }
    result = check_readout_reasoning_preregistration(observed)
    assert result["ok"], result["mismatches"]


def test_negative_control_fails_if_it_matches_the_mlp():
    observed = {
        "lda_per_modality_fused": {
            "accuracy_delta_vs_mlp": -0.010,
            "auroc": 0.991,
            "hot_update_ms": 0.017,
        }
    }
    result = check_readout_reasoning_preregistration(observed)
    assert not result["ok"]
    assert any("accuracy_delta_vs_mlp" in m for m in result["mismatches"])
