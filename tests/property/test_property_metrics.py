import numpy as np
import pytest

from pss.metrics.property import (
    audit_completeness,
    bootstrap_ci,
    budget_adherence,
    budget_adherence_rate,
    conflict_detection_rate,
    cost_attribution_error,
    degradation_correctness,
    downtime,
    forgetting_rate,
    interference,
    locality_violations,
    mean_swap_latency,
    monotonicity_rate,
    paired_differences,
    percentile,
    propagation_correctness,
    resolution_correctness,
    rollback_exactness,
    routing_correctness,
    safety_rejection_rate,
    swap_success_rate,
    tost_equivalence,
    update_success_rate,
    variance_ratio,
    zero_shot_compatibility,
)


def test_swap_metrics():
    assert swap_success_rate([True, True, False]) == pytest.approx(2 / 3)
    assert zero_shot_compatibility(["a", "b"], ["a", "a"]) == 0.5
    assert interference({"vision": 0.9, "tactile": 0.5}, {"vision": 0.8, "tactile": 0.5}) == pytest.approx(0.1)
    assert rollback_exactness("abc", "abc") == 1.0
    assert rollback_exactness("abc", "abd") == 0.0
    assert downtime(1, 100) == 0.01
    assert routing_correctness(9, 10) == 0.9
    assert safety_rejection_rate(3, 3) == 1.0
    assert mean_swap_latency([1.0, 3.0]) == 2.0


def test_kb_metrics():
    assert update_success_rate([True, True, True]) == 1.0
    assert locality_violations(0) == 0
    assert forgetting_rate([True, True, False], [True, False, False]) == pytest.approx(1 / 3)
    assert propagation_correctness([True, False, True]) == pytest.approx(2 / 3)
    assert conflict_detection_rate(4, 5) == 0.8
    assert resolution_correctness(5, 5) == 1.0
    assert audit_completeness(10, 10) == 1.0


def test_compute_metrics():
    assert budget_adherence(10, 10) == 1.0
    assert budget_adherence(11, 10) == 0.0
    assert budget_adherence_rate([(10, 10), (11, 10)]) == 0.5
    assert percentile([0, 1, 2, 3, 4], 50) == 2.0
    assert monotonicity_rate([0.5, 0.6, 0.6, 0.7]) == 1.0
    assert monotonicity_rate([0.5, 0.6, 0.55, 0.54]) == pytest.approx(1 / 3)
    assert degradation_correctness(9, 10) == 0.9
    assert cost_attribution_error([1.0, 2.0], 3.0) == 0.0
    assert cost_attribution_error([1.0, 1.0], 3.0) == pytest.approx(1 / 3)


def test_stats_helpers():
    values = [1.0, 2.0, 3.0, 4.0]
    lo, hi = bootstrap_ci(values, n_boot=200, seed=0)
    assert lo <= np.mean(values) <= hi
    diffs = paired_differences([2.0, 3.0], [1.0, 1.0])
    assert np.allclose(diffs, [1.0, 2.0])
    assert variance_ratio([1.0, 2.0, 3.0], [0.0, 0.1, 0.2]) > 10
    result = tost_equivalence([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], delta=0.05, n_boot=100)
    assert result["equivalent"] is True
