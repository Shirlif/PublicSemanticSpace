"""Property-level metrics for the architecture-validation plan.

All functions are pure and dependency-light (numpy only).  They intentionally
measure *architecture contracts*, not task accuracy: swap success, locality,
rollback exactness, budget adherence, and so on.  Capability metrics stay in
``pss.metrics.entity`` / ``pss.metrics.relation``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _ratio(num: float, den: float, default: float = 0.0) -> float:
    return float(num) / float(den) if den else float(default)


def _mean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(arr.mean()) if arr.size else 0.0


# ── H: hot-swap ────────────────────────────────────────────────────────────


def swap_success_rate(successes: Sequence[bool]) -> float:
    return _ratio(sum(bool(x) for x in successes), len(successes))


def zero_shot_compatibility(predicted: Sequence, expected: Sequence) -> float:
    if len(predicted) != len(expected):
        raise ValueError("predicted and expected must have the same length")
    return _ratio(sum(p == e for p, e in zip(predicted, expected)), len(expected))


def interference(before: Mapping[str, float], after: Mapping[str, float]) -> float:
    """Max absolute change over shared keys (0 = no interference)."""
    keys = set(before) & set(after)
    if not keys:
        return 0.0
    return float(max(abs(float(after[k]) - float(before[k])) for k in keys))


def rollback_exactness(before_hash, after_hash) -> float:
    return 1.0 if before_hash == after_hash else 0.0


def downtime(rejected: int, total: int) -> float:
    return _ratio(rejected, total)


def routing_correctness(correct: int, total: int) -> float:
    return _ratio(correct, total)


def safety_rejection_rate(rejected: int, injected: int) -> float:
    return _ratio(rejected, injected, default=1.0)


def mean_swap_latency(latencies_ms: Sequence[float]) -> float:
    return _mean(latencies_ms)


# ── K: knowledge-base hot update ───────────────────────────────────────────


def update_success_rate(successes: Sequence[bool]) -> float:
    return _ratio(sum(bool(x) for x in successes), len(successes))


def locality_violations(changed_unrelated: int) -> int:
    return int(changed_unrelated)


def forgetting_rate(old_correct_before: Sequence[bool], old_correct_after: Sequence[bool]) -> float:
    """Fraction of facts that were correct before and wrong after."""
    if len(old_correct_before) != len(old_correct_after):
        raise ValueError("before/after sequences must have the same length")
    lost = sum(1 for b, a in zip(old_correct_before, old_correct_after) if b and not a)
    return _ratio(lost, len(old_correct_before))


def propagation_correctness(matches: Sequence[bool]) -> float:
    return _ratio(sum(bool(x) for x in matches), len(matches), default=1.0)


def conflict_detection_rate(detected: int, injected: int) -> float:
    return _ratio(detected, injected, default=1.0)


def resolution_correctness(correct: int, total: int) -> float:
    return _ratio(correct, total, default=1.0)


def audit_completeness(audited: int, changes: int) -> float:
    return _ratio(audited, changes, default=1.0)


def update_latency(latencies_ms: Sequence[float]) -> float:
    return _mean(latencies_ms)


# ── C: bounded inference ───────────────────────────────────────────────────


def budget_adherence(actual: float, bound: float) -> float:
    return 1.0 if float(actual) <= float(bound) else 0.0


def budget_adherence_rate(pairs: Sequence[tuple[float, float]]) -> float:
    """Pairs are ``(actual, bound)``."""
    return _ratio(sum(budget_adherence(a, b) for a, b in pairs), len(pairs), default=1.0)


def percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.percentile(arr, q)) if arr.size else 0.0


def monotonicity_rate(values: Sequence[float], tol: float = 1e-9) -> float:
    """Fraction of adjacent pairs where quality does not decrease."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return 1.0
    return float(np.mean(np.diff(arr) >= -float(tol)))


def degradation_correctness(correct: int, total: int) -> float:
    return _ratio(correct, total, default=1.0)


def cost_attribution_error(module_costs: Sequence[float], total_cost: float) -> float:
    if total_cost <= 0:
        return 0.0
    return abs(float(np.sum(module_costs)) - float(total_cost)) / float(total_cost)


def worst_case_bound(violations: Sequence[float]) -> float:
    return float(max(violations)) if violations else 0.0


# ── descriptive statistics ─────────────────────────────────────────────────


def summarize(values: Sequence[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "p99": None,
                "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def bootstrap_ci(
    values: Sequence[float],
    stat=np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats[i] = float(stat(sample))
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def paired_differences(treated: Sequence[float], control: Sequence[float]) -> np.ndarray:
    if len(treated) != len(control):
        raise ValueError("paired sequences must have the same length")
    return np.asarray(treated, dtype=float) - np.asarray(control, dtype=float)


def variance_ratio(arch_effects: Sequence[float], expert_effects: Sequence[float]) -> float:
    """Var(architecture effect) / Var(expert effect); used for PH5."""
    a = np.asarray(list(arch_effects), dtype=float)
    e = np.asarray(list(expert_effects), dtype=float)
    va = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
    ve = float(np.var(e, ddof=1)) if e.size > 1 else 0.0
    return va / max(ve, 1e-12)


def tost_equivalence(
    treated: Sequence[float],
    control: Sequence[float],
    delta: float,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Two one-sided tests via bootstrap CI on the paired difference."""
    diffs = paired_differences(treated, control)
    lo, hi = bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
    return {
        "delta": float(delta),
        "mean_diff": float(np.mean(diffs)) if diffs.size else float("nan"),
        "ci": [lo, hi],
        "equivalent": bool(lo > -float(delta) and hi < float(delta)),
    }


def scorecard(metrics: Mapping[str, float], thresholds: Mapping[str, float] | None = None) -> dict:
    """Attach pass/fail flags to a metric dictionary."""
    thresholds = thresholds or {}
    out = {}
    for name, value in metrics.items():
        entry = {"value": float(value)}
        if name in thresholds:
            entry["threshold"] = float(thresholds[name])
            entry["pass"] = bool(float(value) >= float(thresholds[name]))
        out[name] = entry
    return out
