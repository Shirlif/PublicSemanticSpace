"""Small statistics helpers for the §15 report.

We avoid a statsmodels dependency: the report needs slope + bootstrap CI,
TOST on the slope, and descriptive variance components.  For a full mixed
model, the plan points at lme4/statsmodels; this module is the reproducible
subset used by the generated report.
"""

from __future__ import annotations

import numpy as np


def slope(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or x.size != y.size:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def bootstrap_slope_ci(
    x,
    y,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values = np.empty(int(n_boot), dtype=float)
    n = x.size
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        values[i] = slope(x[idx], y[idx])
    lo = float(np.percentile(values, 100 * alpha / 2))
    hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return (lo, hi)


def tost_slope(x, y, delta: float = 0.05, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Equivalence test for a slope against zero."""
    observed = slope(x, y)
    lo, hi = bootstrap_slope_ci(x, y, n_boot=n_boot, alpha=0.10, seed=seed)
    return {
        "slope": observed,
        "ci90": [lo, hi],
        "delta": float(delta),
        "equivalent": bool(np.isfinite(lo) and np.isfinite(hi) and lo > -delta and hi < delta),
    }


def variance(values, ddof: int = 1) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= ddof:
        return 0.0
    return float(np.var(arr, ddof=ddof))


def variance_ratio(var_arch: float, var_expert: float) -> float | None:
    if var_expert <= 1e-12:
        return None
    return float(var_arch) / float(var_expert)


def paired_differences(treated, control) -> np.ndarray:
    t = np.asarray(list(treated), dtype=float)
    c = np.asarray(list(control), dtype=float)
    if t.size != c.size:
        raise ValueError("paired sequences must have the same length")
    return t - c


def mean_ci(values, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": None, "ci": [None, None]}
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        means[i] = float(rng.choice(arr, size=arr.size, replace=True).mean())
    return {
        "mean": float(arr.mean()),
        "ci": [
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))),
        ],
    }
