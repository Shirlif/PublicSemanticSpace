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


def two_way_anova(
    records,
    metric: str,
    factor_a: str = "expert_quality",
    factor_b: str = "architecture",
) -> dict:
    """Balanced two-way ANOVA with variance components.

    ``records`` is a list of dicts with ``factor_a``, ``factor_b`` and ``metric``.
    Every (A, B) cell must have the same number of replicates; otherwise the
    caller should aggregate or drop cells.
    """

    from scipy import stats as sps

    cells: dict[tuple, list[float]] = {}
    for row in records:
        key = (row[factor_a], row[factor_b])
        cells.setdefault(key, []).append(float(row[metric]))
    if not cells:
        raise ValueError("no records")
    n = len(next(iter(cells.values())))
    if any(len(values) != n for values in cells.values()):
        raise ValueError("unbalanced cells; equal replicates per cell required")

    a_levels = sorted({key[0] for key in cells})
    b_levels = sorted({key[1] for key in cells})
    a, b = len(a_levels), len(b_levels)
    if len(cells) != a * b:
        raise ValueError("missing cells; balanced full factorial required")

    y = np.asarray([v for values in cells.values() for v in values], dtype=float)
    grand = float(y.mean())
    cell_means = {key: float(np.mean(values)) for key, values in cells.items()}
    mean_a = {lv: float(np.mean([cell_means[(lv, bv)] for bv in b_levels])) for lv in a_levels}
    mean_b = {lv: float(np.mean([cell_means[(av, lv)] for av in a_levels])) for lv in b_levels}

    ss_a = n * b * sum((mean_a[lv] - grand) ** 2 for lv in a_levels)
    ss_b = n * a * sum((mean_b[lv] - grand) ** 2 for lv in b_levels)
    ss_ab = n * sum(
        (cell_means[(av, bv)] - mean_a[av] - mean_b[bv] + grand) ** 2
        for av in a_levels
        for bv in b_levels
    )
    ss_e = sum(
        (float(v) - cell_means[key]) ** 2 for key, values in cells.items() for v in values
    )
    ss_t = ss_a + ss_b + ss_ab + ss_e

    df_a, df_b, df_ab = a - 1, b - 1, (a - 1) * (b - 1)
    df_e = a * b * (n - 1)
    ms_a = ss_a / df_a if df_a else 0.0
    ms_b = ss_b / df_b if df_b else 0.0
    ms_ab = ss_ab / df_ab if df_ab else 0.0
    ms_e = ss_e / df_e if df_e else 0.0

    def _f(ms):
        if ms_e <= 1e-15:
            return 0.0 if ms <= 1e-15 else float("inf")
        return float(ms / ms_e)

    f_a, f_b, f_ab = _f(ms_a), _f(ms_b), _f(ms_ab)
    p_a = float(sps.f.sf(f_a, df_a, df_e)) if df_a and df_e else 1.0
    p_b = float(sps.f.sf(f_b, df_b, df_e)) if df_b and df_e else 1.0
    p_ab = float(sps.f.sf(f_ab, df_ab, df_e)) if df_ab and df_e else 1.0

    # Random-effects variance components (balanced design).
    var_a = max(0.0, (ms_a - ms_e) / (b * n))
    var_b = max(0.0, (ms_b - ms_e) / (a * n))
    var_ab = max(0.0, (ms_ab - ms_e) / n)
    var_e = max(0.0, ms_e)
    total_var = var_a + var_b + var_ab + var_e
    shares = {
        "expert": var_a / total_var if total_var else 0.0,
        "architecture": var_b / total_var if total_var else 0.0,
        "interaction": var_ab / total_var if total_var else 0.0,
        "error": var_e / total_var if total_var else 0.0,
    }

    return {
        "metric": metric,
        "n_per_cell": int(n),
        "levels": {factor_a: a_levels, factor_b: b_levels},
        "SS": {"A": ss_a, "B": ss_b, "AB": ss_ab, "error": ss_e, "total": ss_t},
        "df": {"A": df_a, "B": df_b, "AB": df_ab, "error": df_e},
        "MS": {"A": ms_a, "B": ms_b, "AB": ms_ab, "error": ms_e},
        "F": {"A": f_a, "B": f_b, "AB": f_ab},
        "p": {"A": p_a, "B": p_b, "AB": p_ab},
        "eta2": {
            "A": ss_a / ss_t if ss_t else 0.0,
            "B": ss_b / ss_t if ss_t else 0.0,
            "AB": ss_ab / ss_t if ss_t else 0.0,
        },
        "R2": (ss_a + ss_b + ss_ab) / ss_t if ss_t else 0.0,
        "variance": {"expert": var_a, "architecture": var_b, "interaction": var_ab, "error": var_e},
        "variance_share": shares,
    }
