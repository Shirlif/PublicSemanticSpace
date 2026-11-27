"""Coverage and matching accounting for the g factor.

Executable form of ``docs/plans/PSS-g因子优化实验计划.md`` §1: a task is audited
by splitting its test pool into mutually exclusive cells, then reporting each
cell's support, error and error contribution

.. math::

    E_{\\text{overall}} = \\sum_c \\frac{n_c}{N} e_c, \\qquad
    \\text{contribution}_c = \\frac{n_c}{N} e_c .

The helpers here are pure (no torch, no benchmark import) so the accounting can
be unit-tested and reused by every stage of ``run_g_audit.py``.

Two metrics are deliberately different:

- ``coverage_cells`` decomposes the *binary* error of a predicate, which is
  what localises a missing absence case (a zero delta must suppress the other
  predicate).
- ``worst_cell_exact4`` takes the minimum exact-4 over factor cells (default:
  pair type).  This is the primary metric of the response curves: a mean can
  hide the one cell the plan is trying to repair.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "cell_accuracy",
    "coverage_cells",
    "coverage_dashboard",
    "mean_std",
    "paired_bootstrap_ci",
    "worst_cell_exact4",
]


def coverage_cells(items: Iterable[tuple[str, float, int]]) -> list[dict]:
    """Aggregate ``(cell, correctness, support)`` into support/error/contribution.

    ``correctness`` is in ``[0, 1]`` and may be fractional when several seeds
    are pooled.  Cells are sorted by contribution, then error, so the first row
    is the best candidate for the next targeted augmentation.
    """

    agg: dict[str, dict[str, float]] = {}
    for cell, correct, support in items:
        support = int(support)
        if support <= 0:
            continue
        bucket = agg.setdefault(str(cell), {"support": 0, "correct": 0.0})
        bucket["support"] += support
        bucket["correct"] += float(correct) * support

    total = sum(bucket["support"] for bucket in agg.values())
    cells: list[dict] = []
    for cell, bucket in agg.items():
        error = 1.0 - bucket["correct"] / bucket["support"]
        cells.append(
            {
                "cell": cell,
                "support": int(bucket["support"]),
                "error": float(error),
                "contribution": float(bucket["support"] / total * error) if total else 0.0,
            }
        )
    cells.sort(key=lambda row: (-row["contribution"], -row["error"], row["cell"]))
    return cells


def coverage_dashboard(
    cells: Sequence[dict],
    expected_cells: Iterable[str] = (),
    min_support: int = 1,
    n_worst: int = 5,
) -> dict:
    """Dashboard for one coverage factor (plan §1.1 and §1.3)."""

    support = {row["cell"]: int(row["support"]) for row in cells}
    empty = [str(cell) for cell in expected_cells if support.get(str(cell), 0) == 0]
    supported = [row for row in cells if int(row["support"]) >= int(min_support)]
    worst = sorted(
        supported, key=lambda row: (-row["contribution"], -row["error"], row["cell"])
    )[: int(n_worst)]
    worst_error = sorted(
        supported, key=lambda row: (-row["error"], -row["contribution"], row["cell"])
    )[: int(n_worst)]
    total_error = float(sum(row["contribution"] for row in cells))
    worst_share = (
        float(worst[0]["contribution"] / total_error) if worst and total_error > 0 else 0.0
    )
    return {
        "cells": list(cells),
        "empty_cells": empty,
        "worst_cells": [row["cell"] for row in worst],
        "worst_error_cells": [row["cell"] for row in worst_error],
        "worst_cell": worst[0]["cell"] if worst else None,
        "worst_cell_error": float(worst[0]["error"]) if worst else None,
        "worst_contribution_share": worst_share,
        "total_support": int(sum(row["support"] for row in cells)),
        "total_error": total_error,
        "min_support": int(min_support),
    }


def cell_accuracy(
    rows: Iterable[dict],
    key_fn: Callable[[dict], str],
    *,
    exact_key: str = "exact",
) -> list[dict]:
    """Exact-4 (or binary) accuracy per cell, worst first."""

    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[str(key_fn(row))].append(float(bool(row[exact_key])))

    cells = []
    for cell, values in buckets.items():
        accuracy = float(np.mean(values))
        cells.append(
            {
                "cell": cell,
                "support": int(len(values)),
                "accuracy": accuracy,
                "error": 1.0 - accuracy,
            }
        )
    cells.sort(key=lambda row: (row["accuracy"], -row["support"], row["cell"]))
    return cells


def worst_cell_exact4(
    rows: Iterable[dict],
    key_fn: Callable[[dict], str],
    min_support: int = 1,
) -> dict:
    """Minimum exact-4 over cells with enough support to be meaningful."""

    cells = cell_accuracy(rows, key_fn)
    supported = [row for row in cells if int(row["support"]) >= int(min_support)]
    if not supported:
        return {"value": float("nan"), "cell": None, "cells": cells}
    worst = min(supported, key=lambda row: (row["accuracy"], -row["support"], row["cell"]))
    return {"value": float(worst["accuracy"]), "cell": worst["cell"], "cells": cells}


def paired_bootstrap_ci(
    differences: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI of the mean paired difference."""

    values = np.asarray(list(differences), dtype=float)
    if values.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, values.size, size=(int(n_boot), values.size))
    boots = values[idx].mean(axis=1)
    alpha = (1.0 - float(ci)) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return {
        "mean": float(values.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(values.size),
        "ci": float(ci),
    }


def mean_std(values: Sequence[float]) -> dict:
    """Mean and std, ignoring NaNs; empty input returns NaNs."""

    finite = [float(v) for v in values if not np.isnan(float(v))]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "n": len(finite),
    }
