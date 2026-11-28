"""Pure helpers for the architecture upper-bound review (U1-U4).

Executable form of ``docs/plans/PSS-架构上界复核与下一阶段.md`` §5-§6.
The module deliberately keeps metric definitions free of torch / data loading:
the three stage scripts own the pipelines, this module owns the arithmetic
that the pre-registered bars read.

Cell definition
---------------
A *pair type* is ``horizontal`` when ``|dy| <= eps``, ``vertical`` when
``|dx| <= eps``, and ``diagonal`` otherwise.  The B line calls these the
absence cells: a head trained without one type false-positives the predicates
of the missing axis (BCD's ``above/below`` on horizontal pairs).

``worst_cell_accuracy`` is the minimum exact-4 over the pair types that have
at least one test row.  It is the primary metric for U2 because a mean can
hide a cell that is entirely wrong.
"""

from __future__ import annotations

import numpy as np

PAIR_TYPES = ("horizontal", "vertical", "diagonal")
PREDICATES = ("left", "right", "above", "below")


def pair_labels(pa, pb, eps: float = 0.0) -> np.ndarray:
    """Return ``(..., 4)`` float32 bits for left/right/above/below.

    Order matches ``pss.eval.scripts.run_typed_relations._pair_labels``:
    ``[dx > eps, dx < -eps, dy < -eps, dy > eps]``.
    """

    pa = np.asarray(pa, dtype=np.float32)
    pb = np.asarray(pb, dtype=np.float32)
    dx = pb[..., 0] - pa[..., 0]
    dy = pb[..., 1] - pa[..., 1]
    return np.stack(
        [dx > eps, dx < -eps, dy < -eps, dy > eps], axis=-1
    ).astype(np.float32)


def pair_types(pa, pb, eps: float = 0.0) -> np.ndarray:
    """Vectorised ``horizontal`` / ``vertical`` / ``diagonal`` labels."""

    pa = np.asarray(pa, dtype=np.float32)
    pb = np.asarray(pb, dtype=np.float32)
    dx = np.abs(pb[..., 0] - pa[..., 0])
    dy = np.abs(pb[..., 1] - pa[..., 1])
    out = np.full(dx.shape, "diagonal", dtype=object)
    out[dy <= eps] = "horizontal"
    out[(dx <= eps) & (dy > eps)] = "vertical"
    return out


def exact4(y: np.ndarray, p: np.ndarray) -> float:
    """Fraction of rows where all four predicates match."""

    y = np.asarray(y).astype(bool)
    p = np.asarray(p).astype(bool)
    return float(np.all(y == p, axis=1).mean())


def cell_accuracy(
    types: np.ndarray, y: np.ndarray, p: np.ndarray
) -> dict[str, float]:
    """Per-pair-type exact-4, omitting types with no test rows."""

    types = np.asarray(types)
    y = np.asarray(y).astype(bool)
    p = np.asarray(p).astype(bool)
    rows = np.all(y == p, axis=1)
    out: dict[str, float] = {}
    for name in PAIR_TYPES:
        mask = types == name
        if int(mask.sum()):
            out[name] = float(rows[mask].mean())
    return out


def worst_cell_accuracy(
    types: np.ndarray, y: np.ndarray, p: np.ndarray
) -> float:
    """Minimum per-pair-type exact-4 (0.0 if any declared cell has no rows)."""

    cells = cell_accuracy(types, y, p)
    return min(cells.values()) if cells else 0.0


def absence_cell_error(
    types: np.ndarray, y: np.ndarray, p: np.ndarray, cell: str = "horizontal"
) -> float:
    """Error on one pair type (the missing isolation direction)."""

    cells = cell_accuracy(types, y, p)
    if cell not in cells:
        return 1.0
    return float(1.0 - cells[cell])


def delta_vs_full(arm_worst: float, full_worst: float) -> float:
    """Signed worst-cell gap; negative means the arm is behind full coverage."""

    return float(arm_worst) - float(full_worst)


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Min-max each coordinate to ``[0, 1]`` (CLEVR pixels -> unit square)."""

    points = np.asarray(points, dtype=np.float32)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (points - lo) / span


def snap_pairs(pa: np.ndarray, pb: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a pair with one axis made exactly equal, preserving the other.

    ``kind="horizontal"`` sets ``pb_y = pa_y``; ``kind="vertical"`` sets
    ``pb_x = pa_x``.  ``diagonal`` returns the original pair.
    """

    pa = np.asarray(pa, dtype=np.float32).copy()
    pb = np.asarray(pb, dtype=np.float32).copy()
    if kind == "horizontal":
        pb[..., 1] = pa[..., 1]
    elif kind == "vertical":
        pb[..., 0] = pa[..., 0]
    elif kind != "diagonal":
        raise ValueError(f"unknown pair kind {kind!r}")
    return pa, pb


__all__ = [
    "PAIR_TYPES",
    "PREDICATES",
    "absence_cell_error",
    "cell_accuracy",
    "delta_vs_full",
    "exact4",
    "normalize_points",
    "pair_labels",
    "pair_types",
    "snap_pairs",
    "worst_cell_accuracy",
]
