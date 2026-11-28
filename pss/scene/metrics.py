"""3D relation labels and metrics for the scene-model benchmark.

The labels are computed on **true 3D poses**; the metrics compare them with
queries computed on the **estimated 3D poses**.  This is the opposite of the
1D boundary benchmark, which thresholded a projected pair feature.
"""

from __future__ import annotations

import numpy as np


def axis_aligned(pa: np.ndarray, pb: np.ndarray, eps_xy: float) -> bool:
    d = np.asarray(pb, dtype=np.float64) - np.asarray(pa, dtype=np.float64)
    return bool(abs(d[0]) <= eps_xy or abs(d[1]) <= eps_xy)


def near_boundary(pa: np.ndarray, pb: np.ndarray, eps_xy: float) -> bool:
    d = np.abs(np.asarray(pb, dtype=np.float64) - np.asarray(pa, dtype=np.float64))
    lo, hi = 0.5 * eps_xy, 1.5 * eps_xy
    return bool(lo <= d[0] <= hi or lo <= d[1] <= hi)


def same_level(pa: np.ndarray, pb: np.ndarray, ra: float, rb: float, eps_z: float) -> bool:
    za = float(np.asarray(pa)[2]) - float(ra)
    zb = float(np.asarray(pb)[2]) - float(rb)
    return bool(abs(za - zb) <= eps_z)


def contact(pa: np.ndarray, pb: np.ndarray, ra: float, rb: float, eps_c: float) -> bool:
    d = float(np.linalg.norm(np.asarray(pa) - np.asarray(pb)))
    return bool(d <= float(ra) + float(rb) + eps_c)


def collision(pa: np.ndarray, pb: np.ndarray, ra: float, rb: float) -> bool:
    d = float(np.linalg.norm(np.asarray(pa) - np.asarray(pb)))
    return bool(d < float(ra) + float(rb))


def supported(p: np.ndarray, radius: float, table_z: float, eps_s: float) -> bool:
    return bool(abs((float(np.asarray(p)[2]) - float(radius)) - float(table_z)) <= eps_s)


def _pairs(ids):
    return [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]]


def pair_metrics(
    est: dict[str, np.ndarray],
    true: dict[str, np.ndarray],
    radii: dict[str, float],
    *,
    eps_xy: float,
    eps_z: float,
    eps_c: float,
    table_z: float = 0.0,
    eps_s: float = 0.05,
) -> dict:
    ids = [oid for oid in true if oid in est]
    pairs = _pairs(ids)
    rows = {"all": [], "near": []}
    rel = {name: [] for name in ("same_level", "contact", "collision")}
    for a, b in pairs:
        true_aligned = axis_aligned(true[a], true[b], eps_xy)
        est_aligned = axis_aligned(est[a], est[b], eps_xy)
        rows["all"].append((true_aligned, est_aligned))
        if near_boundary(true[a], true[b], eps_xy):
            rows["near"].append((true_aligned, est_aligned))
        ra, rb = radii.get(a, 0.35), radii.get(b, 0.35)
        rel["same_level"].append(
            (
                same_level(true[a], true[b], ra, rb, eps_z),
                same_level(est[a], est[b], ra, rb, eps_z),
            )
        )
        rel["contact"].append(
            (
                contact(true[a], true[b], ra, rb, eps_c),
                contact(est[a], est[b], ra, rb, eps_c),
            )
        )
        rel["collision"].append(
            (
                collision(true[a], true[b], ra, rb),
                collision(est[a], est[b], ra, rb),
            )
        )

    def _acc(items):
        return float(np.mean([t == e for t, e in items])) if items else 0.0

    out = {
        "overall": _acc(rows["all"]),
        "near_aligned": _acc(rows["near"]),
        "near_support": int(len(rows["near"])),
        "n_pairs": int(len(rows["all"])),
    }
    for name, items in rel.items():
        out[name] = _acc(items)
    out["support"] = float(
        np.mean(
            [
                supported(true[oid], radii.get(oid, 0.35), table_z, eps_s)
                == supported(est[oid], radii.get(oid, 0.35), table_z, eps_s)
                for oid in ids
            ]
        )
        if ids
        else 0.0
    )
    return out


def pose_errors(est: dict[str, np.ndarray], true: dict[str, np.ndarray]) -> dict:
    errors = [
        float(np.linalg.norm(np.asarray(est[oid]) - np.asarray(true[oid])))
        for oid in true
        if oid in est
    ]
    if not errors:
        return {"median": 0.0, "p90": 0.0, "mean": 0.0}
    arr = np.asarray(errors)
    return {
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(arr.mean()),
    }


def binary_ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(probs) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += abs(conf - acc) * mask.mean()
    return float(ece)


def mean_std(rows: list[dict], key: str) -> dict:
    vals = [float(r[key]) for r in rows if key in r]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "n": len(vals),
    }


__all__ = [
    "axis_aligned",
    "binary_ece",
    "collision",
    "contact",
    "mean_std",
    "near_boundary",
    "pair_metrics",
    "pose_errors",
    "same_level",
    "supported",
]
