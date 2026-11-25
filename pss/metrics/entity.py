"""Entity-layer metrics over a gt/detection assignment trace (S1/S2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

METRIC_REGISTRY: dict[str, dict] = {}


def metric(name: str, *, direction: str = "max", uses: tuple[str, ...] = ()):
    def decorator(fn: Callable) -> Callable:
        METRIC_REGISTRY[name] = {
            "fn": fn,
            "direction": direction,
            "uses": list(uses),
        }
        return fn

    return decorator


def _clean_trace(trace: pd.DataFrame) -> pd.DataFrame:
    cols = ["frame", "gt_id", "pred_id"]
    missing = [c for c in cols if c not in trace.columns]
    if missing:
        raise ValueError(f"trace missing columns: {missing}")
    out = trace[cols].dropna().copy()
    out["frame"] = out["frame"].astype(int)
    out["gt_id"] = out["gt_id"].astype(str)
    out["pred_id"] = out["pred_id"].astype(str)
    return out


def _gt_detections(gt_frames: pd.DataFrame) -> pd.DataFrame:
    if gt_frames is None:
        return pd.DataFrame(columns=["frame", "instance_id"])
    cols = ["frame", "instance_id"]
    missing = [c for c in cols if c not in gt_frames.columns]
    if missing:
        raise ValueError(f"gt_frames missing columns: {missing}")
    out = gt_frames[cols].dropna().copy()
    out["frame"] = out["frame"].astype(int)
    out["instance_id"] = out["instance_id"].astype(str)
    return out.drop_duplicates()


@metric("idf1", direction="max", uses=("gt", "detections", "assignments"))
def idf1(gt_frames: pd.DataFrame, trace: pd.DataFrame) -> float:
    """Identity F1 with a global bipartite GT/track matching (Ristani et al.)."""
    trace = _clean_trace(trace)
    gt = _gt_detections(gt_frames)
    gt_total = len(gt)
    pred_total = len(trace)
    if gt_total == 0 and pred_total == 0:
        return 1.0
    if trace.empty:
        return 0.0

    gt_ids = sorted(set(gt["instance_id"]) | set(trace["gt_id"]))
    pred_ids = sorted(set(trace["pred_id"]))
    co = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
    gt_index = {g: i for i, g in enumerate(gt_ids)}
    pred_index = {p: j for j, p in enumerate(pred_ids)}
    for gt_id, pred_id in zip(trace["gt_id"], trace["pred_id"]):
        co[gt_index[gt_id], pred_index[pred_id]] += 1.0

    rows, cols = linear_sum_assignment(-co)
    idtp = float(co[rows, cols].sum())
    idfn = max(0.0, gt_total - idtp)
    idfp = max(0.0, pred_total - idtp)
    denom = 2.0 * idtp + idfn + idfp
    return 2.0 * idtp / denom if denom > 0 else 1.0


@metric("id_retention", direction="max", uses=("assignments",))
def id_retention(trace: pd.DataFrame) -> float:
    trace = _clean_trace(trace)
    if trace.empty:
        return 0.0
    scores = []
    for _, group in trace.groupby("gt_id"):
        counts = group["pred_id"].value_counts()
        scores.append(float(counts.max()) / float(len(group)))
    return float(np.mean(scores)) if scores else 0.0


@metric("id_switches", direction="min", uses=("assignments",))
def id_switches(trace: pd.DataFrame) -> float:
    trace = _clean_trace(trace)
    if trace.empty:
        return 0.0
    switches = []
    for _, group in trace.groupby("gt_id"):
        ordered = group.sort_values("frame")["pred_id"].tolist()
        switches.append(sum(a != b for a, b in zip(ordered, ordered[1:])))
    return float(np.mean(switches)) if switches else 0.0


@metric("duplicate_rate", direction="min", uses=("assignments",))
def duplicate_rate(trace: pd.DataFrame) -> float:
    trace = _clean_trace(trace)
    if trace.empty:
        return 0.0
    extra = 0
    for _, group in trace.groupby("gt_id"):
        extra += max(0, group["pred_id"].nunique() - 1)
    total_pred_ids = max(1, trace["pred_id"].nunique())
    return float(extra) / float(total_pred_ids)


@metric("reappearance_rate", direction="max", uses=("assignments",))
def reappearance_rate(trace: pd.DataFrame, min_gap: int = 1) -> float:
    trace = _clean_trace(trace)
    if trace.empty:
        return 0.0
    outcomes = []
    for _, group in trace.groupby("gt_id"):
        ordered = group.sort_values("frame")
        frames = ordered["frame"].tolist()
        ids = ordered["pred_id"].tolist()
        for (f0, p0), (f1, p1) in zip(zip(frames, ids), zip(frames[1:], ids[1:])):
            if f1 - f0 > min_gap:
                outcomes.append(p0 == p1)
    return float(np.mean(outcomes)) if outcomes else 1.0


@metric("slot_purity", direction="max", uses=("assignments",))
def slot_purity(trace: pd.DataFrame) -> float:
    trace = _clean_trace(trace)
    if trace.empty:
        return 0.0
    purity = []
    for _, group in trace.groupby("pred_id"):
        counts = group["gt_id"].value_counts()
        purity.append(float(counts.max()) / float(len(group)))
    return float(np.mean(purity)) if purity else 0.0


def compute_entity_metrics(
    gt_frames: pd.DataFrame,
    trace: pd.DataFrame,
) -> dict[str, float]:
    return {
        "idf1": idf1(gt_frames, trace),
        "id_retention": id_retention(trace),
        "id_switches": id_switches(trace),
        "duplicate_rate": duplicate_rate(trace),
        "reappearance_rate": reappearance_rate(trace),
        "slot_purity": slot_purity(trace),
        "gt_detections": int(len(_gt_detections(gt_frames))),
        "pred_detections": int(len(_clean_trace(trace))),
        "gt_instances": int(_gt_detections(gt_frames)["instance_id"].nunique()),
        "pred_identities": int(_clean_trace(trace)["pred_id"].nunique()),
    }
