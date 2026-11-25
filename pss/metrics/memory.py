"""S6 memory metrics (T36)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def compute_memory_metrics(
    trace: pd.DataFrame,
    *,
    tolerance_m: float = 0.05,
    rollback_exactness: float = 0.0,
) -> dict[str, Any]:
    if trace.empty:
        return {
            "edited_accuracy": 0.0,
            "old_forgetting": 0.0,
            "forgetting_rate": 1.0,
            "id_retention": 0.0,
            "rollback_exactness": float(rollback_exactness),
            "samples": 0,
        }

    trace = trace.copy()
    trace["gt_xyz"] = trace["gt_xyz"].apply(
        lambda v: None if v is None else [float(x) for x in v]
    )
    trace = trace[trace["gt_xyz"].notna()]

    # Per-GT old position: last GT pose in the previous episode.
    old_positions: dict[str, list[float]] = {}
    edited = []
    old_forget = []
    for episode, group in trace.groupby("episode"):
        for gt_id, rows in group.groupby("gt_id"):
            current = [float(v) for v in rows.iloc[-1]["gt_xyz"]]
            pss = [float(v) for v in rows.iloc[-1]["pss_xyz"]]
            previous = old_positions.get(str(gt_id))
            if previous is not None and _dist(previous, current) > tolerance_m:
                edited.append(_dist(pss, current) <= tolerance_m)
                old_forget.append(_dist(pss, current) < _dist(pss, previous))
            old_positions[str(gt_id)] = current

    retention = []
    for _, rows in trace.groupby("gt_id"):
        counts = rows["pred_id"].value_counts()
        retention.append(float(counts.max()) / float(len(rows)))

    return {
        "edited_accuracy": float(np.mean(edited)) if edited else 1.0,
        "old_forgetting": float(np.mean(old_forget)) if old_forget else 1.0,
        "forgetting_rate": 1.0 - (float(np.mean(old_forget)) if old_forget else 1.0),
        "id_retention": float(np.mean(retention)) if retention else 0.0,
        "rollback_exactness": float(rollback_exactness),
        "samples": int(len(trace)),
        "edited_samples": len(edited),
    }
