"""S7 calibration from detection scores (T37)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _ece(scores: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges, edges[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= 1.0)
        if not mask.any():
            continue
        conf = float(scores[mask].mean())
        acc = float(correct[mask].mean())
        total += float(mask.mean()) * abs(acc - conf)
    return total


def _selective(scores: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    if len(scores) == 0:
        return 0.0
    k = max(1, int(round(len(scores) * coverage)))
    order = np.argsort(-scores)[:k]
    return float(correct[order].mean())


def calibration_bundle(bundle: str | Path, *, gate: float = 0.05) -> dict[str, Any]:
    root = Path(bundle)
    det = pd.read_parquet(root / "obs_detections.parquet")
    if det.empty or "score" not in det.columns:
        return {"samples": 0}
    gt = pd.read_parquet(root / "gt.parquet")
    obs_steps = set(int(s) for s in pd.read_parquet(root / "obs_frames.parquet")["step"])
    gt = gt[(gt["physics_step"].isin(obs_steps)) & (gt["visibility_ratio"] > 0.001)]
    gt_by_step = {int(step): group for step, group in gt.groupby("physics_step")}

    scores: list[float] = []
    correct: list[int] = []
    for step, group in det.groupby("step"):
        g = gt_by_step.get(int(step))
        if g is None or g.empty or group.empty:
            continue
        gpos = np.stack([np.asarray(v, dtype=float) for v in g["pose_xyz"]])
        dpos = np.stack([np.asarray(v, dtype=float) for v in group["pose_xyz"]])
        cost = np.linalg.norm(gpos[:, None, :] - dpos[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        matched = {int(j) for i, j in zip(rows, cols) if cost[i, j] <= gate}
        for j, score in enumerate(group["score"].tolist()):
            scores.append(float(score))
            correct.append(1 if j in matched else 0)
    if not scores:
        return {"samples": 0}
    s = np.asarray(scores)
    c = np.asarray(correct, dtype=float)
    return {
        "samples": int(len(s)),
        "accuracy": float(c.mean()),
        "mean_confidence": float(s.mean()),
        "brier": float(np.mean((s - c) ** 2)),
        "ece": float(_ece(s, c)),
        "selective_accuracy_50": _selective(s, c, 0.5),
        "selective_accuracy_80": _selective(s, c, 0.8),
    }


def calibration_suite(bundles_root: str | Path, suite: str) -> dict[str, Any]:
    root = Path(bundles_root)
    bundles = sorted(p.parent for p in root.glob(f"**/{suite}/**/seed*/manifest.json"))
    reports = [calibration_bundle(b) for b in bundles]
    reports = [r for r in reports if r.get("samples", 0) > 0]
    if not reports:
        raise FileNotFoundError(f"no detection bundles for {suite}")
    out: dict[str, Any] = {"suite": suite, "episodes": len(reports)}
    for key in ("accuracy", "mean_confidence", "brier", "ece", "selective_accuracy_50", "selective_accuracy_80"):
        values = [float(r[key]) for r in reports]
        out[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    out["samples"] = sum(r["samples"] for r in reports)
    out["conflict_f1"] = None
    out["note"] = "conflict F1 requires PSS uncertainty traces; not available in this run"
    return out
