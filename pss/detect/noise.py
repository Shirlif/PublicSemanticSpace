"""Noisy detection tier (T20): GT detections with controlled corruption."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_PROFILE = {
    "miss_rate": {"base": 0.05, "per_visibility_drop": 0.6},
    "class_confusion": [],
    "pose_sigma_m": 0.004,
    "false_positive_per_frame": 0.2,
}


def _visibility_lookup(gt: pd.DataFrame) -> dict[tuple[int, str], float]:
    return {
        (int(step), str(instance)): float(vis)
        for step, instance, vis in zip(
            gt["physics_step"], gt["instance_id"], gt["visibility_ratio"]
        )
    }


def apply_noise(
    rows: list[dict[str, Any]],
    gt: pd.DataFrame,
    *,
    profile: dict[str, Any] | None = None,
    seed: int = 0,
    workspace: list[list[float]] | None = None,
) -> list[dict[str, Any]]:
    profile = {**DEFAULT_PROFILE, **(profile or {})}
    rng = np.random.default_rng(seed)
    visibility = _visibility_lookup(gt)
    miss = profile.get("miss_rate") or {}
    base = float(miss.get("base", 0.05))
    per_drop = float(miss.get("per_visibility_drop", 0.6))
    sigma = float(profile.get("pose_sigma_m", 0.004))
    confusion = {
        str(src): (str(dst), float(p)) for src, dst, p in profile.get("class_confusion", [])
    }

    kept: list[dict[str, Any]] = []
    for row in rows:
        vis = visibility.get((int(row["step"]), str(row.get("emb_ref"))), 1.0)
        p_miss = min(0.95, base + per_drop * (1.0 - vis))
        if rng.random() < p_miss:
            continue
        out = dict(row)
        pos = np.asarray(out["pose_xyz"], dtype=float)
        out["pose_xyz"] = (pos + rng.normal(scale=sigma, size=3)).tolist()
        cls = str(out.get("class_pred", ""))
        if cls in confusion and rng.random() < confusion[cls][1]:
            out["class_pred"] = confusion[cls][0]
        out["det_id"] = f"noisy_{out['det_id']}"
        kept.append(out)

    fp_rate = float(profile.get("false_positive_per_frame", 0.0))
    if fp_rate > 0:
        steps = sorted({int(r["step"]) for r in rows})
        xy = workspace or [[-0.3, -0.25], [0.3, 0.25]]
        for step in steps:
            count = rng.poisson(fp_rate)
            for k in range(count):
                kept.append(
                    {
                        "step": step,
                        "t_s": next(
                            float(r["t_s"]) for r in rows if int(r["step"]) == step
                        ),
                        "camera": rows[0]["camera"],
                        "det_id": f"noise_fp_{step:06d}_{k:03d}",
                        "bbox_xyxy": [0.0, 0.0, 0.0, 0.0],
                        "score": float(rng.uniform(0.3, 0.8)),
                        "class_pred": "cube",
                        "pose_xyz": [
                            float(rng.uniform(xy[0][0], xy[1][0])),
                            float(rng.uniform(xy[0][1], xy[1][1])),
                            0.785,
                        ],
                        "pose_quat": [1.0, 0.0, 0.0, 0.0],
                        "emb": None,
                        "emb_ref": f"noise_fp_{step:06d}_{k:03d}",
                    }
                )
    return kept
