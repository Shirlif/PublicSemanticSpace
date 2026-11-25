"""Twin-asset diagnostics for S2."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def twin_ambiguity(
    detections: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    gate: float = 0.05,
    sim_threshold: float = 0.98,
) -> dict[str, float]:
    """Fraction of frames where same-twin-group detections are appearance-identical.

    Detection-level diagnostic: it says whether the detector/embedding actually
    makes twins indistinguishable.  It is not an abstention policy; abstain
    precision requires the runtime's uncertainty layer (deferred).
    """
    need = {"step", "pose_xyz", "emb"}
    if not need.issubset(detections.columns):
        return {"frames": 0, "ambiguous_frames": 0, "twin_ambiguity_rate": 0.0,
                "mean_max_similarity": 0.0}

    gt_by_step = {int(step): group for step, group in gt.groupby("physics_step")}
    frames = 0
    ambiguous = 0
    max_sims: list[float] = []
    for step, group in detections.groupby("step"):
        step = int(step)
        g = gt_by_step.get(step)
        if g is None or g.empty or group.empty:
            continue
        g = g[g["visibility_ratio"] > 0.001]
        if g.empty:
            continue
        gpos = np.stack([np.asarray(v, dtype=float) for v in g["pose_xyz"]])
        dpos = np.stack([np.asarray(v, dtype=float) for v in group["pose_xyz"]])
        cost = np.linalg.norm(gpos[:, None, :] - dpos[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        matched = [
            (g.iloc[int(i)], group.iloc[int(j)])
            for i, j in zip(rows, cols)
            if cost[i, j] <= gate
        ]
        by_group: dict[str, list[np.ndarray]] = {}
        for gt_row, det_row in matched:
            twin_group = gt_row.get("twin_group")
            if twin_group is None or pd.isna(twin_group):
                continue
            emb = det_row.get("emb")
            if emb is None or len(emb) == 0:
                continue
            by_group.setdefault(str(twin_group), []).append(np.asarray(emb, dtype=float))
        frame_max = 0.0
        for embeddings in by_group.values():
            if len(embeddings) < 2:
                continue
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    a, b = embeddings[i], embeddings[j]
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    if na > 0 and nb > 0:
                        frame_max = max(frame_max, float(np.dot(a, b) / (na * nb)))
        if frame_max > 0:
            frames += 1
            max_sims.append(frame_max)
            if frame_max >= sim_threshold:
                ambiguous += 1
    return {
        "frames": frames,
        "ambiguous_frames": ambiguous,
        "twin_ambiguity_rate": ambiguous / frames if frames else 0.0,
        "mean_max_similarity": float(np.mean(max_sims)) if max_sims else 0.0,
        "abstain_precision": None,
    }
