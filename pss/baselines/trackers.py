"""External baselines for S1 (B1 text-state, B2 SORT-lite)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _rows_for_frame(step: int, group: pd.DataFrame, ids: list[str]) -> list[dict]:
    rows = []
    for (_, row), pred_id in zip(group.iterrows(), ids):
        rows.append(
            {
                "step": int(step),
                "t_s": float(row["t_s"]),
                "pred_id": pred_id,
                "pos_xyz": [float(v) for v in row["pose_xyz"]],
                "category": str(row["class_pred"]),
                "confidence": float(row["score"]),
            }
        )
    return rows


def text_state_predictions(
    detections: pd.DataFrame,
    *,
    steps: Iterable[int] | None = None,
    gate: float = 0.10,
) -> pd.DataFrame:
    """B1: no persistent entity slots; match each frame to the previous frame only."""
    if steps is None:
        steps = sorted(int(s) for s in detections["step"].unique())
    steps = [int(s) for s in steps]
    by_step = {int(step): group for step, group in detections.groupby("step")}
    counter = 0
    prev: list[dict] = []
    output: list[dict] = []
    for step in steps:
        group = by_step.get(step)
        if group is None or group.empty:
            prev = []
            continue
        positions = [np.asarray(v, dtype=float) for v in group["pose_xyz"]]
        categories = [str(v) for v in group["class_pred"]]
        ids = [None] * len(positions)
        if prev:
            cost = np.full((len(prev), len(positions)), 1e6)
            for i, track in enumerate(prev):
                for j, pos in enumerate(positions):
                    if track["category"] != categories[j]:
                        continue
                    distance = float(np.linalg.norm(track["pos"] - pos))
                    if distance <= gate:
                        cost[i, j] = distance
            rows_i, cols_j = linear_sum_assignment(cost)
            for i, j in zip(rows_i, cols_j):
                if cost[i, j] < 1e6:
                    ids[int(j)] = prev[int(i)]["id"]
        for j, pred_id in enumerate(ids):
            if pred_id is None:
                counter += 1
                ids[j] = f"text_{counter:04d}"
        output.extend(_rows_for_frame(step, group, ids))
        prev = [
            {"id": ids[j], "pos": positions[j], "category": categories[j]}
            for j in range(len(positions))
        ]
    return pd.DataFrame(output)


def sort_lite_predictions(
    detections: pd.DataFrame,
    *,
    steps: Iterable[int] | None = None,
    gate: float = 0.15,
    max_age: int = 10,
    appearance_weight: float = 0.0,
) -> pd.DataFrame:
    """B2: constant-velocity SORT-lite; optional appearance term (DeepSORT-lite)."""
    if steps is None:
        steps = sorted(int(s) for s in detections["step"].unique())
    steps = [int(s) for s in steps]
    by_step = {int(step): group for step, group in detections.groupby("step")}
    tracks: list[dict] = []
    counter = 0
    output: list[dict] = []

    def appearance(row) -> np.ndarray | None:
        emb = row.get("emb")
        if emb is None or len(emb) == 0:
            return None
        vec = np.asarray(emb, dtype=float)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

    for step in steps:
        group = by_step.get(step)
        present = group is not None and not group.empty
        positions = (
            [np.asarray(v, dtype=float) for v in group["pose_xyz"]]
            if present
            else []
        )
        embeddings = [appearance(row) for _, row in group.iterrows()] if present else []
        categories = [str(v) for v in group["class_pred"]] if present else []

        predicted = [track["pos"] + track["vel"] for track in tracks]
        ids = [None] * len(positions)
        matched_tracks: set[int] = set()
        if tracks and positions:
            cost = np.full((len(tracks), len(positions)), 1e6)
            for i, track in enumerate(tracks):
                for j, pos in enumerate(positions):
                    if track["category"] != categories[j]:
                        continue
                    distance = float(np.linalg.norm(predicted[i] - pos))
                    if distance > gate:
                        continue
                    c = distance / gate
                    if appearance_weight > 0 and track.get("emb") is not None and embeddings[j] is not None:
                        c = (1 - appearance_weight) * c + appearance_weight * float(
                            1.0 - np.dot(track["emb"], embeddings[j])
                        )
                    cost[i, j] = c
            rows_i, cols_j = linear_sum_assignment(cost)
            for i, j in zip(rows_i, cols_j):
                if cost[i, j] >= 1e6:
                    continue
                track = tracks[int(i)]
                prev_pos = track["pos"]
                track["vel"] = 0.5 * track["vel"] + 0.5 * (positions[int(j)] - prev_pos)
                track["pos"] = positions[int(j)]
                track["category"] = categories[int(j)]
                if embeddings[int(j)] is not None:
                    track["emb"] = embeddings[int(j)]
                track["missed"] = 0
                ids[int(j)] = track["id"]
                matched_tracks.add(int(i))

        for i, track in enumerate(tracks):
            if i in matched_tracks:
                continue
            track["missed"] += 1
        tracks = [track for track in tracks if track["missed"] <= max_age]

        for j, pred_id in enumerate(ids):
            if pred_id is None:
                counter += 1
                pred_id = f"sort_{counter:04d}"
                ids[j] = pred_id
                tracks.append(
                    {
                        "id": pred_id,
                        "pos": positions[j].copy(),
                        "vel": np.zeros(3),
                        "category": categories[j],
                        "emb": embeddings[j],
                        "missed": 0,
                    }
                )
        if positions:
            output.extend(_rows_for_frame(step, group, ids))
    return pd.DataFrame(output)
