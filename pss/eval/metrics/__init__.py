"""Evaluation metrics (Phase 10)."""

from __future__ import annotations

from collections import Counter

import numpy as np


def id_retention(gt_tracks: dict[str, list[tuple[int, str]]]) -> dict:
    retentions, switches, reappear, latencies = [], [], [], []
    observations = 0
    for seq in gt_tracks.values():
        seq = sorted(seq)
        ids = [tid for _, tid in seq]
        if not ids:
            continue
        observations += len(ids)
        counts = Counter(ids)
        retentions.append(max(counts.values()) / len(ids))
        switches.append(sum(1 for a, b in zip(ids, ids[1:]) if a != b))
        for (t0, id0), (t1, id1) in zip(seq, seq[1:]):
            if t1 - t0 > 1:
                reappear.append(id1 == id0)
                latencies.append(t1 - t0 - 1)
    return {
        "objects": len(gt_tracks),
        "observations": observations,
        "id_retention": float(np.mean(retentions)) if retentions else 0.0,
        "id_switches": float(np.mean(switches)) if switches else 0.0,
        "reappearance_success": float(np.mean(reappear)) if reappear else 1.0,
        "wake_latency_frames": float(np.mean(latencies)) if latencies else 0.0,
    }


def relation_prf(computed: set[tuple], truth: set[tuple]) -> dict:
    tp = len(computed & truth)
    fp = len(computed - truth)
    fn = len(truth - computed)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def pose_error(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.mean(np.linalg.norm(pred[:, :3] - truth[:, :3], axis=1)))


def accuracy(pred, truth) -> float:
    pred = list(pred)
    truth = list(truth)
    if not truth:
        return 0.0
    return sum(int(p == t) for p, t in zip(pred, truth)) / len(truth)
