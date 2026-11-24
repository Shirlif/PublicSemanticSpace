"""Metric helpers for the PSS evaluation suite."""

from __future__ import annotations

import numpy as np


def accuracy(pred, truth) -> float:
    pred = list(pred)
    truth = list(truth)
    if not truth:
        return float("nan")
    return float(np.mean([p == t for p, t in zip(pred, truth)]))


def exact_match(pred, truth) -> float:
    return accuracy([_freeze(p) for p in pred], [_freeze(t) for t in truth])


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def auroc(scores, is_positive) -> float:
    """Rank-based AUROC; higher score = positive."""
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(is_positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties.
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    rank_sum = ranks[positive].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fpr_at_95tpr(scores, is_positive) -> float:
    """False-positive rate at the threshold that reaches 95% TPR."""
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(is_positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_pos = positive[order]
    tp = np.cumsum(sorted_pos)
    fp = np.cumsum(~sorted_pos)
    target = int(np.ceil(0.95 * n_pos))
    idx = int(np.searchsorted(tp, target))
    if idx >= len(fp):
        return 1.0
    return float(fp[idx] / n_neg)


def mean(values) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")
