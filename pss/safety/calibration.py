"""Guardrail threshold calibration (Phase 9).

The value vector is fixed; this module calibrates the scalar ``min_score``
gate from synthetic safe/unsafe candidates. It is a proxy calibration, not
human-labeled safety data.
"""

from __future__ import annotations

import numpy as np

from . import ValueVector


def synthetic_guardrail_cases(seed: int = 0, n: int = 200) -> list[tuple[dict, int]]:
    rng = np.random.default_rng(seed)
    cases: list[tuple[dict, int]] = []
    for i in range(n):
        if i % 2 == 0:
            candidate = {
                "id": f"safe_{i}",
                "similarity": float(rng.uniform(0.75, 1.0)),
                "value_domain": "task_fidelity" if i % 4 == 0 else "human_comfort",
                "safety_coeff": float(rng.uniform(0.8, 1.0)),
                "cost_penalty": float(rng.uniform(0.0, 0.08)),
            }
            label = 1
        else:
            candidate = {
                "id": f"unsafe_{i}",
                "similarity": float(rng.uniform(0.05, 0.5)),
                "value_domain": "task_fidelity",
                "safety_coeff": float(rng.choice([0.0, 0.2, 0.4])),
                "cost_penalty": float(rng.uniform(0.0, 0.4)),
            }
            label = 0
        cases.append((candidate, label))
    return cases


def calibrate_min_score(
    cases: list[tuple[dict, int]] | None = None,
    value_vector: ValueVector | None = None,
) -> dict:
    cases = cases or synthetic_guardrail_cases()
    vector = value_vector or ValueVector()
    scores = np.array([vector.score(candidate) for candidate, _ in cases], dtype=float)
    labels = np.array([label for _, label in cases], dtype=int)
    unique = np.unique(scores)
    thresholds = np.concatenate([[0.0], (unique[:-1] + unique[1:]) / 2.0, [unique[-1] + 1e-6]])
    best = {"min_score": 0.0, "balanced_accuracy": -1.0}
    for threshold in thresholds:
        predicted = (scores >= threshold).astype(int)
        tp = int(np.sum((predicted == 1) & (labels == 1)))
        tn = int(np.sum((predicted == 0) & (labels == 0)))
        pos = int(np.sum(labels == 1))
        neg = int(np.sum(labels == 0))
        tpr = tp / max(1, pos)
        tnr = tn / max(1, neg)
        balanced = 0.5 * (tpr + tnr)
        if balanced > best["balanced_accuracy"] or (
            abs(balanced - best["balanced_accuracy"]) < 1e-12
            and threshold > best["min_score"]
        ):
            best = {
                "min_score": float(threshold),
                "balanced_accuracy": float(balanced),
                "true_positive_rate": float(tpr),
                "true_negative_rate": float(tnr),
            }
    return {
        **best,
        "n_cases": len(cases),
        "safe_mean_score": float(scores[labels == 1].mean()),
        "unsafe_mean_score": float(scores[labels == 0].mean()),
        "note": "Synthetic safe/unsafe candidates; not human-labeled safety data.",
    }
