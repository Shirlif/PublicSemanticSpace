"""Offline prototype admission pre-review (Phase 5, L2 proposal simulation).

Thresholds are initial values, not calibrated constants (see 9.10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bank import PrototypeBank, l2


@dataclass
class AdmissionResult:
    decision: str  # accept | review | reject
    metrics: dict
    reasons: list[str]


def calibrated_thresholds(bank: PrototypeBank) -> dict:
    """Data-driven thresholds from the existing bank.

    A single global distance threshold is wrong: per-modality nearest-neighbor
    distances differ by an order of magnitude (tactile is much tighter than
    vision). These values are diagnostics for calibration, not fixed rules.
    """
    per_modality: dict[str, dict] = {}
    for modality, protos in bank.protos.items():
        sims = protos @ protos.T
        np.fill_diagonal(sims, -1.0)
        nearest = 1.0 - sims.max(axis=1)
        per_modality[modality] = {
            "min_nn": float(nearest.min()),
            "median_nn": float(np.median(nearest)),
            "distance_threshold": float(max(0.01, 0.5 * float(nearest.min()))),
        }
    fused_sims = np.mean([bank.protos[m] @ bank.protos[m].T for m in bank.protos], axis=0)
    np.fill_diagonal(fused_sims, -1.0)
    fused_nearest = 1.0 - fused_sims.max(axis=1)
    return {
        "per_modality": per_modality,
        "fused": {
            "min_nn": float(fused_nearest.min()),
            "median_nn": float(np.median(fused_nearest)),
            "distance_threshold": float(max(0.01, 0.5 * float(fused_nearest.min()))),
        },
    }


def pre_review(
    candidate_samples: dict[str, np.ndarray],
    bank: PrototypeBank,
    min_samples: int = 20,
    max_variance: float = 0.35,
    min_inter_distance: float | dict[str, float] = 0.25,
) -> AdmissionResult:
    metrics: dict[str, dict] = {}
    reasons: list[str] = []
    n = min(len(v) for v in candidate_samples.values())
    if n < min_samples:
        reasons.append("insufficient_samples")

    for modality, v in candidate_samples.items():
        v = np.asarray(v, dtype=np.float64)
        proto = l2(v.mean(0))
        variance = float(np.mean(1.0 - v @ proto))
        sims = bank.protos[modality] @ proto
        min_distance = float(1.0 - np.max(sims))
        metrics[modality] = {
            "sample_count": int(len(v)),
            "variance": variance,
            "min_inter_distance": min_distance,
        }
        if variance > max_variance:
            reasons.append(f"variance_high:{modality}")
        threshold = (
            float(min_inter_distance.get(modality, 0.25))
            if isinstance(min_inter_distance, dict)
            else float(min_inter_distance)
        )
        if min_distance < threshold:
            reasons.append(f"inter_distance_low:{modality}")

    if any(r.startswith(("variance_high", "inter_distance_low")) for r in reasons):
        decision = "reject"
    elif "insufficient_samples" in reasons:
        decision = "review"
    else:
        decision = "accept"
    return AdmissionResult(decision=decision, metrics=metrics, reasons=reasons)
