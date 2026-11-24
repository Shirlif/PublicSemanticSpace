"""Conflict-aware prototype recognition.

Per modality, compute top-1 and margin. A modality is reliable when its top
score and top-2 margin pass thresholds. If reliable modalities agree, answer;
if none are reliable or they disagree, abstain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bank import PrototypeBank, l2


@dataclass
class ConflictDecision:
    category: str | None
    conflict: bool
    reason: str
    reliable: dict[str, str] = field(default_factory=dict)
    per_modality: dict[str, dict] = field(default_factory=dict)


def conflict_aware_recognize(
    query: dict[str, np.ndarray],
    bank: PrototypeBank,
    weights: dict[str, float] | None = None,
    min_margin: float = 0.05,
    min_score: float = 0.30,
    high_weight: float = 0.75,
) -> ConflictDecision:
    weights = weights or {m: 1.0 for m in query}
    per: dict[str, dict] = {}
    reliable: dict[str, str] = {}
    present = {
        m for m, z in query.items() if float(np.linalg.norm(np.asarray(z))) > 1e-8
    }
    for modality, z in query.items():
        if modality not in bank.protos or weights.get(modality, 0.0) <= 0:
            continue
        sims = bank.modality_scores(l2(z), modality)
        order = np.argsort(-sims)
        top = bank.categories[int(order[0])]
        score = float(sims[order[0]])
        margin = float(score - sims[order[1]]) if len(order) > 1 else score
        per[modality] = {"top": top, "score": score, "margin": margin}
        if score >= min_score and margin >= min_margin:
            reliable[modality] = top
    high_present = {
        m for m in present if weights.get(m, 0.0) >= high_weight and m in per
    }
    high_reliable = {m: reliable[m] for m in high_present if m in reliable}
    if len(high_present) >= 2:
        if len(high_reliable) < len(high_present):
            return ConflictDecision(
                None, False, "high_modality_uncertain", reliable, per
            )
        if len(set(high_reliable.values())) > 1:
            return ConflictDecision(
                None, True, "high_weight_conflict", reliable, per
            )
        return ConflictDecision(
            next(iter(high_reliable.values())), False, "ok_high_weight", reliable, per
        )
    if len(high_present) == 1:
        modality = next(iter(high_present))
        if modality in reliable:
            return ConflictDecision(
                reliable[modality], False, "ok_single_high_weight", reliable, per
            )
        return ConflictDecision(None, False, "single_high_uncertain", reliable, per)
    if len(reliable) < 2:
        return ConflictDecision(None, False, "no_quorum", reliable, per)
    if len(set(reliable.values())) > 1:
        return ConflictDecision(None, True, "modality_conflict", reliable, per)
    return ConflictDecision(next(iter(reliable.values())), False, "ok_fallback", reliable, per)
