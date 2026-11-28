"""Four-level retrieval and confidence-weighted recognition.

Level 1: scene context / hierarchical tree candidates.
Level 2: modality mask (missing or zero-confidence modalities are skipped).
Level 3: per-modality ANN top-K on the candidate set.
Level 4: exact cosine on the shortlist, fused by modality confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bank import PrototypeBank, l2
from .tree import tree_candidates


@dataclass
class RecognitionResult:
    ranked: list[tuple[str, float]]
    per_modality: dict[str, dict[str, float]]
    modality_top: dict[str, str]
    conflict: bool
    unknown: bool
    candidates: list[str]


def recognize(
    query: dict[str, np.ndarray],
    bank: PrototypeBank,
    tree: dict | None = None,
    context: list[str] | None = None,
    top_k: int = 3,
    ann_k: int = 3,
    weights: dict[str, float] | None = None,
    use_tree: bool = True,
    threshold: float | None = None,
    conflict_weight: float = 0.75,
) -> RecognitionResult:
    weights = dict(weights or {m: 1.0 for m in query})
    modalities = [
        m
        for m in query
        if m in bank.protos
        and weights.get(m, 0.0) > 0.0
        and float(np.linalg.norm(np.asarray(query[m]))) > 1e-8
    ]
    if not modalities:
        return RecognitionResult([], {}, {}, False, True, [])

    # Level 1: candidate categories.
    if context is not None:
        candidates = list(context)
    elif use_tree and tree is not None:
        # Binary tree: take the best branch at each node. top_n=2 degenerates
        # to "keep everything" for a binary tree, so pruning uses top_n=1.
        candidates = tree_candidates(query, tree, top_n=1)
    else:
        candidates = list(bank.categories)
    valid = set(bank.categories)
    candidates = [c for c in candidates if c in valid]
    if not candidates:
        candidates = list(bank.categories)

    # Level 3: per-modality ANN top-K inside the candidate set.
    shortlist = set(candidates)
    for modality in modalities:
        sims = bank.modality_scores(query[modality], modality)
        for i in np.argsort(-sims)[:ann_k]:
            category = bank.categories[int(i)]
            if category in candidates:
                shortlist.add(category)

    # Level 4: exact cosine on the shortlist, weighted fusion.  A category
    # that has no prototype in a modality simply does not contribute that
    # modality's weight; the denominator is per category, not global.
    per_modality: dict[str, dict[str, float]] = {}
    for category in shortlist:
        i = bank.categories.index(category)
        per_modality[category] = {
            m: float(np.dot(l2(query[m]), bank.protos[m][i]))
            for m in modalities
            if bank.has_prototype(m, category)
        }
    scores = {}
    for category in shortlist:
        available = per_modality[category]
        denom = sum(weights[m] for m in available)
        if denom <= 0:
            continue
        scores[category] = (
            sum(weights[m] * value for m, value in available.items()) / denom
        )
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

    # Conflict: high-weight modalities disagree on their own top-1.
    high = [m for m in modalities if weights.get(m, 0.0) >= conflict_weight]
    modality_top: dict[str, str] = {}
    for m in high:
        sims = np.asarray(bank.modality_scores(query[m], m), dtype=np.float64).copy()
        sims[~bank.valid_mask(m)] = -np.inf
        if np.isfinite(sims).any():
            modality_top[m] = bank.categories[int(np.argmax(sims))]
    conflict = len(set(modality_top.values())) > 1

    unknown = bool(
        not ranked or (threshold is not None and ranked[0][1] < threshold)
    )
    return RecognitionResult(
        ranked=ranked,
        per_modality=per_modality,
        modality_top=modality_top,
        conflict=conflict,
        unknown=unknown,
        candidates=sorted(shortlist),
    )
