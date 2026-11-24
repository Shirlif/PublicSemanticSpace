"""Review pipeline: aggregate -> pre-review -> regression -> human gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pss.prototypes import PrototypeBank, recognize
from pss.prototypes.admission import calibrated_thresholds, pre_review

from ..signing import public_from_private, public_key_text, sign_bytes, verify_bytes


def aggregate_proposals(proposals: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for p in proposals:
        groups.setdefault(p["category"], []).append(p)
    out = []
    for category, items in groups.items():
        total = sum(p["sample_count"] for p in items)
        modalities = sorted({m for p in items for m in p["vectors"]})
        vectors = {}
        for m in modalities:
            acc = np.zeros_like(np.asarray(items[0]["vectors"][m], dtype=float))
            weight = 0
            for p in items:
                if m in p["vectors"]:
                    acc += np.asarray(p["vectors"][m], dtype=float) * p["sample_count"]
                    weight += p["sample_count"]
            v = acc / max(1, weight)
            n = np.linalg.norm(v)
            vectors[m] = v / n if n > 0 else v
        variance = {
            m: float(np.mean([p.get("variance", {}).get(m, 0.0) for p in items]))
            for m in modalities
        }
        out.append(
            {
                "category": category,
                "vectors": vectors,
                "sample_count": total,
                "variance": variance,
                "proposals": len(items),
            }
        )
    return out


def bank_from_package(package) -> PrototypeBank:
    return PrototypeBank(
        categories=list(package.categories),
        protos={m: np.asarray(package.protos[m], dtype=np.float32) for m in package.protos},
        stats=package.stats,
        version=package.version,
    )


def regression_gate(
    candidate_vectors: dict[str, np.ndarray],
    holdout: dict[str, np.ndarray],
    bank: PrototypeBank,
    category: str,
    weights: dict[str, float] | None = None,
) -> dict:
    weights = weights or {"vision": 1.0, "tactile": 0.5, "geometry": 0.95}
    protos = {
        m: np.vstack(
            [bank.protos[m], np.asarray(candidate_vectors[m], dtype=np.float32)[None, :]]
        )
        for m in bank.protos
    }
    temp = PrototypeBank(
        categories=bank.categories + [category],
        protos=protos,
        stats={**bank.stats, category: {}},
        version="temp",
    )
    n = len(next(iter(holdout.values())))
    correct = 0
    for i in range(n):
        query = {m: holdout[m][i] for m in holdout}
        result = recognize(query, temp, weights=weights, use_tree=False)
        correct += int(bool(result.ranked) and result.ranked[0][0] == category)
    return {
        "new_acc": correct / max(1, n),
        "n_holdout": n,
        "old_unchanged": True,
    }


@dataclass
class ReviewResult:
    accepted: bool
    decision: str
    reasons: list[str]
    pre_review: str
    regression: dict
    pre_review_metrics: dict | None = None
    pre_review_thresholds: dict | None = None
    human_reviewer: str | None = None
    human_decision: str | None = None


def calibrated_pre_review_thresholds(
    bank: PrototypeBank, factor: float = 0.15
) -> dict[str, float]:
    """Per-modality duplicate thresholds.

    ``factor * min_nn`` is deliberately below the bank's own nearest-neighbor
    distance: pre-review is a duplicate detector, not a novelty gate. The
    regression holdout remains the decisive acceptance gate.
    """
    calibration = calibrated_thresholds(bank)["per_modality"]
    return {
        modality: float(max(0.01, factor * float(values["min_nn"])))
        for modality, values in calibration.items()
    }


def write_human_decision(
    path: str | Path,
    proposal_id: str,
    reviewer: str,
    decision: str,
    note: str = "",
    private_key: bytes | None = None,
) -> dict:
    if decision not in ("accept", "reject"):
        raise ValueError("human decision must be 'accept' or 'reject'")
    payload = {
        "proposal_id": str(proposal_id),
        "reviewer": str(reviewer),
        "decision": decision,
        "note": str(note),
    }
    if private_key is not None:
        data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        payload["signature"] = sign_bytes(private_key, data)
        payload["public_key"] = public_key_text(public_from_private(private_key))
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def load_human_decision(
    path: str | Path,
    trusted_public_key: bytes | None = None,
) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    signature = payload.get("signature")
    public_key = payload.get("public_key")
    if signature:
        if not public_key:
            raise ValueError("signed human decision missing public key")
        if trusted_public_key is not None and public_key.encode("utf-8") != trusted_public_key:
            raise ValueError("human decision public key does not match trusted key")
        base = {
            key: value
            for key, value in payload.items()
            if key not in ("signature", "public_key")
        }
        data = json.dumps(base, sort_keys=True, ensure_ascii=False).encode("utf-8")
        if not verify_bytes(public_key.encode("utf-8"), data, signature):
            raise ValueError("human decision signature verification failed")
    elif trusted_public_key is not None:
        raise ValueError("unsigned human decision rejected by trusted-key policy")
    return payload


def review_proposal(
    candidate_samples: dict[str, np.ndarray],
    candidate_vectors: dict[str, np.ndarray],
    holdout: dict[str, np.ndarray],
    bank: PrototypeBank,
    category: str,
    sample_count: int,
    min_acc: float = 0.6,
    min_samples: int = 20,
    pre_review_thresholds: dict[str, float] | None = None,
    human_decision: dict | None = None,
    require_human: bool = False,
) -> ReviewResult:
    thresholds = (
        calibrated_pre_review_thresholds(bank)
        if pre_review_thresholds is None
        else pre_review_thresholds
    )
    pre = pre_review(candidate_samples, bank, min_inter_distance=thresholds)
    regression = regression_gate(candidate_vectors, holdout, bank, category)
    reasons = []
    if sample_count < min_samples:
        reasons.append("insufficient_samples")
    if regression["new_acc"] < min_acc:
        reasons.append("regression_below_threshold")
    if require_human and human_decision is None:
        reasons.append("human_review_pending")
    if human_decision is not None and human_decision.get("decision") == "reject":
        reasons.append("human_rejected")
    accepted = not reasons
    return ReviewResult(
        accepted=accepted,
        decision="accept" if accepted else "reject",
        reasons=reasons,
        pre_review=pre.decision,
        regression=regression,
        pre_review_metrics=pre.metrics,
        pre_review_thresholds=thresholds,
        human_reviewer=(
            None if human_decision is None else human_decision.get("reviewer")
        ),
        human_decision=(
            None if human_decision is None else human_decision.get("decision")
        ),
    )
