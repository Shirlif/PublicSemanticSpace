"""Expert quality levels and manipulation checks.

The property plan treats expert quality as a controlled factor, not an
observed confound.  For L0 this is done by selecting a stub; for L1/L2 the
same interface can wrap a real encoder with :class:`NoisyEmbeddingWrapper`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import ExpertHealth
from .stubs import (
    AdversarialExpert,
    BaseStub,
    CategoryCodebook,
    ConstantExpert,
    NoisyOracleExpert,
    OracleExpert,
    RandomExpert,
)


@dataclass(frozen=True)
class QualityLevel:
    name: str
    accuracy: float
    ece: float


QUALITY_LEVELS = {
    "Q0": QualityLevel("Q0", 1.00, 0.00),
    "Q1": QualityLevel("Q1", 0.90, 0.05),
    "Q2": QualityLevel("Q2", 0.75, 0.10),
    "Q3": QualityLevel("Q3", 0.55, 0.20),
    "Q4": QualityLevel("Q4", 0.00, 0.50),  # accuracy filled from chance
    "Q5": QualityLevel("Q5", 0.00, 0.00),
}


def make_quality_expert(
    level: str | QualityLevel,
    codebook: CategoryCodebook,
    modality: str = "vision",
    seed: int = 0,
) -> BaseStub:
    """Return a stub matching a quality level."""

    spec = QUALITY_LEVELS[str(level)] if isinstance(level, str) else level
    if spec.name == "Q0":
        return OracleExpert(codebook, seed=seed, modality=modality)
    if spec.name in ("Q1", "Q2", "Q3"):
        return NoisyOracleExpert(
            codebook, accuracy=spec.accuracy, ece=spec.ece, seed=seed, modality=modality
        )
    if spec.name == "Q4":
        return RandomExpert(codebook, seed=seed, modality=modality)
    if spec.name == "Q5":
        return AdversarialExpert(codebook, seed=seed, modality=modality)
    # A custom level: use the requested accuracy, or constant if accuracy is 0.
    if spec.accuracy <= 0.0:
        return ConstantExpert(codebook, seed=seed, modality=modality)
    return NoisyOracleExpert(
        codebook, accuracy=spec.accuracy, ece=spec.ece, seed=seed, modality=modality
    )


def _gt_category(obs):
    if isinstance(obs, dict):
        return obs.get("gt_category") or obs.get("category")
    return getattr(obs, "gt_category", None) or getattr(obs, "category", None)


def manipulation_check(
    expert,
    observations,
    codebook: CategoryCodebook,
    n_bins: int = 10,
) -> dict:
    """Measure the expert's actual accuracy and ECE on a probe set.

    For stubs, the confidence is the max cosine to any codebook vector.  For a
    real encoder this is a rough proxy; the main check is that the measured
    accuracy is close to the requested quality level.
    """

    observations = list(observations)
    if not observations:
        return {"n": 0, "accuracy": None, "ece": None, "mean_confidence": None}

    embeddings = np.asarray(expert.encode(observations), dtype=np.float64)
    correct = 0
    confidences: list[float] = []
    outcomes: list[int] = []
    for obs, vec in zip(observations, embeddings):
        gt = _gt_category(obs)
        if gt is None:
            continue
        # Nearest category under cosine similarity.
        sims = codebook.vectors @ (vec / max(float(np.linalg.norm(vec)), 1e-12))
        pred = codebook.categories[int(np.argmax(sims))]
        conf = float(np.max(sims))
        conf = float(np.clip((conf + 1.0) / 2.0, 0.0, 1.0))
        correct += int(pred == gt)
        confidences.append(conf)
        outcomes.append(int(pred == gt))

    n = len(outcomes)
    if n == 0:
        return {"n": 0, "accuracy": None, "ece": None, "mean_confidence": None}

    accuracy = correct / n
    # Expected calibration error with equal-width confidence bins.
    ece = 0.0
    conf_arr = np.asarray(confidences)
    out_arr = np.asarray(outcomes)
    for lo in np.linspace(0.0, 1.0, n_bins + 1)[:-1]:
        hi = lo + 1.0 / n_bins
        mask = (conf_arr >= lo) & (conf_arr < hi if hi < 1.0 else conf_arr <= hi)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(
            float(np.mean(conf_arr[mask])) - float(np.mean(out_arr[mask]))
        )
    return {
        "n": n,
        "accuracy": float(accuracy),
        "ece": float(ece),
        "mean_confidence": float(np.mean(conf_arr)),
    }


class NoisyEmbeddingWrapper:
    """Quality injection for a real expert.

    With probability ``1 - accuracy`` the output is replaced by a random unit
    vector.  This is an embedding-level degradation, not a label flip; it is
    used in L1 to interpolate between a real expert and chance.
    """

    def __init__(self, inner, accuracy: float, seed: int = 0):
        from .contract import ExpertHealth

        self.inner = inner
        self.accuracy = float(np.clip(accuracy, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)
        self.modality = inner.modality
        self.dim = int(inner.dim)
        self.version = f"{inner.version}+noise-{self.accuracy:.2f}"

    def encode(self, obs):
        out = self.inner.encode(obs)
        arr = np.asarray(out, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        mask = self.rng.random(len(arr)) >= self.accuracy
        if np.any(mask):
            noise = self.rng.normal(size=(int(np.sum(mask)), self.dim))
            noise /= np.linalg.norm(noise, axis=1, keepdims=True)
            arr[mask] = noise
        return arr.astype(np.float32)

    def health(self) -> ExpertHealth:
        base = self.inner.health()
        return ExpertHealth(
            latency_p50=base.latency_p50,
            latency_p99=base.latency_p99,
            calibration_ece=float(1.0 - self.accuracy),
            ood_score=base.ood_score,
            failure_flags=list(base.failure_flags) + ["quality_injected"],
        )
