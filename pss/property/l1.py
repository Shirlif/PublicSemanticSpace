"""L1 controlled integration: real experts, frozen shared space, stubs for quality.

The L1 suite answers one question: do the swap/property metrics survive when the
embeddings come from real frozen experts and real projection heads, instead of
synthetic codebook vectors?

It does **not** run Isaac.  Observations are built from ``data/expert_v1`` (or
``data/category_pretrain``) features after the real expert + projection
pipeline.  The L2 suite is the one that needs the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision import load_vision_expert
from pss.experts import (
    CategoryCodebook,
    ExpertHealth,
    NoisyEmbeddingWrapper,
)
from pss.memory.perception import Observation
from pss.metrics.property import (
    interference,
    mean_swap_latency,
    rollback_exactness,
    routing_correctness,
    swap_success_rate,
    zero_shot_compatibility,
)
from pss.projection.features import load_paired_features
from pss.projection.heads import load_projection
from pss.property.swap import ModuleRegistry
from pss.prototypes.bank import PrototypeBank

MODALITIES = ("vision", "tactile", "geometry")


@dataclass
class L1Assets:
    bank: PrototypeBank
    codebook: CategoryCodebook
    observations: list[Observation]
    heads: dict
    device: str
    metadata: dict


def _as_items(obs):
    if isinstance(obs, (list, tuple)):
        return list(obs)
    return [obs]


class RawFeatureExpert:
    """Expert over precomputed ``Observation.raw[modality]`` embeddings."""

    def __init__(self, modality: str, dim: int = 128, version: str | None = None):
        self.modality = modality
        self.dim = int(dim)
        self.version = version or f"raw-{modality}-v1"

    def encode(self, obs) -> np.ndarray:
        rows = []
        for item in _as_items(obs):
            vec = np.asarray(item.raw.get(self.modality, np.zeros(self.dim)), dtype=np.float32)
            n = float(np.linalg.norm(vec))
            rows.append(vec / n if n > 0 else np.zeros(self.dim, dtype=np.float32))
        return np.stack(rows).astype(np.float32) if rows else np.zeros((0, self.dim), np.float32)

    def health(self) -> ExpertHealth:
        return ExpertHealth()


def _project(feats, heads, device: str) -> dict[str, np.ndarray]:
    with torch.no_grad():
        return {
            m: heads[m](
                torch.as_tensor(getattr(feats, m), dtype=torch.float32, device=device)
            ).cpu().numpy()
            for m in MODALITIES
        }


def _make_observations(
    feats,
    Z: dict[str, np.ndarray],
    categories: set[str],
    split: str | None = None,
) -> list[Observation]:
    out: list[Observation] = []
    for i, category in enumerate(feats.categories):
        if category not in categories:
            continue
        if split is not None and str(feats.splits[i]) != str(split):
            continue
        raw = {m: Z[m][i].astype(np.float32) for m in MODALITIES}
        pos = np.array([(i % 10) * 0.1, ((i // 10) % 10) * 0.1, 0.5], dtype=float)
        out.append(
            Observation(
                gt_id=str(feats.instances[i] or feats.sample_ids[i]),
                pos=pos,
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
                appearance=raw["vision"],
                raw=raw,
                confidence=1.0,
                category=str(category),
                frame=int(i),
                gt_category=str(category),
                category_score=1.0,
                unknown=False,
                conflict=False,
                modality_reliability={m: 1.0 for m in MODALITIES},
            )
        )
    return out


def load_l1_assets(
    data_dir: str | Path = "data/expert_v1",
    expert_dir: str | Path = "checkpoints/expert_v1",
    projection_dir: str | Path = "checkpoints/projection_v1",
    prototype_dir: str | Path = "checkpoints/prototypes_v1",
    device: str | None = None,
    max_samples: int | None = 400,
    split: str | None = "test",
) -> L1Assets:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    expert_dir = Path(expert_dir)
    projection_dir = Path(projection_dir)
    prototype_dir = Path(prototype_dir)

    vision = load_vision_expert(expert_dir / "vision.pt", device=device)
    tactile = load_tactile_expert(expert_dir / "tactile.pt", device=device)
    heads = {m: load_projection(projection_dir / f"{m}.pt", device=device) for m in MODALITIES}
    bank = PrototypeBank.load(prototype_dir / "prototypes.npz")

    feats = load_paired_features(data_dir, vision, tactile)
    Z = _project(feats, heads, device)
    observations = _make_observations(feats, Z, set(bank.categories), split=split)
    if max_samples is not None:
        observations = observations[: int(max_samples)]
    codebook = CategoryCodebook.from_vectors(bank.categories, bank.protos["vision"])
    metadata = {
        "data_dir": str(data_dir),
        "expert_dir": str(expert_dir),
        "projection_dir": str(projection_dir),
        "prototype_dir": str(prototype_dir),
        "n_samples": len(observations),
        "classes": list(bank.categories),
        "device": device,
    }
    return L1Assets(
        bank=bank,
        codebook=codebook,
        observations=observations,
        heads=heads,
        device=device,
        metadata=metadata,
    )


def _predict(codebook: CategoryCodebook, vectors: np.ndarray) -> list[str]:
    return [codebook.predict(v) for v in np.asarray(vectors)]


def run_l1_swap(assets: L1Assets, seed: int = 0) -> dict:
    """Swap real-embedding experts at controlled quality levels."""

    obs = assets.observations
    probe = obs[:8]
    codebook = assets.codebook
    dim = int(codebook.dim)

    registry = ModuleRegistry(public_dim=dim, probe=probe)
    raw = {m: RawFeatureExpert(m, dim=dim) for m in MODALITIES}
    for expert in raw.values():
        registry.register(expert, probe=probe)

    gt = [o.gt_category for o in obs]
    per_level: dict[str, dict] = {}
    successes: list[bool] = []
    zcrs: list[float] = []
    routing: list[bool] = []
    rollbacks: list[float] = []
    latencies: list[float] = []
    interferences: list[float] = []

    for level, accuracy in (("Q0", 1.0), ("Q1", 0.90), ("Q2", 0.75), ("Q3", 0.55), ("Q4", 0.0)):
        candidate = NoisyEmbeddingWrapper(raw["vision"], accuracy=accuracy, seed=seed)
        vision_before = registry.behavior_hash("vision", probe=probe)
        geo_before = registry.behavior_hash("geometry", probe=probe)
        swap = registry.swap(candidate, probe=probe)
        successes.append(swap.success)
        latencies.append(swap.latency_ms)
        preds = _predict(codebook, candidate.encode(obs))
        zcr = zero_shot_compatibility(preds, gt)
        zcrs.append(zcr)
        routing.append(registry.get("vision").version == candidate.version)
        interferences.append(
            0.0 if registry.behavior_hash("geometry", probe=probe) == geo_before else 1.0
        )
        registry.rollback()
        rollbacks.append(
            rollback_exactness(vision_before, registry.behavior_hash("vision", probe=probe))
        )
        per_level[level] = {
            "accuracy_target": float(accuracy),
            "zcr": float(zcr),
            "zcr_ratio": float(
                zcr / max(accuracy * per_level.get("Q0", {}).get("zcr", 1.0), 1e-9)
            ),
            "swap_success": bool(swap.success),
            "swap_latency_ms": float(swap.latency_ms),
            "rollback": float(rollbacks[-1]),
            "int": float(interferences[-1]),
            "dt": 0.0,
        }

    ratios = [per_level[l]["zcr_ratio"] for l in ("Q1", "Q2", "Q3")]
    metrics = {
        "SSR_L1": swap_success_rate(successes),
        "ZCR_L1": float(np.mean(zcrs)),
        "ZCR_same_modality_L1": float(per_level["Q1"]["zcr"]),
        "ZCR_ratio_Q1": float(per_level["Q1"]["zcr_ratio"]),
        "ZCR_ratio_mean": float(np.mean(ratios)),
        "INT_L1": float(max(interferences)),
        "RBE_L1": float(np.mean(rollbacks)),
        "RCR_L1": routing_correctness(sum(routing), len(routing)),
        "swap_latency_ms_L1": mean_swap_latency(latencies),
    }
    return {"metrics": metrics, "per_level": per_level, "metadata": assets.metadata}


def run_l1_family_swap(
    assets: L1Assets,
    alt_data_dir: str | Path = "data/category_pretrain",
    alt_expert_dir: str | Path = "checkpoints/category_pretrain_experts",
    max_per_class: int | None = 40,
    seed: int = 0,
    split: str | None = "test",
) -> dict:
    """Freeze the shared space and swap to a different expert family (H7)."""

    alt_expert_dir = Path(alt_expert_dir)
    vision = load_vision_expert(alt_expert_dir / "vision.pt", device=assets.device)
    tactile = load_tactile_expert(alt_expert_dir / "tactile.pt", device=assets.device)
    feats = load_paired_features(alt_data_dir, vision, tactile)
    Z = _project(feats, assets.heads, assets.device)

    common = set(assets.bank.categories)
    codebooks = {
        modality: CategoryCodebook.from_vectors(
            assets.bank.categories, assets.bank.protos[modality]
        )
        for modality in ("vision", "tactile")
    }
    counts: dict[str, int] = {}
    preds = {m: [] for m in ("vision", "tactile")}
    gt: list[str] = []
    for i, category in enumerate(feats.categories):
        if category not in common:
            continue
        if split is not None and str(feats.splits[i]) != str(split):
            continue
        if max_per_class is not None and counts.get(category, 0) >= max_per_class:
            continue
        counts[category] = counts.get(category, 0) + 1
        gt.append(category)
        for modality in preds:
            preds[modality].append(codebooks[modality].predict(Z[modality][i]))

    out = {
        "alt_data": str(alt_data_dir),
        "alt_experts": str(alt_expert_dir),
        "n_samples": len(gt),
        "common_classes": sorted(common),
        "per_class_counts": counts,
    }
    for modality in preds:
        out[f"ZCR_{modality}"] = zero_shot_compatibility(preds[modality], gt)
    out["chance"] = 1.0 / max(len(assets.bank.categories), 1)
    return out


def _ridge_map(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    """Solve ``X @ W ~= Y`` with a small ridge term."""

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    d = X.shape[1]
    A = X.T @ X + ridge * np.eye(d)
    B = X.T @ Y
    return np.linalg.solve(A, B)


def run_l1_family_calibration(
    assets: L1Assets,
    alt_data_dir: str | Path = "data/category_pretrain",
    alt_expert_dir: str | Path = "checkpoints/category_pretrain_experts",
    ks=(0, 1, 5, 20),
    max_test_per_class: int | None = 40,
    ridge: float = 1e-2,
) -> dict:
    """Few-shot linear calibration of the alt expert family into the frozen space.

    ``k=0`` is the zero-shot condition.  For ``k>0`` we fit ``W`` so that the
    alt embeddings of k training examples per class map close to that class's
    prototype, then evaluate on the alt test split.
    """

    alt_expert_dir = Path(alt_expert_dir)
    vision = load_vision_expert(alt_expert_dir / "vision.pt", device=assets.device)
    tactile = load_tactile_expert(alt_expert_dir / "tactile.pt", device=assets.device)
    feats = load_paired_features(alt_data_dir, vision, tactile)
    Z = _project(feats, assets.heads, assets.device)

    common = set(assets.bank.categories)
    codebooks = {
        modality: CategoryCodebook.from_vectors(
            assets.bank.categories, assets.bank.protos[modality]
        )
        for modality in ("vision", "tactile")
    }
    by_class_train: dict[str, list[int]] = {c: [] for c in common}
    test_indices: list[int] = []
    test_counts: dict[str, int] = {}
    for i, category in enumerate(feats.categories):
        if category not in common:
            continue
        split = str(feats.splits[i])
        if split == "train":
            by_class_train[category].append(i)
        elif split == "test":
            if max_test_per_class is not None and test_counts.get(category, 0) >= max_test_per_class:
                continue
            test_counts[category] = test_counts.get(category, 0) + 1
            test_indices.append(i)

    gt = [str(feats.categories[i]) for i in test_indices]
    out: dict = {
        "alt_data": str(alt_data_dir),
        "alt_experts": str(alt_expert_dir),
        "n_test": len(test_indices),
        "ks": list(ks),
        "chance": 1.0 / max(len(assets.bank.categories), 1),
        "per_modality": {},
    }
    for modality in ("vision", "tactile"):
        codebook = codebooks[modality]
        curves: dict[int, float] = {}
        for k in ks:
            if k == 0:
                W = np.eye(codebook.dim, dtype=np.float64)
            else:
                X, Y = [], []
                for category in common:
                    idx = by_class_train.get(category, [])[: int(k)]
                    for i in idx:
                        X.append(Z[modality][i])
                        Y.append(codebook.vector(category))
                if len(X) < 2:
                    curves[int(k)] = float("nan")
                    continue
                W = _ridge_map(np.stack(X), np.stack(Y), ridge=ridge)
            preds = [codebook.predict(np.asarray(Z[modality][i], dtype=float) @ W) for i in test_indices]
            curves[int(k)] = zero_shot_compatibility(preds, gt)
        out["per_modality"][modality] = {str(k): float(v) for k, v in curves.items()}
    return out

