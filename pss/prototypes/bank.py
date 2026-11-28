"""Per-modality prototype bank (Phase 5).

A prototype is the normalized centroid of a category's training embeddings in
one modality. Prototypes are stored separately per modality; fusion happens at
query time by confidence-weighted cosine, never by summing vectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pss.contracts.schema import SPACE_VERSION


def l2(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_bank(
    feats,
    Z: dict[str, np.ndarray],
    train_idx: list[int] | dict[str, list[int]] | np.ndarray,
    classes: list[str],
    version: str = "prototypes_v1",
    *,
    missing_policy: str | dict[str, str] = "skip",
    modality_versions: dict[str, str] | None = None,
) -> "PrototypeBank":
    """Build per-modality prototypes.

    ``train_idx`` is either one index array shared by every modality, or a
    mapping ``modality -> indices`` for modality-specific batches.  The latter
    is what makes "ingest one modality at a time" possible: a modality may
    cover only a subset of categories.

    ``missing_policy`` controls a category that has no samples in a modality:

    - ``skip`` (default): store a zero row and mark it missing; recognition
      ignores that (category, modality) pair.
    - ``impute``: fill with the modality's global centroid, mark it imputed.
    - ``raise``: preserve the legacy strict behaviour.

    Passing a dict lets each modality choose its own policy.
    """
    if isinstance(train_idx, dict):
        global_idx = sorted({int(i) for values in train_idx.values() for i in values})
    else:
        global_idx = [int(i) for i in train_idx]
    protos: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {c: {} for c in classes}
    coverage: dict[str, dict[str, int]] = {}
    missing: dict[str, list[str]] = {}
    imputed: dict[str, list[str]] = {}
    for modality, arr in Z.items():
        policy = (
            missing_policy.get(modality, "skip")
            if isinstance(missing_policy, dict)
            else missing_policy
        )
        if policy not in {"skip", "impute", "raise"}:
            raise ValueError(f"unknown missing_policy {policy!r} for {modality!r}")
        rows = []
        coverage[modality] = {}
        missing[modality] = []
        imputed[modality] = []
        indices = (
            [int(i) for i in train_idx.get(modality, global_idx)]
            if isinstance(train_idx, dict)
            else global_idx
        )
        if indices:
            global_centroid = l2(arr[indices].astype(np.float64).mean(0))
        else:
            global_centroid = np.zeros(int(np.asarray(arr).shape[1]), dtype=np.float64)
        for c in classes:
            idx = [i for i in indices if feats.categories[i] == c]
            coverage[modality][c] = len(idx)
            if not idx:
                if policy == "raise":
                    raise ValueError(f"no training samples for category {c!r}")
                missing[modality].append(c)
                if policy == "impute":
                    rows.append(global_centroid)
                    imputed[modality].append(c)
                    stats[c][modality] = {
                        "sample_count": 0,
                        "variance": 0.0,
                        "imputed": True,
                    }
                else:
                    width = int(np.asarray(arr).shape[1])
                    rows.append(np.zeros(width, dtype=np.float64))
                    stats[c][modality] = {
                        "sample_count": 0,
                        "variance": 0.0,
                        "missing": True,
                    }
                continue
            v = arr[idx].astype(np.float64)
            centroid = l2(v.mean(0))
            variance = float(np.mean(1.0 - v @ centroid))
            rows.append(centroid)
            stats[c][modality] = {"sample_count": len(idx), "variance": variance}
        protos[modality] = np.stack(rows).astype(np.float32)
    return PrototypeBank(
        categories=list(classes),
        protos=protos,
        stats=stats,
        version=version,
        coverage=coverage,
        missing=missing,
        imputed=imputed,
        modality_versions=dict(modality_versions or {}),
    )


@dataclass
class PrototypeBank:
    categories: list[str]
    protos: dict[str, np.ndarray]  # modality -> (C, D), unit norm
    stats: dict  # category -> modality -> {sample_count, variance}
    version: str = "prototypes_v1"
    # Compatibility fields (defaults keep old banks loadable):
    coverage: dict[str, dict[str, int]] = field(default_factory=dict)
    missing: dict[str, list[str]] = field(default_factory=dict)
    imputed: dict[str, list[str]] = field(default_factory=dict)
    modality_versions: dict[str, str] = field(default_factory=dict)
    space_version: str = SPACE_VERSION

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "categories_json": np.array(json.dumps(self.categories, ensure_ascii=False)),
            "stats_json": np.array(json.dumps(self.stats, ensure_ascii=False)),
            "version": np.array(self.version),
            "coverage_json": np.array(json.dumps(self.coverage, ensure_ascii=False)),
            "missing_json": np.array(json.dumps(self.missing, ensure_ascii=False)),
            "imputed_json": np.array(json.dumps(self.imputed, ensure_ascii=False)),
            "modality_versions_json": np.array(
                json.dumps(self.modality_versions, ensure_ascii=False)
            ),
            "space_version": np.array(self.space_version),
        }
        for modality, arr in self.protos.items():
            payload[f"proto_{modality}"] = np.asarray(arr, dtype=np.float32)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeBank":
        with np.load(Path(path), allow_pickle=False) as f:
            categories = json.loads(str(f["categories_json"]))
            stats = json.loads(str(f["stats_json"]))
            version = str(f["version"])
            coverage = (
                json.loads(str(f["coverage_json"])) if "coverage_json" in f.files else {}
            )
            missing = (
                json.loads(str(f["missing_json"])) if "missing_json" in f.files else {}
            )
            imputed = (
                json.loads(str(f["imputed_json"])) if "imputed_json" in f.files else {}
            )
            modality_versions = (
                json.loads(str(f["modality_versions_json"]))
                if "modality_versions_json" in f.files
                else {}
            )
            space_version = (
                str(f["space_version"]) if "space_version" in f.files else "0.1.1"
            )
            protos = {
                k[len("proto_") :]: f[k].astype(np.float32)
                for k in f.files
                if k.startswith("proto_")
            }
        return cls(
            categories=categories,
            protos=protos,
            stats=stats,
            version=version,
            coverage=coverage,
            missing=missing,
            imputed=imputed,
            modality_versions=modality_versions,
            space_version=space_version,
        )

    def has_prototype(self, modality: str, category: str) -> bool:
        """True when this (modality, category) has a usable prototype."""
        if modality not in self.protos or category not in self.categories:
            return False
        index = self.categories.index(category)
        if category in self.missing.get(modality, []) and category not in self.imputed.get(
            modality, []
        ):
            return False
        return float(np.linalg.norm(self.protos[modality][index])) > 1e-8

    def valid_mask(self, modality: str) -> np.ndarray:
        return np.asarray(
            [self.has_prototype(modality, category) for category in self.categories],
            dtype=bool,
        )

    def modality_scores(self, z, modality: str) -> np.ndarray:
        z = l2(z)
        return self.protos[modality] @ z

    def ann_topk(self, z, modality: str, k: int = 3) -> list[tuple[str, float]]:
        sims = np.asarray(self.modality_scores(z, modality), dtype=np.float64).copy()
        sims[~self.valid_mask(modality)] = -np.inf
        order = np.argsort(-sims)[:k]
        return [
            (self.categories[int(i)], float(sims[i]))
            for i in order
            if np.isfinite(sims[i])
        ]
