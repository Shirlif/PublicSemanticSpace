"""Per-modality prototype bank (Phase 5).

A prototype is the normalized centroid of a category's training embeddings in
one modality. Prototypes are stored separately per modality; fusion happens at
query time by confidence-weighted cosine, never by summing vectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def l2(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_bank(feats, Z: dict[str, np.ndarray], train_idx, classes: list[str], version: str = "prototypes_v1") -> "PrototypeBank":
    train_idx = list(train_idx)
    protos: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {c: {} for c in classes}
    for modality, arr in Z.items():
        rows = []
        for c in classes:
            idx = [i for i in train_idx if feats.categories[i] == c]
            if not idx:
                raise ValueError(f"no training samples for category {c!r}")
            v = arr[idx].astype(np.float64)
            centroid = l2(v.mean(0))
            variance = float(np.mean(1.0 - v @ centroid))
            rows.append(centroid)
            stats[c][modality] = {"sample_count": len(idx), "variance": variance}
        protos[modality] = np.stack(rows).astype(np.float32)
    return PrototypeBank(categories=list(classes), protos=protos, stats=stats, version=version)


@dataclass
class PrototypeBank:
    categories: list[str]
    protos: dict[str, np.ndarray]  # modality -> (C, D), unit norm
    stats: dict  # category -> modality -> {sample_count, variance}
    version: str = "prototypes_v1"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "categories_json": np.array(json.dumps(self.categories, ensure_ascii=False)),
            "stats_json": np.array(json.dumps(self.stats, ensure_ascii=False)),
            "version": np.array(self.version),
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
            protos = {
                k[len("proto_") :]: f[k].astype(np.float32)
                for k in f.files
                if k.startswith("proto_")
            }
        return cls(categories=categories, protos=protos, stats=stats, version=version)

    def modality_scores(self, z, modality: str) -> np.ndarray:
        z = l2(z)
        return self.protos[modality] @ z

    def ann_topk(self, z, modality: str, k: int = 3) -> list[tuple[str, float]]:
        sims = self.modality_scores(z, modality)
        order = np.argsort(-sims)[:k]
        return [(self.categories[int(i)], float(sims[i])) for i in order]
