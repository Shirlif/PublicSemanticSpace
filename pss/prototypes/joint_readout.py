"""Joint discriminative readout for the public semantic space.

The deployed readout is a per-modality prototype vote: each modality ranks
classes by cosine to its centroid, and the modality scores are fused.  That is
cheap and hot-swappable, but it cannot represent cross-modal combinations: on
the 384-class cross-view task the prototype readout scores 0.833 while a
pooled-covariance LDA on the concatenated 128-d features scores 0.927.

This module keeps the same public-space contract (128-d per modality) and adds
a joint readout:

- one pooled-covariance LDA per non-empty modality mask, trained on the
  concatenated standardized features;
- hot updates move a class mean and leave the pooled precision frozen, so an
  update stays O(d) and affects only that class;
- open-set rejection uses the raw max score (``NoveltyMap``-compatible), not a
  per-query normalized fusion.

The prototype bank stays available for per-modality diagnostics and conflict
detection; the joint readout decides the final ranking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from .recognize import RecognitionResult
from .scorers import MahalanobisScorer

MODALITIES = ("vision", "tactile", "geometry")


def _canonical(mask, modalities) -> tuple[str, ...]:
    keep = set(mask)
    return tuple(m for m in modalities if m in keep)


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - scores.max()
    exp = np.exp(shifted)
    return exp / max(exp.sum(), 1e-12)


def _masks(modalities: tuple[str, ...]):
    for size in range(1, len(modalities) + 1):
        for subset in combinations(modalities, size):
            yield tuple(subset)


@dataclass
class JointReadout:
    """Concatenated-feature LDA with one model per available modality mask."""

    classes: list[str]
    modalities: tuple[str, ...]
    scalers: dict[str, tuple[np.ndarray, np.ndarray]]
    models: dict[tuple[str, ...], MahalanobisScorer]
    version: str = "joint_readout_v1"
    threshold: float | None = None
    metadata: dict = field(default_factory=dict)

    # ── fitting ─────────────────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        Z: dict[str, np.ndarray],
        y: np.ndarray,
        classes: list[str],
        modalities: tuple[str, ...] = MODALITIES,
        version: str = "joint_readout_v1",
        shrinkage: float = 1e-3,
    ) -> "JointReadout":
        modalities = tuple(m for m in modalities if m in Z)
        if not modalities:
            raise ValueError("joint readout needs at least one modality")
        y = np.asarray(y, dtype=np.int64)
        scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for modality in modalities:
            arr = np.asarray(Z[modality], dtype=np.float64)
            mean = arr.mean(axis=0)
            scale = arr.std(axis=0)
            scalers[modality] = (mean, np.clip(scale, 1e-8, None))

        models: dict[tuple[str, ...], MahalanobisScorer] = {}
        for mask in _masks(modalities):
            X = cls._concat(Z, scalers, mask)
            models[mask] = MahalanobisScorer(shrinkage=shrinkage).fit(X, y)
        return cls(
            classes=list(classes),
            modalities=modalities,
            scalers=scalers,
            models=models,
            version=version,
            metadata={"shrinkage": float(shrinkage)},
        )

    @staticmethod
    def _concat(Z: dict[str, np.ndarray], scalers, mask) -> np.ndarray:
        blocks = []
        for modality in mask:
            mean, scale = scalers[modality]
            blocks.append((np.asarray(Z[modality], dtype=np.float64) - mean) / scale)
        return np.concatenate(blocks, axis=-1)

    # ── query ───────────────────────────────────────────────────────────────

    def _available(self, query: dict[str, np.ndarray]) -> tuple[str, ...]:
        return _canonical(
            [
                m
                for m in self.modalities
                if m in query and float(np.linalg.norm(np.asarray(query[m]))) > 1e-8
            ],
            self.modalities,
        )

    def _model_mask(self, available: tuple[str, ...]) -> tuple[str, ...]:
        if available in self.models:
            return available
        for size in range(len(available), 0, -1):
            for subset in combinations(available, size):
                if subset in self.models:
                    return tuple(subset)
        return ()

    def scores(self, query: dict[str, np.ndarray]) -> np.ndarray:
        available = self._available(query)
        mask = self._model_mask(available)
        if not mask:
            raise ValueError("joint readout: no usable modality in query")
        z = self._concat(query, self.scalers, mask)
        return np.asarray(self.models[mask].scores(z.reshape(1, -1))[0])

    def scores_batch(self, Z: dict[str, np.ndarray]) -> np.ndarray:
        available = self._available(Z)
        mask = self._model_mask(available)
        if not mask:
            raise ValueError("joint readout: no usable modality in batch")
        X = self._concat(Z, self.scalers, mask)
        return np.asarray(self.models[mask].scores(X))

    def predict(self, query: dict[str, np.ndarray]) -> str:
        return self.classes[int(np.argmax(self.scores(query)))]

    # ── hot update ──────────────────────────────────────────────────────────

    def _index(self, label: str) -> int:
        if label in self.classes:
            return self.classes.index(label)
        self.classes.append(label)
        return len(self.classes) - 1

    def add_class(self, label: str, query: dict[str, np.ndarray]) -> None:
        """Add a class mean to every mask model; the precision stays frozen."""
        idx = self._index(label)
        for mask, scorer in self.models.items():
            z = self._concat(query, self.scalers, mask).reshape(1, -1)
            if idx in set(int(c) for c in np.asarray(scorer.classes_)):
                scorer.update_class(idx, z)
            else:
                scorer.add_class(idx, z)

    def update_class(self, label: str, query: dict[str, np.ndarray]) -> None:
        """Move one class centroid; other classes and the precision are untouched."""
        if label not in self.classes:
            self.add_class(label, query)
            return
        idx = self.classes.index(label)
        for mask, scorer in self.models.items():
            z = self._concat(query, self.scalers, mask).reshape(1, -1)
            scorer.update_class(idx, z)

    def snapshot(self) -> dict:
        return {
            "classes": list(self.classes),
            "models": {
                mask: (
                    np.array(scorer.classes_, copy=True),
                    np.array(scorer.means_, copy=True),
                    np.array(scorer.precision_, copy=True),
                )
                for mask, scorer in self.models.items()
            },
            "threshold": self.threshold,
        }

    def restore(self, snapshot: dict) -> None:
        self.classes = list(snapshot["classes"])
        for mask, (classes, means, precision) in snapshot["models"].items():
            scorer = self.models[mask]
            scorer.classes_ = np.array(classes, copy=True)
            scorer.means_ = np.array(means, copy=True)
            scorer.precision_ = np.array(precision, copy=True)
            scorer.covariance_frozen = True
        self.threshold = snapshot["threshold"]

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "classes": self.classes,
            "modalities": list(self.modalities),
            "version": self.version,
            "threshold": self.threshold,
            "masks": [list(mask) for mask in self.models],
            "metadata": self.metadata,
        }
        payload = {"meta_json": np.array(json.dumps(meta, ensure_ascii=False))}
        for modality, (mean, scale) in self.scalers.items():
            payload[f"scaler_{modality}_mean"] = mean
            payload[f"scaler_{modality}_scale"] = scale
        for i, (_mask, scorer) in enumerate(self.models.items()):
            payload[f"model_{i}_classes"] = np.asarray(scorer.classes_)
            payload[f"model_{i}_means"] = np.asarray(scorer.means_)
            payload[f"model_{i}_precision"] = np.asarray(scorer.precision_)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "JointReadout":
        with np.load(Path(path), allow_pickle=False) as f:
            meta = json.loads(str(f["meta_json"]))
            modalities = tuple(meta["modalities"])
            scalers = {
                m: (
                    np.asarray(f[f"scaler_{m}_mean"], dtype=np.float64),
                    np.asarray(f[f"scaler_{m}_scale"], dtype=np.float64),
                )
                for m in modalities
            }
            models: dict[tuple[str, ...], MahalanobisScorer] = {}
            for i, raw_mask in enumerate(meta["masks"]):
                mask = tuple(raw_mask)
                scorer = MahalanobisScorer()
                scorer.classes_ = np.asarray(f[f"model_{i}_classes"], dtype=np.int64)
                scorer.means_ = np.asarray(f[f"model_{i}_means"], dtype=np.float64)
                scorer.precision_ = np.asarray(
                    f[f"model_{i}_precision"], dtype=np.float64
                )
                scorer.covariance_frozen = True
                models[mask] = scorer
        return cls(
            classes=list(meta["classes"]),
            modalities=modalities,
            scalers=scalers,
            models=models,
            version=str(meta["version"]),
            threshold=meta.get("threshold"),
            metadata=meta.get("metadata", {}),
        )


def recognize_joint(
    query: dict[str, np.ndarray],
    readout: JointReadout,
    bank=None,
    top_k: int = 3,
    weights: dict[str, float] | None = None,
    threshold: float | None = None,
    conflict_weight: float = 0.75,
) -> RecognitionResult:
    """Rank classes with the joint readout, keeping the prototype diagnostics."""
    if not query:
        return RecognitionResult([], {}, {}, False, True, [])
    raw = readout.scores(query)
    probs = _softmax(raw)
    order = np.argsort(-raw)[:top_k]
    ranked = [(readout.classes[int(i)], float(probs[int(i)])) for i in order]

    per_modality: dict[str, dict[str, float]] = {}
    modality_top: dict[str, str] = {}
    conflict = False
    if bank is not None:
        weights = dict(weights or {m: 1.0 for m in query})
        modalities = [
            m
            for m in query
            if m in bank.protos
            and weights.get(m, 0.0) > 0.0
            and float(np.linalg.norm(np.asarray(query[m]))) > 1e-8
        ]
        for category_idx, category in enumerate(bank.categories):
            per_modality[category] = {
                m: float(
                    np.dot(
                        np.asarray(query[m], dtype=np.float64)
                        / max(float(np.linalg.norm(np.asarray(query[m]))), 1e-12),
                        bank.protos[m][category_idx],
                    )
                )
                for m in modalities
            }
        high = [m for m in modalities if weights.get(m, 0.0) >= conflict_weight]
        for modality in high:
            sims = bank.modality_scores(query[modality], modality)
            modality_top[modality] = bank.categories[int(np.argmax(sims))]
        conflict = len(set(modality_top.values())) > 1

    unknown = bool(threshold is not None and raw.max() < threshold)
    return RecognitionResult(
        ranked=ranked,
        per_modality=per_modality,
        modality_top=modality_top,
        conflict=conflict,
        unknown=unknown,
        candidates=list(readout.classes),
    )


__all__ = ["JointReadout", "MODALITIES", "recognize_joint"]
