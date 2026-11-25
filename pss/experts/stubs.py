"""Synthetic expert stubs for property-level validation.

The stubs implement :class:`pss.experts.contract.ExpertContract` and let the
sandbox vary expert quality without touching perception.  They are not models:
they return unit vectors from a category codebook (or random vectors), which is
enough to exercise the runtime's swap / projection / prototype contracts.

Quality levels (Q0-Q5) map to the stubs in :mod:`pss.experts.quality`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .contract import ExpertHealth


class CategoryCodebook:
    """Deterministic category -> unit-vector mapping."""

    def __init__(self, categories, dim: int = 128, seed: int = 0):
        self.categories = list(dict.fromkeys(str(c) for c in categories))
        if not self.categories:
            raise ValueError("codebook needs at least one category")
        self.dim = int(dim)
        rng = np.random.default_rng(seed)
        vectors = rng.normal(size=(len(self.categories), self.dim))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors.astype(np.float32)
        self.index = {c: i for i, c in enumerate(self.categories)}
        self.seed = int(seed)

    @classmethod
    def from_vectors(cls, categories, vectors, seed: int = 0) -> "CategoryCodebook":
        """Build a codebook from existing prototype vectors (e.g. a bank)."""

        cats = list(dict.fromkeys(str(c) for c in categories))
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(cats):
            raise ValueError(
                f"expected ({len(cats)}, dim) vectors, got {arr.shape}"
            )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.clip(norms, 1e-12, None)
        obj = cls(cats, dim=arr.shape[1], seed=seed)
        obj.vectors = arr.astype(np.float32)
        obj.index = {c: i for i, c in enumerate(cats)}
        return obj

    def vector(self, category) -> np.ndarray:
        if category in self.index:
            return self.vectors[self.index[category]]
        # Unknown categories get a stable vector derived from the string.
        rng = np.random.default_rng(abs(hash(str(category))) % (2**32) + self.seed)
        v = rng.normal(size=self.dim)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def predict(self, vec) -> str:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {arr.shape[0]}")
        sims = self.vectors @ (arr / max(float(np.linalg.norm(arr)), 1e-12))
        return self.categories[int(np.argmax(sims))]


def _category(obs):
    if isinstance(obs, dict):
        return obs.get("gt_category") or obs.get("category")
    return getattr(obs, "gt_category", None) or getattr(obs, "category", None)


def _item_id(obs) -> str:
    if isinstance(obs, dict):
        return str(obs.get("gt_id") or obs.get("id") or "")
    return str(getattr(obs, "gt_id", "") or getattr(obs, "id", "") or "")


def _as_items(obs):
    if isinstance(obs, (list, tuple)):
        return list(obs)
    return [obs]


class BaseStub:
    """Shared plumbing: batching, latency tracking, health reporting."""

    modality = "vision"
    version = "stub-v1"

    def __init__(self, codebook: CategoryCodebook, seed: int = 0, dim: int | None = None):
        self.codebook = codebook
        self.dim = int(dim if dim is not None else codebook.dim)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self._latencies: list[float] = []
        self.target_accuracy: float | None = None
        self.target_ece: float = 0.0
        self.ood_score: float = 0.0
        self.failure_flags: list[str] = []

    # -- contract -----------------------------------------------------------
    def encode(self, obs) -> np.ndarray:
        start = time.perf_counter()
        items = _as_items(obs)
        rows = [self._encode_one(item) for item in items]
        out = np.stack(rows).astype(np.float32) if rows else np.zeros((0, self.dim), np.float32)
        self._latencies.append((time.perf_counter() - start) * 1000.0)
        return out

    def health(self) -> ExpertHealth:
        if self._latencies:
            p50 = float(np.percentile(self._latencies, 50))
            p99 = float(np.percentile(self._latencies, 99))
        else:
            p50 = p99 = 0.0
        return ExpertHealth(
            latency_p50=p50,
            latency_p99=p99,
            calibration_ece=float(self.target_ece),
            ood_score=float(self.ood_score),
            failure_flags=list(self.failure_flags),
        )

    # -- to override --------------------------------------------------------
    def _encode_one(self, obs) -> np.ndarray:
        raise NotImplementedError

    # -- helpers ------------------------------------------------------------
    def _random_unit(self) -> np.ndarray:
        v = self.rng.normal(size=self.dim)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def _wrong_category(self, category) -> str:
        if len(self.codebook.categories) <= 1:
            return self.codebook.categories[0]
        others = [c for c in self.codebook.categories if c != category]
        return str(self.rng.choice(others))

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{type(self).__name__}(modality={self.modality!r}, version={self.version!r})"


class OracleExpert(BaseStub):
    """Perfect category embedding (ceiling; not a capability claim)."""

    version = "oracle-v1"

    def __init__(self, codebook, seed=0, modality="vision"):
        super().__init__(codebook, seed=seed)
        self.modality = modality
        self.target_accuracy = 1.0

    def _encode_one(self, obs) -> np.ndarray:
        category = _category(obs)
        return self.codebook.vector(category)


class NoisyOracleExpert(BaseStub):
    """Oracle with a controlled error rate; used for Q1-Q3.

    ``encode`` flips exactly ``round((1 - accuracy) * B)`` items per batch, so
    the measured accuracy is stable across runs instead of sampling noise.
    """

    def __init__(self, codebook, accuracy: float = 0.75, ece: float = 0.1, seed=0, modality="vision"):
        super().__init__(codebook, seed=seed)
        self.modality = modality
        self.accuracy = float(np.clip(accuracy, 0.0, 1.0))
        self.target_accuracy = self.accuracy
        self.target_ece = float(ece)
        self.version = f"noisy-{self.accuracy:.2f}-v1"

    def encode(self, obs) -> np.ndarray:
        items = _as_items(obs)
        n = len(items)
        n_wrong = int(round((1.0 - self.accuracy) * n))
        indices = set(self.rng.permutation(n)[:n_wrong].tolist()) if n_wrong else set()
        rows = []
        for i, item in enumerate(items):
            category = _category(item)
            if i in indices and category is not None:
                rows.append(self.codebook.vector(self._wrong_category(category)))
            else:
                rows.append(self.codebook.vector(category))
        return np.stack(rows).astype(np.float32) if rows else np.zeros((0, self.dim), np.float32)

    def _encode_one(self, obs) -> np.ndarray:
        category = _category(obs)
        if category is None or self.rng.random() < self.accuracy:
            return self.codebook.vector(category)
        return self.codebook.vector(self._wrong_category(category))


class RandomExpert(BaseStub):
    """Format-valid random embeddings (chance level)."""

    version = "random-v1"

    def __init__(self, codebook, seed=0, modality="vision"):
        super().__init__(codebook, seed=seed)
        self.modality = modality
        self.target_accuracy = 1.0 / max(len(codebook.categories), 1)

    def _encode_one(self, obs) -> np.ndarray:
        return self._random_unit()


class ConstantExpert(BaseStub):
    """A single fixed vector; useful for rejection/degradation tests."""

    version = "constant-v1"

    def __init__(self, codebook, seed=0, modality="vision"):
        super().__init__(codebook, seed=seed)
        self.modality = modality
        self._fixed = self._random_unit()
        self.target_accuracy = 1.0 / max(len(codebook.categories), 1)
        self.failure_flags = ["constant_output"]

    def _encode_one(self, obs) -> np.ndarray:
        return self._fixed


class AdversarialExpert(BaseStub):
    """Systematically maps every category to the next one."""

    version = "adversarial-v1"

    def __init__(self, codebook, seed=0, modality="vision"):
        super().__init__(codebook, seed=seed)
        self.modality = modality
        self.target_accuracy = 0.0
        self.failure_flags = ["systematic_error"]

    def _encode_one(self, obs) -> np.ndarray:
        category = _category(obs)
        if category is None or category not in self.codebook.index:
            return self._random_unit()
        i = self.codebook.index[category]
        wrong = self.codebook.categories[(i + 1) % len(self.codebook.categories)]
        return self.codebook.vector(wrong)


class OODExpert(BaseStub):
    """Random embedding outside the codebook manifold."""

    version = "ood-v1"

    def __init__(self, codebook, seed=0, modality="vision"):
        super().__init__(codebook, seed=seed)
        self.modality = modality
        self.ood_score = 1.0
        self.target_accuracy = 1.0 / max(len(codebook.categories), 1)
        self.failure_flags = ["ood"]

    def _encode_one(self, obs) -> np.ndarray:
        return self._random_unit()


class DelayedExpert(BaseStub):
    """Wraps another expert and adds a fixed delay."""

    version = "delayed-v1"

    def __init__(self, inner, delay_ms: float = 50.0, modality=None):
        super().__init__(inner.codebook, seed=inner.seed, dim=inner.dim)
        self.inner = inner
        self.delay_ms = float(delay_ms)
        self.modality = modality or inner.modality
        self.version = f"delayed-{self.delay_ms:.0f}ms-v1"
        self.target_accuracy = getattr(inner, "target_accuracy", None)
        self.target_ece = getattr(inner, "target_ece", 0.0)
        self.failure_flags = list(getattr(inner, "failure_flags", [])) + ["delayed"]

    def _encode_one(self, obs) -> np.ndarray:
        time.sleep(self.delay_ms / 1000.0)
        return self.inner._encode_one(obs)


class DropoutExpert(BaseStub):
    """Randomly returns zero vectors to simulate a missing modality."""

    version = "dropout-v1"

    def __init__(self, inner, p: float = 0.5, seed=0):
        super().__init__(inner.codebook, seed=seed, dim=inner.dim)
        self.inner = inner
        self.p = float(np.clip(p, 0.0, 1.0))
        self.modality = inner.modality
        self.version = f"dropout-{self.p:.2f}-v1"
        self.target_accuracy = getattr(inner, "target_accuracy", None)
        self.target_ece = getattr(inner, "target_ece", 0.0)
        self.failure_flags = list(getattr(inner, "failure_flags", [])) + ["dropout"]

    def _encode_one(self, obs) -> np.ndarray:
        if self.rng.random() < self.p:
            return np.zeros(self.dim, dtype=np.float32)
        return self.inner._encode_one(obs)


# -- deliberately broken modules for safety-rejection tests ----------------


class WrongDimExpert(BaseStub):
    version = "wrong-dim-v1"

    def __init__(self, codebook, dim=64, seed=0):
        super().__init__(codebook, seed=seed)
        self.dim = int(dim)

    def _encode_one(self, obs):
        v = self._random_unit()[: self.dim]
        return v / max(float(np.linalg.norm(v)), 1e-12)


class NonNormalizedExpert(BaseStub):
    version = "non-normalized-v1"

    def _encode_one(self, obs):
        return self._random_unit() * 0.1


class NaNExpert(BaseStub):
    version = "nan-v1"

    def _encode_one(self, obs):
        v = self._random_unit()
        v[0] = np.nan
        return v
