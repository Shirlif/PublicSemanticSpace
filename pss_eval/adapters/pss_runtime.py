"""``SystemUnderTest`` adapter for the PSS cognitive runtime.

This wraps :class:`pss.runtime.runtime.PSSRuntime` -- the real runtime with its
entity layer, memory/tracker and versioned cognitive state -- behind the
standalone :class:`pss_eval.protocol.SystemUnderTest` interface.  Unlike the
E5 flat adapter, this adapter implements the full contract:

* ``snapshot`` / ``rollback`` capture and restore the runtime's cognitive
  state version together with the module registry's swap stack;
* ``swap`` installs a vision module through :class:`pss.property.swap.ModuleRegistry`,
  which validates dimension / finiteness / unit-norm before mutating anything;
  invalid modules are rejected and leave the state untouched;
* ``audit`` exposes the runtime's versioned audit log plus the registry's
  register/swap/rollback history;
* ``identity_assignments`` exposes the tracker's detection -> track id map for
  identity-retention scoring (ground truth stays in the harness);
* ``budget`` / ``budget_stats`` expose an optional :class:`DeadlineGuard` so
  the anytime property can measure caller latency and timeout accounting.

The adapter may import ``pss``; the protocol layer may not.  It never opens
``gt.parquet``: bundle ground truth is attached by ``pss_eval.bundles`` to the
harness-side ``Stream`` only.

One stream observation is one frame.  For bundle streams the frame's
detections live in ``Observation.metadata["detections"]``; for synthetic
streams the single feature vector is treated as one detection with no
category.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from pss.experts import CategoryCodebook, NaNExpert, NonNormalizedExpert, WrongDimExpert
from pss.experts.contract import ExpertHealth
from pss.memory.perception import Observation as PSSObservation
from pss.property.swap import ModuleRegistry
from pss.runtime.deadline import DeadlineGuard
from pss.runtime.variants import build_runtime

from ..protocol import Observation, Prediction, StateSnapshot

__all__ = ["PSSRuntimeAdapter"]


# ── small expert used as a valid, swappable vision module ─────────────────


@dataclass(frozen=True)
class _RawDetection:
    """Probe item handed to an expert: just the raw detector embedding."""

    appearance: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "appearance", np.asarray(self.appearance, dtype=np.float64).reshape(-1)
        )


def _appearance_of(item: Any) -> np.ndarray:
    if isinstance(item, Mapping):
        for key in ("appearance", "embedding", "features"):
            if key in item and item[key] is not None:
                return np.asarray(item[key], dtype=np.float64).reshape(-1)
    for attr in ("appearance", "embedding", "features"):
        value = getattr(item, attr, None)
        if value is not None:
            return np.asarray(value, dtype=np.float64).reshape(-1)
    raise ValueError("cannot extract an appearance vector from probe item")


class _ProjectedAppearanceExpert:
    """Deterministic linear projection of detector embeddings to public space.

    It satisfies :class:`pss.experts.contract.ExpertContract`: ``encode``
    returns finite, L2-normalised rows of width ``dim``.  Different versions
    use different projections, so swapping changes behaviour as well as the
    active version string.
    """

    modality = "vision"

    def __init__(
        self,
        *,
        version: str,
        input_dim: int,
        output_dim: int,
        seed: int = 0,
        modality: str = "vision",
    ) -> None:
        self.version = str(version)
        self.modality = str(modality)
        self.input_dim = int(input_dim)
        self.dim = int(output_dim)
        rng = np.random.default_rng(int(seed) + 104729)
        self._weights = rng.normal(size=(self.input_dim, self.dim)) / np.sqrt(
            max(1, self.input_dim)
        )

    def encode(self, obs: Any) -> np.ndarray:
        items = list(obs) if isinstance(obs, (list, tuple)) else [obs]
        rows: list[np.ndarray] = []
        for item in items:
            vector = _appearance_of(item)
            if vector.shape[0] != self.input_dim:
                raise ValueError(
                    f"expected appearance dim {self.input_dim}, got {vector.shape[0]}"
                )
            out = np.asarray(vector, dtype=np.float64) @ self._weights
            norm = float(np.linalg.norm(out))
            if norm <= 1e-12:
                out = np.zeros(self.dim, dtype=np.float64)
                out[0] = 1.0
            else:
                out = out / norm
            rows.append(out)
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.stack(rows)

    def health(self) -> ExpertHealth:
        return ExpertHealth()


# ── adapter ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _DecodedDetection:
    ref: str
    pos: np.ndarray
    quat: np.ndarray
    score: float
    category: str | None
    embedding: np.ndarray


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:8], 16)


class PSSRuntimeAdapter:
    """``SystemUnderTest`` view of the PSS cognitive runtime."""

    name = "pss-runtime"

    def __init__(
        self,
        *,
        input_dim: int = 32,
        public_dim: int = 32,
        variant: str = "full",
        tier: str = "frozen_real",
        budget_ms: float | None = None,
        reserve_ms: float = 2.0,
        seed: int = 0,
        probe_size: int = 8,
    ) -> None:
        self.input_dim = int(input_dim)
        self.public_dim = int(public_dim)
        self.variant = str(variant)
        self.tier = str(tier)
        self.budget_ms = None if budget_ms is None else float(budget_ms)
        self.reserve_ms = float(reserve_ms)
        self.probe_size = int(probe_size)
        self.reset(seed=seed)

    # -- lifecycle ---------------------------------------------------------

    def reset(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        if getattr(self, "_guard", None) is not None:
            self._guard.close()
        self.runtime = build_runtime(self.variant, tier=self.tier)
        self.registry = ModuleRegistry(public_dim=self.public_dim, probe=None)
        self._codebook = CategoryCodebook(["__probe__"], dim=self.public_dim, seed=0)
        self._base_expert = _ProjectedAppearanceExpert(
            version="pss-vision-base-v1",
            input_dim=self.input_dim,
            output_dim=self.public_dim,
            seed=self.seed,
        )
        self.registry.register(self._base_expert)
        self._probe: list[_RawDetection] = []
        self._assignments: dict[str, str] = {}
        self._frame_count = 0
        self._swap_depth = 0
        self._last_observation: Observation | None = None
        self._last_prediction: Prediction | None = None
        self._guard = (
            DeadlineGuard(self.budget_ms, reserve_ms=self.reserve_ms)
            if self.budget_ms is not None
            else None
        )
        self._degraded_ingests = 0

    # -- core protocol -----------------------------------------------------

    def ingest(self, observation: Observation) -> None:
        detections = self._decode(observation)
        raw_vectors = [det.embedding for det in detections]
        expert = self.registry.get("vision")
        if expert is None:  # pragma: no cover - registry always has the base
            raise RuntimeError("no active vision module")
        encoded = np.asarray(
            expert.encode([_RawDetection(appearance=vector) for vector in raw_vectors]),
            dtype=np.float64,
        )
        if encoded.ndim != 2 or encoded.shape[0] != len(detections):
            raise RuntimeError("vision module returned the wrong batch shape")

        pss_observations = [
            PSSObservation(
                gt_id=det.ref,
                pos=det.pos.copy(),
                quat=det.quat.copy(),
                appearance=np.asarray(encoded[i], dtype=np.float64),
                raw={
                    "vision": np.asarray(encoded[i], dtype=np.float64),
                    "geometry": np.zeros(4, dtype=np.float64),
                    "tactile": np.zeros(4, dtype=np.float64),
                },
                confidence=float(det.score),
                category=det.category,
                frame=self._frame_count,
                gt_category=det.category,
            )
            for i, det in enumerate(detections)
        ]

        if len(self._probe) < self.probe_size:
            for vector in raw_vectors:
                if len(self._probe) >= self.probe_size:
                    break
                self._probe.append(_RawDetection(appearance=vector))

        frame = self._frame_count
        if self._guard is not None:
            result, timed_out, _elapsed = self._guard.call(
                lambda: self.runtime.ingest(pss_observations, frame), frame=frame
            )
            if timed_out:
                self._degraded_ingests += 1
                result = None
        else:
            result = self.runtime.ingest(pss_observations, frame)

        if result is not None:
            events = result.get("events", {})
            for kind in ("matched", "new", "reid"):
                for track_id, identity in events.get(kind, []):
                    if identity is not None:
                        self._assignments[str(identity)] = str(track_id)

        self._last_observation = observation
        self._last_prediction = self._predict(observation, detections, encoded)
        self._frame_count += 1

    def step(self) -> Prediction:
        if self._last_prediction is None:
            raise RuntimeError("step() called before ingest()")
        return self._last_prediction

    def state(self) -> StateSnapshot:
        self._settle()
        state = self.runtime.state
        entities = []
        for entity_id in sorted(state.entities):
            entity = state.entities[entity_id]
            entities.append(
                {
                    "id": entity.id,
                    "category": entity.category,
                    "pos": np.round(
                        np.asarray(entity.pos, dtype=np.float64).reshape(-1)[:3], 6
                    ),
                    "quat": np.round(
                        np.asarray(entity.quat, dtype=np.float64).reshape(-1)[:4], 6
                    ),
                    "confidence": round(float(entity.confidence), 6),
                    "status": str(entity.status),
                    "last_seen": int(entity.last_seen),
                    "observations": int(entity.observations),
                    "active": bool(entity.active),
                }
            )
        # The append-only audit log is deliberately excluded: PSS rollback
        # appends an audit entry by design, and the contract property compares
        # the *cognitive* state digest before and after rollback.
        return {
            "system": self.name,
            "seed": self.seed,
            "variant": self.variant,
            "tier": self.tier,
            "frame_count": int(self._frame_count),
            "version": int(state.version),
            "entities": entities,
            "n_relations": int(len(state.relations)),
            "n_facts": int(len(state.facts)),
            "goals": sorted(str(goal) for goal in state.goals),
            "plans": sorted(str(plan) for plan in state.plans),
            "active_versions": dict(sorted(self.registry.active_versions().items())),
            "swap_depth": int(self._swap_depth),
        }

    # -- optional capabilities --------------------------------------------

    def snapshot(self) -> str:
        self._settle()
        payload = {
            "version": int(self.runtime.state.version),
            "swap_depth": int(self._swap_depth),
            "active_versions": self.registry.active_versions(),
            "frame_count": int(self._frame_count),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def rollback(self, token: str) -> bool:
        self._settle()
        try:
            payload = json.loads(token)
            target_depth = int(payload["swap_depth"])
            while self._swap_depth > target_depth:
                result = self.registry.rollback()
                if not result.success:
                    return False
                self._swap_depth -= 1
            if self._swap_depth != target_depth:
                return False
            self.runtime.rollback(int(payload["version"]))
            self._frame_count = int(payload.get("frame_count", self._frame_count))
            return dict(self.registry.active_versions()) == dict(
                payload.get("active_versions", {})
            )
        except Exception:  # noqa: BLE001 - a failed rollback must report False
            return False

    def swap(self, module: Mapping[str, Any]) -> bool:
        self._settle()
        try:
            expert = self._make_module(module)
            probe = self._probe or [
                _RawDetection(
                    appearance=np.full(self.input_dim, 1.0 / np.sqrt(self.input_dim))
                )
            ]
            result = self.registry.swap(expert, probe=probe)
        except Exception:  # noqa: BLE001 - rejected descriptors are a False result
            return False
        if result.success:
            self._swap_depth += 1
        return bool(result.success)

    def audit(self) -> list[dict[str, Any]]:
        self._settle()
        entries = [entry.to_dict() for entry in self.runtime.state.audit]
        entries.extend(dict(entry) for entry in self.registry.history())
        return entries

    def identity_assignments(self) -> Mapping[str, str]:
        self._settle()
        return dict(self._assignments)

    def budget(self) -> Mapping[str, Any]:
        if self._guard is None:
            return {}
        return {
            "budget_ms": float(self._guard.budget_ms),
            "reserve_ms": float(self._guard.reserve_ms),
        }

    def budget_stats(self) -> Mapping[str, Any]:
        if self._guard is None:
            return {}
        return self._guard.snapshot().to_dict()

    # -- helpers -----------------------------------------------------------

    def _settle(self) -> None:
        """Join a timed-out ingest worker before inspecting/mutating state."""

        if self._guard is not None:
            self._guard.wait_pending()

    def _make_module(self, module: Mapping[str, Any]):
        kind = str(module.get("kind", "quality"))
        modality = str(module.get("modality", "vision"))
        if kind == "invalid_dim":
            return WrongDimExpert(self._codebook, dim=self.public_dim + 32)
        if kind == "invalid_norm":
            return NonNormalizedExpert(self._codebook)
        if kind == "invalid_nan":
            return NaNExpert(self._codebook)
        version = str(module.get("version", "pss-eval-v1"))
        return _ProjectedAppearanceExpert(
            version=version,
            input_dim=self.input_dim,
            output_dim=self.public_dim,
            seed=_stable_seed(version),
            modality=modality,
        )

    def _decode(self, observation: Observation) -> list[_DecodedDetection]:
        metadata = observation.metadata or {}
        raw = metadata.get("detections")
        if raw:
            detections = [
                _DecodedDetection(
                    ref=str(item["ref"]),
                    pos=np.asarray(item["pos"], dtype=np.float64).reshape(3),
                    quat=np.asarray(item["quat"], dtype=np.float64).reshape(4),
                    score=float(item.get("score", 1.0)),
                    category=(
                        None if item.get("category") is None else str(item["category"])
                    ),
                    embedding=np.asarray(
                        item["embedding"], dtype=np.float64
                    ).reshape(-1),
                )
                for item in raw
            ]
        else:
            detections = [
                _DecodedDetection(
                    ref=observation.observation_id,
                    pos=np.zeros(3, dtype=np.float64),
                    quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                    score=1.0,
                    category=None,
                    embedding=np.asarray(
                        observation.features, dtype=np.float64
                    ).reshape(-1),
                )
            ]
        for det in detections:
            if det.embedding.shape[0] != self.input_dim:
                raise ValueError(
                    f"{self.name} expects embedding dim {self.input_dim}, "
                    f"got {det.embedding.shape[0]}"
                )
        return detections

    def _predict(
        self,
        observation: Observation,
        detections: list[_DecodedDetection],
        encoded: np.ndarray,
    ) -> Prediction:
        if encoded.size == 0:
            embedding = np.zeros(self.public_dim, dtype=np.float64)
        else:
            embedding = np.asarray(encoded, dtype=np.float64).mean(axis=0)
            norm = float(np.linalg.norm(embedding))
            if norm > 1e-12:
                embedding = embedding / norm
        categories = [det.category for det in detections if det.category is not None]
        label = None
        scores = None
        if categories:
            counts = Counter(categories)
            label = counts.most_common(1)[0][0]
            total = float(sum(counts.values()))
            scores = {name: count / total for name, count in sorted(counts.items())}
        return Prediction(
            observation_id=observation.observation_id,
            embedding=embedding,
            label=label,
            scores=scores,
        )
