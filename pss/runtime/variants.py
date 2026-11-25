"""Ablation variants for the PSS runtime (design §7).

Variants are switched by constructing a different runtime, never by changing
the Isaac scene or the bundle.  Offline suites can therefore replay the same
bundle under all variants.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pss.memory.perception import Observation
from pss.memory.scene_graph import SceneGraph, Track
from pss.memory.tracking import Tracker, TrackerConfig
from pss.runtime.layers import EntityLayer, RelationFactLayer
from pss.runtime.runtime import PSSRuntime
from pss.runtime.state import CognitiveState, Entity

VARIANTS = (
    "full",
    "no_entity",
    "no_relations",
    "no_memory",
    "no_active_set",
    "no_uncertainty",
    "no_audit",
    "flat_context",
    "no_tools",
)


class AppearanceMetric:
    """Cosine distance on appearance embeddings."""

    @staticmethod
    def _distance(track, obs) -> float:
        a = np.asarray(track.appearance, dtype=float)
        b = np.asarray(obs.appearance, dtype=float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na <= 1e-12 or nb <= 1e-12:
            return 1.0
        return float(1.0 - np.dot(a, b) / (na * nb))

    def cost(self, track, obs) -> float:
        return self._distance(track, obs)

    def cost_matrix(self, tracks, observations) -> np.ndarray:
        matrix = np.ones((len(tracks), len(observations)), dtype=float)
        for i, track in enumerate(tracks):
            for j, obs in enumerate(observations):
                matrix[i, j] = self._distance(track, obs)
        return matrix


class FlatEntityLayer:
    """No persistent slots: each frame gets fresh IDs (no_entity / flat_context)."""

    def __init__(self):
        self.graph = SceneGraph()
        self.active_capacity = -1
        self._ids: list[str] = []

    def ingest(self, observations, frame: int) -> dict[str, Any]:
        self.graph = SceneGraph()
        self._ids = []
        events = {"matched": [], "new": [], "dormant": [], "reid": []}
        for j, obs in enumerate(observations):
            track_id = f"flat_{frame:04d}_{j:02d}"
            track = Track(
                id=track_id,
                pos=obs.pos.copy(),
                quat=obs.quat.copy(),
                appearance=np.asarray(obs.appearance, dtype=float).copy(),
                category=obs.category,
                confidence=obs.confidence,
                first_seen=frame,
                last_seen=frame,
                raw=dict(obs.raw),
            )
            self.graph.add(track)
            self._ids.append(track_id)
            events["new"].append((track_id, obs.gt_id))
        return events

    def active_ids(self, capacity: int = 4, observations=None) -> list[str]:
        return list(self._ids)

    def sync(self, state: CognitiveState, active_ids: list[str] | None = None) -> CognitiveState:
        state.entities.clear()
        for track in self.graph.tracks.values():
            state.add_entity(
                Entity(
                    id=track.id,
                    category=track.category,
                    pos=np.asarray(track.pos, dtype=float).copy(),
                    quat=np.asarray(track.quat, dtype=float).copy(),
                    confidence=float(track.confidence),
                    status="active",
                    last_seen=int(track.last_seen),
                    active=True,
                )
            )
        return state


class NoMemoryEntityLayer(EntityLayer):
    """Memoryless entity layer: a track exists only while it is observed.

    Any track that misses a frame is dropped immediately, so an object that
    disappears behind an occluder and reappears is a new entity.  This is the
    architecture-level ablation for the memory tasks (``--variant no_memory``).
    """

    def ingest(self, observations, frame: int) -> dict[str, Any]:
        events = super().ingest(observations, frame)
        for track in list(self.graph.tracks.values()):
            if track.status != "active":
                self.graph.remove(track.id)
        return events

    def sync(self, state: CognitiveState, active_ids: list[str] | None = None) -> CognitiveState:
        state = super().sync(state, active_ids)
        valid = set(self.graph.tracks)
        for entity_id in list(state.entities):
            if entity_id not in valid:
                del state.entities[entity_id]
        return state


class NullRelationLayer:
    """No relation/fact store (no_relations / flat_context)."""

    def __init__(self):
        self.write_log: list[dict] = []

    def add_symbolic(self, *args, **kwargs) -> None:
        return None

    def add_learned(self, *args, **kwargs) -> None:
        return None

    def sync_graph(self, *args, **kwargs) -> None:
        return None

    def sync(self, *args, **kwargs) -> None:
        return None

    def conflicts(self) -> dict:
        return {}

    def snapshot(self) -> dict:
        return {}

    def restore(self, payload) -> None:
        return None

    def resolve(self, entity, attribute):
        return None

    def assert_fact(self, *args, **kwargs) -> str:
        return "write"


def _tracker(tier: str, graph: SceneGraph, *, memory: bool = True) -> Tracker:
    metric = AppearanceMetric()
    if tier == "oracle":
        config = TrackerConfig(
            gate_radius=0.6,
            max_cost=0.5,
            w_pose=0.05,
            w_app=0.9,
            w_time=0.05,
            w_category=0.0,
            metric=metric,
            metric_max_cost=0.15,
            reid_pose_gate=0.6,
            reid_app_min=0.5,
            reid_w_pose=0.0,
            reid_w_app=1.0,
            reid_max_age=150 if memory else 0,
            dormancy_missed=2 if memory else 1,
        )
    else:
        config = TrackerConfig(
            gate_radius=0.25,
            max_cost=1.0,
            w_pose=0.6,
            w_app=0.3,
            w_time=0.1,
            w_category=0.05,
            metric=metric,
            metric_max_cost=0.7,
            reid_max_age=150 if memory else 0,
            dormancy_missed=2 if memory else 1,
        )
    return Tracker(config, graph)


def build_runtime(variant: str, *, tier: str = "oracle") -> PSSRuntime:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    flat = variant in ("no_entity", "flat_context")
    if flat:
        entity_layer = FlatEntityLayer()
    elif variant == "no_memory":
        graph = SceneGraph()
        entity_layer = NoMemoryEntityLayer(
            graph=graph,
            tracker=_tracker(tier, graph, memory=False),
            use_default_relations=False,
            use_default_active=False,
            active_capacity=4,
        )
    else:
        graph = SceneGraph()
        entity_layer = EntityLayer(
            graph=graph,
            tracker=_tracker(tier, graph),
            use_default_relations=False,
            use_default_active=False,
            active_capacity=-1 if variant == "no_active_set" else 4,
        )
    relation_layer = (
        NullRelationLayer()
        if variant in ("no_relations", "flat_context")
        else RelationFactLayer(use_default_write=False)
    )
    return PSSRuntime(
        entity_layer=entity_layer,
        relation_layer=relation_layer,
        audit=variant != "no_audit",
        uncertainty_enabled=variant != "no_uncertainty",
        use_default_relations=False,
        use_default_active=False,
        use_default_write=False,
    )
