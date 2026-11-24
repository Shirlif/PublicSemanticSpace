"""Working memory: 4 slots + attention scheduling (Phase 6).

A slot is a working-memory position. Activating an entity binds its persistent
raw features into a slot; eviction removes only the binding. Reactivating the
same entity in a different slot rebuilds it from raw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pss.binding import SlotPool, build_entity, cosine, make_slots, unbind_modality
from pss.memory.scene_graph import Track


@dataclass
class AttentionScheduler:
    capacity: int = 4
    w_task: float = 0.5
    w_salience: float = 0.3
    w_affect: float = 0.2

    def score(self, task: float = 0.0, salience: float = 0.0, affect: float = 0.0) -> float:
        return self.w_task * float(task) + self.w_salience * float(salience) + self.w_affect * float(affect)

    def select(
        self,
        tracks: list[Track],
        task_scores: dict[str, float] | None = None,
        salience: dict[str, float] | None = None,
        affect: dict[str, float] | None = None,
    ) -> tuple[list[str], dict[str, float]]:
        task_scores = task_scores or {}
        salience = salience or {}
        affect = affect or {}
        scored = [
            (
                self.score(
                    task_scores.get(tr.id, 0.0),
                    salience.get(tr.id, 0.0),
                    affect.get(tr.id, 0.0),
                ),
                tr.id,
            )
            for tr in tracks
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))
        selected = [tid for _, tid in scored[: self.capacity]]
        return selected, dict(scored)


class WorkingMemory:
    def __init__(self, scheduler: AttentionScheduler | None = None, dim: int = 128):
        self.scheduler = scheduler or AttentionScheduler()
        self.dim = int(dim)
        self.slots = make_slots("hadamard", self.dim, self.scheduler.capacity)
        self.pool = SlotPool(self.slots)
        self.entities: dict[str, object] = {}
        self.raw: dict[str, dict[str, np.ndarray]] = {}
        self.priorities: dict[str, float] = {}

    def active_ids(self) -> list[str]:
        return list(self.entities)

    def activate(self, track: Track, priority: float = 0.0):
        eid = track.id
        if eid in self.entities:
            return self.entities[eid]
        raw = track.raw or (
            {"vision": track.appearance} if track.appearance is not None else {}
        )
        if not raw:
            raise ValueError(f"track {eid!r} has no raw features to bind")
        slot_idx, evicted = self.pool.acquire(eid, priority=priority)
        if evicted is not None:
            self._drop(evicted)
        entity = build_entity(eid, raw, self.slots[slot_idx], slot_index=slot_idx)
        self.entities[eid] = entity
        self.raw[eid] = {k: np.asarray(v, dtype=float) for k, v in raw.items()}
        self.priorities[eid] = float(priority)
        return entity

    def evict(self, eid: str) -> None:
        self.pool.release(eid)
        self._drop(eid)

    def _drop(self, eid: str) -> None:
        self.entities.pop(eid, None)
        self.raw.pop(eid, None)
        self.priorities.pop(eid, None)

    def set_scene(
        self,
        tracks: list[Track],
        task_scores: dict[str, float] | None = None,
        salience: dict[str, float] | None = None,
        affect: dict[str, float] | None = None,
    ) -> list[str]:
        selected, scores = self.scheduler.select(tracks, task_scores, salience, affect)
        for eid in list(self.entities):
            if eid not in selected:
                self.evict(eid)
        by_id = {tr.id: tr for tr in tracks}
        for eid in selected:
            if eid not in self.entities and eid in by_id:
                self.activate(by_id[eid], priority=scores.get(eid, 0.0))
        return selected

    def feature(self, eid: str, modality: str = "vision") -> np.ndarray:
        entity = self.entities[eid]
        return unbind_modality(entity, modality, self.slots[entity.slot])

    def recall(self, eid: str, modality: str = "vision") -> float:
        if eid not in self.raw or modality not in self.raw[eid]:
            return 0.0
        return cosine(self.feature(eid, modality), self.raw[eid][modality])
