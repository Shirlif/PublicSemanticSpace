"""Working-memory slot pool.

A slot is a **working-memory position**, not an entity identity. Entities keep
their IDs in the scene graph; the pool only decides which few entities are
currently bound into the small number of active slots. Eviction returns the
evicted entity ID so the caller can persist its raw state.
"""

from __future__ import annotations

import numpy as np


class SlotPool:
    def __init__(self, slots):
        self.slots = np.asarray(slots, dtype=np.float64)
        if self.slots.ndim != 2:
            raise ValueError("slots must be a 2-D array (K, D)")
        self.k = self.slots.shape[0]
        self._slot_to_entity: list[str | None] = [None] * self.k
        self._entity_to_slot: dict[str, int] = {}
        self._priority: dict[str, float] = {}

    def active(self) -> dict[str, int]:
        """Map entity id -> slot index for currently active entities."""
        return dict(self._entity_to_slot)

    def slot_index(self, entity_id: str) -> int | None:
        return self._entity_to_slot.get(entity_id)

    def is_active(self, entity_id: str) -> bool:
        return entity_id in self._entity_to_slot

    def acquire(self, entity_id: str, priority: float = 0.0) -> tuple[int, str | None]:
        """Assign a slot to ``entity_id``.

        Returns ``(slot_index, evicted_entity_id)``. If the entity is already
        active, its slot is reused and nothing is evicted.
        """
        if entity_id in self._entity_to_slot:
            self._priority[entity_id] = float(priority)
            return self._entity_to_slot[entity_id], None

        for i in range(self.k):
            if self._slot_to_entity[i] is None:
                self._assign(i, entity_id, priority)
                return i, None

        victim = min(self._entity_to_slot, key=lambda e: (self._priority[e], e))
        victim_slot = self._entity_to_slot[victim]
        self.release(victim)
        self._assign(victim_slot, entity_id, priority)
        return victim_slot, victim

    def release(self, entity_id: str) -> int | None:
        slot = self._entity_to_slot.pop(entity_id, None)
        if slot is not None:
            self._slot_to_entity[slot] = None
            self._priority.pop(entity_id, None)
        return slot

    def _assign(self, slot: int, entity_id: str, priority: float) -> None:
        self._slot_to_entity[slot] = entity_id
        self._entity_to_slot[entity_id] = slot
        self._priority[entity_id] = float(priority)

    def __len__(self) -> int:
        return len(self._entity_to_slot)
