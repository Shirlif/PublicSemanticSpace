"""Rebind policy: when a bound vector has drifted too far, rebuild it.

Hadamard round-trip error grows ~``noise * sqrt(cycles)``. The contract sets a
10% relative-error budget; once exceeded, re-extract the feature from
perception (or, in Phase 0/1, from the stored raw feature) and bind again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pss.contracts import REBIND_ERROR_BUDGET, Entity

from .entity_ops import rebuild_for_slot


def relative_error(reference, recovered) -> float:
    """``||recovered - reference|| / ||reference||``."""
    ref = np.asarray(reference, dtype=np.float64)
    rec = np.asarray(recovered, dtype=np.float64)
    denom = np.linalg.norm(ref)
    if denom == 0.0:
        raise ValueError("reference vector has zero norm")
    return float(np.linalg.norm(rec - ref) / denom)


@dataclass
class RebindPolicy:
    error_budget: float = REBIND_ERROR_BUDGET

    def should_rebind(self, error: float) -> bool:
        return float(error) > self.error_budget

    def rebind(self, entity: Entity, slot: np.ndarray, slot_index: int | None = None) -> Entity:
        return rebuild_for_slot(entity, slot, slot_index=slot_index)


@dataclass
class DriftMonitor:
    """Track per-entity error estimates and flag rebind candidates."""

    error_budget: float = REBIND_ERROR_BUDGET
    errors: dict[str, float] = field(default_factory=dict)

    def update(self, entity_id: str, error: float) -> bool:
        self.errors[entity_id] = float(error)
        return self.needs_rebind(entity_id)

    def get(self, entity_id: str) -> float:
        return self.errors.get(entity_id, 0.0)

    def needs_rebind(self, entity_id: str) -> bool:
        return self.get(entity_id) > self.error_budget

    def clear(self, entity_id: str) -> None:
        self.errors.pop(entity_id, None)
