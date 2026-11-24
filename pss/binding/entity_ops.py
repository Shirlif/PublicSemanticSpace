"""Per-modality entity assembly (contract route A).

An ``Entity`` stores one bound vector per modality. Cross-modality features are
never summed into a single vector: addition is not separable (9.2.3). The
persistent representation is ``raw`` + pose; ``bound`` is valid only for the
slot it was built with.
"""

from __future__ import annotations

import numpy as np

from pss.contracts import Entity

from .ops import bind, cosine, unbind


def build_entity(
    entity_id: str,
    modality_features: dict[str, np.ndarray],
    slot: np.ndarray,
    slot_index: int | None = None,
    confidences: dict[str, float] | None = None,
    **meta,
) -> Entity:
    """Bind each modality's public feature to ``slot`` and build an Entity."""
    raw = {
        name: np.asarray(vec, dtype=np.float64)
        for name, vec in modality_features.items()
    }
    bound = {name: bind(vec, slot) for name, vec in raw.items()}
    conf = dict(confidences) if confidences is not None else {name: 1.0 for name in raw}
    return Entity(
        id=entity_id,
        bound=bound,
        raw=raw,
        conf=conf,
        slot=slot_index,
        **meta,
    )


def unbind_modality(entity: Entity, modality: str, slot: np.ndarray) -> np.ndarray:
    """Recover one modality's public feature from its bound vector."""
    if modality not in entity.bound:
        raise KeyError(f"entity {entity.id!r} has no modality {modality!r}")
    return unbind(entity.bound[modality], slot)


def match_modality(
    entity: Entity,
    modality: str,
    prototypes: dict[str, np.ndarray],
    slot: np.ndarray,
) -> list[tuple[str, float]]:
    """Rank per-modality prototypes by cosine similarity."""
    feature = unbind_modality(entity, modality, slot)
    ranked = [
        (name, cosine(feature, proto)) for name, proto in prototypes.items()
    ]
    return sorted(ranked, key=lambda kv: -kv[1])


def rebuild_for_slot(
    entity: Entity,
    slot: np.ndarray,
    slot_index: int | None = None,
) -> Entity:
    """Re-bind an entity's persistent raw features to a new working-memory slot.

    This is the correct way to move a dormant entity into a different slot:
    never reinterpret an old bound vector with a new slot.
    """
    return build_entity(
        entity.id,
        entity.raw,
        slot,
        slot_index=slot_index,
        confidences=entity.conf,
        pose=entity.pose,
        prototype=entity.prototype,
        state=entity.state,
        parent=entity.parent,
        provenance=entity.provenance,
        last_seen=entity.last_seen,
        status=entity.status,
    )
