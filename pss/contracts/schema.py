"""Contract v0 data structures for the public semantic space.

These dataclasses are the *interface* between pipeline stages. They are
intentionally strict about dimensions and confidence ranges, because the
pipeline's error budget (5.6) depends on every stage emitting the same shape
and the same normalization convention.

Changing ``PUBLIC_DIM``, the normalization rule, or the binding operator is a
**major** contract change: every stored entity vector is invalidated (9.1.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .modality import active_modalities

SPACE_VERSION = "0.1.1"
PUBLIC_DIM = 128
SLOT_COUNT = 4
# Derived from contracts/modalities.yaml (active modalities only).  Kept as a
# module-level constant for backward compatibility; new code should call
# ``get_modalities()`` instead.
MODALITIES = active_modalities()
BINDING_SCHEME = "hadamard"

# Capacity and drift budgets measured by bench/binding_stability.py (Phase 0).
# At D=128, top-1 retrieval is 1.000 for K=5 superposed components and 0.769
# for K=20. Keep per-vector superposition small, or raise D to 256/512.
RECOMMENDED_MAX_SUPERPOSITION = 8
# Hadamard round-trip error grows ~sqrt(cycles): 1% noise x 100 cycles -> 10%.
# Rebind (or re-extract from perception) before the relative error exceeds this.
REBIND_ERROR_BUDGET = 0.10

EntityStatus = Literal["active", "dormant", "fragmented", "missing", "historical"]
_STATUSES = ("active", "dormant", "fragmented", "missing", "historical")


def l2_normalize(vec, eps: float = 1e-12) -> np.ndarray:
    """Normalize a vector to unit L2 norm, rejecting near-zero vectors."""
    v = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("cannot normalize a near-zero vector")
    return v / n


def _check_vec(name: str, vec) -> None:
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got shape {arr.shape}")
    if arr.shape[0] != PUBLIC_DIM:
        raise ValueError(f"{name} has dim {arr.shape[0]}, expected {PUBLIC_DIM}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")


@dataclass
class Entity:
    """A scene entity: identity + per-modality bound vectors + metadata.

    ``bound`` is an **ephemeral working-memory representation**: it is valid
    only for ``slot``. The persistent representation is ``raw`` + ``pose``.
    When an entity is evicted and reloaded into a different slot, rebuild
    ``bound`` from ``raw`` with the new slot (see ``pss.binding.rebind``).

    ``bound`` holds one bound vector **per modality**. It must never contain a
    cross-modality sum (9.2.3); ``raw`` holds the unbound public-space feature
    for each modality.
    """

    id: str
    pose: np.ndarray | None = None
    slot: int | None = None
    bound: dict[str, np.ndarray] = field(default_factory=dict)
    raw: dict[str, np.ndarray] = field(default_factory=dict)
    conf: dict[str, float] = field(default_factory=dict)
    prototype: str | None = None
    state: dict = field(default_factory=dict)
    parent: str | None = None
    provenance: dict = field(default_factory=dict)
    last_seen: float = 0.0
    status: EntityStatus = "active"

    def validate(self) -> None:
        if not self.id:
            raise ValueError("entity id must be non-empty")
        if self.slot is not None and self.slot < 0:
            raise ValueError(f"slot index must be non-negative, got {self.slot}")
        for name, vec in self.bound.items():
            _check_vec(f"bound[{name}]", vec)
        for name, vec in self.raw.items():
            _check_vec(f"raw[{name}]", vec)
        for name, c in self.conf.items():
            if not 0.0 <= float(c) <= 1.0:
                raise ValueError(f"confidence for {name} outside [0,1]: {c}")
        if self.status not in _STATUSES:
            raise ValueError(f"unknown status: {self.status}")


@dataclass
class Relation:
    """A relation triple with three separately stored bound components.

    The summed ``fact`` vector is a lookup shortcut only. Never use it to
    recover the individual components: addition is not separable (9.2.3).
    """

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    bound_parts: dict[str, np.ndarray] = field(default_factory=dict)
    fact: np.ndarray | None = None

    def validate(self) -> None:
        if not self.subject or not self.predicate or not self.object:
            raise ValueError("subject, predicate and object must be non-empty")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence outside [0,1]: {self.confidence}")
        if self.bound_parts and set(self.bound_parts) != {"subj", "rel", "obj"}:
            raise ValueError("bound_parts must contain exactly subj/rel/obj")
        for name, vec in self.bound_parts.items():
            _check_vec(f"bound_parts[{name}]", vec)
        if self.fact is not None:
            _check_vec("fact", self.fact)


@dataclass
class PrototypeNode:
    """A node in the hierarchical prototype tree (9.6.3).

    ``modality_protos`` maps a modality name to a unit-norm prototype vector.
    Prototype vectors come from real sample clusters; they are not hand-written.
    """

    node_id: str
    category_name: str
    parent_node: str | None = None
    modality_protos: dict[str, np.ndarray] = field(default_factory=dict)
    sample_count: int = 0
    variance: float = 0.0
    confidence_threshold: float = 0.7
    confusable_with: list[str] = field(default_factory=list)
    safety_tag: dict = field(default_factory=dict)
    version: str = "proto_pkg_v0.1.0"

    def validate(self) -> None:
        if not self.node_id or not self.category_name:
            raise ValueError("node_id and category_name must be non-empty")
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        if self.variance < 0.0:
            raise ValueError("variance must be non-negative")
        if not 0.0 <= float(self.confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in [0,1]")
        for name, vec in self.modality_protos.items():
            _check_vec(f"modality_protos[{name}]", vec)
            norm = float(np.linalg.norm(np.asarray(vec, dtype=np.float64)))
            if abs(norm - 1.0) > 1e-3:
                raise ValueError(f"prototype {name} is not unit-norm (norm={norm:.6f})")
