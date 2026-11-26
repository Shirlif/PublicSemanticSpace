"""Type and attribute constraints as composable operators (Phase 7).

A *type constraint* restricts the allowed category of a query.  The allowed
categories live in a :class:`TypeNode` (a leaf holds one category; an internal
node holds the union of its leaves).  Scoring is per modality: for every
modality shared with the query, the node is scored by the best cosine against
any of its categories, and those scores are averaged (optionally weighted).

An *attribute constraint* restricts a scalar (for example a screw diameter) to
a closed interval.  :func:`compatibility` turns two intervals into the
"does this screw fit this hole" decision, reporting the clearance when they do
not overlap.

Both constraint flavours implement :class:`pss.binding.protocol.Operator`, so
the inference engine can treat them like every other operator family:
``self.compose(other)`` means ``self ∘ other``.

This module deliberately imports only numpy and light PSS helpers; it never
imports :mod:`pss.prototypes` (whose package pulls in scipy).  Prototype data
is therefore accepted duck-typed through :class:`PrototypeView`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pss.binding.ops import cosine
from pss.binding.protocol import Operator


def safe_l2(v: Any) -> np.ndarray:
    """Return ``v`` scaled to unit L2 norm; a zero vector is returned as-is."""
    arr = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return arr.copy()
    return arr / norm


class PrototypeView:
    """Read-only, normalized view over per-modality category prototypes.

    Accepts either:

    - a ``PrototypeBank``-like object with ``.categories`` (``list[str]``) and
      ``.protos`` (``dict[modality, ndarray (C, d)]``), or
    - a plain mapping ``category -> {modality: vector}``.

    Every stored vector is normalized with :func:`safe_l2`.
    """

    def __init__(self, source: Any) -> None:
        protos: dict[str, dict[str, np.ndarray]] = {}
        if hasattr(source, "categories") and hasattr(source, "protos"):
            categories = [str(c) for c in source.categories]
            for modality, matrix in source.protos.items():
                arr = np.asarray(matrix, dtype=np.float64)
                if arr.ndim != 2:
                    raise ValueError(
                        f"prototypes for modality {modality!r} must be 2-D, got shape {arr.shape}"
                    )
                if arr.shape[0] != len(categories):
                    raise ValueError(
                        f"modality {modality!r} has {arr.shape[0]} rows for "
                        f"{len(categories)} categories"
                    )
                for i, category in enumerate(categories):
                    protos.setdefault(category, {})[str(modality)] = safe_l2(arr[i])
        elif isinstance(source, Mapping):
            for category, per_modality in source.items():
                if not isinstance(per_modality, Mapping):
                    raise TypeError(
                        f"category {category!r} must map to a {{modality: vector}} mapping"
                    )
                protos[str(category)] = {
                    str(modality): safe_l2(vector)
                    for modality, vector in per_modality.items()
                }
            categories = list(protos)
        else:
            raise TypeError(
                "PrototypeView expects a bank-like object or a "
                "category -> {modality: vector} mapping"
            )

        # Keep declaration order and retain categories that own no prototypes.
        self.categories: list[str] = list(dict.fromkeys(categories))
        self._protos = protos

    def protos_for(self, category: str) -> dict[str, np.ndarray]:
        """Return copies of this category's per-modality unit vectors."""
        return {m: v.copy() for m, v in self._protos.get(category, {}).items()}

    def modalities(self) -> set[str]:
        """Return every modality that appears in the view."""
        out: set[str] = set()
        for per_modality in self._protos.values():
            out.update(per_modality)
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PrototypeView(categories={self.categories!r})"


@dataclass(frozen=True)
class TypeNode:
    """A node in the category hierarchy: a set of categories."""

    node_id: str
    categories: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.categories, frozenset):
            object.__setattr__(self, "categories", frozenset(self.categories))

    @classmethod
    def leaf(cls, node_id: str, category: str) -> "TypeNode":
        """Build a leaf node holding exactly one category."""
        return cls(node_id=node_id, categories=frozenset({category}))

    def intersects(self, other: "TypeNode") -> bool:
        """Whether the two nodes share at least one category."""
        return bool(self.categories & other.categories)

    def intersection(self, other: "TypeNode") -> "TypeNode | None":
        """Return the shared-category node, or ``None`` when disjoint."""
        common = self.categories & other.categories
        if not common:
            return None
        return TypeNode(
            node_id=f"{self.node_id}&{other.node_id}",
            categories=frozenset(common),
        )

    def __contains__(self, category: object) -> bool:
        return category in self.categories


@dataclass(frozen=True)
class TypeConstraint(Operator):
    """Restrict a multi-modal query to the categories of one :class:`TypeNode`.

    ``membership`` scores the query against the node, ``apply``/``project``
    produce a projection target, and ``compose`` intersects two constraints.
    """

    # ``field()`` masks the ``Operator.name`` class attribute so this stays a
    # required argument rather than inheriting a dataclass default.
    name: str = field()
    node: TypeNode
    # The view is a helper, not identity: keep it out of eq/repr/hash.
    view: PrototypeView = field(compare=False, repr=False, hash=False)
    # Dicts are unhashable, so exclude weights from the generated __hash__.
    weights: dict[str, float] | None = field(default=None, compare=True, hash=False)
    kind: str = "constraint"

    def _modalities(self) -> set[str]:
        out: set[str] = set()
        for category in self.node.categories:
            out.update(self.view.protos_for(category))
        return out

    def _weight(self, modality: str) -> float:
        if self.weights is None:
            return 1.0
        return float(self.weights.get(modality, 1.0))

    def membership(self, query: Mapping[str, Any]) -> float:
        """Weighted mean of the best per-modality cosine inside the node.

        Returns ``-1.0`` when the node is empty or no query modality is
        covered by the node.
        """
        if not self.node.categories:
            return -1.0
        covered = self._modalities()
        scores: list[tuple[float, float]] = []
        for modality, vector in query.items():
            if modality not in covered:
                continue
            best: float | None = None
            for category in self.node.categories:
                proto = self.view.protos_for(category).get(modality)
                if proto is None:
                    continue
                sim = cosine(vector, proto)
                if best is None or sim > best:
                    best = sim
            if best is not None:
                scores.append((self._weight(modality), best))
        total = sum(w for w, _ in scores)
        if not scores or total <= 0.0:
            return -1.0
        return float(sum(w * s for w, s in scores) / total)

    def apply(self, query: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Return the representative prototype per shared modality.

        The representative is the prototype of the node category with the
        highest cosine for that modality, renormalized to unit length.
        Modalities the node cannot cover are omitted.
        """
        out: dict[str, np.ndarray] = {}
        covered = self._modalities()
        for modality, vector in query.items():
            if modality not in covered:
                continue
            best_category: str | None = None
            best_sim = -np.inf
            for category in self.node.categories:
                proto = self.view.protos_for(category).get(modality)
                if proto is None:
                    continue
                sim = cosine(vector, proto)
                if sim > best_sim:
                    best_sim = sim
                    best_category = category
            if best_category is not None:
                out[modality] = safe_l2(self.view.protos_for(best_category)[modality])
        return out

    def project(self, query: Mapping[str, Any], alpha: float = 1.0) -> dict[str, np.ndarray]:
        """Blend the query toward the representative prototype.

        ``alpha=1.0`` returns the pure representative (renormalized);
        ``alpha=0.0`` keeps the query.  Modalities without a representative are
        copied from the query unchanged.
        """
        representatives = self.apply(query)
        out = {m: np.asarray(v, dtype=np.float64).copy() for m, v in query.items()}
        for modality, representative in representatives.items():
            blended = alpha * representative + (1.0 - alpha) * safe_l2(query[modality])
            out[modality] = safe_l2(blended)
        return out

    def residual(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
        """``1 - weighted cosine`` over shared modalities, else ``1.0``."""
        shared = [m for m in a if m in b]
        if not shared:
            return 1.0
        total = 0.0
        acc = 0.0
        for modality in shared:
            weight = self._weight(modality)
            acc += weight * cosine(a[modality], b[modality])
            total += weight
        if total <= 0.0:
            return 1.0
        return float(1.0 - acc / total)

    def compatible(self, other: Any) -> bool:
        """Whether the two constraints share a category."""
        node = other.node if isinstance(other, TypeConstraint) else other
        return self.node.intersects(node)

    def compose(self, other: Any) -> "TypeConstraint | None":
        """Return ``self ∘ other`` as the intersection node, or ``None``.

        Weights merge with ``self`` taking precedence; both constraints are
        normalised against ``self``'s view.
        """
        if not isinstance(other, TypeConstraint):
            return None
        node = self.node.intersection(other.node)
        if node is None:
            return None
        if self.weights is None and other.weights is None:
            weights = None
        else:
            weights = dict(other.weights or {})
            weights.update(self.weights or {})
        return TypeConstraint(
            name=f"{self.name}&{other.name}",
            node=node,
            view=self.view,
            weights=weights,
            kind=self.kind,
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "node_id": self.node.node_id,
            "categories": sorted(self.node.categories),
            "weights": dict(self.weights) if self.weights else {},
        }


@dataclass(frozen=True)
class AttributeInterval:
    """A closed scalar interval ``[low, high]``."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(
                f"AttributeInterval requires low <= high, got [{self.low}, {self.high}]"
            )

    def contains(self, v: float) -> bool:
        """Whether ``v`` lies inside the closed interval."""
        return self.low <= float(v) <= self.high

    def overlaps(self, other: "AttributeInterval") -> bool:
        """Whether the two closed intervals share at least one point."""
        return self.low <= other.high and other.low <= self.high

    def intersection(self, other: "AttributeInterval") -> "AttributeInterval | None":
        """Return the overlapping interval, or ``None`` when disjoint."""
        low = max(self.low, other.low)
        high = min(self.high, other.high)
        if low > high:
            return None
        return AttributeInterval(low=low, high=high)

    def residual(self, v: float) -> float:
        """Distance from ``v`` to the interval (``0.0`` when inside)."""
        value = float(v)
        if value < self.low:
            return float(self.low - value)
        if value > self.high:
            return float(value - self.high)
        return 0.0


@dataclass(frozen=True)
class AttributeConstraint(Operator):
    """Restrict a scalar to an :class:`AttributeInterval`."""

    # ``field()`` masks the ``Operator.name`` class attribute (see above).
    name: str = field()
    interval: AttributeInterval
    kind: str = "attribute"

    def apply(self, v: float) -> float:
        """Clip ``v`` into the interval."""
        return float(min(max(float(v), self.interval.low), self.interval.high))

    def residual(self, v: float, w: float) -> float:
        """Absolute difference of the two clipped values."""
        return float(abs(self.apply(v) - self.apply(w)))

    def compatible(self, other: Any) -> bool:
        """Whether the two attribute intervals overlap."""
        interval = other.interval if isinstance(other, AttributeConstraint) else other
        return self.interval.overlaps(interval)

    def compose(self, other: Any) -> "AttributeConstraint | None":
        """Return the interval intersection, or ``None`` when disjoint."""
        if not isinstance(other, AttributeConstraint):
            return None
        interval = self.interval.intersection(other.interval)
        if interval is None:
            return None
        return AttributeConstraint(
            name=f"{self.name}&{other.name}",
            interval=interval,
            kind=self.kind,
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "low": self.interval.low,
            "high": self.interval.high,
        }


def compatibility(screw: AttributeInterval, hole: AttributeInterval) -> dict:
    """Report whether a screw interval fits a hole interval.

    ``clearance`` is the gap between the intervals (``0.0`` when they overlap).
    ``low``/``high`` are the feasible overlap window (``low > high`` when the
    parts are incompatible).
    """
    low = max(screw.low, hole.low)
    high = min(screw.high, hole.high)
    compatible = low <= high
    if compatible:
        clearance = 0.0
    elif screw.high < hole.low:
        clearance = float(hole.low - screw.high)
    else:
        clearance = float(screw.low - hole.high)
    return {
        "compatible": bool(compatible),
        "clearance": float(clearance),
        "low": float(low),
        "high": float(high),
    }
