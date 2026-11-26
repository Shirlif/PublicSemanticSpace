"""Typed operator protocol shared by every operator family.

The inference engine never branches on family.  It only calls ``apply``,
``inverse``, ``compose`` and ``residual``.  Four families implement this
protocol:

- ``pss.binding.frames.FrameOp``           reference-frame transforms (SE(3))
- ``pss.binding.operators.*Operator``      semantic relation transforms
- ``pss.world_model.conditional.*``        conditional action effects
- ``pss.reasoning.constraints.*``          type and attribute constraints

Composition order
-----------------
``self.compose(other)`` means ``self ∘ other``: ``other`` is applied first and
``self`` second, matching function composition and matrix multiplication
``self.matrix @ other.matrix``.  This matters because several of the operator
families are deliberately non-commutative.

Residual convention
-------------------
``residual(x, y)`` is a non-negative consistency distance where ``0.0`` means
perfectly consistent.  Scales differ by family (metres for poses, ``1 - cos``
for embeddings, Jaccard distance for fact sets), so thresholds must be set per
family.  The inference engine only compares residuals within one family.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class Operator(ABC):
    """Minimal interface every operatory family must satisfy."""

    kind: str = "abstract"
    name: str = "operator"

    @abstractmethod
    def apply(self, x: Any) -> Any:
        """Transform ``x``.  Input/output types are family-specific."""

    def inverse(self) -> "Operator | None":
        """Return the inverse operator, or ``None`` when not invertible."""
        return None

    def compose(self, other: "Operator") -> "Operator | None":
        """Return ``self ∘ other``, or ``None`` when the pair is not composable."""
        return None

    def residual(self, x: Any, y: Any) -> float:
        """Consistency distance between two values of this family's type."""
        raise NotImplementedError(f"{type(self).__name__} does not define residual")

    def composable(self, other: "Operator") -> bool:
        """Families only compose with themselves unless they override this."""
        return isinstance(other, Operator) and self.kind == other.kind

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r}, kind={self.kind!r})"


@dataclass(frozen=True)
class OperatorChain(Operator):
    """A generic left-to-right chain: ``apply`` runs ``parts[0]`` first.

    ``OperatorChain((a, b)).apply(x) == a.apply(b.apply(x))`` because ``a`` is
    the outer composition.  Use :meth:`from_sequence` when you have a list in
    application order.
    """

    parts: tuple[Operator, ...]
    kind: str = "chain"
    name: str = "chain"

    def apply(self, x: Any) -> Any:
        for part in self.parts:
            x = part.apply(x)
        return x

    def inverse(self) -> "Operator | None":
        inverses = [part.inverse() for part in self.parts]
        if any(inv is None for inv in inverses):
            return None
        return OperatorChain(tuple(reversed(inverses)))

    def compose(self, other: "Operator") -> "Operator | None":
        if isinstance(other, OperatorChain):
            return OperatorChain(self.parts + other.parts)
        return OperatorChain(self.parts + (other,))

    def residual(self, x: Any, y: Any) -> float:
        """End-to-end residual, using cosine distance for vector outputs."""
        out = np.asarray(self.apply(x), dtype=np.float64).ravel()
        ref = np.asarray(y, dtype=np.float64).ravel()
        if out.shape != ref.shape:
            return 1.0
        na = float(np.linalg.norm(out))
        nb = float(np.linalg.norm(ref))
        if na == 0.0 or nb == 0.0:
            return 1.0
        return float(1.0 - np.dot(out, ref) / (na * nb))

    @classmethod
    def from_sequence(cls, parts) -> "OperatorChain":
        """Build from a list in *application* order (opposite of ``parts``)."""
        return cls(tuple(reversed(tuple(parts))))
