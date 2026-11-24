"""Relation triples as three separately stored bound components.

Why three parts: adding the bound components destroys separability (9.2.3).
``unbind(bind(A, r) + bind(B, r), r) = A + B`` and A, B cannot be recovered.
So the encoder always stores ``subj``/``rel``/``obj`` separately, and the
summed ``fact`` vector is a lookup shortcut only -- never a storage format.

The default scheme is Hadamard slot binding. ``scheme="hrr"`` is available for
the Phase 1 comparison and is not the adopted path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hrr import hrr_bind, hrr_unbind, make_roles
from .ops import bind, cosine, make_slots, unbind

ROLE_NAMES = ("subj", "rel", "obj")


@dataclass
class EncodedRelation:
    parts: dict[str, np.ndarray]
    fact: np.ndarray

    def validate(self) -> None:
        if set(self.parts) != set(ROLE_NAMES):
            raise ValueError("parts must contain exactly subj/rel/obj")
        d = np.asarray(self.fact).shape[0]
        for name, vec in self.parts.items():
            arr = np.asarray(vec)
            if arr.ndim != 1 or arr.shape[0] != d:
                raise ValueError(f"part {name} has shape {arr.shape}, expected ({d},)")
        if not np.all(np.isfinite(self.fact)):
            raise ValueError("fact contains non-finite values")


class RelationEncoder:
    """Encode/decode ``(subject, predicate, object)`` triples.

    ``decode(parts, role)`` reads one component from the separately stored
    parts and is exact for Hadamard binding. ``decode_fact(fact, role)`` reads
    from the summed vector and is intentionally noisy: it is only for fast
    similarity lookup.
    """

    def __init__(self, d: int, seed: int = 0, scheme: str = "hadamard"):
        self.d = d
        self.scheme = scheme
        if scheme == "hrr":
            self.roles = make_roles(d, len(ROLE_NAMES), seed=seed)
            self._bind = hrr_bind
            self._unbind = hrr_unbind
        else:
            self.roles = make_slots(scheme, d, len(ROLE_NAMES), seed=seed)
            self._bind = bind
            self._unbind = unbind

    def encode(self, subject, predicate, obj) -> EncodedRelation:
        fillers = (subject, predicate, obj)
        parts = {
            name: self._bind(np.asarray(vec, dtype=np.float64), self.roles[i])
            for i, (name, vec) in enumerate(zip(ROLE_NAMES, fillers))
        }
        fact = parts["subj"] + parts["rel"] + parts["obj"]
        return EncodedRelation(parts=parts, fact=fact)

    def decode(self, parts: dict[str, np.ndarray], role: str) -> np.ndarray:
        i = ROLE_NAMES.index(role)
        return self._unbind(parts[role], self.roles[i])

    def decode_fact(self, fact, role: str) -> np.ndarray:
        i = ROLE_NAMES.index(role)
        return self._unbind(fact, self.roles[i])

    def crosstalk(self, parts: dict[str, np.ndarray]) -> float:
        """Max absolute cosine when a part is decoded with the wrong role."""
        true = {role: self.decode(parts, role) for role in ROLE_NAMES}
        worst = 0.0
        for i, role_i in enumerate(ROLE_NAMES):
            for j, role_j in enumerate(ROLE_NAMES):
                if i == j:
                    continue
                wrong = self._unbind(parts[role_i], self.roles[j])
                worst = max(worst, abs(cosine(wrong, true[role_j])))
        return worst
