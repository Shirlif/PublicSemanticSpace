"""Relation-algebra consistency for geometric predicates.

The relation head is trained per ordered pair, so nothing stops it from
returning a set of facts that no single point layout can satisfy:
``left(a, b)`` together with ``left(b, a)``, ``left`` together with ``right``,
or a chain ``left(a, b) & left(b, c)`` whose consequence ``left(a, c)`` is
missing.  The P5b failure mode (a head that is 0.571 exact-4 on held-out
regions) is exactly the kind of thing these checks should localise.

This module is deliberately dependency-free: facts are plain
``(subject, predicate, object)`` tuples and penalties operate on numpy arrays.
It reports violations and closes facts; it does not decide which fact to drop.
That repair step belongs to the caller, which owns confidences.

Predicate order follows the P5b label vector:
``("left", "right", "above", "below")``, where ``left(a, b)`` means
``x_a < x_b`` and ``above(a, b)`` means ``y_a > y_b``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DEFAULT_ALGEBRA",
    "INVERSE",
    "PREDICATES",
    "RelationAlgebra",
    "Violation",
    "antisymmetry_penalty",
    "closure",
    "consistency",
    "exclusivity_penalty",
    "inverse_gap",
    "violations",
]

Fact = tuple[str, str, str]

PREDICATES = ("left", "right", "above", "below")
INVERSE = {"left": "right", "right": "left", "above": "below", "below": "above"}


@dataclass(frozen=True)
class RelationAlgebra:
    """Which algebraic laws a predicate family is expected to obey."""

    inverse: Mapping[str, str] = field(default_factory=lambda: dict(INVERSE))
    exclusive: tuple[tuple[str, str], ...] = (("left", "right"), ("above", "below"))
    transitive: tuple[str, ...] = PREDICATES


DEFAULT_ALGEBRA = RelationAlgebra()


@dataclass(frozen=True)
class Violation:
    kind: str
    facts: tuple[Fact, ...]
    detail: str = ""


def _normalise(facts: Iterable) -> set[Fact]:
    out: set[Fact] = set()
    for fact in facts:
        if len(fact) < 3:
            raise ValueError(f"relation facts are (subject, predicate, object), got {fact!r}")
        out.add((str(fact[0]), str(fact[1]), str(fact[2])))
    return out


def _exclusive_partners(predicate: str, algebra: RelationAlgebra) -> tuple[str, ...]:
    partners = []
    for a, b in algebra.exclusive:
        if predicate == a:
            partners.append(b)
        elif predicate == b:
            partners.append(a)
    return tuple(partners)


def violations(
    facts: Iterable,
    algebra: RelationAlgebra = DEFAULT_ALGEBRA,
) -> list[Violation]:
    """All algebraic violations in ``facts``, deduplicated and sorted."""

    facts = _normalise(facts)
    found: set[tuple] = set()

    for a, p, b in facts:
        for q in _exclusive_partners(p, algebra):
            if (a, q, b) in facts:
                pair = tuple(sorted(((a, p, b), (a, q, b))))
                found.add(("exclusivity", pair, ""))
        if p in algebra.inverse and (b, p, a) in facts:
            pair = tuple(sorted(((a, p, b), (b, p, a))))
            found.add(("antisymmetry", pair, ""))
        inv = algebra.inverse.get(p)
        if inv is not None and (b, inv, a) not in facts:
            found.add(("inverse_gap", ((a, p, b),), f"missing {(b, inv, a)}"))
        if p in algebra.transitive:
            for b2, q, c in facts:
                if b2 == b and q == p and a != c and (a, p, c) not in facts:
                    chain = ((a, p, b), (b, p, c))
                    found.add(("transitive_gap", chain, f"missing {(a, p, c)}"))

    return [
        Violation(kind=kind, facts=tuple(sorted(facts_)), detail=detail)
        for kind, facts_, detail in sorted(found)
    ]


def inverse_gap(facts: Iterable, algebra: RelationAlgebra = DEFAULT_ALGEBRA) -> list[Fact]:
    """Facts whose inverse is absent (``left(a, b)`` without ``right(b, a)``)."""

    facts = _normalise(facts)
    out = []
    for a, p, b in sorted(facts):
        inv = algebra.inverse.get(p)
        if inv is not None and (b, inv, a) not in facts:
            out.append((a, p, b))
    return out


def closure(
    facts: Iterable,
    algebra: RelationAlgebra = DEFAULT_ALGEBRA,
    max_facts: int = 10000,
) -> frozenset[Fact]:
    """Add inverse and transitive consequences until a fixpoint or ``max_facts``."""

    current = _normalise(facts)
    changed = True
    while changed and len(current) < max_facts:
        changed = False
        snapshot = list(current)
        for a, p, b in snapshot:
            inv = algebra.inverse.get(p)
            if inv is not None:
                implied = (b, inv, a)
                if implied not in current:
                    current.add(implied)
                    changed = True
            if p in algebra.transitive:
                for b2, q, c in snapshot:
                    if b2 != b or q != p or a == c:
                        continue
                    implied = (a, p, c)
                    if implied not in current:
                        current.add(implied)
                        changed = True
        if len(current) >= max_facts:
            break
    return frozenset(current)


def consistency(
    facts: Iterable,
    algebra: RelationAlgebra = DEFAULT_ALGEBRA,
) -> dict:
    """Violation counts plus a contradiction-only score.

    ``score`` only counts exclusivity and antisymmetry, i.e. fact sets that no
    point layout can realise.  Missing inverses and transitive consequences are
    reported but not penalised, because a caller may deliberately store only
    half of each symmetric pair.
    """

    facts = _normalise(facts)
    counts = Counter(v.kind for v in violations(facts, algebra))
    contradictions = counts.get("exclusivity", 0) + counts.get("antisymmetry", 0)
    return {
        "n_facts": len(facts),
        "exclusivity": counts.get("exclusivity", 0),
        "antisymmetry": counts.get("antisymmetry", 0),
        "inverse_gap": counts.get("inverse_gap", 0),
        "transitive_gap": counts.get("transitive_gap", 0),
        "score": max(0.0, 1.0 - contradictions / max(len(facts), 1)),
    }


def _as_probabilities(probs) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.shape[-1] < len(PREDICATES):
        raise ValueError(
            f"expected at least {len(PREDICATES)} predicate columns, got {arr.shape}"
        )
    return arr[..., : len(PREDICATES)]


def antisymmetry_penalty(probs, reverse_probs) -> float:
    """``|left(a,b) - right(b,a)| + ...``, the numpy form of the bilevel loss.

    ``probs`` and ``reverse_probs`` are aligned batches of ordered pairs; the
    reverse batch holds the same pairs with endpoints swapped.
    """

    p = _as_probabilities(probs)
    q = _as_probabilities(reverse_probs)
    return float(
        np.mean(np.abs(p[..., 0] - q[..., 1]))
        + np.mean(np.abs(p[..., 1] - q[..., 0]))
        + np.mean(np.abs(p[..., 2] - q[..., 3]))
        + np.mean(np.abs(p[..., 3] - q[..., 2]))
    )


def exclusivity_penalty(probs) -> float:
    """``relu(left + right - 1) + relu(above + below - 1)``."""

    p = _as_probabilities(probs)
    return float(
        np.mean(np.maximum(p[..., 0] + p[..., 1] - 1.0, 0.0))
        + np.mean(np.maximum(p[..., 2] + p[..., 3] - 1.0, 0.0))
    )
