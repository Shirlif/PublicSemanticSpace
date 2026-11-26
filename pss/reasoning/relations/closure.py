"""Composition and inverse tables for typed relational reasoning.

A single relation edge is cheap; the value is in *composition*: given
``(a, r1, b)`` and ``(b, r2, c)``, which predicates ``p`` may hold for
``(a, p, c)``?  For qualitative spatial calculi (RCC-8, cardinal directions)
and for small ontologies these tables are known and can be written down; they
do not need to be learned.  Learning only sharpens the priors.

Path order
----------
``compose(r1, r2)`` reads a *path*: ``a -r1-> b -r2-> c``.  In operator terms
this is ``T_r2 ∘ T_r1`` (``r1`` applied first).  The table itself is
order-sensitive: ``compose("left_of", "right_of")`` and
``compose("right_of", "left_of")`` are different entries.

Empty result
------------
An empty frozenset means "no deduction is licensed", not "false".  The engine
must never turn ``compose(...) == frozenset()`` into a negative fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

Entry = frozenset[tuple[str, float]]

PREDICATES = (
    # spatial / directional
    "left_of",
    "right_of",
    "front_of",
    "behind",
    "above",
    "below",
    "inside",
    "contains",
    "on",
    "under",
    "contact",
    "near",
    # semantic
    "isa",
    "has_instance",
    "part_of",
    "has_part",
    "same_category",
)

INVERSE: dict[str, str] = {
    "left_of": "right_of",
    "right_of": "left_of",
    "front_of": "behind",
    "behind": "front_of",
    "above": "below",
    "below": "above",
    "inside": "contains",
    "contains": "inside",
    "on": "under",
    "under": "on",
    "contact": "contact",
    "near": "near",
    "isa": "has_instance",
    "has_instance": "isa",
    "part_of": "has_part",
    "has_part": "part_of",
    "same_category": "same_category",
}

SYMMETRIC = frozenset({"contact", "near", "same_category"})

# (r1, r2) -> {(p, prior)} for the path a -r1-> b -r2-> c.
# Empty tuples are intentional and mean "not licensed".
CLOSURE_BASE: dict[tuple[str, str], tuple[tuple[str, float], ...]] = {
    # directional transitivity (same frame, no intervening rotation)
    ("left_of", "left_of"): (("left_of", 0.95),),
    ("right_of", "right_of"): (("right_of", 0.95),),
    ("front_of", "front_of"): (("front_of", 0.95),),
    ("behind", "behind"): (("behind", 0.95),),
    ("above", "above"): (("above", 0.95),),
    ("below", "below"): (("below", 0.95),),
    # opposite directions cancel; nothing is licensed
    ("left_of", "right_of"): (),
    ("right_of", "left_of"): (),
    ("above", "below"): (),
    ("below", "above"): (),
    ("front_of", "behind"): (),
    ("behind", "front_of"): (),
    # topological (RCC-8: NTPP is transitive)
    ("inside", "inside"): (("inside", 0.99),),
    ("contains", "contains"): (("contains", 0.99),),
    ("inside", "contains"): (),
    ("contains", "inside"): (),
    # stacking: on top of something that is on a third thing is above it,
    # but not necessarily resting on it.
    ("on", "on"): (("above", 0.80),),
    ("under", "under"): (("below", 0.80),),
    # contact is not transitive; it only licenses a weak near
    ("contact", "contact"): (("near", 0.90),),
    ("near", "near"): (),
    # semantic / ontological
    ("isa", "isa"): (("isa", 0.99),),
    ("part_of", "part_of"): (("part_of", 0.98),),
    ("part_of", "isa"): (("part_of", 0.90),),
    ("has_part", "has_part"): (("has_part", 0.98),),
    ("isa", "part_of"): (),
    ("same_category", "same_category"): (("same_category", 0.99),),
    ("isa", "same_category"): (),
    ("same_category", "isa"): (),
}


def _normalise(entries) -> dict[tuple[str, str], Entry]:
    out: dict[tuple[str, str], Entry] = {}
    for key, value in entries.items():
        pairs = tuple(sorted((str(p), float(w)) for p, w in value))
        for _, weight in pairs:
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"prior for {key} out of [0, 1]: {weight}")
        out[(str(key[0]), str(key[1]))] = frozenset(pairs)
    return out


def default_closure() -> "ClosureTable":
    return ClosureTable(_normalise(CLOSURE_BASE), source="hand")


@dataclass
class ClosureTable:
    """Versioned composition/inverse table, mirroring ``RelationRuleSet``."""

    entries: dict[tuple[str, str], Entry]
    version: int = 1
    source: str = "hand"
    history: list[dict] = field(default_factory=list)

    # ── lookup ──────────────────────────────────────────────────────────────

    def compose(self, r1: str, r2: str) -> Entry:
        """Predicates licensed by the path ``a -r1-> b -r2-> c``."""
        return self.entries.get((r1, r2), frozenset())

    def compose_path(self, predicates) -> Entry:
        """Fold a path left to right; empty as soon as nothing is licensed."""
        predicates = list(predicates)
        if not predicates:
            return frozenset()
        current: Entry = frozenset({(predicates[0], 1.0)})
        for nxt in predicates[1:]:
            nxt_entries: set[tuple[str, float]] = set()
            for pred, prior in current:
                for out, out_prior in self.compose(pred, nxt):
                    nxt_entries.add((out, prior * out_prior))
            current = frozenset(nxt_entries)
            if not current:
                return frozenset()
        return current

    def inverse_of(self, predicate: str) -> str | None:
        return INVERSE.get(predicate)

    def required_inverse(self, predicate: str) -> str:
        try:
            return INVERSE[predicate]
        except KeyError as exc:
            raise KeyError(f"no inverse registered for predicate {predicate!r}") from exc

    # ── confidence ──────────────────────────────────────────────────────────

    @staticmethod
    def derived_confidence(c1: float, c2: float, prior: float = 1.0) -> float:
        """Confidence of a two-premise derivation.

        The geometric mean is used instead of a product so that a long chain
        decays gracefully rather than collapsing to zero in two hops.
        """
        c1 = max(0.0, min(1.0, float(c1)))
        c2 = max(0.0, min(1.0, float(c2)))
        prior = max(0.0, min(1.0, float(prior)))
        return float(prior * (c1 * c2) ** 0.5)

    # ── validation ──────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Return human-readable problems; empty means the table is coherent."""
        problems: list[str] = []
        for predicate in INVERSE:
            inv = INVERSE[predicate]
            if inv not in INVERSE:
                problems.append(f"inverse of {predicate!r} is not a predicate: {inv!r}")
            elif INVERSE[inv] != predicate:
                problems.append(
                    f"inverse is not an involution: {predicate!r} -> {inv!r} -> {INVERSE[inv]!r}"
                )
        for predicate in SYMMETRIC:
            if INVERSE.get(predicate) != predicate:
                problems.append(f"symmetric predicate {predicate!r} is not self-inverse")
        for (r1, r2), outs in self.entries.items():
            for predicate in (r1, r2, *(p for p, _ in outs)):
                if predicate not in INVERSE:
                    problems.append(f"unknown predicate {predicate!r} in entry ({r1!r}, {r2!r})")
        problems.extend(self._associativity_warnings())
        return problems

    def _associativity_warnings(self) -> list[str]:
        """Warn when two bracketing orders license disjoint predicates.

        Associativity is a warning, not an error: priors are learned and the
        tables are qualitative approximations.
        """
        warnings: list[str] = []
        for (r1, r2), first in self.entries.items():
            for p, _ in first:
                for (p2, r3), second in self.entries.items():
                    if p2 != p:
                        continue
                    left = {q for q, _ in second}
                    direct = self.compose(r2, r3)
                    if not direct:
                        continue
                    right = {q for s, _ in direct for q, _ in self.compose(r1, s)}
                    if left and right and not (left & right):
                        warnings.append(
                            f"associativity: ({r1!r}∘{r2!r})∘{r3!r} -> {sorted(left)}, "
                            f"{r1!r}∘({r2!r}∘{r3!r}) -> {sorted(right)}"
                        )
        return warnings

    # ── learning ────────────────────────────────────────────────────────────

    @classmethod
    def learn_from(
        cls,
        observations,
        min_support: int = 3,
        base: "ClosureTable | None" = None,
    ) -> "ClosureTable":
        """Count path conclusions in ``observations``.

        ``observations`` yields ``(a, r1, b, r2, c, p)`` tuples: the two
        premise edges and the observed conclusion predicate ``p``.  Only
        counts at or above ``min_support`` are added; the base table's priors
        are kept as a fallback.
        """
        counts: dict[tuple[str, str], dict[str, int]] = {}
        totals: dict[tuple[str, str], int] = {}
        for _a, r1, _b, r2, _c, p in observations:
            key = (str(r1), str(r2))
            totals[key] = totals.get(key, 0) + 1
            counts.setdefault(key, {})
            counts[key][str(p)] = counts[key].get(str(p), 0) + 1

        entries = dict((base or default_closure()).entries)
        for key, pred_counts in counts.items():
            total = totals[key]
            observed = [
                (pred, n / total)
                for pred, n in pred_counts.items()
                if n >= min_support
            ]
            if observed:
                entries[key] = frozenset(observed)
        table = cls(_normalise(entries), source="learned")
        return table

    # ── versioning ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "entries": {f"{r1}|{r2}": sorted(v) for (r1, r2), v in self.entries.items()},
            "version": int(self.version),
            "source": self.source,
        }

    def update(self, entries: dict[tuple[str, str], Entry], reason: str = "") -> int:
        before = self.snapshot()
        merged = dict(self.entries)
        merged.update(_normalise(entries))
        self.entries = merged
        self.version += 1
        self.source = "learned"
        self.history.append(
            {"version": self.version, "before": before, "after": self.snapshot(), "reason": reason}
        )
        return self.version

    def rollback(self, snapshot: dict) -> int:
        restored: dict[tuple[str, str], Entry] = {}
        for key, value in snapshot["entries"].items():
            r1, r2 = key.split("|", 1)
            restored[(r1, r2)] = frozenset((p, float(w)) for p, w in value)
        self.entries = restored
        self.version = int(snapshot["version"])
        self.source = str(snapshot["source"])
        self.history.append(
            {
                "version": self.version,
                "before": None,
                "after": self.snapshot(),
                "reason": "rollback",
            }
        )
        return self.version

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ClosureTable":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        entries: dict[tuple[str, str], Entry] = {}
        for key, value in payload["entries"].items():
            r1, r2 = key.split("|", 1)
            entries[(r1, r2)] = frozenset((p, float(w)) for p, w in value)
        return cls(entries, version=int(payload["version"]), source=str(payload["source"]))
