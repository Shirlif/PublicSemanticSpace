"""Symbolic set, quantifier and temporal queries over ``(s, p, o)`` facts.

The helpers here operate on explicit Python sets of triples. They are
deliberately *not* vectorized: superposing a fact set into a single VSA vector
and counting hits is unreliable because the number of recoverable
superposition components grows only like ``O(d / log d)`` for dimension ``d``.
Set membership, quantification and time filtering therefore stay symbolic.

A "fact" is a ``(subject, predicate, object)`` tuple of strings, optionally
followed by a confidence float. :class:`~pss.reasoning.relations.RelationRecord`
instances are also accepted anywhere a fact is expected.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from pss.reasoning.relations import RelationRecord

    FactLike = tuple[str, str, str] | tuple[str, str, str, float] | RelationRecord
else:
    FactLike = object


Triple = tuple[str, str, str]


def _split_fact(item: object) -> tuple[str, str, str, float]:
    """Normalize a tuple-like or RelationRecord-like fact to ``(s, p, o, conf)``."""
    if isinstance(item, (tuple, list)):
        if len(item) == 3:
            return item[0], item[1], item[2], 1.0
        if len(item) == 4:
            return item[0], item[1], item[2], float(item[3])
        raise ValueError(f"fact must have 3 or 4 fields, got {len(item)}")
    if hasattr(item, "subject") and hasattr(item, "predicate"):
        return (
            item.subject,
            item.predicate,
            getattr(item, "object", None),
            float(getattr(item, "confidence", 1.0)),
        )
    raise TypeError(f"unsupported fact type: {type(item)!r}")


def _triple_of(item: object) -> Triple:
    """Return only the ``(s, p, o)`` part of a fact, ignoring confidence."""
    subject, predicate, obj, _ = _split_fact(item)
    return subject, predicate, obj


class FactSet:
    """A deduplicated set of facts, each carrying a confidence in ``[0, 1]``.

    Iteration yields ``(subject, predicate, object, confidence)``. Adding the
    same triple again replaces its confidence (last write wins); the set keeps
    insertion order.
    """

    def __init__(self, facts: Iterable[FactLike] | None = None):
        self._facts: dict[Triple, float] = {}
        if facts is None:
            return
        for item in facts:
            subject, predicate, obj, confidence = _split_fact(item)
            self.add(subject, predicate, obj, confidence)

    def add(self, subject: str, predicate: str, obj: str, conf: float = 1.0) -> None:
        """Insert or update a fact."""
        self._facts[(subject, predicate, obj)] = float(conf)

    def remove(self, subject: str, predicate: str, obj: str) -> None:
        """Drop a fact if present; absent facts are ignored."""
        self._facts.pop((subject, predicate, obj), None)

    def __contains__(self, fact: object) -> bool:
        try:
            return _triple_of(fact) in self._facts
        except (TypeError, ValueError):
            return False

    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self) -> Iterator[tuple[str, str, str, float]]:
        for (subject, predicate, obj), confidence in self._facts.items():
            yield subject, predicate, obj, confidence

    def facts(self, min_conf: float = 0.0) -> set[Triple]:
        """Return the bare triples whose confidence is at least ``min_conf``."""
        return {
            triple
            for triple, confidence in self._facts.items()
            if confidence >= min_conf
        }

    def confidence(self, subject: str, predicate: str, obj: str) -> float | None:
        """Return the confidence of a fact, or ``None`` when it is absent."""
        return self._facts.get((subject, predicate, obj))

    def select(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        min_conf: float = 0.0,
    ) -> set[Triple]:
        """Return triples matching any provided field and the confidence floor."""
        out: set[Triple] = set()
        for (s, p, o), confidence in self._facts.items():
            if subject is not None and s != subject:
                continue
            if predicate is not None and p != predicate:
                continue
            if object is not None and o != object:
                continue
            if confidence < min_conf:
                continue
            out.add((s, p, o))
        return out

    def subjects(
        self,
        predicate: str | None,
        object: str | None = None,
        min_conf: float = 0.0,
    ) -> set[str]:
        """Return subjects of matching facts."""
        return {
            s
            for (s, p, o), confidence in self._facts.items()
            if (predicate is None or p == predicate)
            and (object is None or o == object)
            and confidence >= min_conf
        }

    def objects(
        self,
        predicate: str | None,
        subject: str | None = None,
        min_conf: float = 0.0,
    ) -> set[str]:
        """Return objects of matching facts."""
        return {
            o
            for (s, p, o), confidence in self._facts.items()
            if (predicate is None or p == predicate)
            and (subject is None or s == subject)
            and confidence >= min_conf
        }

    def copy(self) -> "FactSet":
        """Return an independent shallow copy."""
        clone = FactSet()
        clone._facts = dict(self._facts)
        return clone

    def to_dict(self) -> dict:
        """Return a JSON-friendly snapshot of the set."""
        return {
            "count": len(self._facts),
            "facts": [
                {
                    "subject": s,
                    "predicate": p,
                    "object": o,
                    "confidence": confidence,
                }
                for (s, p, o), confidence in self._facts.items()
            ],
        }


@dataclass(frozen=True)
class QuantifierResult:
    """Outcome of a set-level quantifier over a collection of entities."""

    kind: str
    value: bool | int
    total: int
    confidence: float
    counterexamples: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "value": self.value,
            "total": self.total,
            "confidence": self.confidence,
            "counterexamples": list(self.counterexamples),
        }


def _as_factset(facts: FactSet | Iterable[FactLike]) -> FactSet:
    if isinstance(facts, FactSet):
        return facts
    return FactSet(facts)


def _unique_entities(entities: Iterable[str]) -> list[str]:
    """Deduplicate entities while preserving their original order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for entity in entities:
        if entity not in seen:
            seen.add(entity)
            ordered.append(entity)
    return ordered


def _match_confidences(
    facts: FactSet,
    entities: Iterable[str],
    predicate: str,
    target: str,
    min_conf: float,
) -> list[tuple[str, float | None]]:
    return [
        (entity, facts.confidence(entity, predicate, target))
        for entity in _unique_entities(entities)
    ]


def all_of(
    facts: FactSet | Iterable[FactLike],
    entities: Iterable[str],
    predicate: str,
    target: str,
    min_conf: float = 0.5,
) -> QuantifierResult:
    """Check that every entity has ``(entity, predicate, target)``.

    ``confidence`` is the minimum confidence among the supporting facts, or
    ``0.0`` when no entity supports the query (including an empty entity set).
    Failing entities are reported in order as ``counterexamples``.
    """
    fs = _as_factset(facts)
    matches = _match_confidences(fs, entities, predicate, target, min_conf)
    supporting = [conf for _, conf in matches if conf is not None and conf >= min_conf]
    counterexamples = tuple(entity for entity, conf in matches if conf is None or conf < min_conf)
    confidence = min(supporting) if supporting else 0.0
    return QuantifierResult(
        kind="all",
        value=not counterexamples,
        total=len(matches),
        confidence=confidence,
        counterexamples=counterexamples,
    )


def any_of(
    facts: FactSet | Iterable[FactLike],
    entities: Iterable[str],
    predicate: str,
    target: str,
    min_conf: float = 0.5,
) -> QuantifierResult:
    """Check that at least one entity has ``(entity, predicate, target)``.

    ``confidence`` is the strongest supporting confidence (``max``); entities
    without a qualifying fact are reported as counterexamples.
    """
    fs = _as_factset(facts)
    matches = _match_confidences(fs, entities, predicate, target, min_conf)
    supporting = [conf for _, conf in matches if conf is not None and conf >= min_conf]
    counterexamples = tuple(entity for entity, conf in matches if conf is None or conf < min_conf)
    return QuantifierResult(
        kind="any",
        value=bool(supporting),
        total=len(matches),
        confidence=max(supporting) if supporting else 0.0,
        counterexamples=counterexamples,
    )


def none_of(
    facts: FactSet | Iterable[FactLike],
    entities: Iterable[str],
    predicate: str,
    target: str,
    min_conf: float = 0.5,
) -> QuantifierResult:
    """Check that no entity has ``(entity, predicate, target)``.

    ``confidence`` is ``1.0`` minus the strongest violating confidence, so a
    high-confidence violation lowers confidence in the negative answer.
    """
    fs = _as_factset(facts)
    matches = _match_confidences(fs, entities, predicate, target, min_conf)
    violating = [conf for _, conf in matches if conf is not None and conf >= min_conf]
    counterexamples = tuple(
        entity for entity, conf in matches if conf is not None and conf >= min_conf
    )
    return QuantifierResult(
        kind="none",
        value=not violating,
        total=len(matches),
        confidence=1.0 - max(violating) if violating else 1.0,
        counterexamples=counterexamples,
    )


def count_of(
    facts: FactSet | Iterable[FactLike],
    entities: Iterable[str],
    predicate: str,
    target: str,
    min_conf: float = 0.5,
) -> QuantifierResult:
    """Count how many entities have ``(entity, predicate, target)``.

    ``confidence`` is the minimum confidence among the counted facts, or
    ``0.0`` when nothing was counted.
    """
    fs = _as_factset(facts)
    matches = _match_confidences(fs, entities, predicate, target, min_conf)
    supporting = [conf for _, conf in matches if conf is not None and conf >= min_conf]
    counterexamples = tuple(entity for entity, conf in matches if conf is None or conf < min_conf)
    return QuantifierResult(
        kind="count",
        value=len(supporting),
        total=len(matches),
        confidence=min(supporting) if supporting else 0.0,
        counterexamples=counterexamples,
    )


def unique_object(
    facts: FactSet | Iterable[FactLike],
    subject: str,
    predicate: str,
    min_conf: float = 0.5,
) -> str | None:
    """Return the object of ``(subject, predicate, ?)`` when exactly one holds.

    Returns ``None`` both when no object qualifies and when several do
    (ambiguous). A single unary fact whose object is ``None`` is therefore
    indistinguishable from an absent fact.
    """
    fs = _as_factset(facts)
    objects = fs.objects(predicate, subject=subject, min_conf=min_conf)
    if len(objects) == 1:
        return next(iter(objects))
    return None


def select_entities(
    facts: FactSet | Iterable[FactLike],
    predicate: str | None = None,
    object: str | None = None,
    min_conf: float = 0.0,
) -> set[str]:
    """Return subjects matching an optional predicate/object filter."""
    return _as_factset(facts).subjects(predicate, object=object, min_conf=min_conf)


class TemporalHistory:
    """A time-indexed log of fact snapshots.

    Each :meth:`record` call stores a snapshot at one time; recording an
    existing time replaces that snapshot and its explicit removals. Queries
    such as :meth:`at` merge every snapshot at or before the requested time and
    then apply the explicit removals recorded at those times. Snapshots are
    therefore *additive*: a fact stays visible until a later record lists it in
    ``removed``. Emit ``removed`` explicitly when an object moves, otherwise
    old and new locations will both be visible after the move.
    """

    def __init__(self) -> None:
        self.entries: list[tuple[float, FactSet]] = []
        # Explicit removals applied after the snapshot recorded at each time.
        self.removals: dict[float, set[Triple]] = {}

    def record(
        self,
        time: float,
        facts: FactSet | Iterable[FactLike],
        removed: Iterable[FactLike] = (),
    ) -> None:
        """Store or replace the snapshot recorded at ``time``."""
        stamp = float(time)
        snapshot = facts.copy() if isinstance(facts, FactSet) else FactSet(facts)
        removed_triples = {_triple_of(item) for item in removed}

        index = bisect.bisect_left([entry[0] for entry in self.entries], stamp)
        if index < len(self.entries) and self.entries[index][0] == stamp:
            self.entries[index] = (stamp, snapshot)
        else:
            self.entries.insert(index, (stamp, snapshot))
        self.removals[stamp] = removed_triples

    def times(self) -> list[float]:
        return [stamp for stamp, _ in self.entries]

    def _apply(self, state: FactSet, snapshot: FactSet, stamp: float) -> None:
        for subject, predicate, obj, confidence in snapshot:
            state.add(subject, predicate, obj, confidence)
        for triple in self.removals.get(stamp, ()):
            state.remove(*triple)

    def at(self, time: float) -> FactSet:
        """Return the merged state of all snapshots recorded at or before ``time``.

        Snapshots are additive; see the class docstring. Explicit ``removed``
        triples are applied after each snapshot's additions.
        """
        stamp = float(time)
        state = FactSet()
        for entry_time, snapshot in self.entries:
            if entry_time > stamp:
                break
            self._apply(state, snapshot, entry_time)
        return state

    def between(self, t0: float, t1: float) -> FactSet:
        """Merge only the snapshots whose times fall in the closed interval."""
        start, end = float(t0), float(t1)
        state = FactSet()
        for entry_time, snapshot in self.entries:
            if entry_time < start:
                continue
            if entry_time > end:
                break
            self._apply(state, snapshot, entry_time)
        return state

    def added_between(self, t0: float, t1: float) -> set[Triple]:
        """Return facts present at ``t1`` but not at ``t0`` (net additions)."""
        return self.at(t1).facts() - self.at(t0).facts()

    def removed_between(self, t0: float, t1: float) -> set[Triple]:
        """Return facts present at ``t0`` but not at ``t1`` (net removals).

        This is only meaningful when the intervening record declares the
        removed facts explicitly; purely additive snapshots never remove.
        """
        return self.at(t0).facts() - self.at(t1).facts()

    def first_seen(self, fact: FactLike) -> float | None:
        """Return the earliest recorded time at which the fact is present."""
        triple = _triple_of(fact)
        state = FactSet()
        for entry_time, snapshot in self.entries:
            self._apply(state, snapshot, entry_time)
            if triple in state:
                return entry_time
        return None

    def last_seen(self, fact: FactLike) -> float | None:
        """Return the latest recorded time at which the fact is present."""
        triple = _triple_of(fact)
        state = FactSet()
        last: float | None = None
        for entry_time, snapshot in self.entries:
            self._apply(state, snapshot, entry_time)
            if triple in state:
                last = entry_time
        return last

    def where_was(
        self,
        entity: str,
        time: float,
        location_predicates: Iterable[str] = ("in", "on", "at", "near"),
    ) -> set[str]:
        """Return the locations of ``entity`` in the merged state at ``time``.

        Only facts whose predicate is in ``location_predicates`` are considered.
        Because snapshots are additive, a moved object is reported at both its
        old and new locations unless the move recorded an explicit removal.
        """
        predicates = set(location_predicates)
        state = self.at(time)
        return {
            obj
            for subject, predicate, obj, _ in state
            if subject == entity and predicate in predicates and obj is not None
        }

    def history_of(self, entity: str) -> list[tuple[float, set[Triple]]]:
        """Return ``(time, facts)`` for every recorded time.

        Facts are included when ``entity`` appears as either subject or object.
        """
        out: list[tuple[float, set[Triple]]] = []
        state = FactSet()
        for entry_time, snapshot in self.entries:
            self._apply(state, snapshot, entry_time)
            facts = {
                (subject, predicate, obj)
                for subject, predicate, obj, _ in state
                if subject == entity or obj == entity
            }
            out.append((entry_time, facts))
        return out


def resolve_location_reference(
    history: TemporalHistory,
    entity: str,
    time: float,
) -> dict:
    """Convenience wrapper: where was ``entity`` at ``time``?

    Returns ``{"locations": [...], "at": time, "found": bool}`` with locations
    sorted for stable output. ``found`` is false when no location fact exists.
    """
    locations = sorted(history.where_was(entity, time))
    return {"locations": locations, "at": float(time), "found": bool(locations)}
