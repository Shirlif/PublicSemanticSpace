"""Forward and backward inference over typed relation edges (Phase 7).

The forward engine (:func:`derive`) is a small semi-naive fixpoint: it pairs
every incoming edge ``(a, r1, b)`` with every outgoing edge ``(b, r2, c)`` at a
shared middle entity and asks the :class:`~.closure.ClosureTable` which
predicates ``r1 ∘ r2`` licenses.  Derived facts join the working set and act as
premises for later rounds.

Design notes
------------
* Indices are kept per ``(subject, predicate)`` and ``(predicate, object)`` so
  the engine never scans the full relation set pairwise.
* Each round only pairs facts where at least one premise is new (semi-naive),
  and a global ``seen_pairs`` set keeps a stable pair count across rounds.
* Empty closure entries never produce negative facts; an unlicensed pair is
  simply rejected.
* ``max_derived`` bounds both the output and the in-round candidate buffer, so
  a dense hub cannot make one round arbitrarily expensive.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from pss.reasoning.relations import RelationRecord

from .closure import ClosureTable, default_closure

Triple = tuple[str, str, str]
Premise = tuple[Triple, Triple]


@dataclass(frozen=True)
class DerivedFact:
    """A fact produced by composition rather than observed directly."""

    subject: str
    predicate: str
    object: str
    confidence: float
    premises: tuple[Triple, ...]
    rule: str = "closure"
    residual: float = 0.0

    def as_record(self) -> RelationRecord:
        """Return the derived fact as a :class:`RelationRecord`."""
        return RelationRecord(
            subject=self.subject,
            predicate=self.predicate,
            object=self.object,
            confidence=self.confidence,
            source="inference",
        )

    def as_tuple(self) -> Triple:
        return (self.subject, self.predicate, self.object)


@dataclass
class InferenceResult:
    """Output of :func:`derive` with counters for auditing the search."""

    derived: list[DerivedFact] = field(default_factory=list)
    rounds: int = 0
    considered: int = 0
    rejected: int = 0

    def facts(self) -> set[Triple]:
        """Triples of the derived facts (asserted facts are not included)."""
        return {fact.as_tuple() for fact in self.derived}

    def records(self) -> list[RelationRecord]:
        return [fact.as_record() for fact in self.derived]

    def to_dict(self) -> dict:
        return {
            "derived": [
                {
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.object,
                    "confidence": fact.confidence,
                    "premises": [list(premise) for premise in fact.premises],
                    "rule": fact.rule,
                    "residual": fact.residual,
                }
                for fact in self.derived
            ],
            "rounds": self.rounds,
            "considered": self.considered,
            "rejected": self.rejected,
        }


def _as_fact(record: RelationRecord | tuple) -> tuple[str, str, str, float]:
    """Normalise a record or a 3/4-tuple to ``(subject, predicate, object, conf)``.

    ``RelationRecord`` instances and plain ``(s, p, o)`` / ``(s, p, o, conf)``
    tuples are accepted.  Confidence defaults to ``1.0``.  Any other shape
    raises :class:`ValueError`.
    """
    if isinstance(record, RelationRecord):
        if record.subject is None or record.predicate is None or record.object is None:
            raise ValueError(f"RelationRecord has a missing field: {record!r}")
        return (
            str(record.subject),
            str(record.predicate),
            str(record.object),
            float(record.confidence),
        )

    if isinstance(record, (tuple, list)):
        if len(record) == 3:
            subject, predicate, obj = record
            confidence = 1.0
        elif len(record) == 4:
            subject, predicate, obj, confidence = record
            confidence = 1.0 if confidence is None else float(confidence)
        else:
            raise ValueError(f"expected a 3- or 4-tuple, got {len(record)} elements")
        if subject is None or predicate is None or obj is None:
            raise ValueError("subject, predicate and object must not be None")
        return (str(subject), str(predicate), str(obj), confidence)

    raise ValueError(f"cannot interpret {record!r} as a relation fact")


def derive(
    records: Iterable[RelationRecord | tuple],
    closure: ClosureTable | None = None,
    max_rounds: int = 2,
    min_conf: float = 0.5,
    max_derived: int = 1000,
    type_check: Callable[[str, str, str], bool] | None = None,
    allowed_predicates: Iterable[str] | None = None,
    exclude_self: bool = True,
) -> InferenceResult:
    """Forward-chain asserted relation facts.

    Parameters
    ----------
    records:
        Asserted facts as :class:`RelationRecord` or ``(s, p, o[, conf])``.
        Inputs are read once and never mutated.
    closure:
        Composition table; defaults to :func:`~.closure.default_closure`.
    max_rounds:
        Maximum number of semi-naive passes.  Facts derived in round ``k`` are
        premises for round ``k + 1``, so a chain of length ``n`` needs
        ``n - 1`` rounds.
    min_conf:
        Derived facts below this confidence are discarded.
    max_derived:
        Hard cap on accepted facts.  Also stops pair scanning within a round
        once that many candidates have been buffered.
    type_check:
        Optional ``(subject, predicate, object) -> bool`` veto.
    allowed_predicates:
        Optional whitelist applied to conclusion predicates.
    exclude_self:
        Reject conclusions whose subject and object are identical.

    Counter semantics
    -----------------
    ``rounds`` counts passes that produced at least one new fact.
    ``considered`` counts distinct premise pairs evaluated.  ``rejected``
    counts candidate conclusions discarded by a filter (self-loop, empty
    closure, whitelist, type check, low confidence, or deduplication); a pair
    that licenses several predicates can therefore contribute more than once.
    """
    table = default_closure() if closure is None else closure
    max_rounds = max(0, int(max_rounds))
    max_derived = max(0, int(max_derived))
    allowed = None if allowed_predicates is None else frozenset(allowed_predicates)

    asserted: dict[Triple, float] = {}
    order: list[Triple] = []
    for record in records:
        subject, predicate, obj, confidence = _as_fact(record)
        key = (subject, predicate, obj)
        if key not in asserted:
            asserted[key] = confidence
            order.append(key)
        elif confidence > asserted[key]:
            asserted[key] = confidence

    result = InferenceResult()
    if max_derived == 0 or not asserted:
        return result

    working: dict[Triple, float] = dict(asserted)
    derived: list[DerivedFact] = []

    outgoing: dict[tuple[str, str], list[str]] = {}
    incoming: dict[tuple[str, str], list[str]] = {}
    out_predicates: dict[str, set[str]] = {}
    in_predicates: dict[str, set[str]] = {}

    def add_edge(key: Triple) -> None:
        subject, predicate, obj = key
        outgoing.setdefault((subject, predicate), []).append(obj)
        incoming.setdefault((predicate, obj), []).append(subject)
        out_predicates.setdefault(subject, set()).add(predicate)
        in_predicates.setdefault(obj, set()).add(predicate)

    for key in order:
        add_edge(key)

    seen_pairs: set[Premise] = set()
    considered = 0
    rejected = 0
    rounds = 0
    frontier = list(order)
    stopped = False

    for _ in range(max_rounds):
        if not frontier or stopped:
            break
        candidates: dict[Triple, tuple[float, Premise]] = {}
        stopped = False

        def consider_pair(premise1: Triple, premise2: Triple) -> None:
            nonlocal considered, rejected, stopped
            if stopped:
                return
            pair = (premise1, premise2)
            if pair in seen_pairs:
                return
            seen_pairs.add(pair)
            considered += 1

            a, r1, middle = premise1
            middle2, r2, c = premise2
            if middle != middle2:  # defensive; indices should guarantee this
                rejected += 1
                return
            if exclude_self and a == c:
                rejected += 1
                return

            entries = table.compose(r1, r2)
            if not entries:
                rejected += 1
                return

            c1 = working[premise1]
            c2 = working[premise2]
            for predicate, prior in sorted(entries):
                if allowed is not None and predicate not in allowed:
                    rejected += 1
                    continue
                conclusion = (a, predicate, c)
                if conclusion in working:
                    rejected += 1
                    continue
                if type_check is not None and not type_check(a, predicate, c):
                    rejected += 1
                    continue
                confidence = table.derived_confidence(c1, c2, prior)
                if confidence < min_conf:
                    rejected += 1
                    continue
                previous = candidates.get(conclusion)
                if previous is None or confidence > previous[0]:
                    candidates[conclusion] = (confidence, pair)
                if len(derived) + len(candidates) >= max_derived:
                    stopped = True
                    return

        for fact in frontier:
            subject, predicate, middle = fact
            # Fact used as the incoming edge: pair with edges leaving `middle`.
            for next_predicate in sorted(out_predicates.get(middle, ())):
                for obj in outgoing.get((middle, next_predicate), ()):
                    consider_pair(fact, (middle, next_predicate, obj))

            # Fact used as the outgoing edge: pair with edges arriving at it.
            for prev_predicate in sorted(in_predicates.get(subject, ())):
                for other in incoming.get((prev_predicate, subject), ()):
                    consider_pair((other, prev_predicate, subject), fact)

        if not candidates:
            break

        # Highest confidence first, then triple for deterministic output.
        ranked = sorted(candidates.items(), key=lambda item: (-item[1][0], item[0]))
        new_keys: list[Triple] = []
        for conclusion, (confidence, premises) in ranked:
            if len(derived) >= max_derived:
                break
            if conclusion in working:
                continue
            fact = DerivedFact(
                subject=conclusion[0],
                predicate=conclusion[1],
                object=conclusion[2],
                confidence=confidence,
                premises=premises,
                rule="closure",
                residual=0.0,
            )
            derived.append(fact)
            working[conclusion] = confidence
            add_edge(conclusion)
            new_keys.append(conclusion)

        if not new_keys:
            break
        rounds += 1
        frontier = new_keys
        if len(derived) >= max_derived:
            break

    result.derived = derived
    result.rounds = rounds
    result.considered = considered
    result.rejected = rejected
    return result


def derive_records(
    records: Iterable[RelationRecord | tuple],
    **kwargs,
) -> list[RelationRecord]:
    """Convenience wrapper returning only the derived relation records."""
    return [fact.as_record() for fact in derive(records, **kwargs).derived]


def backward_chain(
    records: Iterable[RelationRecord | tuple],
    subject: str,
    object: str,
    closure: ClosureTable | None = None,
    max_depth: int = 3,
    min_conf: float = 0.5,
) -> list[tuple[str, float, tuple[Triple, ...]]]:
    """Approximate backward search for predicates linking ``subject`` to ``object``.

    This is a bounded breadth-first enumeration, not full resolution.  It
    explores only simple paths (a node never repeats within a path), stops
    after ``max_paths = 10000`` edge expansions, and composes along the way
    with the same table used by :func:`derive`.

    Returns ``(predicate, confidence, path)`` candidates, best per predicate.
    The confidence of a path with edge confidences ``c_i`` and composed prior
    ``p`` is ``p * (prod(c_i)) ** (1 / len(c_i))``, which reduces exactly to
    :meth:`ClosureTable.derived_confidence` for two-edge paths.
    """
    table = default_closure() if closure is None else closure
    max_depth = max(0, int(max_depth))
    subject = str(subject)
    object = str(object)
    if max_depth == 0 or subject == object:
        return []

    adjacency: dict[str, list[tuple[str, str, float]]] = {}
    seen_edges: set[Triple] = set()
    for record in records:
        src, predicate, dst, confidence = _as_fact(record)
        key = (src, predicate, dst)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        adjacency.setdefault(src, []).append((predicate, dst, confidence))

    max_paths = 10000
    explored = 0
    best: dict[str, tuple[float, tuple[Triple, ...]]] = {}
    # Queue items: (node, composed entry or None, edge confidences, path, visited)
    queue: deque = deque()
    queue.append((subject, None, (), (), frozenset({subject})))

    while queue:
        node, entry, confidences, path, visited = queue.popleft()
        if len(path) >= max_depth:
            continue
        for predicate, next_node, edge_conf in adjacency.get(node, ()):
            explored += 1
            if explored > max_paths:
                queue.clear()
                break
            if next_node in visited:
                continue
            if entry is None:
                composed: tuple[tuple[str, float], ...] = ((predicate, 1.0),)
                next_confidences = (edge_conf,)
            else:
                merged: dict[str, float] = {}
                for current_predicate, current_prior in entry:
                    for out_predicate, step_prior in table.compose(current_predicate, predicate):
                        prior = current_prior * step_prior
                        if prior > merged.get(out_predicate, -1.0):
                            merged[out_predicate] = prior
                if not merged:
                    continue
                composed = tuple(sorted(merged.items()))
                next_confidences = confidences + (edge_conf,)

            next_path = path + ((node, predicate, next_node),)
            if next_node == object:
                geometric = 1.0
                for confidence in next_confidences:
                    geometric *= max(0.0, min(1.0, confidence))
                geometric **= 1.0 / len(next_confidences)
                for out_predicate, prior in composed:
                    confidence = max(0.0, min(1.0, prior * geometric))
                    if confidence < min_conf:
                        continue
                    previous = best.get(out_predicate)
                    if (
                        previous is None
                        or confidence > previous[0]
                        or (confidence == previous[0] and len(next_path) < len(previous[1]))
                    ):
                        best[out_predicate] = (confidence, next_path)
            else:
                queue.append(
                    (next_node, composed, next_confidences, next_path, visited | {next_node})
                )

    return [(predicate, confidence, path) for predicate, (confidence, path) in sorted(best.items())]


__all__ = [
    "DerivedFact",
    "InferenceResult",
    "backward_chain",
    "derive",
    "derive_records",
]
