"""Typed composition of primitive operators (P0-P3 Lego layer).

An :class:`Atom` is a primitive with a :class:`Signature` (input type, output
type, optional reference frame) and a cost.  Composition is *sequencing* of
atoms: the output type of one atom must match the input type of the next.
This is the "stud" that makes free composition safe.

The search is best-first over typed payloads.  It is deliberately small: the
point is to make the closed loop measurable, not to replace the A* planner in
``pss.reasoning.planner``.

Selection atoms
---------------
``SelectAtom``, ``ProjectAtom``, ``ConstraintAtom`` and ``PickAtom`` compose a
candidate filter chain: select by a categorical attribute, project onto a
numeric interval, enforce a feasibility constraint, then pick.  A plan is only
a plan if every filter ran; otherwise the picked object is a different one.

Verified cache
--------------
:class:`PlanCache` writes an atom sequence back **only after the caller
confirms the goal was verified**.  Failed attempts lower confidence but never
overwrite a verified plan.  Replay re-checks every prerequisite and the goal,
so a stale plan is rejected rather than executed.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Protocol, Sequence

import numpy as np

Payload = dict[str, Any]
AtomMap = dict[str, "Atom"]


@dataclass(frozen=True)
class Signature:
    """What an atom consumes and produces.

    ``frame`` is part of the signature on purpose: a relation atom computed in
    the world frame is not interchangeable with the same predicate computed in
    a camera frame.
    """

    input_type: str
    output_type: str
    frame: str | None = None
    cost: float = 1.0


class Atom(Protocol):
    name: str
    signature: Signature

    def applicable(self, payload: Payload) -> bool:
        """Whether the atom can run on this payload."""

    def apply(self, payload: Payload) -> Payload:
        """Return a new payload; must not mutate the input."""


@dataclass(frozen=True)
class SearchNode:
    payload: Payload
    stype: str
    cost: float
    steps: tuple[str, ...]
    depth: int


@dataclass
class CompositionResult:
    success: bool
    steps: list[str]
    cost: float
    expansions: int
    payload: Payload | None = None
    reason: str = "ok"

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "steps": list(self.steps),
            "cost": self.cost,
            "expansions": self.expansions,
            "reason": self.reason,
        }


def _freeze(value: Any) -> Hashable:
    """Deterministic hashable projection for visited-state deduplication."""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze(v) for v in value))
    if isinstance(value, np.ndarray):
        return tuple(np.round(np.asarray(value, dtype=float).ravel(), 6))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def best_first(
    initial: Payload,
    atoms: Sequence[Atom],
    goal_test: Callable[[Payload], bool],
    max_expansions: int = 2000,
    max_steps: int = 12,
    check_types: bool = True,
    state_key: Callable[[Payload], Hashable] | None = None,
) -> CompositionResult:
    """Best-first search over type-compatible atom sequences.

    ``check_types=False`` disables the signature check; it exists only so the
    evaluation can measure what happens without the "stud".
    """
    if "type" not in initial:
        raise ValueError("initial payload needs a 'type' field")

    counter = itertools.count()
    start = SearchNode(dict(initial), str(initial["type"]), 0.0, (), 0)
    heap: list[tuple[float, int, SearchNode]] = [(0.0, next(counter), start)]
    visited: set[Hashable] = set()
    expansions = 0

    while heap:
        cost, _, node = heapq.heappop(heap)
        expansions += 1
        if expansions > max_expansions:
            return CompositionResult(
                False,
                list(node.steps),
                cost,
                expansions,
                node.payload,
                "expansion_limit",
            )
        if goal_test(node.payload):
            return CompositionResult(True, list(node.steps), cost, expansions, node.payload, "ok")
        if node.depth >= max_steps:
            continue

        key = state_key(node.payload) if state_key is not None else _freeze(node.payload)
        if key in visited:
            continue
        visited.add(key)

        for atom in atoms:
            if check_types and atom.signature.input_type != node.stype:
                continue
            if not atom.applicable(node.payload):
                continue
            child = atom.apply(dict(node.payload))
            child_type = str(child.get("type", ""))
            penalty = float(child.get("residual", 0.0))
            child_cost = cost + atom.signature.cost + penalty
            child_node = SearchNode(
                child,
                child_type,
                child_cost,
                node.steps + (atom.name,),
                node.depth + 1,
            )
            heapq.heappush(heap, (child_cost, next(counter), child_node))

    return CompositionResult(False, [], 0.0, expansions, None, "exhausted")


# ── selection / projection / constraint atoms ───────────────────────────────


@dataclass(frozen=True)
class Candidate:
    """A selectable entity with categorical, numeric and embedding attributes."""

    id: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "id":
            return self.id
        return self.attributes.get(key, default)


class SelectAtom:
    """Keep candidates whose categorical ``attribute`` equals ``value``."""

    def __init__(self, name: str, attribute: str, value: Any):
        self.name = name
        self.attribute = attribute
        self.value = value
        self.signature = Signature("candidates", "candidates", cost=0.1)

    def applicable(self, payload: Payload) -> bool:
        return "candidates" in payload

    def apply(self, payload: Payload) -> Payload:
        out = dict(payload)
        out["type"] = "candidates"
        out["candidates"] = [
            candidate
            for candidate in payload["candidates"]
            if candidate.get(self.attribute) == self.value
        ]
        return out


class ProjectAtom:
    """Keep candidates that satisfy a numeric/attribute predicate."""

    def __init__(self, name: str, keep: Callable[[Candidate], bool]):
        self.name = name
        self.keep = keep
        self.signature = Signature("candidates", "candidates", cost=0.1)

    def applicable(self, payload: Payload) -> bool:
        return "candidates" in payload

    def apply(self, payload: Payload) -> Payload:
        out = dict(payload)
        out["type"] = "candidates"
        out["candidates"] = [
            candidate for candidate in payload["candidates"] if self.keep(candidate)
        ]
        return out


class ConstraintAtom(ProjectAtom):
    """Hard feasibility filter; same mechanics as projection, different intent."""


class TypeMatchAtom:
    """Keep candidates whose embedding matches a type constraint.

    ``constraint`` is duck-typed: it needs ``membership(query) -> float``.
    This is how ``pss.reasoning.constraints.TypeConstraint`` joins a plan
    without making the composition layer depend on it at import time.
    """

    def __init__(
        self,
        name: str,
        constraint,
        threshold: float = 0.5,
        embedding_key: str = "embedding",
    ):
        self.name = name
        self.constraint = constraint
        self.threshold = float(threshold)
        self.embedding_key = embedding_key
        self.signature = Signature("candidates", "candidates", cost=0.2)

    def applicable(self, payload: Payload) -> bool:
        return "candidates" in payload

    def apply(self, payload: Payload) -> Payload:
        kept = []
        for candidate in payload["candidates"]:
            query = candidate.get(self.embedding_key) or {}
            if self.constraint.membership(query) >= self.threshold:
                kept.append(candidate)
        out = dict(payload)
        out["type"] = "candidates"
        out["candidates"] = kept
        return out


class PickAtom:
    """Pick the first surviving candidate.

    With ``require_unique=True`` the atom refuses to run unless exactly one
    candidate survives.  That is what forces a selection plan to apply every
    filter instead of stopping once the target happens to be first.
    """

    def __init__(
        self,
        name: str = "pick",
        attribute: str = "id",
        require_unique: bool = False,
    ):
        self.name = name
        self.attribute = attribute
        self.require_unique = bool(require_unique)
        self.signature = Signature("candidates", "selected", cost=1.0)

    def applicable(self, payload: Payload) -> bool:
        candidates = payload.get("candidates") or []
        if not candidates:
            return False
        if self.require_unique:
            return len(candidates) == 1
        return True

    def apply(self, payload: Payload) -> Payload:
        out = dict(payload)
        out["type"] = "selected"
        out["selected"] = payload["candidates"][0].get(self.attribute)
        return out


# ── plan templates and verified cache ───────────────────────────────────────


def plan_template(steps: Sequence[str], subject: str | None = None) -> tuple[str, ...]:
    """Normalize a plan so it can be replayed for a different subject."""
    if subject is None:
        return tuple(steps)
    marker = f"move:{subject}"
    return tuple("move:$subject" if step == marker else step for step in steps)


def instantiate(template: Sequence[str], subject: str | None = None) -> list[str]:
    if subject is None:
        return list(template)
    return [step.replace("$subject", subject) for step in template]


@dataclass
class CachedPlan:
    steps: tuple[str, ...] = ()
    support: int = 0
    attempts: int = 0
    failures: int = 0
    version: int = 1

    def confidence(self) -> float:
        """Laplace-smoothed success rate; 0.5 before any observation."""
        return (self.support + 1) / (self.attempts + 2)


@dataclass
class CacheStats:
    hits: int = 0
    stale: int = 0
    misses: int = 0
    stores: int = 0
    verified: int = 0
    rejected: int = 0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "stale": self.stale,
            "misses": self.misses,
            "stores": self.stores,
            "verified": self.verified,
            "rejected": self.rejected,
        }


@dataclass
class PlanCache:
    """Verified, versioned composition cache.

    ``record`` is the only write path.  It stores a plan only when the caller
    passes ``verified=True`` and the plan is non-empty; failures lower the
    entry's confidence but never replace a verified plan.
    """

    entries: dict[Hashable, CachedPlan] = field(default_factory=dict)
    stats: CacheStats = field(default_factory=CacheStats)
    min_support: int = 1
    history: list[dict] = field(default_factory=list)

    def get(
        self,
        key: Hashable,
        atom_map: AtomMap,
        initial: Payload,
        goal_test: Callable[[Payload], bool],
        subject: str | None = None,
    ) -> CompositionResult | None:
        entry = self.entries.get(key)
        if entry is None or not entry.steps:
            self.stats.misses += 1
            return None
        steps = instantiate(entry.steps, subject)
        payload = dict(initial)
        for name in steps:
            atom = atom_map.get(name)
            if atom is None or not atom.applicable(payload):
                self.stats.stale += 1
                return None
            payload = atom.apply(dict(payload))
        if not goal_test(payload):
            self.stats.stale += 1
            return None
        self.stats.hits += 1
        return CompositionResult(True, steps, 0.0, 0, payload, "cache")

    def record(
        self,
        key: Hashable,
        steps: Sequence[str],
        verified: bool,
        subject: str | None = None,
    ) -> None:
        entry = self.entries.get(key)
        if entry is None:
            entry = CachedPlan()
            self.entries[key] = entry
        entry.attempts += 1
        if verified and steps:
            entry.support += 1
            entry.steps = plan_template(steps, subject)
            entry.version += 1
            self.stats.verified += 1
            if entry.support >= self.min_support:
                self.stats.stores += 1
        else:
            entry.failures += 1
            self.stats.rejected += 1
            if entry.support == 0:
                entry.steps = ()

    def put(self, key: Hashable, steps: Sequence[str], subject: str | None = None) -> None:
        """Backward-compatible alias for a verified write."""
        self.record(key, steps, verified=True, subject=subject)

    def confidence(self, key: Hashable) -> float:
        entry = self.entries.get(key)
        return entry.confidence() if entry is not None else 0.0

    def coverage(self, keys: Sequence[Hashable]) -> float:
        keys = list(keys)
        if not keys:
            return 0.0
        covered = sum(1 for key in keys if self.entries.get(key) and self.entries[key].steps)
        return covered / len(keys)

    def snapshot(self) -> dict:
        return {
            "entries": [
                {
                    "key": key,
                    "steps": list(entry.steps),
                    "support": entry.support,
                    "attempts": entry.attempts,
                    "failures": entry.failures,
                    "version": entry.version,
                }
                for key, entry in self.entries.items()
            ],
            "stats": self.stats.to_dict(),
        }

    def rollback(self, snapshot: dict) -> None:
        self.history.append(self.snapshot())
        self.entries = {}
        for item in snapshot["entries"]:
            self.entries[item["key"]] = CachedPlan(
                steps=tuple(item["steps"]),
                support=int(item["support"]),
                attempts=int(item["attempts"]),
                failures=int(item["failures"]),
                version=int(item["version"]),
            )
        stats = snapshot["stats"]
        self.stats = CacheStats(
            hits=int(stats["hits"]),
            stale=int(stats["stale"]),
            misses=int(stats["misses"]),
            stores=int(stats["stores"]),
            verified=int(stats["verified"]),
            rejected=int(stats["rejected"]),
        )
