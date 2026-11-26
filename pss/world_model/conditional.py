"""Conditional action operators with resource accounting (Phase 7 extension).

This module implements the ``pss.binding.protocol.Operator`` family for
conditional action effects.  It extends the flat add/remove operators in
``pss.world_model.actions`` in three ways:

- effects carry their own conditions and are applied sequentially;
- operators declare consumable / producible resources;
- operators may carry an explicit rollback effect set.

Facts are tuples of strings, e.g. ``("clothes", "in", "basket")``, and states
are frozensets of facts.  Nothing here mutates an input state.

Semantics worth remembering
---------------------------
* An effect's conditions are checked against the state produced by all
  previous effects, not against the operator's input state.
* ``inverse_effects`` stores the delta in the direction the *undo* operator
  applies it.  It is not inverted again by :meth:`ConditionalOperator.inverse`;
  use :meth:`Effect.inverted` when deriving it from a forward effect.
* Every inverse operator has empty preconditions.  The caller must validate
  the live state after the undo.  A conditional adverse effect is only undone
  when the author lists that fact in ``inverse_effects``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from pss.binding.protocol import Operator

Fact = tuple[str, ...]


@dataclass(frozen=True)
class Effect:
    """A conditional add/remove delta on a fact set.

    ``conditions`` must hold for the effect to fire.  When it fires ``remove``
    is subtracted first and ``add`` is unioned second, so a fact present in
    both sets ends up in the result.
    """

    conditions: frozenset[Fact] = frozenset()
    add: frozenset[Fact] = frozenset()
    remove: frozenset[Fact] = frozenset()

    def applicable(self, facts: frozenset[Fact]) -> bool:
        """Return whether every condition holds in ``facts``."""
        return self.conditions <= frozenset(facts)

    def apply(self, facts: frozenset[Fact]) -> frozenset[Fact]:
        """Apply remove-then-add when applicable; otherwise return a copy."""
        state = frozenset(facts)
        if not self.applicable(state):
            return state
        return frozenset((state - self.remove) | self.add)

    def inverted(self) -> Effect:
        """Return the effect that swaps ``add`` and ``remove``.

        Conditions are preserved: an effect that only fires under a condition
        inverts into one that only fires under the same condition.
        """
        return Effect(conditions=self.conditions, add=self.remove, remove=self.add)


@dataclass(frozen=True)
class ConditionalOperator(Operator):
    """An action whose effects may be conditional and sequential.

    ``consumes`` and ``produces`` use ``(resource_name, amount)`` pairs and
    are interpreted by :class:`ResourcePool`.  ``inverse_effects`` is written
    in rollback direction; see the module docstring.
    """

    # ``field()`` keeps ``name`` required; the ABC supplies ``name =
    # "operator"`` as a class attribute that would otherwise become a default.
    name: str = field()
    preconditions: frozenset[Fact] = frozenset()
    effects: tuple[Effect, ...] = ()
    cost: float = 1.0
    duration: float = 0.0
    consumes: tuple[tuple[str, int], ...] = ()
    produces: tuple[tuple[str, int], ...] = ()
    inverse_effects: tuple[Effect, ...] | None = None
    kind: str = "action"

    def applicable(
        self,
        facts: frozenset[Fact],
        confidence: Mapping[Fact, float] | None = None,
        threshold: float = 0.5,
    ) -> bool:
        """Check hard preconditions and, optionally, soft confidences.

        Every precondition must be present.  When ``confidence`` is supplied,
        each precondition's score must be at least ``threshold``; facts absent
        from the mapping count as ``1.0``.
        """
        state = frozenset(facts)
        if not self.preconditions <= state:
            return False
        if confidence is not None:
            for precondition in self.preconditions:
                if confidence.get(precondition, 1.0) < threshold:
                    return False
        return True

    def apply(self, facts: frozenset[Fact]) -> frozenset[Fact]:
        """Apply effects in order, re-checking conditions against the state.

        Effects are sequential: a later effect can depend on a fact added by
        an earlier one.  The input state is never mutated.
        """
        state = frozenset(facts)
        for effect in self.effects:
            if effect.applicable(state):
                state = effect.apply(state)
        return state

    def resource_ok(self, pool: ResourcePool) -> bool:
        """Return whether ``pool`` can satisfy every ``consumes`` entry."""
        needed: dict[str, int] = {}
        for name, amount in self.consumes:
            needed[name] = needed.get(name, 0) + amount
        return all(pool.can_consume(name, amount) for name, amount in needed.items())

    def inverse(self) -> ConditionalOperator | None:
        """Return the rollback operator, or ``None`` when not invertible.

        The returned operator is named ``undo_<name>`` and has empty
        preconditions, so callers must validate the live state before applying
        it.  The effects come from ``inverse_effects`` as written; they are not
        inverted a second time.
        """
        if self.inverse_effects is None:
            return None
        return ConditionalOperator(
            name=f"undo_{self.name}",
            preconditions=frozenset(),
            effects=tuple(self.inverse_effects),
            kind=self.kind,
        )

    def compose(self, other: Operator) -> None:
        """Actions do not compose into a single conditional operator."""
        return None

    def residual(self, x: frozenset[Fact], y: frozenset[Fact]) -> float:
        """Jaccard distance between two fact sets, in ``[0.0, 1.0]``.

        Two empty sets are treated as identical, so the distance is ``0.0``.
        """
        left = frozenset(x)
        right = frozenset(y)
        union = left | right
        if not union:
            return 0.0
        return 1.0 - len(left & right) / len(union)

    def consumes_dict(self) -> dict[str, int]:
        """Return ``consumes`` as a plain dict."""
        return dict(self.consumes)

    def produces_dict(self) -> dict[str, int]:
        """Return ``produces`` as a plain dict."""
        return dict(self.produces)


@dataclass(frozen=True)
class Resource:
    """A named resource with a fixed capacity."""

    name: str
    capacity: int

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError(f"resource {self.name!r} capacity must be >= 0")


class ResourcePool:
    """Mutable used/available accounting over fixed resource capacities.

    ``resources`` may be an iterable of :class:`Resource` objects or a mapping
    from resource name to capacity.  Recording snapshots capture used counts;
    capacities are fixed at construction.
    """

    def __init__(
        self,
        resources: Iterable[Resource] | Mapping[str, int] = (),
    ) -> None:
        self._capacity: dict[str, int] = {}
        self._used: dict[str, int] = {}
        if isinstance(resources, Mapping):
            for name, capacity in resources.items():
                self._add(Resource(name, capacity))
        else:
            for resource in resources:
                self._add(resource)

    def _add(self, resource: Resource) -> None:
        if resource.name in self._capacity:
            raise ValueError(f"duplicate resource {resource.name!r}")
        self._capacity[resource.name] = resource.capacity
        self._used[resource.name] = 0

    def _require(self, name: str) -> None:
        if name not in self._capacity:
            raise KeyError(name)

    @staticmethod
    def _check_amount(n: int) -> None:
        if n < 0:
            raise ValueError("amount must be >= 0")

    def available(self, name: str) -> int:
        """Return capacity minus used for ``name``."""
        self._require(name)
        return self._capacity[name] - self._used[name]

    def used(self, name: str) -> int:
        """Return the currently used amount for ``name``."""
        self._require(name)
        return self._used[name]

    def can_consume(self, name: str, n: int = 1) -> bool:
        """Return whether ``n`` units of ``name`` are available."""
        self._require(name)
        self._check_amount(n)
        return self.available(name) >= n

    def consume(self, name: str, n: int = 1) -> bool:
        """Consume ``n`` units; return ``False`` and change nothing otherwise."""
        self._require(name)
        self._check_amount(n)
        if self.available(name) < n:
            return False
        self._used[name] += n
        return True

    def release(self, name: str, n: int = 1) -> None:
        """Return ``n`` units to ``name``; used never drops below zero."""
        self._require(name)
        self._check_amount(n)
        self._used[name] = max(0, self._used[name] - n)

    def snapshot(self) -> dict[str, int]:
        """Return the mutable state as ``{resource_name: used}``."""
        return dict(self._used)

    def restore(self, snapshot: Mapping[str, int]) -> None:
        """Restore used counts from :meth:`snapshot`.

        Unknown resources, negative counts, and counts above capacity raise
        ``ValueError`` / ``KeyError``.  Validation happens before any change,
        so a rejected snapshot leaves the pool untouched.
        """
        pending: dict[str, int] = {}
        for name, used in snapshot.items():
            self._require(name)
            if used < 0 or used > self._capacity[name]:
                raise ValueError(f"snapshot value for {name!r} is out of range")
            pending[name] = used
        self._used.update(pending)


class ConditionalLibrary:
    """A named registry of :class:`ConditionalOperator` objects."""

    def __init__(
        self,
        operators: Mapping[str, ConditionalOperator] | Iterable[ConditionalOperator] = (),
    ) -> None:
        if isinstance(operators, Mapping):
            self.operators: dict[str, ConditionalOperator] = dict(operators)
        else:
            self.operators = {}
            for operator in operators:
                if operator.name in self.operators:
                    raise ValueError(f"duplicate operator {operator.name!r}")
                self.operators[operator.name] = operator

    def get(self, name: str) -> ConditionalOperator:
        """Return the operator called ``name``."""
        return self.operators[name]

    def names(self) -> list[str]:
        """Return operator names in library order."""
        return list(self.operators)

    def applicable(
        self,
        facts: frozenset[Fact],
        pool: ResourcePool | None = None,
        confidence: Mapping[Fact, float] | None = None,
        threshold: float = 0.5,
    ) -> list[tuple[ConditionalOperator, float]]:
        """Return applicable operators with scores, best first.

        The score is the mean precondition confidence (``1.0`` without a
        confidence mapping).  Sorting is by descending score and then by name.
        When ``pool`` is supplied, operators lacking resources are excluded.
        """
        state = frozenset(facts)
        ranked: list[tuple[ConditionalOperator, float]] = []
        for operator in self.operators.values():
            if not operator.applicable(state, confidence=confidence, threshold=threshold):
                continue
            if pool is not None and not operator.resource_ok(pool):
                continue
            score = 1.0
            if confidence is not None and operator.preconditions:
                score = sum(
                    confidence.get(precondition, 1.0)
                    for precondition in operator.preconditions
                ) / len(operator.preconditions)
            ranked.append((operator, score))
        ranked.sort(key=lambda item: (-item[1], item[0].name))
        return ranked

    def apply(
        self,
        name: str,
        facts: frozenset[Fact],
        pool: ResourcePool | None = None,
    ) -> frozenset[Fact]:
        """Apply one operator by name and update resources when given.

        Raises ``ValueError`` when the operator's preconditions or resource
        needs are not met.  Resource consumption is checked before any state
        is returned, so a failed call leaves ``pool`` unchanged.
        """
        operator = self.get(name)
        state = frozenset(facts)
        if not operator.applicable(state):
            raise ValueError(f"operator {name!r} is not applicable")
        if pool is not None and not operator.resource_ok(pool):
            raise ValueError(f"operator {name!r} lacks required resources")
        result = operator.apply(state)
        if pool is not None:
            for resource, amount in operator.consumes:
                if not pool.consume(resource, amount):
                    raise RuntimeError(
                        f"resource {resource!r} was expected to be available"
                    )
            for resource, amount in operator.produces:
                pool.release(resource, amount)
        return result


def _facts(*facts: Fact) -> frozenset[Fact]:
    return frozenset(facts)


def laundry_conditional_library() -> ConditionalLibrary:
    """Build the laundry task as conditional operators.

    The sequence mirrors the 12-step demo in
    ``pss.demos.laundry_task_demo`` and exercises all three extensions:
    conditional adverse effects, sequential effects, and hand resources.
    ``open_washer`` and ``pick_clothes`` are invertible; the rest are not.
    """
    pick_effect = Effect(
        add=_facts(("clothes", "held")),
        remove=_facts(("clothes", "in", "basket")),
    )
    open_effect = Effect(
        add=_facts(("door", "open")),
        remove=_facts(("door", "closed")),
    )
    return ConditionalLibrary(
        [
            ConditionalOperator(
                name="pick_clothes",
                preconditions=_facts(("clothes", "in", "basket")),
                effects=(pick_effect,),
                consumes=(("hand", 1),),
                inverse_effects=(pick_effect.inverted(),),
            ),
            ConditionalOperator(
                name="open_washer",
                preconditions=_facts(("door", "closed"), ("machine", "stopped")),
                effects=(
                    open_effect,
                    Effect(
                        conditions=_facts(("water", "in", "washer")),
                        add=_facts(("floor", "wet")),
                    ),
                ),
                inverse_effects=(open_effect.inverted(),),
            ),
            ConditionalOperator(
                name="pick_detergent",
                preconditions=_facts(("detergent", "on", "counter")),
                effects=(
                    Effect(
                        add=_facts(("detergent", "held")),
                        remove=_facts(("detergent", "on", "counter")),
                    ),
                ),
                consumes=(("hand", 1),),
            ),
            ConditionalOperator(
                name="put_in_clothes",
                preconditions=_facts(("clothes", "held"), ("door", "open")),
                effects=(
                    Effect(
                        add=_facts(("clothes", "in", "washer")),
                        remove=_facts(("clothes", "held")),
                    ),
                ),
                produces=(("hand", 1),),
            ),
            ConditionalOperator(
                name="pour_detergent",
                preconditions=_facts(("detergent", "held"), ("door", "open")),
                effects=(
                    Effect(
                        add=_facts(("detergent", "in", "washer")),
                        remove=_facts(("detergent", "held")),
                    ),
                ),
                produces=(("hand", 1),),
            ),
            ConditionalOperator(
                name="close_washer",
                preconditions=_facts(("door", "open")),
                effects=(
                    Effect(
                        add=_facts(("door", "closed")),
                        remove=_facts(("door", "open")),
                    ),
                ),
            ),
            ConditionalOperator(
                name="fill_water",
                preconditions=_facts(("door", "closed")),
                effects=(
                    Effect(
                        add=_facts(("water", "in", "washer")),
                        remove=_facts(("water", "drained")),
                    ),
                ),
            ),
            ConditionalOperator(
                name="press_start",
                preconditions=_facts(
                    ("door", "closed"),
                    ("water", "in", "washer"),
                    ("clothes", "in", "washer"),
                    ("detergent", "in", "washer"),
                ),
                effects=(
                    Effect(
                        add=_facts(("machine", "running")),
                        remove=_facts(("machine", "stopped")),
                    ),
                ),
            ),
            ConditionalOperator(
                name="wait_done",
                preconditions=_facts(("machine", "running")),
                effects=(
                    Effect(
                        add=_facts(
                            ("clothes", "clean"),
                            ("water", "drained"),
                            ("machine", "stopped"),
                        ),
                        remove=_facts(
                            ("water", "in", "washer"),
                            ("machine", "running"),
                        ),
                    ),
                ),
            ),
            ConditionalOperator(
                name="take_out",
                preconditions=_facts(("clothes", "in", "washer"), ("door", "open")),
                effects=(
                    Effect(
                        add=_facts(("clothes", "held")),
                        remove=_facts(("clothes", "in", "washer")),
                    ),
                ),
                consumes=(("hand", 1),),
            ),
            ConditionalOperator(
                name="hang",
                preconditions=_facts(("clothes", "held"), ("drying_rack", "empty")),
                effects=(
                    Effect(
                        add=_facts(("clothes", "on", "drying_rack")),
                        remove=_facts(("clothes", "held"), ("drying_rack", "empty")),
                    ),
                ),
                produces=(("hand", 1),),
            ),
        ]
    )


def laundry_resource_pool() -> ResourcePool:
    """Return the one-hand pool used by the laundry library.

    Capacity is ``1`` so picking a second object while already holding one is
    blocked until the held object is put down.
    """
    return ResourcePool({"hand": 1})


__all__ = [
    "ConditionalLibrary",
    "ConditionalOperator",
    "Effect",
    "Fact",
    "Resource",
    "ResourcePool",
    "laundry_conditional_library",
    "laundry_resource_pool",
]
