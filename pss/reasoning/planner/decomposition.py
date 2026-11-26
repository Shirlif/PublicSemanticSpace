"""Goal decomposition for the action planner.

The A* planner searches forward over fact states.  This module adds the other
direction: given a goal set, it recursively satisfies each fact by finding an
operator that adds it, then recursively satisfying that operator's
preconditions.  The result is an ordered action list.

The decomposition is not complete.  It processes goal facts in sorted order
and commits to the first producer it finds, so a different goal order or a
different producer may be required.  :class:`CompositionPlanner` therefore
compares the decomposition against A* and returns the cheaper plan; both are
validated by simulation before they are returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pss.reasoning.composition import best_first
from pss.reasoning.planner import AStarPlanner, PlanResult

Fact = tuple


@dataclass(frozen=True)
class Subgoal:
    fact: Fact
    operator: str | None
    preconditions: frozenset = frozenset()
    depth: int = 0


@dataclass
class Decomposition:
    success: bool
    actions: list[str] = field(default_factory=list)
    subgoals: list[Subgoal] = field(default_factory=list)
    unresolved: frozenset = frozenset()
    nodes: int = 0
    reason: str = "ok"


def _adds(operator) -> set[Fact]:
    if hasattr(operator, "effects"):
        return {fact for effect in operator.effects for fact in effect.add}
    return set(getattr(operator, "add", ()))


def _apply(operator, state: frozenset) -> frozenset:
    state_set = set(state)
    if hasattr(operator, "effects"):
        for effect in operator.effects:
            if effect.conditions <= state_set:
                state_set -= effect.remove
                state_set |= effect.add
    else:
        state_set -= set(getattr(operator, "remove", ()))
        state_set |= set(getattr(operator, "add", ()))
    return frozenset(state_set)


def decompose(
    goal,
    operators,
    initial,
    max_depth: int = 12,
    max_nodes: int = 2000,
) -> Decomposition:
    """Backward-chain a goal set into an ordered action list."""
    goal = frozenset(goal)
    initial = frozenset(initial)
    operators = list(operators)
    producers: dict[Fact, list] = {}
    for operator in operators:
        for fact in _adds(operator):
            producers.setdefault(fact, []).append(operator)

    nodes = [0]
    subgoals: list[Subgoal] = []

    def solve(fact, state, depth, stack):
        if fact in state:
            return True, state, []
        if depth >= max_depth or fact in stack:
            return False, state, []
        if nodes[0] >= max_nodes:
            return False, state, []
        nodes[0] += 1

        for operator in producers.get(fact, []):
            current = state
            actions: list[str] = []
            ok = True
            for precondition in sorted(operator.preconditions):
                if precondition in current:
                    continue
                ok, current, sub_actions = solve(
                    precondition, current, depth + 1, stack | {fact}
                )
                if not ok:
                    break
                actions.extend(sub_actions)
            if not ok:
                continue
            new_state = _apply(operator, current)
            if fact not in new_state:
                continue
            actions.append(operator.name)
            subgoals.append(
                Subgoal(fact, operator.name, frozenset(operator.preconditions), depth)
            )
            return True, new_state, actions
        return False, state, []

    state = initial
    actions: list[str] = []
    unresolved: set[Fact] = set()
    for fact in sorted(goal):
        if fact in state:
            continue
        ok, state, sub_actions = solve(fact, state, 0, frozenset())
        if ok:
            actions.extend(sub_actions)
        else:
            unresolved.add(fact)

    return Decomposition(
        success=not unresolved,
        actions=actions,
        subgoals=subgoals,
        unresolved=frozenset(unresolved),
        nodes=nodes[0],
        reason="ok" if not unresolved else "unresolved",
    )


def simulate(initial, actions, library) -> tuple[list[frozenset], bool]:
    facts = frozenset(initial)
    states = [facts]
    for name in actions:
        operator = library.get(name)
        if not operator.preconditions <= facts:
            return states, False
        facts = operator.apply(facts)
        states.append(facts)
    return states, True


class CompositionPlanner:
    """Plan by decomposition and A*, then return the cheaper valid plan."""

    def __init__(
        self,
        library,
        max_depth: int = 12,
        max_nodes: int = 2000,
        max_expansions: int = 20000,
    ):
        self.library = library
        self.operators = list(library.operators.values())
        self.max_depth = int(max_depth)
        self.max_nodes = int(max_nodes)
        self.astar = AStarPlanner(library, max_expansions=max_expansions)

    def plan(self, initial, goal, confidence=None, threshold: float = 0.5) -> PlanResult:
        initial = frozenset(initial)
        goal = frozenset(goal)

        candidates: list[PlanResult] = []
        decomposition = decompose(
            goal,
            self.operators,
            initial,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
        )
        if decomposition.success:
            states, ok = simulate(initial, decomposition.actions, self.library)
            if ok and goal <= states[-1]:
                candidates.append(
                    PlanResult(
                        success=True,
                        actions=list(decomposition.actions),
                        states=states,
                        nodes=decomposition.nodes,
                        cost=float(len(decomposition.actions)),
                        reason="decomposition",
                    )
                )

        astar = self.astar.plan(initial, goal, confidence=confidence, threshold=threshold)
        if astar.success:
            candidates.append(astar)

        if not candidates:
            return PlanResult(
                success=False,
                actions=[],
                states=[],
                nodes=decomposition.nodes + astar.nodes,
                cost=float("inf"),
                reason=f"decomposition:{decomposition.reason}; astar:{astar.reason}",
            )
        return min(candidates, key=lambda result: (result.cost, result.reason != "decomposition"))

    def plan_typed(
        self,
        initial_payload,
        atoms,
        goal_test,
        max_steps: int = 12,
        max_expansions: int = 5000,
        state_key=None,
    ) -> PlanResult:
        """Run the typed composition search and return a planner result."""
        result = best_first(
            initial_payload,
            atoms,
            goal_test,
            max_expansions=max_expansions,
            max_steps=max_steps,
            check_types=True,
            state_key=state_key,
        )
        return PlanResult(
            success=result.success,
            actions=list(result.steps),
            states=[result.payload] if result.payload is not None else [],
            nodes=result.expansions,
            cost=result.cost,
            reason=result.reason,
        )
