"""Runtime services: uncertainty, planning, and audit-facing checks.

These services read a ``CognitiveState`` and return decisions/reports. They do
not own state; the runtime commits changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from pss.knowledge import RuleEngine
from pss.reasoning.planner import AStarPlanner
from pss.world_model.actions import OperatorLibrary

from .state import CognitiveState, Goal, Plan, UncertaintyReport


class UncertaintyService:
    def __init__(
        self,
        object_threshold: float = 0.5,
        relation_threshold: float = 0.5,
    ):
        self.object_threshold = float(object_threshold)
        self.relation_threshold = float(relation_threshold)

    def assess(
        self,
        state: CognitiveState,
        fact_conflicts: dict | None = None,
        missing_modalities: dict | None = None,
        modality_reliability: dict | None = None,
    ) -> UncertaintyReport:
        object_confidence = {e.id: float(e.confidence) for e in state.entities.values()}
        flags: list[str] = []
        if any(c < self.object_threshold for c in object_confidence.values()):
            flags.append("low_object_confidence")

        conflicts = []
        for key, values in (fact_conflicts or {}).items():
            conflicts.append({"key": list(key), "values": sorted(map(str, values))})
        if conflicts:
            flags.append("fact_conflict")

        relation_conflicts = []
        grouped: dict[tuple, dict[str, float]] = {}
        for r in state.relations:
            if r.relation_type != "spatial":
                continue
            key = (r.source, r.target)
            grouped.setdefault(key, {})[r.predicate] = max(
                grouped.setdefault(key, {}).get(r.predicate, 0.0), float(r.probability)
            )
        for key, predicates in grouped.items():
            if (
                predicates.get("left", 0.0) >= self.relation_threshold
                and predicates.get("right", 0.0) >= self.relation_threshold
            ) or (
                predicates.get("above", 0.0) >= self.relation_threshold
                and predicates.get("below", 0.0) >= self.relation_threshold
            ):
                relation_conflicts.append(
                    {"source": key[0], "target": key[1], "predicates": predicates}
                )
        if relation_conflicts:
            flags.append("relation_conflict")

        missing = {k: list(v) for k, v in (missing_modalities or {}).items()}
        if missing:
            flags.append("missing_modality")

        reliability = dict(modality_reliability or {})
        if any(float(v) < 0.5 for v in reliability.values()):
            flags.append("low_modality_reliability")

        if ("fact_conflict" in flags or "relation_conflict" in flags) and (
            "low_object_confidence" in flags
        ):
            recommendation = "abstain"
        elif "fact_conflict" in flags or "relation_conflict" in flags:
            recommendation = "review"
        elif (
            "low_object_confidence" in flags
            or "missing_modality" in flags
            or "low_modality_reliability" in flags
        ):
            recommendation = "review"
        else:
            recommendation = "accept"

        report = UncertaintyReport(
            object_confidence={k: round(v, 3) for k, v in object_confidence.items()},
            modality_reliability={k: round(float(v), 3) for k, v in reliability.items()},
            fact_conflicts=conflicts,
            relation_conflicts=relation_conflicts,
            missing_modalities=missing,
            flags=flags,
            recommendation=recommendation,
        )
        state.uncertainty = report
        return report


class RuntimePlanner:
    def __init__(
        self,
        library: OperatorLibrary,
        max_expansions: int = 20000,
        heuristic=None,
        heuristic_weight: float = 1.0,
    ):
        self.library = library
        self.heuristic = heuristic
        self.heuristic_name = "hand" if heuristic is None else "learned"
        self.planner = AStarPlanner(
            library,
            max_expansions=max_expansions,
            heuristic=heuristic,
            heuristic_weight=heuristic_weight,
        )

    def initial_facts(self, state: CognitiveState) -> frozenset:
        return frozenset(state.fact_set())

    def plan(
        self,
        state: CognitiveState,
        goal: Goal,
        confidence: dict | None = None,
        threshold: float = 0.5,
    ) -> Plan:
        result = self.planner.plan(
            self.initial_facts(state),
            goal.facts,
            confidence=confidence,
            threshold=threshold,
        )
        plan_id = f"plan_{len(state.plans) + 1:04d}"
        plan = Plan(
            id=plan_id,
            goal_id=goal.id,
            actions=list(result.actions),
            states=list(result.states),
            confidence=1.0 if result.success else 0.0,
            audit_version=int(state.version),
            reason=result.reason if not result.success else "",
            operator_version=int(getattr(self.library, "version", 1)),
        )
        state.plans[plan_id] = plan
        return plan

    def plan_relational(
        self,
        state: CognitiveState,
        goal: Goal,
        regions: list[str],
        rel_fn,
        max_nodes: int = 20000,
    ) -> Plan:
        """BFS over region placements for a relational goal.

        ``rel_fn(source_region, target_region, predicate)`` is either the exact
        geometric predicate or a learned relation head. The plan is evaluated
        with the same predicate by the caller; the benchmark also checks it
        against the exact predicate to expose learned-relation errors.
        """
        if not goal.relation:
            return self.plan(state, goal)
        source, target, predicate = goal.relation
        current = {
            f.entity: f.value
            for (entity, attribute), f in state.facts.items()
            if attribute == "location" and isinstance(f.value, str)
        }
        if source not in current or target not in current:
            return self._failed_plan(state, goal, "missing location fact")
        entities = sorted(current)
        start = tuple(current[e] for e in entities)
        index = {e: i for i, e in enumerate(entities)}

        def satisfied(assign):
            return bool(
                rel_fn(assign[index[source]], assign[index[target]], predicate)
            )

        def to_facts(assign):
            return frozenset(
                (entity, "location", assign[i]) for entity, i in index.items()
            )

        def to_actions(parent, cur):
            actions = []
            while parent[cur] is not None:
                prev, action = parent[cur]
                actions.append(action)
                cur = prev
            return list(reversed(actions))

        if satisfied(start):
            plan = Plan(
                id=f"plan_{len(state.plans) + 1:04d}",
                goal_id=goal.id,
                actions=[],
                states=[to_facts(start)],
                confidence=1.0,
                audit_version=int(state.version),
                operator_version=int(getattr(self.library, "version", 1)),
            )
            state.plans[plan.id] = plan
            return plan

        from collections import deque

        frontier = deque([start])
        parent = {start: None}
        states = {start: to_facts(start)}
        while frontier and len(parent) < max_nodes:
            cur = frontier.popleft()
            occupied = set(cur)
            for entity in entities:
                i = index[entity]
                for region in regions:
                    if region in occupied:
                        continue
                    nxt = list(cur)
                    nxt[i] = region
                    nxt = tuple(nxt)
                    if nxt in parent:
                        continue
                    parent[nxt] = (cur, f"move_{entity}_{cur[i]}_{region}")
                    states[nxt] = to_facts(nxt)
                    if satisfied(nxt):
                        plan = Plan(
                            id=f"plan_{len(state.plans) + 1:04d}",
                            goal_id=goal.id,
                            actions=to_actions(parent, nxt),
                            states=[states[nxt]],
                            confidence=1.0,
                            audit_version=int(state.version),
                            operator_version=int(getattr(self.library, "version", 1)),
                        )
                        state.plans[plan.id] = plan
                        return plan
                    frontier.append(nxt)
        return self._failed_plan(state, goal, "no relational path")

    def _failed_plan(self, state: CognitiveState, goal: Goal, reason: str) -> Plan:
        plan = Plan(
            id=f"plan_{len(state.plans) + 1:04d}",
            goal_id=goal.id,
            actions=[],
            states=[],
            confidence=0.0,
            audit_version=int(state.version),
            reason=reason,
            operator_version=int(getattr(self.library, "version", 1)),
        )
        state.plans[plan.id] = plan
        return plan

    def validate(self, plan: Plan, goal: Goal, initial: frozenset) -> bool:
        state = frozenset(initial)
        try:
            for name in plan.actions:
                state = self.library.apply(self.library.get(name), state)
        except Exception:
            return False
        return set(goal.facts) <= set(state)


@dataclass
class RuntimeResult:
    action: dict | None
    language: str
    citations: list[dict]
    safety: dict
    uncertainty: dict | None
    version: int
    plan: dict | None = None
