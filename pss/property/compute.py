"""Budgeted inference helpers for the C-series experiments.

The controller clips entity/relation sets and hands the planner a hard expansion
budget.  ``AStarPlanner`` reports ``max_expansions + 1`` when it stops, so the
wrapper passes ``budget.max_expansions - 1`` to keep the observed node count
inside the configured bound.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from pss.reasoning.planner import AStarPlanner, PlanResult


@dataclass
class Budget:
    max_entities: int = 200
    max_relations: int = 1000
    max_expansions: int = 20000
    max_retrieval_levels: int = 4
    max_latency_ms: float | None = None


@dataclass
class BudgetReport:
    within_budget: bool
    violations: list[str] = field(default_factory=list)
    entities: int = 0
    relations: int = 0
    expansions: int = 0
    retrieval_levels: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "within_budget": bool(self.within_budget),
            "violations": list(self.violations),
            "entities": int(self.entities),
            "relations": int(self.relations),
            "expansions": int(self.expansions),
            "retrieval_levels": int(self.retrieval_levels),
            "latency_ms": float(self.latency_ms),
        }


class BudgetController:
    def __init__(self, budget: Budget):
        self.budget = budget

    def active_set(self, entities):
        return list(entities)[: max(0, int(self.budget.max_entities))]

    def relation_set(self, relations):
        return list(relations)[: max(0, int(self.budget.max_relations))]

    def retrieval_levels(self, requested: int) -> int:
        return int(min(max(0, int(requested)), int(self.budget.max_retrieval_levels)))

    def check(
        self,
        *,
        entities: int = 0,
        relations: int = 0,
        expansions: int = 0,
        retrieval_levels: int = 0,
        latency_ms: float = 0.0,
    ) -> BudgetReport:
        violations: list[str] = []
        if entities > self.budget.max_entities:
            violations.append("entities")
        if relations > self.budget.max_relations:
            violations.append("relations")
        if expansions > self.budget.max_expansions:
            violations.append("expansions")
        if retrieval_levels > self.budget.max_retrieval_levels:
            violations.append("retrieval_levels")
        if self.budget.max_latency_ms is not None and latency_ms > self.budget.max_latency_ms:
            violations.append("latency")
        return BudgetReport(
            within_budget=not violations,
            violations=violations,
            entities=int(entities),
            relations=int(relations),
            expansions=int(expansions),
            retrieval_levels=int(retrieval_levels),
            latency_ms=float(latency_ms),
        )

    def planner(self, library, heuristic=None) -> "BudgetedPlanner":
        return BudgetedPlanner(library, self.budget, heuristic=heuristic)


class BudgetedPlanner:
    def __init__(self, library, budget: Budget, heuristic=None):
        self.budget = budget
        self.controller = BudgetController(budget)
        # The planner returns max_expansions + 1 on stop; reserve one node.
        inner_budget = max(0, int(budget.max_expansions) - 1)
        self.planner = AStarPlanner(
            library,
            max_expansions=inner_budget,
            heuristic=heuristic,
        )

    def plan(self, initial, goal, confidence=None, threshold: float = 0.5):
        start = time.perf_counter()
        result: PlanResult = self.planner.plan(
            initial, goal, confidence=confidence, threshold=threshold
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        report = self.controller.check(
            expansions=int(result.nodes),
            latency_ms=latency_ms,
        )
        return result, report


def measure_costs(stages: dict) -> dict:
    """Time a mapping of ``name -> callable`` and return costs in ms."""

    costs: dict[str, float] = {}
    total = 0.0
    for name, fn in stages.items():
        start = time.perf_counter()
        fn()
        cost = (time.perf_counter() - start) * 1000.0
        costs[name] = cost
        total += cost
    return {"stages": costs, "total_ms": total, "attribution_error": 0.0}


def anytime_curve(plan_fn, budgets) -> list[dict]:
    """Run ``plan_fn(budget)`` for a sequence of expansion budgets.

    Returns a list of ``{budget, success, nodes, latency_ms}``.  This is the raw
    material for the C3 quality-vs-time curve.
    """

    rows = []
    for budget in budgets:
        start = time.perf_counter()
        result, report = plan_fn(int(budget))
        latency = (time.perf_counter() - start) * 1000.0
        rows.append(
            {
                "budget": int(budget),
                "success": bool(result.success),
                "nodes": int(result.nodes),
                "latency_ms": float(latency),
                "within_budget": bool(report.within_budget),
            }
        )
    return rows


def degradation_scenario(monitor, signals_sequence) -> list[dict]:
    """Feed signals to a ``HealthMonitor`` and record the resulting levels."""

    rows = []
    for signals in signals_sequence:
        status = monitor.check(signals)
        rows.append(
            {
                "level": status.level.value,
                "proposal_allowed": bool(status.proposal_allowed),
                "motion_allowed": bool(status.motion_allowed),
                "manual_required": bool(status.manual_required),
            }
        )
    return rows


def monotonic_quality(successes) -> float:
    """Fraction of adjacent budget steps that do not lose success."""

    arr = np.asarray([1.0 if s else 0.0 for s in successes], dtype=float)
    if arr.size < 2:
        return 1.0
    return float(np.mean(np.diff(arr) >= -1e-9))
