"""Three-layer guardrails: hard blacklist, value vector, scenario prototypes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_VALUE_VECTOR = {
    "human_safety": 1.0,
    "human_comfort": 0.85,
    "task_fidelity": 0.6,
    "self_preservation": 0.4,
    "resource_efficiency": 0.3,
    "exploration_restraint": 0.2,
}


@dataclass
class SafetyVerdict:
    allowed: bool
    rule: str | None = None
    reason: str = "ok"
    layer: str = "hard"


class HardSafetyBlacklist:
    """Layer 1: independent, non-bypassable vetoes."""

    def __init__(self, rules=None):
        self.rules = rules if rules is not None else self.default_rules()

    @staticmethod
    def default_rules():
        return [
            (
                "human_in_zone",
                lambda a, s: float(s.get("human_distance", 1.0)) < 0.4
                and bool(a.get("target_is_human", False)),
            ),
            (
                "sharp_tool_fast",
                lambda a, s: s.get("target_category") == "tool"
                and a.get("kind") == "push"
                and float(a.get("params", {}).get("force", 0.0)) > 5.0,
            ),
            ("overheat", lambda a, s: float(s.get("temperature", 0.0)) > 80.0),
            (
                "privacy_zone",
                lambda a, s: bool(s.get("in_private_zone", False))
                and a.get("kind") == "record",
            ),
        ]

    def check(self, action: dict, state: dict) -> SafetyVerdict:
        for name, predicate in self.rules:
            try:
                hit = predicate(action, state)
            except Exception:
                hit = False
            if hit:
                return SafetyVerdict(False, name, f"blocked by {name}")
        return SafetyVerdict(True)


@dataclass
class ValueVector:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_VALUE_VECTOR))

    def score(self, candidate: dict) -> float:
        similarity = float(candidate.get("similarity", 1.0))
        domain = candidate.get("value_domain", "task_fidelity")
        weight = float(self.weights.get(domain, 0.5))
        safety = float(candidate.get("safety_coeff", 1.0))
        cost = float(candidate.get("cost_penalty", 0.0))
        return similarity * weight * safety - cost


@dataclass
class ScenarioValuePrototype:
    scenario: str
    weight_modifiers: dict = field(default_factory=dict)
    safety_tags: list = field(default_factory=list)

    def apply(self, weights: dict) -> dict:
        out = dict(weights)
        for key, delta in self.weight_modifiers.items():
            out[key] = float(np.clip(out.get(key, 0.5) + delta, 0.0, 1.0))
        return out


class ScenarioValueStore:
    def __init__(self, nodes: list[ScenarioValuePrototype] | None = None):
        self.nodes = {n.scenario: n for n in (nodes or [])}

    def add(self, node: ScenarioValuePrototype) -> None:
        self.nodes[node.scenario] = node

    def apply(self, scenario: str, weights: dict) -> dict:
        node = self.nodes.get(scenario)
        return node.apply(weights) if node else dict(weights)


@dataclass
class GuardrailResult:
    allowed: bool
    hard_veto: bool
    score: float
    reasons: list
    details: dict


class GuardrailEvaluator:
    def __init__(self, blacklist=None, value_vector=None, scenario_store=None, min_score: float = 0.2):
        self.blacklist = blacklist or HardSafetyBlacklist()
        self.value_vector = value_vector or ValueVector()
        self.scenario_store = scenario_store or ScenarioValueStore()
        self.min_score = float(min_score)

    def evaluate(self, action: dict, state: dict, candidate: dict, scenario: str | None = None) -> GuardrailResult:
        verdict = self.blacklist.check(action, state)
        if not verdict.allowed:
            return GuardrailResult(
                allowed=False,
                hard_veto=True,
                score=0.0,
                reasons=[verdict.reason],
                details={"hard_rule": verdict.rule},
            )
        weights = (
            self.scenario_store.apply(scenario, self.value_vector.weights)
            if scenario
            else dict(self.value_vector.weights)
        )
        score = ValueVector(weights).score(candidate)
        safety_coeff = float(candidate.get("safety_coeff", 1.0))
        reasons = []
        if safety_coeff <= 0.0:
            reasons.append("safety_coeff_zero")
        if score < self.min_score:
            reasons.append("score_below_threshold")
        return GuardrailResult(
            allowed=not reasons,
            hard_veto=False,
            score=score,
            reasons=reasons,
            details={"value_vector": weights, "scenario": scenario},
        )


from .calibration import calibrate_min_score, synthetic_guardrail_cases  # noqa: E402

__all__ = [
    "GuardrailEvaluator",
    "HardSafetyBlacklist",
    "ScenarioValuePrototype",
    "ScenarioValueStore",
    "ValueVector",
    "calibrate_min_score",
    "synthetic_guardrail_cases",
]
