"""PSS cognitive runtime: the orchestrator that owns state and services.

The runtime does not define new perception or language models. It owns the
typed state, routes observations into the entity/relation layers, runs the
uncertainty and planning services, and routes results to the output heads.

Layering (matching ``docs/architecture/PSS-认知层架构定位.md``):

    observations
      -> EntityLayer          (object slots / persistent IDs)
      -> RelationFactLayer    (typed relations / versioned facts)
      -> UncertaintyService   (conflict / abstention)
      -> RuntimePlanner       (goal / plan)
      -> OutputRouter         (action / language / safety)
      -> CognitiveState       (versioned, auditable)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pss.heads.text import LLMLanguageHead, ToolAgent
from pss.reasoning.relations import DerivedFact, RelationRecord
from pss.safety import GuardrailEvaluator

from .layers import EntityLayer, RelationFactLayer, components
from .permissions import PermissionDenied, PermissionService
from .services import RuntimePlanner, RuntimeResult, UncertaintyService
from .state import CognitiveState, Goal, Plan


class OutputRouter:
    def __init__(
        self,
        guardrails: GuardrailEvaluator | None = None,
        allow_write: bool = False,
        decoder=None,
    ):
        self.guardrails = guardrails or GuardrailEvaluator()
        self.allow_write = bool(allow_write)
        self.decoder = decoder

    def route(
        self,
        state: CognitiveState,
        question: str,
        plan: Plan | None = None,
        action: dict | None = None,
        safety_state: dict | None = None,
    ) -> RuntimeResult:
        pss_state = state.to_pss_state(
            active_ids=[e.id for e in state.entities.values() if e.active]
        )
        if self.decoder is not None:
            language = LLMLanguageHead(
                pss_state, decoder=self.decoder, allow_write=self.allow_write
            ).answer(question)
        else:
            language = ToolAgent(pss_state, allow_write=self.allow_write).answer(question)
        safety = self.guardrails.evaluate(
            action or {},
            safety_state or {},
            candidate={"similarity": 1.0, "value_domain": "task_fidelity"},
        )
        return RuntimeResult(
            action=action,
            language=language["answer"],
            citations=language.get("citations", []),
            safety={
                "allowed": bool(safety.allowed),
                "score": float(safety.score),
                "reasons": list(safety.reasons),
                "hard_veto": bool(safety.hard_veto),
            },
            uncertainty=state.uncertainty.to_dict() if state.uncertainty else None,
            version=int(state.version),
            plan=plan.to_dict() if plan else None,
        )


class PSSRuntime:
    def __init__(
        self,
        entity_layer: EntityLayer | None = None,
        relation_layer: RelationFactLayer | None = None,
        planner: RuntimePlanner | None = None,
        outputs: OutputRouter | None = None,
        uncertainty: UncertaintyService | None = None,
        state: CognitiveState | None = None,
        audit: bool = True,
        uncertainty_enabled: bool = True,
        relation_head=None,
        active_policy=None,
        write_policy=None,
        use_default_relations: bool = True,
        use_default_active: bool = True,
        use_default_write: bool = True,
        perception=None,
        associator=None,
        metric_max_cost: float = 10.0,
        permissions: str = "admin",
        axis_mode: str = "off",
        axis_boundary=None,
        axis_sigma: float = 0.0,
        axis_eps: float = 0.02,
        axis_abstain: float = 0.0,
    ):
        self.state = state or CognitiveState()
        self.perception = perception
        self.permissions = PermissionService(permissions)
        self.entities = entity_layer or EntityLayer(
            relation_head=relation_head,
            active_policy=active_policy,
            use_default_relations=use_default_relations,
            use_default_active=use_default_active,
            associator=associator,
            metric_max_cost=metric_max_cost,
        )
        self.relations = relation_layer or RelationFactLayer(
            write_policy=write_policy,
            use_default_write=use_default_write,
            axis_mode=axis_mode,
            axis_boundary=axis_boundary,
            axis_sigma=axis_sigma,
            axis_eps=axis_eps,
            axis_abstain=axis_abstain,
        )
        if relation_layer is not None and axis_mode != "off":
            relation_layer.rule_set.axis_mode = axis_mode
            relation_layer.rule_set.axis_boundary = axis_boundary
            relation_layer.rule_set.axis_sigma = float(axis_sigma)
            relation_layer.rule_set.axis_eps = float(axis_eps)
            relation_layer.rule_set.axis_abstain = float(axis_abstain)
        self.planner = planner
        self.outputs = outputs or OutputRouter()
        self.uncertainty = uncertainty or UncertaintyService()
        self.audit_enabled = bool(audit)
        self.uncertainty_enabled = bool(uncertainty_enabled)
        self._relation_snapshots = [self.relations.snapshot()]

    def _require(self, action: str, target: str) -> None:
        try:
            self.permissions.require(action)
        except PermissionDenied as exc:
            if self.audit_enabled:
                self.state.commit(
                    "permission_denied",
                    target,
                    actor=self.permissions.profile.name,
                    reason=str(exc),
                )
            raise

    def _commit(self, action: str, target: str, **kwargs) -> int:
        if not self.audit_enabled:
            return int(self.state.version)
        version = self.state.commit(action, target, **kwargs)
        while len(self._relation_snapshots) <= version:
            self._relation_snapshots.append(self.relations.snapshot())
        return version

    # ── ingest ──────────────────────────────────────────────────────────────

    def ingest(
        self,
        observations,
        frame: int,
        relation_records: list[RelationRecord] | None = None,
        learned_head=None,
        modalities=("vision", "tactile", "geometry"),
        derive_frame: str | None = None,
        derive_max_rounds: int = 2,
        derive_min_conf: float = 0.5,
    ) -> dict:
        self._require("ingest", f"frame:{frame}")
        events = self.entities.ingest(observations, frame)
        active = self.entities.active_ids(observations=observations)
        self.entities.sync(self.state, active_ids=active)
        missing = {}
        reliability: dict[str, list[float]] = {}
        for obs in observations:
            raw = (
                obs.valid_raw()
                if getattr(obs, "streams", None)
                else (getattr(obs, "raw", {}) or {})
            )
            absent = [m for m in modalities if m not in raw]
            if absent:
                missing[str(getattr(obs, "gt_id", "unknown"))] = absent
            for modality, value in (getattr(obs, "modality_reliability", {}) or {}).items():
                reliability.setdefault(modality, []).append(float(value))
        mean_reliability = {
            modality: float(np.mean(values)) for modality, values in reliability.items()
        }
        if relation_records:
            self.relations.add_symbolic(relation_records, self.state)
        if learned_head is not None:
            self.relations.add_learned(self.entities.graph, learned_head, self.state)
        else:
            self.relations.sync_graph(self.entities.graph, self.state)
        self.relations.sync(self.state)
        derived: list[DerivedFact] = []
        if derive_frame is not None:
            derived = self.relations.derive_facts(
                frame=derive_frame,
                max_rounds=derive_max_rounds,
                min_conf=derive_min_conf,
            )
            self.relations.assert_derived(derived, self.state)
        report = (
            self.uncertainty.assess(
                self.state,
                fact_conflicts=self.relations.conflicts(),
                missing_modalities=missing,
                modality_reliability=mean_reliability,
            )
            if self.uncertainty_enabled
            else None
        )
        version = self._commit(
            "ingest",
            f"frame:{frame}",
            after={
                "events": {k: len(v) for k, v in events.items()},
                "derived": len(derived),
                "uncertainty": (report.recommendation if report else "disabled"),
            },
            reason="perception frame",
        )
        return {
            "events": events,
            "version": version,
            "derived": [fact.as_tuple() for fact in derived],
            "uncertainty": report.to_dict() if report else None,
        }

    def ingest_sample(
        self,
        sample,
        frame: int,
        relation_records=None,
        learned_head=None,
    ) -> dict:
        if self.perception is None:
            raise RuntimeError("runtime has no perception service")
        observations = self.perception.observe(sample, frame)
        return self.ingest(
            observations,
            frame,
            relation_records=relation_records,
            learned_head=learned_head,
        )

    # ── facts / goals ───────────────────────────────────────────────────────

    def assert_fact(
        self,
        entity: str,
        attribute: str,
        value,
        source: str = "runtime",
        priority: int = 0,
        reason: str = "",
        gate=None,
        gate_features=None,
        use_default_gate: bool = True,
        reliability: float = 1.0,
        age_ratio: float = 0.0,
        key_frequency: float | None = None,
    ) -> int:
        self._require("write_facts", f"{entity}.{attribute}")
        before = self.relations.resolve(entity, attribute)
        decision = self.relations.assert_fact(
            entity,
            attribute,
            value,
            source=source,
            priority=priority,
            reason=reason,
            gate=gate,
            gate_features=gate_features,
            use_default_gate=use_default_gate,
            reliability=reliability,
            age_ratio=age_ratio,
            key_frequency=key_frequency,
        )
        if decision != "write":
            return self._commit(
                "fact_rejected",
                f"{entity}.{attribute}",
                after={"value": value, "source": source, "decision": decision},
                actor=source,
                reason=reason or f"write policy {decision}",
            )
        self.relations.sync(self.state)
        if self.uncertainty_enabled:
            self.uncertainty.assess(
                self.state, fact_conflicts=self.relations.conflicts()
            )
        return self._commit(
            "assert_fact",
            f"{entity}.{attribute}",
            before=before,
            after=value,
            actor=source,
            reason=reason,
        )

    def set_goal(self, goal: Goal) -> int:
        self._require("plan", f"goal:{goal.id}")
        self.state.goals[goal.id] = goal
        return self._commit(
            "set_goal", goal.id, after=goal.to_dict(), reason=goal.description
        )

    def plan(self, goal_id: str, confidence: dict | None = None, threshold: float = 0.5) -> Plan:
        self._require("plan", f"goal:{goal_id}")
        if self.planner is None:
            raise RuntimeError("runtime has no planner")
        goal = self.state.goals.get(goal_id)
        if goal is None:
            raise KeyError(f"unknown goal {goal_id!r}")
        plan = self.planner.plan(self.state, goal, confidence, threshold)
        self._commit(
            "plan",
            plan.id,
            after=plan.to_dict(),
            reason="planner",
        )
        return plan

    # ── output ──────────────────────────────────────────────────────────────

    def explain(self, question: str, action: dict | None = None, plan: Plan | None = None) -> RuntimeResult:
        self._require("read", "explain")
        return self.outputs.route(
            self.state,
            question,
            plan=plan,
            action=action,
            safety_state={"target_category": action.get("category") if action else None},
        )

    def act(
        self,
        action: dict,
        question: str = "执行这个动作是否安全？",
        plan: Plan | None = None,
    ) -> RuntimeResult:
        self._require("act", str(action.get("kind", "action")))
        result = self.outputs.route(self.state, question, plan=plan, action=action)
        self._commit(
            "act",
            str(action.get("kind", "action")),
            after=action,
            reason=f"safety={result.safety['allowed']}",
        )
        result.version = int(self.state.version)
        return result

    # ── versions ────────────────────────────────────────────────────────────

    def rollback(self, version: int) -> int:
        self._require("rollback", f"v{version}")
        if not self.audit_enabled:
            raise RuntimeError("audit disabled; rollback unavailable")
        self.state.rollback(version)
        if 0 <= version < len(self._relation_snapshots):
            self.relations.restore(self._relation_snapshots[version])
        else:
            self.relations.sync(self.state)
        while len(self._relation_snapshots) <= self.state.version:
            self._relation_snapshots.append(self.relations.snapshot())
        return int(self.state.version)

    def summary(self) -> dict:
        self._require("read", "summary")
        payload = self.state.summary()
        payload["components"] = components(self.entities, self.relations)
        if self.perception is not None:
            payload["perception"] = self.perception.describe()
        if self.planner is not None:
            payload["planning"] = {
                "heuristic": getattr(self.planner, "heuristic_name", "hand")
            }
        payload["permissions"] = self.permissions.describe()
        return payload
