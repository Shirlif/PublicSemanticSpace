"""Typed cognitive state for the PSS runtime (Phase 16).

This is the layer the architecture document calls the "cognitive runtime":
entities, relations, facts, goals, plans, uncertainty and audit, with explicit
versions and snapshots. Perception, relations and outputs are services over
this state; none of them own state directly.

The state is deliberately serializable and auditable. ``to_entities`` /
``to_relations`` / ``to_facts`` are the only views the output heads and tools
consume.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

STATUSES = ("active", "dormant", "missing", "historical")


@dataclass
class Entity:
    id: str
    category: str | None
    pos: np.ndarray
    quat: np.ndarray
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    provenance: dict = field(default_factory=dict)
    status: str = "active"
    last_seen: int = 0
    observations: int = 1
    active: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "pos": [round(float(x), 3) for x in np.asarray(self.pos)[:3]],
            "quat": [round(float(x), 3) for x in np.asarray(self.quat)[:4]],
            "attributes": dict(self.attributes),
            "confidence": round(float(self.confidence), 3),
            "provenance": dict(self.provenance),
            "status": self.status,
            "last_seen": int(self.last_seen),
            "observations": int(self.observations),
            "active": bool(self.active),
        }


@dataclass
class Relation:
    source: str
    target: str
    relation_type: str
    predicate: str
    probability: float = 1.0
    origin: str = "symbolic"  # symbolic | learned | inference
    frame: str = "world"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type,
            "predicate": self.predicate,
            "probability": round(float(self.probability), 3),
            "origin": self.origin,
            "frame": self.frame,
        }


@dataclass
class Fact:
    entity: str
    attribute: str
    value: Any
    source: str = "runtime"
    priority: int = 0
    confidence: float = 1.0
    version: int = 0

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "attribute": self.attribute,
            "value": self.value,
            "source": self.source,
            "priority": int(self.priority),
            "confidence": round(float(self.confidence), 3),
            "version": int(self.version),
        }


@dataclass
class Goal:
    id: str
    facts: frozenset[tuple] = frozenset()
    description: str = ""
    priority: int = 0
    relation: tuple | None = None  # (source, target, predicate)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "facts": [list(f) for f in sorted(self.facts)],
            "relation": list(self.relation) if self.relation else None,
            "description": self.description,
            "priority": int(self.priority),
        }


@dataclass
class Plan:
    id: str
    goal_id: str
    actions: list[str]
    states: list[frozenset]
    confidence: float = 1.0
    audit_version: int = 0
    reason: str = ""
    operator_version: int = 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "actions": list(self.actions),
            "n_states": len(self.states),
            "confidence": round(float(self.confidence), 3),
            "audit_version": int(self.audit_version),
            "operator_version": int(self.operator_version),
            "reason": self.reason,
        }


@dataclass
class UncertaintyReport:
    object_confidence: dict[str, float] = field(default_factory=dict)
    modality_reliability: dict[str, float] = field(default_factory=dict)
    fact_conflicts: list[dict] = field(default_factory=list)
    relation_conflicts: list[dict] = field(default_factory=list)
    missing_modalities: dict[str, list[str]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    recommendation: str = "accept"  # accept | review | abstain | replan

    def to_dict(self) -> dict:
        return {
            "object_confidence": dict(self.object_confidence),
            "modality_reliability": dict(self.modality_reliability),
            "fact_conflicts": list(self.fact_conflicts),
            "relation_conflicts": list(self.relation_conflicts),
            "missing_modalities": dict(self.missing_modalities),
            "flags": list(self.flags),
            "recommendation": self.recommendation,
        }


@dataclass
class AuditEntry:
    version: int
    action: str
    target: str
    before: Any = None
    after: Any = None
    actor: str = "runtime"
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "version": int(self.version),
            "action": self.action,
            "target": self.target,
            "before": self.before,
            "after": self.after,
            "actor": self.actor,
            "reason": self.reason,
        }


class CognitiveState:
    """Typed state with versions and snapshots.

    Every mutation is committed through ``PSSRuntime`` (which appends a
    snapshot and an audit entry). Direct mutation is possible but then the
    caller owns versioning.
    """

    def __init__(self, history=None):
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self.facts: dict[tuple[str, str], Fact] = {}
        self.goals: dict[str, Goal] = {}
        self.plans: dict[str, Plan] = {}
        self.uncertainty: UncertaintyReport | None = None
        self.audit: list[AuditEntry] = []
        self.version: int = 0
        self.snapshots: list[dict] = [self._snapshot()]
        if history is None:
            from pss.reasoning.queries import TemporalHistory

            history = TemporalHistory()
        self.history = history

    # ── snapshots / versions ────────────────────────────────────────────────

    def _snapshot(self) -> dict:
        return {
            "entities": copy.deepcopy(self.entities),
            "relations": copy.deepcopy(self.relations),
            "facts": copy.deepcopy(self.facts),
            "goals": copy.deepcopy(self.goals),
            "plans": copy.deepcopy(self.plans),
            "uncertainty": copy.deepcopy(self.uncertainty),
            "audit": copy.deepcopy(self.audit),
            "version": self.version,
        }

    def commit(self, action: str, target: str, before=None, after=None, actor="runtime", reason="") -> int:
        previous = self.snapshots[-1]["facts"] if self.snapshots else {}
        before_facts = {
            (fact.entity, fact.attribute, fact.value) for fact in previous.values()
        }
        self.version += 1
        entry = AuditEntry(self.version, action, target, before, after, actor, reason)
        self.audit.append(entry)
        after_facts = self.fact_set()
        self.snapshots.append(self._snapshot())
        self.history.record(
            float(self.version),
            after_facts,
            removed=before_facts - after_facts,
        )
        return self.version

    def fact_at(self, version: int) -> set[tuple]:
        """Facts as they stood at ``version``, from the temporal history."""
        return self.history.at(float(version)).facts()

    def rollback(self, version: int) -> None:
        if version < 0 or version >= len(self.snapshots):
            raise IndexError(f"unknown version {version}")
        payload = copy.deepcopy(self.snapshots[version])
        self.entities = payload["entities"]
        self.relations = payload["relations"]
        self.facts = payload["facts"]
        self.goals = payload["goals"]
        self.plans = payload["plans"]
        self.uncertainty = payload["uncertainty"]
        self.audit = payload["audit"]
        self.version = payload["version"]
        self.audit.append(
            AuditEntry(self.version, "rollback", f"v{version}", None, None, "runtime", f"to v{version}")
        )
        self.snapshots.append(self._snapshot())

    # ── queries ─────────────────────────────────────────────────────────────

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity

    def add_relation(self, relation: Relation) -> None:
        for i, existing in enumerate(self.relations):
            if (
                existing.source == relation.source
                and existing.target == relation.target
                and existing.relation_type == relation.relation_type
                and existing.predicate == relation.predicate
            ):
                self.relations[i] = relation
                return
        self.relations.append(relation)

    def set_fact(self, fact: Fact) -> None:
        self.facts[(fact.entity, fact.attribute)] = fact

    def get_fact(self, entity: str, attribute: str) -> Fact | None:
        return self.facts.get((entity, attribute))

    def query_entities(self, category=None, status=None) -> list[Entity]:
        return [
            e
            for e in self.entities.values()
            if (category is None or e.category == category)
            and (status is None or e.status == status)
        ]

    def query_relations(self, source=None, target=None, predicate=None) -> list[Relation]:
        return [
            r
            for r in self.relations
            if (source is None or r.source == source)
            and (target is None or r.target == target)
            and (predicate is None or r.predicate == predicate)
        ]

    def fact_set(self) -> set[tuple]:
        return {(f.entity, f.attribute, f.value) for f in self.facts.values()}

    # ── views for output heads / tools ─────────────────────────────────────

    def to_pss_state(self, active_ids: list[str] | None = None):
        from pss.heads.text import PSSState

        entities = [e.to_dict() for e in self.entities.values()]
        relations = [r.to_dict() for r in self.relations]
        facts = [f.to_dict() for f in self.facts.values()]
        audit = [e.to_dict() for e in self.audit[-20:]]
        return PSSState(
            objects=entities,
            relations=relations,
            facts=facts,
            audit=audit,
            memory={"active_ids": list(active_ids or []), "n_objects": len(entities)},
            uncertainty=self.uncertainty.to_dict() if self.uncertainty else {},
        )

    def summary(self) -> dict:
        return {
            "version": int(self.version),
            "entities": len(self.entities),
            "relations": len(self.relations),
            "facts": len(self.facts),
            "goals": len(self.goals),
            "plans": len(self.plans),
            "audit": len(self.audit),
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
        }
