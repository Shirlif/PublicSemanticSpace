"""Entity and relation/fact layers over ``CognitiveState``.

These are the two lower layers of the PSS cognitive runtime:

- ``EntityLayer`` owns persistent entities (object slots, IDs, attributes,
  confidence, provenance). It wraps the existing tracker/scene graph so the
  runtime does not re-implement identity. It also runs the learned typed
  relation head on active tracks and can select object slots with the learned
  active-set policy.
- ``RelationFactLayer`` owns typed relations and versioned facts, including
  multi-source conflicts, rule derivation, and the learned write gate.

Defaults are learned when the corresponding checkpoint exists; otherwise the
hand-written fallback (confidence top-k, write-all, symbolic-only relations)
is used. Learned and fallback behavior is recorded in ``components()`` so the
runtime can report which path is active.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from pss.knowledge import MultiSourceFactStore, RuleEngine
from pss.memory import SceneGraph, Tracker, TrackerConfig
from pss.memory.active_policy import ActiveSetPolicy, WritePolicy, tracking_pair_features, write_features
from pss.memory.learned_metric import entity_token
from pss.reasoning.relations import RelationRecord
from pss.reasoning.relations.rules import RelationRuleSet

from .state import CognitiveState, Entity, Fact, Relation

RELATION_CKPT = Path("checkpoints/typed_relations/relation_head.pt")
ACTIVE_CKPT = Path("checkpoints/active_policy/active_policy.pt")
WRITE_CKPT = Path("checkpoints/active_policy/write_policy.pt")
RULES_CKPT = Path("checkpoints/relation_rules/rules.json")


def _load_relation_head():
    if not RELATION_CKPT.exists():
        return None
    try:
        from pss.memory.typed_relations import load_relation_head

        return load_relation_head(RELATION_CKPT)
    except Exception:
        return None


def _load_active_policy():
    if not ACTIVE_CKPT.exists():
        return None
    try:
        import torch

        policy = ActiveSetPolicy()
        payload = torch.load(ACTIVE_CKPT, map_location="cpu", weights_only=False)
        policy.load_state_dict(payload["state_dict"])
        policy.eval()
        return policy
    except Exception:
        return None


def _load_relation_rules():
    if not RULES_CKPT.exists():
        return RelationRuleSet()
    try:
        import json

        payload = json.loads(RULES_CKPT.read_text(encoding="utf-8"))
        return RelationRuleSet(
            near_dist=float(payload["near_dist"]),
            contact_pad=float(payload["contact_pad"]),
            bin_half=float(payload["bin_half"]),
            bin_z=float(payload["bin_z"]),
            version=int(payload.get("version", 1)),
            source=str(payload.get("source", "learned")),
        )
    except Exception:
        return RelationRuleSet()


def _load_write_policy():
    if not WRITE_CKPT.exists():
        return None
    try:
        import torch

        policy = WritePolicy()
        payload = torch.load(WRITE_CKPT, map_location="cpu", weights_only=False)
        policy.load_state_dict(payload["state_dict"])
        policy.eval()
        return policy
    except Exception:
        return None


class EntityLayer:
    def __init__(
        self,
        tracker: Tracker | None = None,
        graph: SceneGraph | None = None,
        relation_head=None,
        active_policy=None,
        relation_threshold: float = 0.5,
        use_default_relations: bool = True,
        use_default_active: bool = True,
        associator=None,
        metric_max_cost: float = 10.0,
    ):
        self.graph = graph or SceneGraph()
        self.associator = associator
        self.associator_name = type(associator).__name__ if associator is not None else "hand"
        if tracker is not None:
            self.tracker = tracker
        else:
            config = TrackerConfig(metric=associator, metric_max_cost=metric_max_cost)
            self.tracker = Tracker(config, self.graph)
        self.relation_head = (
            relation_head
            if relation_head is not None
            else (_load_relation_head() if use_default_relations else None)
        )
        self.active_policy = (
            active_policy
            if active_policy is not None
            else (_load_active_policy() if use_default_active else None)
        )
        self.relation_threshold = float(relation_threshold)
        self.last_learned_edges = 0

    def ingest(self, observations, frame: int) -> dict:
        events = self.tracker.step(observations, frame)
        if self.relation_head is not None and len(self.graph.active()) >= 2:
            from pss.memory.typed_relations import write_relation_edges

            edges = write_relation_edges(
                self.graph,
                self.relation_head,
                tracks=self.graph.active(),
                threshold=self.relation_threshold,
            )
            self.last_learned_edges = len(edges)
        else:
            self.last_learned_edges = 0
        return events

    def _tokens(self, tracks, observations=None):
        if not tracks:
            return None, None
        track_tokens = np.stack(
            [
                entity_token(t.pos, t.raw, t.category, include_tactile=True)
                for t in tracks
            ]
        )
        if not observations:
            return track_tokens, None
        obs_tokens = np.stack(
            [
                entity_token(
                    o.pos, o.raw, o.category or o.gt_category, include_tactile=True
                )
                for o in observations
            ]
        )
        return track_tokens, obs_tokens

    def active_ids(self, capacity: int = 4, observations=None) -> list[str]:
        active = [t for t in self.graph.tracks.values() if t.status == "active"]
        if not active:
            return []
        if self.active_policy is not None and observations:
            try:
                track_tokens, obs_tokens = self._tokens(active, observations)
                features = tracking_pair_features(track_tokens, obs_tokens)
                scores = self.active_policy.scores(features, device="cpu")  # (M, N)
                track_scores = scores.max(axis=0)
                order = np.argsort(-track_scores)
                return [active[int(i)].id for i in order[: max(0, capacity)]]
            except Exception:
                pass
        active.sort(key=lambda t: (-float(t.confidence), -int(t.last_seen), t.id))
        return [t.id for t in active[: max(0, capacity)]]

    def slots(self, capacity: int = 4, observations=None) -> list[dict]:
        by_id = {t.id: t for t in self.graph.tracks.values()}
        return [
            {
                "slot": i,
                "id": eid,
                "category": by_id[eid].category,
                "pos": [round(float(x), 3) for x in by_id[eid].pos[:3]],
                "confidence": round(float(by_id[eid].confidence), 3),
            }
            for i, eid in enumerate(self.active_ids(capacity, observations))
            if eid in by_id
        ]

    def sync(self, state: CognitiveState, active_ids: list[str] | None = None) -> CognitiveState:
        active = set(active_ids if active_ids is not None else self.active_ids())
        for track in self.graph.tracks.values():
            state.add_entity(
                Entity(
                    id=track.id,
                    category=track.category,
                    pos=np.asarray(track.pos, dtype=float).copy(),
                    quat=np.asarray(track.quat, dtype=float).copy(),
                    attributes={
                        "velocity": [round(float(x), 3) for x in np.asarray(track.velocity)[:3]],
                        "raw_modalities": sorted(track.raw.keys()),
                    },
                    confidence=float(track.confidence),
                    provenance={
                        "first_seen": int(track.first_seen),
                        "last_seen": int(track.last_seen),
                        "missed": int(track.missed),
                        "parent": track.parent,
                        **dict(track.provenance),
                    },
                    status=track.status,
                    last_seen=int(track.last_seen),
                    observations=int(track.observations),
                    active=track.id in active,
                )
            )
        return state


class RelationFactLayer:
    def __init__(
        self,
        facts: MultiSourceFactStore | None = None,
        rules: RuleEngine | None = None,
        write_policy=None,
        use_default_write: bool = True,
        rule_set: RelationRuleSet | None = None,
    ):
        self.facts = facts or MultiSourceFactStore()
        self.rules = rules
        self.rule_set = rule_set if rule_set is not None else _load_relation_rules()
        self.write_policy = (
            write_policy
            if write_policy is not None
            else (_load_write_policy() if use_default_write else None)
        )
        self.write_log: list[dict] = []

    # ── facts ───────────────────────────────────────────────────────────────

    def _write_decision(
        self,
        entity: str,
        attribute: str,
        value,
        source: str,
        priority: int,
        reliability: float,
        age_ratio: float,
        key_frequency: float | None,
        gate,
        gate_features,
        use_default_gate: bool,
    ) -> str:
        if gate is not None and gate_features is not None:
            return str(gate(gate_features))
        if self.write_policy is None or not use_default_gate:
            return "write"
        current = self.resolve(entity, attribute)
        key = (entity, attribute)
        assertions = self.facts.assertions.get(key, [])
        conflict = 0.0 if current is None else float(current != value)
        novelty = 1.0 if current is None else 0.0
        # Unknown query frequency is treated as rare (0.0); callers that know
        # the application frequency should pass it explicitly.
        frequency = float(key_frequency) if key_frequency is not None else 0.0
        features = write_features(
            reliability=float(reliability),
            conflict=conflict,
            novelty=novelty,
            key_frequency=frequency,
            age_ratio=float(age_ratio),
            is_new=novelty,
            value_ratio=0.0,
        )
        return self.write_policy.decide(features, device="cpu")

    def assert_fact(
        self,
        entity: str,
        attribute: str,
        value,
        source: str = "runtime",
        priority: int = 0,
        reason: str = "",
        reliability: float = 1.0,
        age_ratio: float = 0.0,
        key_frequency: float | None = None,
        gate=None,
        gate_features=None,
        use_default_gate: bool = True,
    ) -> str:
        decision = self._write_decision(
            entity, attribute, value, source, priority,
            reliability, age_ratio, key_frequency,
            gate, gate_features, use_default_gate,
        )
        if decision != "write":
            self.write_log.append(
                {
                    "entity": entity,
                    "attribute": attribute,
                    "value": value,
                    "source": source,
                    "decision": decision,
                    "reliability": float(reliability),
                }
            )
            return decision
        self.facts.assert_fact(entity, attribute, value, source, priority, reason)
        return "write"

    def resolve(self, entity: str, attribute: str):
        return self.facts.resolve(entity, attribute)

    def conflicts(self) -> dict[tuple, list]:
        return self.facts.conflicts()

    def derive(self, entity: str, attribute: str):
        if self.rules is None:
            return None
        return self.rules.derive(self.facts, entity, attribute)

    def snapshot(self) -> dict:
        return copy.deepcopy(
            {
                "assertions": self.facts.assertions,
                "version": self.facts.version,
                "history": self.facts.history,
                "audit": self.facts.audit,
                "write_log": self.write_log,
            }
        )

    def restore(self, payload: dict) -> None:
        self.facts.assertions = copy.deepcopy(payload["assertions"])
        self.facts.version = int(payload["version"])
        self.facts.history = copy.deepcopy(payload["history"])
        self.facts.audit = copy.deepcopy(payload["audit"])
        self.write_log = copy.deepcopy(payload.get("write_log", []))

    def sync(self, state: CognitiveState) -> CognitiveState:
        for (entity, attribute), assertions in self.facts.assertions.items():
            if not assertions:
                continue
            best = max(assertions, key=lambda a: (a.priority, a.timestamp))
            distinct = {a.value for a in assertions}
            state.set_fact(
                Fact(
                    entity=entity,
                    attribute=attribute,
                    value=best.value,
                    source=best.source,
                    priority=int(best.priority),
                    confidence=1.0 if len(distinct) == 1 else 0.5,
                    version=int(self.facts.version),
                )
            )
        return state

    # ── relations ───────────────────────────────────────────────────────────

    def add_geometry(self, instances: dict[str, dict], state: CognitiveState) -> int:
        records = self.rule_set.relations(instances)
        return self.add_symbolic(records, state)

    def add_symbolic(self, records: list[RelationRecord], state: CognitiveState) -> int:
        added = 0
        for record in records:
            state.add_relation(
                Relation(
                    source=record.subject,
                    target=record.object or "world",
                    relation_type="symbolic",
                    predicate=record.predicate,
                    probability=float(record.confidence),
                    origin="symbolic",
                )
            )
            added += 1
        return added

    def add_learned(self, graph: SceneGraph, head, state: CognitiveState, threshold: float = 0.5) -> int:
        from pss.memory.typed_relations import write_relation_edges

        edges = write_relation_edges(graph, head, threshold=threshold)
        for edge in edges:
            self._add_relation(edge, state)
        return len(edges)

    def sync_graph(self, graph: SceneGraph, state: CognitiveState) -> int:
        """Copy learned relation edges already written into the scene graph."""
        added = 0
        for record in graph.get_relations():
            state.add_relation(
                Relation(
                    source=record["source"],
                    target=record["target"],
                    relation_type=str(record.get("type") or "learned"),
                    predicate=record["predicate"],
                    probability=float(record.get("probability", 0.0)),
                    origin="learned",
                )
            )
            added += 1
        return added

    @staticmethod
    def _add_relation(edge, state: CognitiveState) -> None:
        state.add_relation(
            Relation(
                source=edge.source,
                target=edge.target,
                relation_type=edge.relation_type,
                predicate=edge.predicate,
                probability=float(edge.probability),
                origin="learned",
            )
        )


def components(entity_layer: EntityLayer, relation_layer: RelationFactLayer) -> dict:
    """Audit which learned defaults are active."""
    return {
        "relation_head": entity_layer.relation_head is not None,
        "active_policy": entity_layer.active_policy is not None,
        "write_policy": relation_layer.write_policy is not None,
        "associator": getattr(entity_layer, "associator_name", "hand"),
        "relation_rules": {
            "version": int(relation_layer.rule_set.version),
            "source": relation_layer.rule_set.source,
            "thresholds": relation_layer.rule_set.snapshot(),
        },
        "fallback": {
            "relation_head": "symbolic records only" if entity_layer.relation_head is None else None,
            "active_policy": "confidence top-k" if entity_layer.active_policy is None else None,
            "write_policy": "write-all" if relation_layer.write_policy is None else None,
        },
    }
