"""Versioned knowledge base with TMS-style justification tracking.

The store is the existing ``MultiSourceFactStore`` (priority/timestamp
resolution, conflict reporting, audit, rollback).  The TMS adds the missing
piece: every asserted value carries a justification, so removing a source
retracts the dependent conclusions instead of only deleting a key.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass

from pss.knowledge.facts import MultiSourceFactStore
from pss.knowledge.tms import Justification, Node, TruthMaintenanceSystem


@dataclass
class KBChange:
    action: str
    entity: str
    attribute: str
    value: object = None
    source: str = ""
    reason: str = ""


class VersionedKnowledgeBase:
    def __init__(self, assertions=None):
        self.store = MultiSourceFactStore(assertions)
        self.tms = TruthMaintenanceSystem()
        self._snapshots: list[dict] = []
        self._changes = 0
        self._commit()

    # ── mutation ────────────────────────────────────────────────────────────

    def assert_fact(
        self,
        entity: str,
        attribute: str,
        value,
        source: str = "runtime",
        priority: int = 0,
        reason: str = "",
    ) -> Node:
        node = (entity, attribute, value)
        # A source re-asserting the same key replaces its previous assertion.
        for old_value in self._values_for(entity, attribute, source):
            self.tms.remove_source_for((entity, attribute, old_value), source)
        key = (entity, attribute)
        if key in self.store.assertions:
            self.store.assertions[key] = [
                a for a in self.store.assertions[key] if a.source != source
            ]
        self.store.assert_fact(entity, attribute, value, source, priority, reason)
        self.tms.justify(node, source=source, confidence=1.0)
        self._changes += 1
        self._commit()
        return node

    def retract(self, entity: str, attribute: str, source: str, reason: str = "") -> int:
        for value in self._values_for(entity, attribute, source):
            self.tms.remove_source_for((entity, attribute, value), source)
        self.store.retract(entity, attribute, source, reason)
        self._changes += 1
        self._commit()
        return 1

    def retract_source(self, source: str, reason: str = "") -> int:
        removed = 0
        for key, assertions in list(self.store.assertions.items()):
            if any(a.source == source for a in assertions):
                self.store.retract(key[0], key[1], source, reason)
                removed += 1
        self.tms.remove_source(source)
        self._changes += max(1, removed)
        self._commit()
        return removed

    def update(
        self,
        entity: str,
        attribute: str,
        value,
        source: str = "runtime",
        priority: int = 0,
        reason: str = "",
    ) -> Node:
        self.retract(entity, attribute, source, reason=f"update:{reason}")
        return self.assert_fact(entity, attribute, value, source, priority, reason)

    def add_justification(self, justification: Justification) -> Justification:
        return self.tms.add_justification(justification)

    # ── queries ─────────────────────────────────────────────────────────────

    def _values_for(self, entity: str, attribute: str, source: str) -> list:
        return [
            a.value
            for a in self.store.assertions.get((entity, attribute), [])
            if a.source == source
        ]

    def resolve(self, entity: str, attribute: str):
        return self.store.resolve(entity, attribute)

    def conflicts(self) -> dict:
        return self.store.conflicts()

    def conflict_set(self) -> dict:
        return self.tms.conflict_keys()

    def audit_count(self) -> int:
        return len(self.store.audit)

    def change_count(self) -> int:
        return int(self._changes)

    # ── state hashes ────────────────────────────────────────────────────────

    def business_state(self) -> dict:
        """Canonical state excluding timestamps, audit and version counters."""
        out: dict[str, list] = {}
        for (entity, attribute), assertions in self.store.assertions.items():
            if not assertions:
                continue
            values = sorted(
                (
                    {
                        "value": str(a.value),
                        "source": str(a.source),
                        "priority": int(a.priority),
                    }
                    for a in assertions
                ),
                key=lambda d: (d["value"], d["source"], d["priority"]),
            )
            out[f"{entity}.{attribute}"] = values
        return out

    def state_hash(self) -> str:
        payload = json.dumps(self.business_state(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ── snapshots / rollback ────────────────────────────────────────────────

    def _snapshot(self) -> dict:
        return {
            "assertions": copy.deepcopy(self.store.assertions),
            "store_version": int(self.store.version),
            "store_history": copy.deepcopy(self.store.history),
            "store_audit": copy.deepcopy(self.store.audit),
            "tms": self.tms.snapshot(),
        }

    def _commit(self) -> None:
        self._snapshots.append(self._snapshot())

    def snapshot(self) -> dict:
        return {
            "state_hash": self.state_hash(),
            "store_version": int(self.store.version),
            "tms_version": int(self.tms.version),
            "index": len(self._snapshots) - 1,
        }

    def rollback(self, index: int) -> None:
        if index < 0 or index >= len(self._snapshots):
            raise IndexError(f"unknown KB snapshot {index}")
        payload = self._snapshots[index]
        self.store.assertions = copy.deepcopy(payload["assertions"])
        self.store.version = int(payload["store_version"])
        self.store.history = copy.deepcopy(payload["store_history"])
        self.store.audit = copy.deepcopy(payload["store_audit"])
        self.tms.restore(payload["tms"])
        self._snapshots = self._snapshots[: index + 1]
        self._changes = 0
