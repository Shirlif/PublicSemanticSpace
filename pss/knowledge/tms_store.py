"""TMS-backed ``MultiSourceFactStore`` (claim-gap E10).

The property suite validated TMS on a standalone ``VersionedKnowledgeBase``.
Production code uses ``MultiSourceFactStore`` (assertions from several sources,
priority resolution).  This wrapper puts the same truth-maintenance layer behind
that store:

- every assertion creates a justification whose ``source`` is the asserting
  source;
- a derived fact is a justification over in-nodes (rules);
- retracting a source removes its justifications and propagates;
- resolve() only returns a value the TMS labels IN;
- snapshots/rollback restore both the assertion table and the TMS graph.

It is intentionally a wrapper, not a rewrite, so existing call sites can adopt
it without changing the fact schema.
"""

from __future__ import annotations

import copy

from pss.knowledge.facts import Assertion, MultiSourceFactStore
from pss.knowledge.tms import Node, TruthMaintenanceSystem


class TMSFactStore:
    """A ``MultiSourceFactStore`` whose derived labels are TMS-backed."""

    def __init__(
        self,
        assertions: dict[tuple[str, str], list[Assertion]] | None = None,
    ) -> None:
        self.store = MultiSourceFactStore(assertions)
        self.tms = TruthMaintenanceSystem()
        for (entity, attribute), items in self.store.assertions.items():
            for item in items:
                self.tms.justify(
                    (entity, attribute, item.value),
                    source=item.source,
                    rule="assert",
                )
        self.version = 0
        self._snapshots: list[dict] = []
        self._record()

    # ── snapshots ───────────────────────────────────────────────────────

    def _record(self) -> None:
        self._snapshots.append(
            {
                "assertions": copy.deepcopy(self.store.assertions),
                "store_version": int(self.store.version),
                "tms": self.tms.snapshot(),
            }
        )

    def state_hash(self) -> str:
        import hashlib
        import json

        payload = {
            "assertions": {
                f"{k[0]}.{k[1]}": [(a.value, a.source, a.priority) for a in v]
                for k, v in sorted(self.store.assertions.items())
            },
            "labels": {
                "|".join(str(x) for x in node): bool(self.tms.label(node))
                for node in sorted(self.tms._all_nodes(), key=lambda n: tuple(map(str, n)))
            },
        }
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def rollback(self, version: int) -> None:
        if version < 0 or version >= len(self._snapshots):
            raise IndexError(f"unknown TMSFactStore version {version}")
        snap = self._snapshots[version]
        self.store.assertions = copy.deepcopy(snap["assertions"])
        self.store.version = int(snap["store_version"])
        self.tms.restore(snap["tms"])
        self.version = int(version)

    # ── mutations ───────────────────────────────────────────────────────

    def assert_fact(
        self,
        entity: str,
        attribute: str,
        value: str,
        source: str,
        priority: int = 0,
        reason: str = "",
    ) -> None:
        self.store.assert_fact(entity, attribute, value, source, priority, reason)
        self.tms.justify((entity, attribute, value), source=source, rule="assert")
        self.version += 1
        self._record()

    def retract(self, entity: str, attribute: str, source: str, reason: str = "") -> None:
        key = (entity, attribute)
        values = [
            a.value
            for a in self.store.assertions.get(key, [])
            if a.source == source
        ]
        self.store.retract(entity, attribute, source, reason)
        for value in values:
            self.tms.remove_source_for((entity, attribute, value), source)
        self.version += 1
        self._record()

    def retract_source(self, source: str, reason: str = "") -> None:
        keys = [
            key
            for key, items in list(self.store.assertions.items())
            if any(a.source == source for a in items)
        ]
        for entity, attribute in keys:
            self.store.retract(entity, attribute, source, reason)
        self.tms.remove_source(source)
        self.version += 1
        self._record()

    def justify(
        self,
        conclusion: Node,
        in_nodes=(),
        out_nodes=(),
        source: str = "rule",
        rule: str = "",
        confidence: float = 1.0,
    ):
        justification = self.tms.justify(
            conclusion,
            in_nodes=in_nodes,
            out_nodes=out_nodes,
            source=source,
            rule=rule,
            confidence=confidence,
        )
        self.version += 1
        self._record()
        return justification

    # ── queries ─────────────────────────────────────────────────────────

    def resolve(self, entity: str, attribute: str):
        value = self.store.resolve(entity, attribute)
        if value is None:
            return None
        return value if self.tms.label((entity, attribute, value)) else None

    def get(self, entity: str, attribute: str):
        return self.resolve(entity, attribute)

    def label(self, node: Node) -> bool:
        return bool(self.tms.label(tuple(node)))

    def conflicts(self) -> dict:
        return self.store.conflicts()

    def tms_conflicts(self) -> dict:
        return self.tms.conflict_keys()

    def audit_count(self) -> int:
        return len(self.store.audit)

    def change_count(self) -> int:
        return int(self.store.version)

    def justifications_for(self, node: Node):
        return self.tms.justifications_for(tuple(node))

    def all_facts(self) -> dict:
        return {
            key: [
                {"value": a.value, "source": a.source, "priority": a.priority}
                for a in items
            ]
            for key, items in self.store.assertions.items()
        }
