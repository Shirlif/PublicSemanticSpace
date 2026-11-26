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

Snapshot modes mirror ``TruthMaintenanceSystem``:

- ``full``: deep-copy the assertion table and a full TMS snapshot per change;
- ``cow``: shallow copy of the assertion table (``Assertion`` objects are
  treated as immutable) plus a COW TMS snapshot;
- ``delta``: record only the keys touched by the mutation, plus a periodic COW
  checkpoint.  This is the default because the runtime snapshots on every
  committed version.

``snapshot()`` returns a compact token into the append-only snapshot log.  A
token stays valid after later mutations, so the runtime can keep one token per
version and restore any of them.
"""

from __future__ import annotations

import copy

from pss.knowledge.facts import Assertion, MultiSourceFactStore
from pss.knowledge.tms import Node, TruthMaintenanceSystem

SNAPSHOT_MODES = ("full", "cow", "delta", "none")


class TMSFactStore:
    """A ``MultiSourceFactStore`` whose derived labels are TMS-backed."""

    def __init__(
        self,
        assertions: dict[tuple[str, str], list[Assertion]] | None = None,
        snapshot_mode: str = "delta",
        checkpoint_interval: int = 256,
    ) -> None:
        if snapshot_mode not in SNAPSHOT_MODES:
            raise ValueError(
                f"unknown snapshot_mode {snapshot_mode!r}; choose from {SNAPSHOT_MODES}"
            )
        self.store = MultiSourceFactStore(assertions, record_history=False)
        self.snapshot_mode = snapshot_mode
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        # Bulk-load justifications without per-item history; a single
        # checkpoint is taken afterwards.
        self.tms = TruthMaintenanceSystem(snapshot_mode="none")
        for (entity, attribute), items in self.store.assertions.items():
            for item in items:
                self.tms.justify(
                    (entity, attribute, item.value),
                    source=item.source,
                    rule="assert",
                )
        self.tms.reset_history(snapshot_mode)
        self.version = 0
        self._snapshot_log: list[dict] = []
        self._version_entries: dict[int, int] = {}
        # Entry index that represents the current state.  A restore moves it
        # back; the next mutation appends a branch whose ``parent`` is this
        # entry, so abandoned-branch deltas are never replayed.
        self._current_entry = -1
        self._record(force_full=True)

    # ── compatibility surface ───────────────────────────────────────────

    @property
    def assertions(self) -> dict[tuple[str, str], list[Assertion]]:
        return self.store.assertions

    @property
    def audit(self) -> list:
        return self.store.audit

    @property
    def history(self) -> list[dict]:
        return self._snapshot_log

    # ── snapshots ───────────────────────────────────────────────────────

    def _record(self, changed=(), force_full: bool = False) -> None:
        mode = self.snapshot_mode
        if mode == "none":
            return
        changed = tuple(changed)
        tms_token = self.tms.snapshot()
        parent = self._current_entry
        kind = mode
        depth = 0
        if mode == "delta":
            if force_full or parent < 0:
                kind = "cow"
            else:
                depth = int(self._snapshot_log[parent].get("depth", 0)) + 1
                if depth >= self.checkpoint_interval:
                    kind = "cow"
                    depth = 0
        if kind == "full":
            entry = {
                "kind": "full",
                "assertions": copy.deepcopy(self.store.assertions),
                "store_version": int(self.store.version),
                "tms": tms_token,
                "version": int(self.version),
                "parent": parent,
                "depth": 0,
            }
        elif kind == "cow":
            entry = {
                "kind": "cow",
                "assertions": {
                    key: tuple(items) for key, items in self.store.assertions.items()
                },
                "store_version": int(self.store.version),
                "tms": tms_token,
                "version": int(self.version),
                "parent": parent,
                "depth": 0,
            }
        else:  # delta
            entry = {
                "kind": "delta",
                "delta": {
                    key: (
                        tuple(self.store.assertions[key])
                        if key in self.store.assertions
                        else None
                    )
                    for key in changed
                },
                "store_version": int(self.store.version),
                "tms": tms_token,
                "version": int(self.version),
                "parent": parent,
                "depth": depth,
            }
        self._snapshot_log.append(entry)
        self._current_entry = len(self._snapshot_log) - 1
        self._version_entries[int(self.version)] = self._current_entry

    def _reconstruct(self, entry_index: int) -> dict:
        if entry_index < 0 or entry_index >= len(self._snapshot_log):
            raise IndexError(f"unknown TMSFactStore snapshot {entry_index}")
        # Follow parent links, not list position, so a rollback followed by
        # new mutations reconstructs the new branch only.
        chain: list[int] = []
        current = entry_index
        while (
            current >= 0 and self._snapshot_log[current]["kind"] == "delta"
        ):
            chain.append(current)
            current = int(
                self._snapshot_log[current].get("parent", current - 1)
            )
        if current < 0:
            assertions: dict[tuple[str, str], list[Assertion]] = {}
            store_version = 0
        else:
            base = self._snapshot_log[current]
            assertions = {
                key: list(items) for key, items in base["assertions"].items()
            }
            store_version = int(base["store_version"])
        tms_token = self._snapshot_log[entry_index]["tms"]
        for position in reversed(chain):
            entry = self._snapshot_log[position]
            for key, value in entry["delta"].items():
                if value is None:
                    assertions.pop(key, None)
                else:
                    assertions[key] = list(value)
            store_version = int(entry["store_version"])
            tms_token = entry["tms"]
        return {
            "assertions": assertions,
            "store_version": store_version,
            "tms": tms_token,
        }

    def _restore_entry(self, entry_index: int) -> None:
        state = self._reconstruct(entry_index)
        self.store.assertions = state["assertions"]
        self.store.version = int(state["store_version"])
        self.tms.restore(state["tms"])
        self._current_entry = entry_index

    def snapshot(self) -> dict:
        if self._current_entry < 0:
            return {
                "tms_fact_store": True,
                "materialized": {
                    "assertions": copy.deepcopy(self.store.assertions),
                    "store_version": int(self.store.version),
                    "tms": self.tms.snapshot(),
                },
            }
        return {
            "tms_fact_store": True,
            "entry": self._current_entry,
            "version": int(self.version),
        }

    def restore(self, payload: dict) -> None:
        if payload.get("tms_fact_store") and "entry" in payload:
            self._restore_entry(int(payload["entry"]))
            self.version = int(payload.get("version", self.version))
            return
        if "materialized" in payload:
            state = payload["materialized"]
            self.store.assertions = copy.deepcopy(state["assertions"])
            self.store.version = int(state["store_version"])
            self.tms.restore(state["tms"])
            self._current_entry = -1
            return
        # Legacy full payload (assertions + tms materialised state).
        if "assertions" in payload:
            self.store.assertions = copy.deepcopy(payload["assertions"])
        self.store.version = int(payload.get("store_version", self.store.version))
        if "tms" in payload:
            self.tms.restore(payload["tms"])
        self.version = int(payload.get("version", self.version))
        self._current_entry = -1

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
        if version < 0 or version not in self._version_entries:
            raise IndexError(f"unknown TMSFactStore version {version}")
        self._restore_entry(self._version_entries[version])
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
        self._record(changed=((entity, attribute),))

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
        self._record(changed=(key,))

    def update(
        self,
        entity: str,
        attribute: str,
        value: str,
        source: str = "runtime",
        priority: int = 0,
        reason: str = "",
    ) -> None:
        """Replace one source's value for a key (retract + assert)."""
        self.retract(entity, attribute, source, reason=f"update:{reason}")
        self.assert_fact(entity, attribute, value, source, priority, reason)

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
        self._record(changed=tuple(keys))

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
        self._record(changed=())
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
