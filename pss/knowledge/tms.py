"""A small truth-maintenance system (TMS-lite) for PSS knowledge updates.

Doyle (1979) is the conceptual ancestor: every belief carries a justification,
nodes are IN or OUT, and adding/removing a justification propagates only to the
affected subgraph.  This implementation is deliberately small:

- a node is ``(entity, attribute, value)``;
- a justification is ``(conclusion, in_nodes, out_nodes, source, rule)``;
- ``in_nodes`` must be supported, ``out_nodes`` must be unsupported;
- assumptions are always supported;
- cycles are treated as unsupported (well-foundedness);
- ``conflict_keys`` reports keys with more than one supported value.

It is not a full ATMS: there is a single context.  That is enough for K3
(dependency propagation) and K4 (conflict detection) in the property plan.

Propagation is incremental: a reverse index (antecedent -> conclusions) is
maintained on every edge change, and a mutation marks only the changed nodes
and their transitive dependents dirty.  ``propagate()`` re-evaluates the dirty
set; ``propagate_full()`` is kept as the reference full recomputation used by
tests and the scale benchmark.

Snapshots have three modes:

- ``full``: deep-copy the whole justification graph (the original behaviour);
- ``cow``: copy-on-write, a shallow dict of immutable justification tuples;
- ``delta``: store only the nodes changed by each mutation, with a periodic
  copy-on-write checkpoint so rollback replay stays bounded.

``none`` disables history (used while bulk-loading, before the first
checkpoint).  All modes expose the same query semantics.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

Node = tuple[str, str, Any]
Key = tuple[str, str]

SNAPSHOT_MODES = ("full", "cow", "delta", "none")


@dataclass(frozen=True)
class Justification:
    conclusion: Node
    in_nodes: frozenset[Node] = frozenset()
    out_nodes: frozenset[Node] = frozenset()
    source: str = "runtime"
    rule: str = ""
    confidence: float = 1.0


class TruthMaintenanceSystem:
    def __init__(self, snapshot_mode: str = "full", checkpoint_interval: int = 256):
        if snapshot_mode not in SNAPSHOT_MODES:
            raise ValueError(
                f"unknown snapshot_mode {snapshot_mode!r}; choose from {SNAPSHOT_MODES}"
            )
        # Values are immutable tuples so a shallow dict copy is a real COW
        # snapshot: later mutations replace tuples, never mutate them in place.
        self._justifications: dict[Node, tuple[Justification, ...]] = {}
        self._assumptions: set[Node] = set()
        self._labels: dict[Node, bool] = {}
        # antecedent -> {conclusion: number of justifications using the edge}
        self._reverse: dict[Node, dict[Node, int]] = {}
        self._dirty: set[Node] = set()
        self.version = 0
        self.history: list[dict] = []
        self.snapshot_mode = snapshot_mode
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        # Delta bookkeeping: nodes changed since the previous history entry.
        self._pending: dict[Node, tuple[Justification, ...] | None] = {}
        self._last_assumptions: set[Node] = set()
        # Index of the history entry that represents the current state.  A
        # restore moves it back; the next mutation appends a new branch entry
        # whose ``parent`` is this index.
        self._current_index = -1
        self._snapshot(force_full=True)

    # ── mutation ────────────────────────────────────────────────────────────

    def assume(self, node: Node) -> None:
        self._assumptions.add(tuple(node))
        self._changed((node,))

    def unassume(self, node: Node) -> None:
        self._assumptions.discard(tuple(node))
        self._changed((node,))

    def add_justification(self, justification: Justification) -> Justification:
        if not isinstance(justification, Justification):
            raise TypeError("expected a Justification")
        node = tuple(justification.conclusion)
        self._justifications[node] = self._justifications.get(node, ()) + (
            justification,
        )
        self._link(justification)
        # New antecedent nodes have never been labelled; include them in the
        # dirty set so propagate() does not need a full-node scan.
        changed = [node]
        changed.extend(
            antecedent
            for antecedent in set(justification.in_nodes)
            | set(justification.out_nodes)
            if antecedent not in self._labels
        )
        self._changed(tuple(changed))
        return justification

    def justify(
        self,
        conclusion: Node,
        in_nodes=(),
        out_nodes=(),
        source: str = "runtime",
        rule: str = "",
        confidence: float = 1.0,
    ) -> Justification:
        return self.add_justification(
            Justification(
                conclusion=tuple(conclusion),
                in_nodes=frozenset(tuple(n) for n in in_nodes),
                out_nodes=frozenset(tuple(n) for n in out_nodes),
                source=str(source),
                rule=str(rule),
                confidence=float(confidence),
            )
        )

    def remove_source(self, source: str) -> int:
        removed = 0
        changed: set[Node] = set()
        for node, justifications in list(self._justifications.items()):
            kept = tuple(j for j in justifications if j.source != source)
            dropped = len(justifications) - len(kept)
            if not dropped:
                continue
            removed += dropped
            for j in justifications:
                if j.source == source:
                    self._unlink(j)
            if kept:
                self._justifications[node] = kept
            else:
                del self._justifications[node]
            changed.add(node)
        if changed:
            self._changed(tuple(changed))
        return removed

    def remove_source_for(self, node: Node, source: str) -> int:
        """Remove justifications for one conclusion from one source."""
        node = tuple(node)
        justifications = self._justifications.get(node, ())
        kept = tuple(j for j in justifications if j.source != source)
        removed = len(justifications) - len(kept)
        if removed:
            for j in justifications:
                if j.source == source:
                    self._unlink(j)
            if kept:
                self._justifications[node] = kept
            else:
                del self._justifications[node]
            self._changed((node,))
        return removed

    def remove_conclusion(self, node: Node) -> int:
        node = tuple(node)
        justifications = self._justifications.pop(node, ())
        was_assumption = node in self._assumptions
        self._assumptions.discard(node)
        if justifications or was_assumption:
            for j in justifications:
                self._unlink(j)
            self._changed((node,))
        return len(justifications)

    # ── reverse index / dirty propagation ───────────────────────────────────

    def _link(self, justification: Justification) -> None:
        for antecedent in set(justification.in_nodes) | set(justification.out_nodes):
            counts = self._reverse.setdefault(antecedent, {})
            conclusion = justification.conclusion
            counts[conclusion] = counts.get(conclusion, 0) + 1

    def _unlink(self, justification: Justification) -> None:
        for antecedent in set(justification.in_nodes) | set(justification.out_nodes):
            counts = self._reverse.get(antecedent)
            if not counts:
                continue
            conclusion = justification.conclusion
            remaining = counts.get(conclusion, 0) - 1
            if remaining > 0:
                counts[conclusion] = remaining
            else:
                counts.pop(conclusion, None)
                if not counts:
                    del self._reverse[antecedent]

    def _rebuild_reverse(self) -> None:
        self._reverse = {}
        for justifications in self._justifications.values():
            for justification in justifications:
                self._link(justification)

    def _dependents(self, node: Node, transitive: bool = True) -> set[Node]:
        node = tuple(node)
        direct = set(self._reverse.get(node, ()))
        if not transitive:
            return direct
        seen: set[Node] = set()
        stack = list(direct)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(set(self._reverse.get(current, ())) - seen)
        return seen

    def _mark_dirty(self, nodes) -> None:
        for node in nodes:
            node = tuple(node)
            self._dirty.add(node)
            self._dirty.update(self._dependents(node, transitive=True))

    def _changed(self, nodes=()) -> None:
        self.version += 1
        nodes = tuple(tuple(n) for n in nodes)
        if nodes:
            self._mark_dirty(nodes)
            if self.snapshot_mode == "delta":
                for node in nodes:
                    self._pending[node] = self._justifications.get(node)
        else:
            # Conservative fallback for callers that cannot name the change.
            self._dirty.update(self._all_nodes())
            if self.snapshot_mode == "delta":
                self._pending = dict(self._justifications)
        self._snapshot()

    # ── queries ─────────────────────────────────────────────────────────────

    def _all_nodes(self) -> set[Node]:
        nodes: set[Node] = set(self._assumptions)
        for conclusion, justifications in self._justifications.items():
            nodes.add(conclusion)
            for justification in justifications:
                nodes.update(justification.in_nodes)
                nodes.update(justification.out_nodes)
        return nodes

    def _evaluate(self, node: Node, visiting: frozenset[Node]) -> bool:
        node = tuple(node)
        if node in self._assumptions:
            return True
        # Cached labels are always produced by a top-level evaluation, so a
        # clean (non-dirty) node's label is valid even inside a cycle.
        if node not in self._dirty and node in self._labels:
            return self._labels[node]
        if node in visiting:
            # A cycle with no external support is not well-founded.
            return False
        visiting = visiting | {node}
        for justification in self._justifications.get(node, ()):
            if all(self._evaluate(n, visiting) for n in justification.in_nodes) and all(
                not self._evaluate(n, visiting) for n in justification.out_nodes
            ):
                return True
        return False

    def supported(self, node: Node, _visiting: frozenset[Node] | None = None) -> bool:
        """Uncached well-founded support check (reference semantics)."""
        node = tuple(node)
        if node in self._assumptions:
            return True
        visiting = _visiting or frozenset()
        if node in visiting:
            # A cycle with no external support is not well-founded.
            return False
        visiting = visiting | {node}
        for justification in self._justifications.get(node, ()):
            if all(self.supported(n, visiting) for n in justification.in_nodes) and all(
                not self.supported(n, visiting) for n in justification.out_nodes
            ):
                return True
        return False

    def _propagate_dirty(self) -> None:
        if not self._dirty and self._labels:
            return
        for node in set(self._dirty):
            self._labels[node] = self._evaluate(node, frozenset())
        self._dirty.clear()

    def propagate(self, copy_labels: bool = True) -> dict[Node, bool]:
        """Re-evaluate only dirty nodes and their dependents.

        ``copy_labels=False`` returns the live label map; use it when the
        caller only needs the post-propagation state and wants to avoid an
        O(N) dict copy.  ``label()`` uses the internal dirty path directly.
        """
        self._propagate_dirty()
        return dict(self._labels) if copy_labels else self._labels

    def propagate_full(self) -> dict[Node, bool]:
        """Full recomputation (reference path, used for equivalence checks)."""
        return {node: self.supported(node) for node in self._all_nodes()}

    def label(self, node: Node) -> bool:
        node = tuple(node)
        if self._dirty or not self._labels:
            self._propagate_dirty()
        if node not in self._labels:
            self._labels[node] = self._evaluate(node, frozenset())
        return bool(self._labels[node])

    def in_nodes(self) -> set[Node]:
        return {node for node, val in self.propagate().items() if val}

    def conflict_keys(self) -> dict[Key, list[Any]]:
        grouped: dict[Key, set[Any]] = {}
        for node in self.in_nodes():
            entity, attribute, value = node
            grouped.setdefault((entity, attribute), set()).add(value)
        return {
            key: sorted(values, key=lambda v: repr(v))
            for key, values in grouped.items()
            if len(values) > 1
        }

    def dependents(self, node: Node, transitive: bool = True) -> set[Node]:
        return self._dependents(node, transitive=transitive)

    def justifications_for(self, node: Node) -> list[Justification]:
        return list(self._justifications.get(tuple(node), ()))

    # ── snapshots ───────────────────────────────────────────────────────────

    def reset_history(self, mode: str | None = None) -> None:
        """Discard history and take one checkpoint of the current state.

        Used after bulk loading so construction does not create one history
        entry per justification.
        """
        if mode is not None:
            if mode not in SNAPSHOT_MODES:
                raise ValueError(f"unknown snapshot_mode {mode!r}")
            self.snapshot_mode = mode
        self.history = []
        self._current_index = -1
        self._pending.clear()
        self._last_assumptions = set(self._assumptions)
        self._snapshot(force_full=True)

    def _snapshot(self, force_full: bool = False) -> None:
        mode = self.snapshot_mode
        if mode == "none":
            return
        parent = self._current_index
        kind = mode
        depth = 0
        if mode == "delta":
            if force_full or parent < 0:
                kind = "cow"
            else:
                depth = int(self.history[parent].get("depth", 0)) + 1
                if depth >= self.checkpoint_interval:
                    kind = "cow"
                    depth = 0
        if kind == "full":
            payload = {
                "kind": "full",
                "justifications": copy.deepcopy(self._justifications),
                "assumptions": set(self._assumptions),
                "version": int(self.version),
                "parent": parent,
                "depth": 0,
            }
        elif kind == "cow":
            payload = {
                "kind": "cow",
                "justifications": dict(self._justifications),
                "assumptions": frozenset(self._assumptions),
                "version": int(self.version),
                "parent": parent,
                "depth": 0,
            }
        else:  # delta
            payload = {
                "kind": "delta",
                "delta": dict(self._pending),
                "assumptions_add": set(self._assumptions - self._last_assumptions),
                "assumptions_remove": set(
                    self._last_assumptions - self._assumptions
                ),
                "version": int(self.version),
                "parent": parent,
                "depth": depth,
            }
        self._last_assumptions = set(self._assumptions)
        self._pending.clear()
        self.history.append(payload)
        self._current_index = len(self.history) - 1

    def snapshot(self) -> dict:
        """A compact token for the current state.

        History is append-only and entries carry a ``parent`` link, so a token
        stays valid after later mutations and after a rollback onto another
        branch.  ``none`` mode has no history and falls back to a materialised
        payload.
        """
        if self._current_index >= 0:
            return {
                "index": self._current_index,
                "version": int(self.version),
            }
        return {
            "justifications": copy.deepcopy(self._justifications),
            "assumptions": set(self._assumptions),
            "version": int(self.version),
        }

    def _restore_index(self, index: int) -> None:
        if index < 0 or index >= len(self.history):
            raise IndexError(f"unknown TMS version {index}")
        # Follow parent links rather than list position so a restore after a
        # rollback does not replay entries from the abandoned branch.
        chain: list[int] = []
        current = index
        while current >= 0 and self.history[current].get("kind") == "delta":
            chain.append(current)
            current = int(self.history[current].get("parent", current - 1))
        if current < 0:
            justifications: dict[Node, tuple[Justification, ...]] = {}
            assumptions: set[Node] = set()
        else:
            base = self.history[current]
            if base["kind"] == "full":
                justifications = copy.deepcopy(base["justifications"])
            else:
                justifications = dict(base["justifications"])
            assumptions = set(base["assumptions"])
        for position in reversed(chain):
            payload = self.history[position]
            for node, value in payload["delta"].items():
                if value is None:
                    justifications.pop(node, None)
                else:
                    justifications[node] = value
            assumptions |= payload["assumptions_add"]
            assumptions -= payload["assumptions_remove"]
        self._justifications = {
            node: tuple(items) for node, items in justifications.items()
        }
        self._assumptions = set(assumptions)
        self.version = int(self.history[index].get("version", self.version))
        self._rebuild_reverse()
        self._labels.clear()
        self._dirty.update(self._all_nodes())
        self._last_assumptions = set(self._assumptions)
        self._pending.clear()
        self._current_index = index

    def restore(self, payload: dict) -> None:
        if "index" in payload and "justifications" not in payload:
            self._restore_index(int(payload["index"]))
            return
        self._justifications = {
            node: tuple(items)
            for node, items in copy.deepcopy(payload["justifications"]).items()
        }
        self._assumptions = set(payload["assumptions"])
        self.version = int(payload.get("version", self.version))
        self._rebuild_reverse()
        self._labels.clear()
        self._dirty.update(self._all_nodes())
        self._last_assumptions = set(self._assumptions)
        self._pending.clear()
        self._current_index = -1

    def rollback(self, version: int) -> None:
        self._restore_index(int(version))
