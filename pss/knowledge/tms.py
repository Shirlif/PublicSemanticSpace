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
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

Node = tuple[str, str, Any]
Key = tuple[str, str]


@dataclass(frozen=True)
class Justification:
    conclusion: Node
    in_nodes: frozenset[Node] = frozenset()
    out_nodes: frozenset[Node] = frozenset()
    source: str = "runtime"
    rule: str = ""
    confidence: float = 1.0


class TruthMaintenanceSystem:
    def __init__(self):
        self._justifications: dict[Node, list[Justification]] = {}
        self._assumptions: set[Node] = set()
        self._labels: dict[Node, bool] = {}
        self.version = 0
        self.history: list[dict] = []
        self._snapshot()

    # ── mutation ────────────────────────────────────────────────────────────

    def assume(self, node: Node) -> None:
        self._assumptions.add(tuple(node))
        self._changed()

    def unassume(self, node: Node) -> None:
        self._assumptions.discard(tuple(node))
        self._changed()

    def add_justification(self, justification: Justification) -> Justification:
        if not isinstance(justification, Justification):
            raise TypeError("expected a Justification")
        self._justifications.setdefault(tuple(justification.conclusion), []).append(justification)
        self._changed()
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
        for node, justifications in list(self._justifications.items()):
            kept = [j for j in justifications if j.source != source]
            removed += len(justifications) - len(kept)
            if kept:
                self._justifications[node] = kept
            else:
                del self._justifications[node]
        if removed:
            self._changed()
        return removed

    def remove_source_for(self, node: Node, source: str) -> int:
        """Remove justifications for one conclusion from one source."""
        node = tuple(node)
        justifications = self._justifications.get(node, [])
        kept = [j for j in justifications if j.source != source]
        removed = len(justifications) - len(kept)
        if removed:
            if kept:
                self._justifications[node] = kept
            else:
                del self._justifications[node]
            self._changed()
        return removed

    def remove_conclusion(self, node: Node) -> int:
        node = tuple(node)
        justifications = self._justifications.pop(node, [])
        self._assumptions.discard(node)
        if justifications:
            self._changed()
        return len(justifications)

    # ── queries ─────────────────────────────────────────────────────────────

    def _all_nodes(self) -> set[Node]:
        nodes: set[Node] = set(self._assumptions)
        for conclusion, justifications in self._justifications.items():
            nodes.add(conclusion)
            for j in justifications:
                nodes.update(j.in_nodes)
                nodes.update(j.out_nodes)
        return nodes

    def supported(self, node: Node, _visiting: frozenset[Node] | None = None) -> bool:
        node = tuple(node)
        if node in self._assumptions:
            return True
        visiting = _visiting or frozenset()
        if node in visiting:
            # A cycle with no external support is not well-founded.
            return False
        visiting = visiting | {node}
        for j in self._justifications.get(node, ()):
            if all(self.supported(n, visiting) for n in j.in_nodes) and all(
                not self.supported(n, visiting) for n in j.out_nodes
            ):
                return True
        return False

    def propagate(self) -> dict[Node, bool]:
        self._labels = {node: self.supported(node) for node in self._all_nodes()}
        return dict(self._labels)

    def label(self, node: Node) -> bool:
        node = tuple(node)
        if not self._labels:
            self.propagate()
        if node not in self._labels:
            self._labels[node] = self.supported(node)
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
        node = tuple(node)
        reverse: dict[Node, set[Node]] = {}
        for conclusion, justifications in self._justifications.items():
            for j in justifications:
                for antecedent in set(j.in_nodes) | set(j.out_nodes):
                    reverse.setdefault(antecedent, set()).add(conclusion)
        if not transitive:
            return set(reverse.get(node, set()))
        seen: set[Node] = set()
        stack = list(reverse.get(node, set()))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(reverse.get(cur, set()) - seen)
        return seen

    def justifications_for(self, node: Node) -> list[Justification]:
        return list(self._justifications.get(tuple(node), ()))

    # ── snapshots ───────────────────────────────────────────────────────────

    def _changed(self) -> None:
        self.version += 1
        self._labels.clear()
        self._snapshot()

    def _snapshot(self) -> None:
        self.history.append(
            {
                "justifications": copy.deepcopy(self._justifications),
                "assumptions": set(self._assumptions),
                "version": int(self.version),
            }
        )

    def snapshot(self) -> dict:
        return {
            "justifications": copy.deepcopy(self._justifications),
            "assumptions": set(self._assumptions),
            "version": int(self.version),
            "history_len": len(self.history),
        }

    def restore(self, payload: dict) -> None:
        self._justifications = copy.deepcopy(payload["justifications"])
        self._assumptions = set(payload["assumptions"])
        self.version = int(payload.get("version", self.version))
        self._labels.clear()

    def rollback(self, version: int) -> None:
        if version < 0 or version >= len(self.history):
            raise IndexError(f"unknown TMS version {version}")
        self.restore(self.history[version])
