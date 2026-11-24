"""Versioned operator library for the runtime planner.

The architecture asks for versioned logic, not only versioned facts. This wraps
``OperatorLibrary`` with an explicit version, update/remove operations and
rollback. Plans record the operator version they were produced under.
"""

from __future__ import annotations

import copy

from pss.world_model.actions import Operator, OperatorLibrary


class VersionedOperatorLibrary:
    def __init__(self, operators: list[Operator], version: int = 1):
        self.operators: dict[str, Operator] = {op.name: op for op in operators}
        self.version = int(version)
        self.snapshots: dict[int, dict[str, Operator]] = {}
        self.audit: list[dict] = []
        self._snapshot()

    def _snapshot(self) -> None:
        self.snapshots[self.version] = copy.deepcopy(self.operators)

    def _library(self) -> OperatorLibrary:
        return OperatorLibrary(list(self.operators.values()))

    def get(self, name: str) -> Operator:
        return self.operators[name]

    def applicable(self, facts, confidence=None, threshold: float = 0.5):
        return self._library().applicable(facts, confidence, threshold)

    def apply(self, op: Operator, facts):
        return self._library().apply(op, facts)

    def update(self, op: Operator, reason: str = "") -> int:
        before = self.operators.get(op.name)
        self.operators[op.name] = op
        self.version += 1
        self.audit.append(
            {
                "version": self.version,
                "action": "update",
                "operator": op.name,
                "before": None if before is None else before.name,
                "reason": reason,
            }
        )
        self._snapshot()
        return self.version

    def remove(self, name: str, reason: str = "") -> int:
        before = self.operators.pop(name, None)
        self.version += 1
        self.audit.append(
            {
                "version": self.version,
                "action": "remove",
                "operator": name,
                "before": None if before is None else before.name,
                "reason": reason,
            }
        )
        self._snapshot()
        return self.version

    def rollback(self, version: int) -> int:
        if version not in self.snapshots:
            raise IndexError(f"unknown operator version {version}")
        self.operators = copy.deepcopy(self.snapshots[version])
        self.version = int(version)
        self.audit.append(
            {"version": self.version, "action": "rollback", "operator": "*", "reason": f"to v{version}"}
        )
        self._snapshot()
        return self.version
