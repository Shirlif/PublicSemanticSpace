"""Versioned fact store and explicit rule engine.

Knowledge (facts) is stored separately from logic (rules). Facts can be
updated, deleted, rolled back, and audited without touching the rules.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AuditEntry:
    version: int
    action: str
    key: tuple
    old: object
    new: object
    reason: str
    timestamp: float = field(default_factory=time.time)


class FactStore:
    def __init__(self, facts: dict[tuple[str, str], str] | None = None):
        self.facts = dict(facts or {})
        self.version = 0
        self.history: list[dict] = []
        self.audit: list[AuditEntry] = []
        self._snapshot()

    def _snapshot(self):
        self.history.append(copy.deepcopy(self.facts))

    def get(self, entity: str, attribute: str):
        return self.facts.get((entity, attribute))

    def update(self, entity: str, attribute: str, value: str, reason: str = ""):
        key = (entity, attribute)
        old = self.facts.get(key)
        self.facts[key] = value
        self.version += 1
        self.audit.append(
            AuditEntry(self.version, "update", key, old, value, reason)
        )
        self._snapshot()

    def add(self, entity: str, attribute: str, value: str, reason: str = ""):
        if (entity, attribute) in self.facts:
            raise KeyError(f"fact already exists: {entity}.{attribute}")
        self.update(entity, attribute, value, reason or "add")

    def delete(self, entity: str, attribute: str, reason: str = ""):
        key = (entity, attribute)
        old = self.facts.pop(key, None)
        self.version += 1
        self.audit.append(
            AuditEntry(self.version, "delete", key, old, None, reason)
        )
        self._snapshot()

    def rollback(self, version: int):
        if version < 0 or version >= len(self.history):
            raise IndexError("unknown fact version")
        self.facts = copy.deepcopy(self.history[version])
        self.version = version
        self.audit.append(
            AuditEntry(self.version, "rollback", (), None, None, f"to v{version}")
        )

    def all_facts(self):
        return dict(self.facts)


@dataclass
class Rule:
    name: str
    condition: Callable[[FactStore, str], bool]
    conclusion: str  # attribute name; value is "yes"/"no"

    def applies(self, store: FactStore, entity: str) -> bool:
        return bool(self.condition(store, entity))


class RuleEngine:
    def __init__(self, rules: list[Rule]):
        self.rules = list(rules)

    def derive(self, store: FactStore, entity: str, attribute: str):
        for rule in self.rules:
            if rule.conclusion == attribute:
                return "yes" if rule.applies(store, entity) else "no"
        return None

    def answer(self, store: FactStore, entity: str, attribute: str):
        direct = store.get(entity, attribute)
        if direct is not None:
            return direct
        return self.derive(store, entity, attribute)


@dataclass
class Assertion:
    value: str
    source: str
    priority: int = 0
    timestamp: float = field(default_factory=time.time)


class MultiSourceFactStore:
    """Facts asserted by multiple sources with priority and conflict reporting."""

    def __init__(
        self,
        assertions: dict[tuple[str, str], list[Assertion]] | None = None,
        record_history: bool = True,
    ):
        self.assertions = {
            k: list(v) for k, v in (assertions or {}).items()
        }
        self.version = 0
        # ``TMSFactStore`` owns versioning/rollback and disables this inner
        # history so a wrapped mutation is not O(N) in the assertion table.
        self.record_history = bool(record_history)
        self.history: list[dict] = []
        self.audit: list[AuditEntry] = []
        self._snapshot()

    def _snapshot(self):
        if not self.record_history:
            return
        self.history.append(copy.deepcopy(self.assertions))

    def assert_fact(
        self,
        entity: str,
        attribute: str,
        value: str,
        source: str,
        priority: int = 0,
        reason: str = "",
    ):
        key = (entity, attribute)
        self.assertions.setdefault(key, []).append(
            Assertion(value, source, int(priority))
        )
        self.version += 1
        self.audit.append(
            AuditEntry(self.version, "assert", key, None, value, f"{source}:{priority} {reason}")
        )
        self._snapshot()

    def retract(self, entity: str, attribute: str, source: str, reason: str = ""):
        key = (entity, attribute)
        old = self.assertions.get(key, [])
        self.assertions[key] = [a for a in old if a.source != source]
        self.version += 1
        self.audit.append(
            AuditEntry(self.version, "retract", key, source, None, reason)
        )
        self._snapshot()

    def delete_entity(self, entity: str, reason: str = ""):
        keys = [k for k in self.assertions if k[0] == entity]
        for key in keys:
            self.assertions.pop(key, None)
        self.version += 1
        self.audit.append(
            AuditEntry(self.version, "delete_entity", (entity,), len(keys), 0, reason)
        )
        self._snapshot()

    def resolve(self, entity: str, attribute: str):
        values = self.assertions.get((entity, attribute), [])
        if not values:
            return None
        best = max(values, key=lambda a: (a.priority, a.timestamp))
        return best.value

    def get(self, entity: str, attribute: str):
        return self.resolve(entity, attribute)

    def conflicts(self):
        out = {}
        for key, values in self.assertions.items():
            distinct = {a.value for a in values}
            if len(distinct) > 1:
                out[key] = sorted(distinct)
        return out

    def rollback(self, version: int):
        if version < 0 or version >= len(self.history):
            raise IndexError("unknown fact version")
        self.assertions = copy.deepcopy(self.history[version])
        self.version = version
        self.audit.append(
            AuditEntry(self.version, "rollback", (), None, None, f"to v{version}")
        )


@dataclass
class RuleVersion:
    version: int
    rule: Rule


class VersionedRuleEngine:
    def __init__(self, rules: list[Rule]):
        self.rules = {r.name: r for r in rules}
        self.version = 0
        self.history: list[dict] = []
        self.audit: list[AuditEntry] = []
        self._snapshot()

    def _snapshot(self):
        self.history.append(copy.deepcopy(self.rules))

    def update(self, rule: Rule, reason: str = ""):
        old = self.rules.get(rule.name)
        self.rules[rule.name] = rule
        self.version += 1
        self.audit.append(
            AuditEntry(self.version, "update_rule", (rule.name,), old, rule, reason)
        )
        self._snapshot()

    def delete(self, name: str, reason: str = ""):
        old = self.rules.pop(name, None)
        self.version += 1
        self.audit.append(
            AuditEntry(self.version, "delete_rule", (name,), old, None, reason)
        )
        self._snapshot()

    def answer(self, store, entity: str, attribute: str):
        direct = store.get(entity, attribute)
        if direct is not None:
            return direct
        rule = self.rules.get(attribute)
        if rule is None:
            return None
        return "yes" if rule.condition(store, entity) else "no"

    def rollback(self, version: int):
        if version < 0 or version >= len(self.history):
            raise IndexError("unknown rule version")
        self.rules = copy.deepcopy(self.history[version])
        self.version = version
        self.audit.append(
            AuditEntry(self.version, "rollback_rules", (), None, None, f"to v{version}")
        )
