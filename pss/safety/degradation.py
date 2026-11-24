"""Health monitoring and four-level degradation (Phase 9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class HealthLevel(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"


@dataclass
class Fault:
    name: str
    severity: int  # 1 warning, 2 degraded, 3 emergency
    detail: str = ""


@dataclass
class HealthStatus:
    level: HealthLevel
    faults: list[Fault]
    proposal_allowed: bool
    motion_allowed: bool
    exploration_allowed: bool
    manual_required: bool
    offline: bool = False


def _target_level(faults: list[Fault]) -> HealthLevel:
    if not faults:
        return HealthLevel.NORMAL
    severity = max(f.severity for f in faults)
    if severity >= 3:
        return HealthLevel.EMERGENCY
    if severity >= 2:
        return HealthLevel.DEGRADED
    return HealthLevel.WARNING


class DegradationFSM:
    def __init__(self, debounce: int = 2):
        self.debounce = max(1, int(debounce))
        self.level = HealthLevel.NORMAL
        self._pending = HealthLevel.NORMAL
        self._count = 0

    def update(self, faults: list[Fault], offline: bool = False) -> HealthStatus:
        target = _target_level(faults)
        if target == self.level:
            self._pending = self.level
            self._count = 0
        else:
            if target == self._pending:
                self._count += 1
            else:
                self._pending = target
                self._count = 1
            if self._count >= self.debounce:
                self.level = target
                self._count = 0
        return self.status(faults, offline)

    def status(self, faults: list[Fault], offline: bool = False) -> HealthStatus:
        level = self.level
        return HealthStatus(
            level=level,
            faults=faults,
            proposal_allowed=level in (HealthLevel.NORMAL, HealthLevel.WARNING) and not offline,
            motion_allowed=level != HealthLevel.EMERGENCY,
            exploration_allowed=level in (HealthLevel.NORMAL, HealthLevel.WARNING),
            manual_required=level == HealthLevel.EMERGENCY,
            offline=offline,
        )


class HealthMonitor:
    def __init__(self, debounce: int = 2):
        self.fsm = DegradationFSM(debounce)

    def check(self, signals: dict) -> HealthStatus:
        faults: list[Fault] = []
        if not signals.get("camera_ok", True):
            faults.append(Fault("camera_dropout", 2))
        if float(signals.get("projection_drift", 0.0)) > 0.1:
            faults.append(Fault("projection_drift", 1))
        if signals.get("sensor_conflict", False):
            faults.append(Fault("sensor_conflict", 1))
        if signals.get("package_corrupt", False):
            faults.append(Fault("package_corrupt", 2))
        if float(signals.get("temperature", 0.0)) > 80.0:
            faults.append(Fault("overheat", 3))
        if signals.get("llm_injection", False):
            faults.append(Fault("llm_injection", 1))
        return self.fsm.update(faults, offline=bool(signals.get("offline", False)))
