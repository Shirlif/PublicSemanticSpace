"""PSS cognitive runtime (Phase 16).

Owns typed state (entities, relations, facts, goals, plans, uncertainty,
audit) and routes perception/relations/planning/outputs as services. This is
the layering from ``docs/architecture/PSS-认知层架构定位.md``, not another model.
"""

from .layers import EntityLayer, RelationFactLayer
from .permissions import (
    ACTIONS,
    PROFILES,
    PermissionDenied,
    PermissionProfile,
    PermissionService,
)
from .runtime import OutputRouter, PSSRuntime
from .services import RuntimePlanner, RuntimeResult, UncertaintyService
from .state import (
    AuditEntry,
    CognitiveState,
    Entity,
    Fact,
    Goal,
    Plan,
    Relation,
    UncertaintyReport,
)

__all__ = [
    "ACTIONS",
    "AuditEntry",
    "PROFILES",
    "PermissionDenied",
    "PermissionProfile",
    "PermissionService",
    "CognitiveState",
    "Entity",
    "EntityLayer",
    "Fact",
    "Goal",
    "OutputRouter",
    "PSSRuntime",
    "Plan",
    "Relation",
    "RelationFactLayer",
    "RuntimePlanner",
    "RuntimeResult",
    "UncertaintyReport",
    "UncertaintyService",
]
