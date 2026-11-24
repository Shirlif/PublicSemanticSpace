"""PSS evaluation suite (Phase A skeleton).

Unified episode schema + benchmark adapters + metrics + runner. Raw benchmark
data stays under ``data/public/<bench>/``; the suite only holds adapters and
manifests.
"""

from .adapter import BenchmarkAdapter, BenchmarkUnavailable
from .registry import get_adapter, list_adapters
from .schema import (
    EntitySpec,
    EvalEpisode,
    EvalResult,
    FactSpec,
    Frame,
    GoalSpec,
    Prediction,
    RelationSpec,
)

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkUnavailable",
    "EntitySpec",
    "EvalEpisode",
    "EvalResult",
    "FactSpec",
    "Frame",
    "GoalSpec",
    "Prediction",
    "RelationSpec",
    "get_adapter",
    "list_adapters",
]
