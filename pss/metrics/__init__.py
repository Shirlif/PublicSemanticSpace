"""Offline evaluation metrics.

Metrics are pure functions over bundle tables: no simulator, no GPU, no PSS
runtime.  The same implementation is shared by the Isaac layer, the MuJoCo fast
layer and the cross-machine recomputation check (A9).
"""

from .entity import (
    METRIC_REGISTRY,
    compute_entity_metrics,
    duplicate_rate,
    id_retention,
    id_switches,
    idf1,
    metric,
    reappearance_rate,
    slot_purity,
)
from .twin import twin_ambiguity
from .relation import consistency_violations, per_predicate_prf
from .memory import compute_memory_metrics

__all__ = [
    "METRIC_REGISTRY",
    "compute_entity_metrics",
    "duplicate_rate",
    "id_retention",
    "id_switches",
    "idf1",
    "metric",
    "reappearance_rate",
    "slot_purity",
    "twin_ambiguity",
    "consistency_violations",
    "per_predicate_prf",
    "compute_memory_metrics",
]
