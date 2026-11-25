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
]
