"""Common episode schema for the PSS evaluation suite.

One episode is one testable unit. Not every benchmark fills every field; the
``capabilities`` set says what is present. The adapters convert their native
format into this schema; the runner only knows this schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Frame:
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    segmentation: np.ndarray | None = None
    tactile: np.ndarray | None = None
    poses: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class EntitySpec:
    id: str
    category: str | None = None
    pos: np.ndarray | None = None
    attributes: dict = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class RelationSpec:
    source: str
    target: str
    relation_type: str = "symbolic"
    predicate: str = ""
    probability: float = 1.0


@dataclass
class FactSpec:
    entity: str
    attribute: str
    value: Any
    source: str = "benchmark"
    priority: int = 0


@dataclass
class GoalSpec:
    facts: frozenset[tuple] = frozenset()
    relation: tuple | None = None
    description: str = ""


@dataclass
class EvalEpisode:
    id: str
    benchmark: str
    layer: str  # language | open_set | memory | entity | relation | planning | audit
    split: str = "test"
    capabilities: set[str] = field(default_factory=set)
    frames: list[Frame] = field(default_factory=list)
    entities: list[EntitySpec] = field(default_factory=list)
    relations: list[RelationSpec] = field(default_factory=list)
    facts: list[FactSpec] = field(default_factory=list)
    goal: GoalSpec | None = None
    query: str | None = None
    tools: list[dict] = field(default_factory=list)
    expected: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


@dataclass
class Prediction:
    episode_id: str
    output: dict
    meta: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    benchmark: str
    layer: str
    split: str
    metric: str
    value: float | None
    n: int
    meta: dict = field(default_factory=dict)
