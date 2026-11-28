"""Contact-event data model.

A contact is an **event**, not a boolean: it has participants, onset/offset,
location, intensity, confidence, evidence sources and a lifecycle status.  The
same object pair can have multiple events over time; events are kept separate
by the ``event_id`` and merged only by the segmenter when they overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STATUS = ("predicted", "confirmed", "contradicted", "expired", "resolved")


@dataclass
class ContactEvidence:
    source: str  # geometry | force | tactile | audio | vision | proprio
    timestamp: float
    score: float
    intensity: float | None = None
    location: np.ndarray | None = None
    reliability: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ContactEvent:
    event_id: str
    subject: str
    object: str
    onset: float
    offset: float | None = None
    location: np.ndarray | None = None
    intensity: float | None = None
    confidence: float = 0.0
    status: str = "predicted"
    evidence: list[ContactEvidence] = field(default_factory=list)
    version: int = 1
    provenance: dict = field(default_factory=dict)

    @property
    def pair(self) -> tuple[str, str]:
        return tuple(sorted((self.subject, self.object)))

    @property
    def duration(self) -> float:
        return max(0.0, float(self.offset or self.onset) - float(self.onset))

    def with_status(self, status: str) -> "ContactEvent":
        if status not in STATUS:
            raise ValueError(f"unknown status {status!r}")
        self.status = status
        self.version += 1
        return self


@dataclass
class ContactFrame:
    t: float
    contacts: list[dict]
    tactile: dict[str, np.ndarray]
    poses: dict[str, np.ndarray]
    scene_model: object | None = None
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    segmentation: np.ndarray | None = None
    camera: object | None = None


__all__ = [
    "STATUS",
    "ContactEvent",
    "ContactEvidence",
    "ContactFrame",
]
