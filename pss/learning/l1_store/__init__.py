"""L1 instance memory: local-only records and scrubbed proposals (Phase 9)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

import numpy as np

ALLOWED_ENV_KEYS = ("lighting", "device_class", "temperature_band")


@dataclass
class L1Record:
    kind: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )
    entity_id: str | None = None
    category: str | None = None
    vectors: dict[str, list[float]] = field(default_factory=dict)
    facts: list[list[str]] = field(default_factory=list)
    confidence: float = 1.0
    environment: dict = field(default_factory=dict)
    health: str = "normal"


class L1Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: list[L1Record] = []

    def append(self, record: L1Record) -> None:
        self.records.append(record)

    def extend(self, records) -> None:
        self.records.extend(records)

    def query(self, kind: str | None = None, category: str | None = None, min_confidence: float = 0.0):
        out = []
        for r in self.records:
            if kind is not None and r.kind != kind:
                continue
            if category is not None and r.category != category:
                continue
            if r.confidence < min_confidence:
                continue
            out.append(r)
        return out

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "L1Store":
        store = cls(path)
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    store.records.append(L1Record(**json.loads(line)))
        return store

    def make_proposal(
        self,
        category: str,
        health: str = "normal",
        min_confidence: float = 0.5,
        max_items: int = 100,
    ) -> dict | None:
        if health not in ("normal", "warning"):
            return None
        items = [
            r
            for r in self.records
            if r.category == category
            and r.kind in ("observation", "conflict")
            and r.confidence >= min_confidence
        ][:max_items]
        if not items:
            return None

        modalities = sorted({m for r in items for m in r.vectors})
        vectors: dict[str, list[float]] = {}
        variance: dict[str, float] = {}
        for m in modalities:
            arr = np.stack([np.asarray(r.vectors[m], dtype=float) for r in items if m in r.vectors])
            mean = arr.mean(0)
            n = np.linalg.norm(mean)
            mean = mean / n if n > 0 else mean
            vectors[m] = mean.tolist()
            variance[m] = float(np.mean(1.0 - arr @ mean))

        environment = {}
        for key in ALLOWED_ENV_KEYS:
            values = [r.environment.get(key) for r in items if key in r.environment]
            if values:
                environment[key] = values[0]

        return {
            "proposal_id": str(uuid.uuid4()),
            "category": category,
            "sample_count": len(items),
            "vectors": vectors,
            "variance": variance,
            "confidence": float(np.mean([r.confidence for r in items])),
            "environment": environment,
            "scrubbed": True,
            "source": "edge_l1",
        }
