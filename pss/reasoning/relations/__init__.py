"""Relation layer (Phase 7): geometric/attribute records + vector encoding.

Structured records drive the planner; the bound vector encoding is the
retrieval/grounding path (Phase 1's three-part relation storage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pss.binding import RelationEncoder, cosine

from .closure import (
    INVERSE,
    PREDICATES,
    SYMMETRIC,
    ClosureTable,
    default_closure,
)
from .contact import contact_pairs

BIN_HALF = 0.06
# Matches SensorRig.relations: BIN_HEIGHT (0.06) + 0.08 headroom.
BIN_Z = 0.14
DEFAULT_BINS = {
    "bin0": np.array([0.38, 0.20, 0.0]),
    "bin1": np.array([0.38, -0.20, 0.0]),
}


@dataclass(frozen=True)
class RelationRecord:
    subject: str
    predicate: str
    object: str | None = None
    confidence: float = 1.0
    source: str = "geometry"
    frame: str = "world"

    def as_tuple(self) -> tuple:
        return (self.subject, self.predicate, self.object)

    def as_frame_tuple(self) -> tuple:
        """Include the reference frame; use this for frame-scoped reasoning."""
        return (self.subject, self.predicate, self.object, self.frame)


def geometric_relations(
    instances: dict[str, dict],
    bins: dict[str, np.ndarray] | None = None,
    near_dist: float = 0.12,
    contact_pad: float = 0.01,
) -> list[RelationRecord]:
    """Compute `in` / `contact` / `near` from poses and rough radii.

    ``instances`` maps id -> {"pos": (3,), "radius": float}.
    """
    bins = DEFAULT_BINS if bins is None else bins
    out: list[RelationRecord] = []
    ids = list(instances)

    for sid, inst in instances.items():
        p = np.asarray(inst["pos"], dtype=float)
        for bname, bpos in bins.items():
            if (
                abs(p[0] - bpos[0]) < BIN_HALF - 0.01
                and abs(p[1] - bpos[1]) < BIN_HALF - 0.01
                and p[2] < BIN_Z
            ):
                out.append(RelationRecord(sid, "in", bname, 1.0))

    for i, a in enumerate(ids):
        pa = np.asarray(instances[a]["pos"], dtype=float)
        ra = float(instances[a].get("radius", 0.05))
        for b in ids[i + 1 :]:
            pb = np.asarray(instances[b]["pos"], dtype=float)
            rb = float(instances[b].get("radius", 0.05))
            d = float(np.linalg.norm(pa - pb))
            if d <= ra + rb + contact_pad:
                out.append(RelationRecord(a, "contact", b, 1.0))
                out.append(RelationRecord(b, "contact", a, 1.0))
            if d < near_dist:
                conf = float(np.clip(1.0 - d / near_dist, 0.05, 1.0))
                out.append(RelationRecord(a, "near", b, conf))
                out.append(RelationRecord(b, "near", a, conf))
    return out


def attribute_relations(
    sizes: dict[str, float], min_ratio: float = 0.15
) -> list[RelationRecord]:
    out: list[RelationRecord] = []
    ids = list(sizes)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            sa, sb = float(sizes[a]), float(sizes[b])
            denom = max(sa, sb, 1e-9)
            ratio = abs(sa - sb) / denom
            if ratio < min_ratio:
                continue
            if sa > sb:
                out.append(RelationRecord(a, "larger_than", b, float(np.clip(ratio, 0.05, 1.0)), "attribute"))
                out.append(RelationRecord(b, "smaller_than", a, float(np.clip(ratio, 0.05, 1.0)), "attribute"))
            else:
                out.append(RelationRecord(b, "larger_than", a, float(np.clip(ratio, 0.05, 1.0)), "attribute"))
                out.append(RelationRecord(a, "smaller_than", b, float(np.clip(ratio, 0.05, 1.0)), "attribute"))
    return out


class RelationStore:
    def __init__(self, dim: int = 128, seed: int = 0):
        self.dim = int(dim)
        self.records: list[RelationRecord] = []
        self.encoder = RelationEncoder(self.dim, seed=seed)
        self.predicate_basis: dict[str, np.ndarray] = {}
        self.encoded: dict[tuple, object] = {}
        self._rng = np.random.default_rng(seed)

    def add(self, record: RelationRecord) -> None:
        self.records.append(record)

    def extend(self, records) -> None:
        self.records.extend(records)

    def facts(self) -> set[tuple]:
        return {r.as_tuple() for r in self.records}

    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        min_conf: float = 0.0,
    ) -> list[RelationRecord]:
        out = []
        for r in self.records:
            if subject is not None and r.subject != subject:
                continue
            if predicate is not None and r.predicate != predicate:
                continue
            if object is not None and r.object != object:
                continue
            if r.confidence < min_conf:
                continue
            out.append(r)
        return out

    def _predicate_vector(self, predicate: str) -> np.ndarray:
        if predicate not in self.predicate_basis:
            v = self._rng.normal(size=self.dim)
            self.predicate_basis[predicate] = v / np.linalg.norm(v)
        return self.predicate_basis[predicate]

    def encode(self, record: RelationRecord, subject_vec, object_vec=None):
        pred_vec = self._predicate_vector(record.predicate)
        obj_vec = subject_vec if object_vec is None else object_vec
        encoded = self.encoder.encode(subject_vec, pred_vec, obj_vec)
        self.encoded[record.as_tuple()] = encoded
        return encoded

    def vector_query(self, fact_vec, top_k: int = 3) -> list[tuple[float, tuple]]:
        scored = [
            (cosine(fact_vec, encoded.fact), key)
            for key, encoded in self.encoded.items()
        ]
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]


def precision_recall_f1(computed: set[tuple], truth: set[tuple]) -> dict:
    tp = len(computed & truth)
    fp = len(computed - truth)
    fn = len(truth - computed)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def relations_from_frame(
    sample: dict,
    frame: int,
    category_sizes: dict[str, tuple],
    contact_margin: float = 1e-5,
) -> list[RelationRecord]:
    instances = {}
    for i, instance_id in enumerate(sample["instance_ids"]):
        category = sample["categories"][i]
        size = category_sizes.get(category, (0.05,))
        radius = max(float(x) for x in size) / 2.0
        instances[str(instance_id)] = {
            "pos": np.asarray(sample["poses"][frame, i, :3], dtype=float),
            "radius": radius,
        }
    records = geometric_relations(instances)

    specs = sample.get("object_specs")
    if specs:
        # Replace the bounding-sphere contact proxy with collision geometry.
        records = [r for r in records if r.predicate != "contact"]
        for i, j in contact_pairs(specs, sample["poses"][frame], margin=contact_margin):
            a = str(sample["instance_ids"][i])
            b = str(sample["instance_ids"][j])
            records.append(RelationRecord(a, "contact", b, 1.0, "collision_geometry"))
            records.append(RelationRecord(b, "contact", a, 1.0, "collision_geometry"))
    return records


# Imported last on purpose: ``inference`` imports ``RelationRecord`` from this
# package, so the symbol must already be defined when this line executes.
from .inference import (  # noqa: E402
    DerivedFact,
    InferenceResult,
    backward_chain,
    derive,
    derive_records,
)

