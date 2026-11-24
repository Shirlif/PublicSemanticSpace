"""Persistent 3D scene graph (Phase 6).

The scene graph owns entity identity. Working-memory slots are only a view over
the few entities currently in focus; eviction never deletes the entity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STATUSES = ("active", "dormant", "missing", "fragmented", "historical")


@dataclass
class Track:
    id: str
    pos: np.ndarray
    quat: np.ndarray
    appearance: np.ndarray | None = None
    category: str | None = None
    confidence: float = 1.0
    first_seen: int = 0
    last_seen: int = 0
    missed: int = 0
    status: str = "active"
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw: dict[str, np.ndarray] = field(default_factory=dict)
    parent: str | None = None
    provenance: dict = field(default_factory=dict)
    observations: int = 1


class SpatialIndex:
    """Coarse 3D grid. Good enough for hundreds of entities; swap for an octree."""

    def __init__(self, cell_size: float = 0.1):
        self.cell_size = float(cell_size)
        self.cells: dict[tuple[int, int, int], set[str]] = {}

    def _key(self, pos) -> tuple[int, int, int]:
        return tuple(int(np.floor(float(x) / self.cell_size)) for x in pos[:3])

    def add(self, track_id: str, pos) -> None:
        self.cells.setdefault(self._key(pos), set()).add(track_id)

    def remove(self, track_id: str, pos) -> None:
        key = self._key(pos)
        bucket = self.cells.get(key)
        if bucket is not None:
            bucket.discard(track_id)
            if not bucket:
                self.cells.pop(key, None)

    def update(self, track_id: str, old_pos, new_pos) -> None:
        self.remove(track_id, old_pos)
        self.add(track_id, new_pos)

    def query(self, pos, radius: float) -> list[str]:
        r = float(radius)
        cs = self.cell_size
        lo = [int(np.floor((float(x) - r) / cs)) for x in pos[:3]]
        hi = [int(np.floor((float(x) + r) / cs)) for x in pos[:3]]
        found: set[str] = set()
        for ix in range(lo[0], hi[0] + 1):
            for iy in range(lo[1], hi[1] + 1):
                for iz in range(lo[2], hi[2] + 1):
                    found |= self.cells.get((ix, iy, iz), set())
        return list(found)


class SceneGraph:
    def __init__(self, cell_size: float = 0.1):
        self.tracks: dict[str, Track] = {}
        # Typed relation logits/records, keyed by (source, target).
        # Values are predicate -> record dicts written by the relation layer.
        self.relations: dict[tuple[str, str], dict[str, dict]] = {}
        self.index = SpatialIndex(cell_size)
        self._next = 1

    def new_id(self) -> str:
        track_id = f"obj_{self._next:04d}"
        self._next += 1
        return track_id

    def add(self, track: Track) -> Track:
        self.tracks[track.id] = track
        self.index.add(track.id, track.pos)
        return track

    def remove(self, track_id: str) -> None:
        track = self.tracks.pop(track_id, None)
        if track is not None:
            self.index.remove(track_id, track.pos)
        for key in [k for k in self.relations if track_id in k]:
            self.relations.pop(key, None)

    def get(self, track_id: str) -> Track | None:
        return self.tracks.get(track_id)

    def all(self) -> list[Track]:
        return list(self.tracks.values())

    def active(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.status == "active"]

    def dormant(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.status in ("dormant", "missing")]

    def reid_candidates(self, t: int | None = None, max_age: int | None = None) -> list[Track]:
        """Tracks eligible for re-ID, including long-term historical entities.

        ``historical`` used to be excluded, which made any occlusion longer
        than ``historical_missed`` unrecoverable. A persistent graph should
        keep the entity searchable, with age as a soft bound.
        """
        out = []
        for track in self.tracks.values():
            if track.status not in ("dormant", "missing", "historical"):
                continue
            if (
                t is not None
                and max_age is not None
                and t - track.last_seen > max_age
            ):
                continue
            out.append(track)
        return out

    def query_near(self, pos, radius: float) -> list[Track]:
        out = []
        for track_id in self.index.query(pos, radius):
            track = self.tracks.get(track_id)
            if track is not None and np.linalg.norm(track.pos - np.asarray(pos)) <= radius:
                out.append(track)
        return out

    def set_relation(self, edge) -> None:
        """Store a typed relation record (duck-typed ``RelationEdge``).

        Records are keyed by ``(source, target, type, predicate)`` so a later
        update overwrites the previous logit instead of duplicating it.
        """
        entry = edge.to_dict() if hasattr(edge, "to_dict") else dict(edge)
        key = (str(entry.get("source")), str(entry.get("target")))
        relation_type = str(entry.get("type") or entry.get("relation_type"))
        predicate = str(entry["predicate"])
        self.relations.setdefault(key, {})[f"{relation_type}:{predicate}"] = entry

    def get_relations(
        self,
        source: str | None = None,
        target: str | None = None,
        relation_type: str | None = None,
        min_probability: float = 0.0,
    ) -> list[dict]:
        out = []
        for (src, dst), records in self.relations.items():
            if source is not None and src != source:
                continue
            if target is not None and dst != target:
                continue
            for record in records.values():
                if relation_type is not None and record.get("type") != relation_type:
                    continue
                if float(record.get("probability", 0.0)) < min_probability:
                    continue
                out.append(record)
        return out

    def relation_logits(self, source: str, target: str) -> dict[str, dict]:
        return dict(self.relations.get((source, target), {}))

    def fragment(self, parent_id: str, children: list[Track], t: int) -> list[Track]:
        """Mark a parent as fragmented and attach child shards with provenance.

        This is the deterministic branch of 7.4: the tracker saw the event.
        Result-only hypotheses (no event observation) need local sub-feature
        matching against a prototype bank and are deferred to Phase 7.
        """
        parent = self.get(parent_id)
        if parent is None or parent.status == "historical":
            return []
        parent.status = "fragmented"
        for child in children:
            child.parent = parent_id
            child.provenance = {
                **child.provenance,
                "fractured_from": parent_id,
                "frame": int(t),
            }
            self.add(child)
        return children

    def save(self, path: str | Path) -> None:
        payload = {"next_id": self._next, "tracks": [], "relations": []}
        for records in self.relations.values():
            payload["relations"].extend(records.values())
        for t in self.tracks.values():
            payload["tracks"].append(
                {
                    "id": t.id,
                    "pos": t.pos.tolist(),
                    "quat": t.quat.tolist(),
                    "appearance": None if t.appearance is None else t.appearance.tolist(),
                    "category": t.category,
                    "confidence": t.confidence,
                    "first_seen": t.first_seen,
                    "last_seen": t.last_seen,
                    "missed": t.missed,
                    "status": t.status,
                    "velocity": t.velocity.tolist(),
                    "raw": {k: np.asarray(v).tolist() for k, v in t.raw.items()},
                    "parent": t.parent,
                    "provenance": t.provenance,
                    "observations": t.observations,
                }
            )
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "SceneGraph":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = cls()
        graph._next = int(payload["next_id"])
        for d in payload["tracks"]:
            track = Track(
                id=d["id"],
                pos=np.asarray(d["pos"], dtype=float),
                quat=np.asarray(d["quat"], dtype=float),
                appearance=None
                if d["appearance"] is None
                else np.asarray(d["appearance"], dtype=float),
                category=d["category"],
                confidence=float(d["confidence"]),
                first_seen=int(d["first_seen"]),
                last_seen=int(d["last_seen"]),
                missed=int(d["missed"]),
                status=d["status"],
                velocity=np.asarray(d["velocity"], dtype=float),
                raw={k: np.asarray(v, dtype=float) for k, v in d["raw"].items()},
                parent=d["parent"],
                provenance=d["provenance"],
                observations=int(d["observations"]),
            )
            graph.add(track)
        for record in payload.get("relations", []):
            key = (str(record["source"]), str(record["target"]))
            relation_type = str(record.get("type", "unknown"))
            predicate = str(record["predicate"])
            graph.relations.setdefault(key, {})[f"{relation_type}:{predicate}"] = dict(record)
        return graph
