"""Spatial suite schema: scenes, cameras, queries, and leakage-free serialization.

The suite is deliberately numpy-only. It does not import MuJoCo or torch, so it
runs on any machine and stays independent of the perception stack.

World frame convention (documented in every serialized scene):

    +x = right, +y = away from the viewer, +z = up, units = meters.

A scene contains object poses and sizes plus camera poses. It never contains
pairwise relations, distances, or derived answers; those are computed by
``predicates.py`` and live only in the query's ``answer`` field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import numpy as np

EPS = 1e-6


@dataclass(frozen=True)
class Camera:
    name: str
    pos: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (forward, right, true_up) unit vectors."""
        forward = np.asarray(self.target, dtype=float) - np.asarray(self.pos, dtype=float)
        norm = float(np.linalg.norm(forward))
        if norm < EPS:
            raise ValueError(f"camera {self.name} has zero-length look direction")
        forward = forward / norm
        up = np.asarray(self.up, dtype=float)
        right = np.cross(forward, up)
        rnorm = float(np.linalg.norm(right))
        if rnorm < EPS:
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
            rnorm = float(np.linalg.norm(right))
        right = right / rnorm
        true_up = np.cross(right, forward)
        return forward, right, true_up


@dataclass(frozen=True)
class ObjectSpec:
    id: str
    category: str
    pos: tuple[float, float, float]
    half: tuple[float, float, float]
    yaw: float = 0.0  # radians, object-frame rotation around +z
    container: bool = False
    inner_radius: float = 0.0

    @property
    def radius(self) -> float:
        return float(max(self.half[0], self.half[1]))

    @property
    def height(self) -> float:
        return float(self.half[2] * 2.0)

    @property
    def bottom(self) -> float:
        return float(self.pos[2] - self.half[2])

    @property
    def top(self) -> float:
        return float(self.pos[2] + self.half[2])

    def moved(self, pos: Iterable[float]) -> "ObjectSpec":
        return replace(self, pos=tuple(float(v) for v in pos))


@dataclass(frozen=True)
class Scene:
    scene_id: str
    objects: tuple[ObjectSpec, ...]
    cameras: tuple[Camera, ...]
    table: tuple[float, float, float, float]  # x_min, x_max, y_min, y_max
    split: str = "train"
    ood: str = "id"

    def obj(self, oid: str) -> ObjectSpec:
        for item in self.objects:
            if item.id == oid:
                return item
        raise KeyError(f"scene {self.scene_id} has no object {oid!r}")

    def camera(self, name: str) -> Camera:
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise KeyError(f"scene {self.scene_id} has no camera {name!r}")

    def with_objects(self, objects: Iterable[ObjectSpec]) -> "Scene":
        return replace(self, objects=tuple(objects))


@dataclass(frozen=True)
class Query:
    query_id: str
    family: str
    kind: str  # "bool" | "count" | "object"
    question: str
    answer: str
    answer_format: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Item:
    scene: Scene
    query: Query

    def to_record(self) -> dict[str, Any]:
        return {
            "split": self.scene.split,
            "ood": self.scene.ood,
            "scene": scene_payload(self.scene),
            "query": {
                "query_id": self.query.query_id,
                "family": self.query.family,
                "kind": self.query.kind,
                "question": self.query.question,
                "answer": self.query.answer,
                "answer_format": self.query.answer_format,
                "meta": self.query.meta,
            },
        }


def _r4(value: float) -> float:
    """Round serialized numbers; full float64 noise is wasted tokens."""
    return round(float(value), 4)


def scene_payload(scene: Scene) -> dict[str, Any]:
    """Leakage-free scene JSON. No relations, distances, or answers."""
    return {
        "units": "meters",
        "world_frame": {
            "x": "right",
            "y": "away_from_viewer",
            "z": "up",
            "note": "positions are object centers; half_extents are per axis",
        },
        "table": {
            "x_min": _r4(scene.table[0]),
            "x_max": _r4(scene.table[1]),
            "y_min": _r4(scene.table[2]),
            "y_max": _r4(scene.table[3]),
            "z": 0.0,
        },
        "cameras": [
            {
                "id": cam.name,
                "position": [_r4(v) for v in cam.pos],
                "look_at": [_r4(v) for v in cam.target],
                "up": [_r4(v) for v in cam.up],
            }
            for cam in scene.cameras
        ],
        "objects": [
            {
                "id": obj.id,
                "category": obj.category,
                "position": [_r4(v) for v in obj.pos],
                "half_extents": [_r4(v) for v in obj.half],
                "yaw_deg": _r4(np.degrees(obj.yaw)),
                "is_container": bool(obj.container),
                **({"inner_radius": _r4(obj.inner_radius)} if obj.container else {}),
            }
            for obj in scene.objects
        ],
    }


def scene_to_json(scene: Scene, indent: int | None = None) -> str:
    return json.dumps(scene_payload(scene), ensure_ascii=False, sort_keys=True, indent=indent)


def answer_instruction(kind: str) -> str:
    if kind == "bool":
        return "Answer with exactly one word: yes or no."
    if kind == "count":
        return "Answer with a single integer."
    if kind == "object":
        return "Answer with exactly one object id, for example obj_2."
    raise ValueError(f"unknown answer kind: {kind}")


def render_prompt(item: Item, few_shot: Iterable[Item] = ()) -> str:
    """Render one prompt. The scene is serialized fresh for every item."""
    parts = [
        "You are answering a spatial question about a tabletop scene.",
        "All positions are in meters in the world frame: +x right, +y away from the viewer, +z up.",
        "Object positions are centers; half_extents are per axis. Cameras are given by position and look_at.",
        "Frame definitions:",
        "- world: right = +x, front = +y, up = +z.",
        "- viewer: right/forward/up are the camera's own axes (cam0 unless another camera is named).",
        "- object: forward = (cos(yaw), sin(yaw), 0), right = forward x up.",
        "- 'A left_of B' means A lies on the left side of B; 'front_of' uses the frame's forward axis.",
        "",
        "Scene:",
        scene_to_json(item.scene),
    ]
    examples = list(few_shot)
    if examples:
        parts += ["", "Examples:"]
        for ex in examples:
            parts += [
                f"Question: {ex.query.question}",
                f"Answer: {ex.query.answer}",
            ]
    parts += [
        "",
        f"Question: {item.query.question}",
        item.query.answer_format or answer_instruction(item.query.kind),
    ]
    return "\n".join(parts)


def parse_answer(raw: str, kind: str) -> str | None:
    """Strict answer parser. Returns None for anything ambiguous."""
    if raw is None:
        return None
    text = raw.strip().lower()
    if not text:
        return None
    if kind == "bool":
        tokens = text.replace(".", " ").replace(",", " ").split()
        if not tokens:
            return None
        first = tokens[0]
        if first in {"yes", "true"}:
            return "yes"
        if first in {"no", "false"}:
            return "no"
        return None
    if kind == "count":
        tokens = text.replace(".", " ").replace(",", " ").split()
        if len(tokens) != 1:
            return None
        try:
            return str(int(tokens[0]))
        except ValueError:
            return None
    if kind == "object":
        tokens = text.replace(".", " ").replace(",", " ").split()
        if len(tokens) != 1 or not tokens[0].startswith("obj_"):
            return None
        return tokens[0]
    return None


def parse_final_answer(raw: str, kind: str) -> str | None:
    """Parse a CoT answer that ends with ``Answer: ...``.

    Falls back to the strict parser when no marker is present, so the same
    function works for zero-shot and chain-of-thought prompts.
    """
    if raw is None:
        return None
    marker = raw.lower().rfind("answer:")
    if marker == -1:
        return parse_answer(raw, kind)
    return parse_answer(raw[marker + len("answer:") :], kind)
