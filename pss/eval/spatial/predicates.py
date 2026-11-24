"""Exact geometric predicates for the spatial suite.

These are the PSS "exact" channel. They are intentionally simple and
documented: sphere/box approximations, no mesh intersection, no learned
thresholds. The point is a reproducible upper bound on relation accuracy, not
a claim of physical fidelity.

Frame conventions:

- world:  right = +x, front = +y, up = +z
- viewer: right = camera right, front = camera forward, up = camera true up
- object: right = subject's local +right (forward x up), front = subject yaw
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .schema import ObjectSpec, Scene, EPS

NEAR_DIST = 0.12
CONTACT_PAD = 0.012
ON_PAD = 0.012
INSIDE_PAD = 0.005


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > EPS else v


def frame_axes(
    scene: Scene, frame: str, subject: ObjectSpec | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (right, front, up) for a named frame."""
    if frame == "world":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    if frame == "viewer":
        forward, right, up = scene.camera("cam0").basis()
        return right, forward, up
    if frame == "object":
        if subject is None:
            raise ValueError("object frame needs a subject")
        forward = np.array([math.cos(subject.yaw), math.sin(subject.yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        right = _unit(np.cross(forward, up))
        return right, forward, up
    if frame.startswith("viewer:"):
        forward, right, up = scene.camera(frame.split(":", 1)[1]).basis()
        return right, forward, up
    raise ValueError(f"unknown frame: {frame}")


def _project(a: ObjectSpec, b: ObjectSpec, axis: np.ndarray) -> float:
    return float(np.dot(np.asarray(a.pos) - np.asarray(b.pos), axis))


def left_of(scene: Scene, a: ObjectSpec, b: ObjectSpec, frame: str = "world") -> bool:
    right, _front, _up = frame_axes(scene, frame, a)
    return _project(a, b, -right) > EPS


def right_of(scene: Scene, a: ObjectSpec, b: ObjectSpec, frame: str = "world") -> bool:
    right, _front, _up = frame_axes(scene, frame, a)
    return _project(a, b, right) > EPS


def front_of(scene: Scene, a: ObjectSpec, b: ObjectSpec, frame: str = "world") -> bool:
    _right, front, _up = frame_axes(scene, frame, a)
    return _project(a, b, front) > EPS


def behind(scene: Scene, a: ObjectSpec, b: ObjectSpec, frame: str = "world") -> bool:
    _right, front, _up = frame_axes(scene, frame, a)
    return _project(a, b, -front) > EPS


def above(a: ObjectSpec, b: ObjectSpec, eps: float = EPS) -> bool:
    return float(a.pos[2] - b.pos[2]) > eps


def below(a: ObjectSpec, b: ObjectSpec, eps: float = EPS) -> bool:
    return float(b.pos[2] - a.pos[2]) > eps


def center_distance(a: ObjectSpec, b: ObjectSpec) -> float:
    return float(np.linalg.norm(np.asarray(a.pos) - np.asarray(b.pos)))


def contact(a: ObjectSpec, b: ObjectSpec) -> bool:
    return center_distance(a, b) <= a.radius + b.radius + CONTACT_PAD


def near(a: ObjectSpec, b: ObjectSpec, threshold: float = NEAR_DIST) -> bool:
    return center_distance(a, b) < threshold


def _horizontal_distance(a: ObjectSpec, b: ObjectSpec) -> float:
    pa = np.asarray(a.pos, dtype=float)
    pb = np.asarray(b.pos, dtype=float)
    return float(np.linalg.norm(pa[:2] - pb[:2]))


def on(a: ObjectSpec, b: ObjectSpec) -> bool:
    """a rests on b (b is a support, usually a box)."""
    return abs(a.bottom - b.top) <= ON_PAD and _horizontal_distance(a, b) <= b.radius


def inside(a: ObjectSpec, b: ObjectSpec) -> bool:
    """a is inside container b."""
    if not b.container:
        return False
    return (
        _horizontal_distance(a, b) <= b.inner_radius - a.radius + INSIDE_PAD
        and a.bottom >= b.bottom - INSIDE_PAD
        and a.top <= b.top + INSIDE_PAD
    )


def fits(a: ObjectSpec, b: ObjectSpec) -> bool:
    if not b.container:
        return False
    return a.radius <= b.inner_radius + INSIDE_PAD and a.height <= b.height + INSIDE_PAD


def _point_segment_distance(p: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    seg = end - start
    denom = float(np.dot(seg, seg))
    if denom < EPS:
        return float(np.linalg.norm(p - start)), 0.0
    t = float(np.dot(p - start, seg) / denom)
    t_clamped = min(1.0, max(0.0, t))
    closest = start + t_clamped * seg
    return float(np.linalg.norm(p - closest)), t


def occluded(scene: Scene, a: ObjectSpec, b: ObjectSpec, camera: str = "cam0") -> bool:
    """a is hidden behind b from the camera (sphere approximation)."""
    cam = np.asarray(scene.camera(camera).pos, dtype=float)
    pa = np.asarray(a.pos, dtype=float)
    pb = np.asarray(b.pos, dtype=float)
    dist, t = _point_segment_distance(pb, cam, pa)
    return 0.02 < t < 0.98 and dist <= b.radius * 0.90


def closer_than(scene: Scene, a: ObjectSpec, b: ObjectSpec, camera: str = "cam0") -> bool:
    cam = np.asarray(scene.camera(camera).pos, dtype=float)
    da = float(np.linalg.norm(np.asarray(a.pos, dtype=float) - cam))
    db = float(np.linalg.norm(np.asarray(b.pos, dtype=float) - cam))
    return da < db


def relation(scene: Scene, a: ObjectSpec, b: ObjectSpec, predicate: str, frame: str = "world") -> bool:
    if predicate == "left_of":
        return left_of(scene, a, b, frame)
    if predicate == "right_of":
        return right_of(scene, a, b, frame)
    if predicate == "front_of":
        return front_of(scene, a, b, frame)
    if predicate == "behind":
        return behind(scene, a, b, frame)
    if predicate == "above":
        return above(a, b)
    if predicate == "below":
        return below(a, b)
    if predicate == "contact":
        return contact(a, b)
    if predicate == "near":
        return near(a, b)
    if predicate == "inside":
        return inside(a, b)
    if predicate == "on":
        return on(a, b)
    raise ValueError(f"unknown predicate: {predicate}")


def count_relation(
    scene: Scene,
    reference: ObjectSpec,
    predicate: str,
    frame: str = "world",
    exclude: Iterable[str] = (),
) -> int:
    skip = set(exclude) | {reference.id}
    if predicate == "inside":
        return sum(1 for obj in scene.objects if obj.id not in skip and inside(obj, reference))
    if predicate == "on":
        return sum(1 for obj in scene.objects if obj.id not in skip and on(obj, reference))
    return sum(
        1 for obj in scene.objects if obj.id not in skip and relation(scene, obj, reference, predicate, frame)
    )


def ordinal_from_left(
    scene: Scene, category: str, k: int, frame: str = "viewer"
) -> str | None:
    """Return the k-th object of ``category`` ordered left-to-right."""
    right, _front, _up = frame_axes(scene, frame)
    candidates = [obj for obj in scene.objects if obj.category == category]
    if len(candidates) < k or k < 1:
        return None
    # More left means a larger projection onto the left axis (-right).
    candidates.sort(key=lambda obj: float(np.dot(np.asarray(obj.pos), -right)), reverse=True)
    return candidates[k - 1].id


PREDICATE_PHRASES = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "front_of": "in front of",
    "behind": "behind",
    "above": "above",
    "below": "below",
    "contact": "touching",
    "near": "near",
    "inside": "inside",
    "on": "on top of",
}


FRAME_PHRASES = {
    "world": "in the world frame (+x right, +y away from the viewer)",
    "viewer": "from the viewer's perspective (cam0)",
    "object": "in the subject's own frame",
}
