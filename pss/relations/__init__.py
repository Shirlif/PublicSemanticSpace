"""Geometric relation GT (T24).

Relations are computed only from GT poses/boxes, never from PSS state.  The
same predicate functions are used by the evaluation side to score PSS's
relation inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class RelObject:
    name: str
    pose: Sequence[float]  # xyz + quat
    size: Sequence[float]  # full extents

    @property
    def center(self) -> Vec3:
        return (float(self.pose[0]), float(self.pose[1]), float(self.pose[2]))

    @property
    def half(self) -> Vec3:
        return (
            float(self.size[0]) / 2.0,
            float(self.size[1]) / 2.0,
            float(self.size[2]) / 2.0,
        )


def _aabb(obj: RelObject) -> tuple[Vec3, Vec3]:
    c, h = obj.center, obj.half
    return (
        (c[0] - h[0], c[1] - h[1], c[2] - h[2]),
        (c[0] + h[0], c[1] + h[1], c[2] + h[2]),
    )


def _aabb_gap(a: RelObject, b: RelObject) -> float:
    amin, amax = _aabb(a)
    bmin, bmax = _aabb(b)
    gaps = []
    for i in range(3):
        gap = max(0.0, amin[i] - bmax[i], bmin[i] - amax[i])
        gaps.append(gap)
    return math.sqrt(sum(g * g for g in gaps))


def _xy_overlap(a: RelObject, b: RelObject) -> bool:
    amin, amax = _aabb(a)
    bmin, bmax = _aabb(b)
    return amin[0] <= bmax[0] and bmin[0] <= amax[0] and amin[1] <= bmax[1] and bmin[1] <= amax[1]


def _inside(a: RelObject, b: RelObject) -> bool:
    amin, amax = _aabb(a)
    bmin, bmax = _aabb(b)
    return all(amin[i] >= bmin[i] - 1e-9 and amax[i] <= bmax[i] + 1e-9 for i in range(3))


def relation_holds(
    subj: RelObject,
    pred: str,
    obj: RelObject,
    *,
    tolerance_m: float = 0.02,
    near_m: float = 0.10,
) -> bool:
    a, b = subj.center, obj.center
    tol = tolerance_m
    if pred == "left_of":
        return a[0] < b[0] - tol
    if pred == "right_of":
        return a[0] > b[0] + tol
    if pred == "in_front_of":
        return a[1] < b[1] - tol
    if pred == "behind":
        return a[1] > b[1] + tol
    if pred == "above":
        return a[2] > b[2] + tol
    if pred == "below":
        return a[2] < b[2] - tol
    if pred == "near":
        return math.dist(a, b) <= near_m
    if pred == "touching":
        return _aabb_gap(subj, obj) <= tol
    if pred == "on_top_of":
        subj_min, _ = _aabb(subj)
        _, obj_max = _aabb(obj)
        return (
            a[2] > b[2]
            and _xy_overlap(subj, obj)
            and abs(subj_min[2] - obj_max[2]) <= tol
        )
    if pred == "inside":
        return _inside(subj, obj)
    raise ValueError(f"unknown predicate {pred!r}")


def compute_relations(
    objects: Iterable[RelObject],
    predicates: Sequence[str],
    *,
    tolerance_m: float = 0.02,
    near_m: float = 0.10,
) -> list[tuple[str, str, str]]:
    items = list(objects)
    triples: list[tuple[str, str, str]] = []
    for subj in items:
        for obj in items:
            if subj.name == obj.name:
                continue
            for pred in predicates:
                if relation_holds(subj, pred, obj, tolerance_m=tolerance_m, near_m=near_m):
                    triples.append((subj.name, pred, obj.name))
    return triples


def format_triples(triples: Iterable[tuple[str, str, str]]) -> list[str]:
    return [f"{pred}({obj})" for subj, pred, obj in triples]
