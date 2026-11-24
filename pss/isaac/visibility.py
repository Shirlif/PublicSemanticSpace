"""Analytic visibility sampling for GT export (W1).

Visibility is the fraction of sample points on an object that are not blocked
by another object from the camera.  For W1 every object is a box, so ray/box
intersection gives an exact, deterministic answer without a renderer or PhysX
query -- which matters on 8 GB where every extra render pass steals memory
from the simulation.

W2 replaces this with renderer/PhysX-based occlusion once the G1 arm and the
real occlusion faults arrive; the GT column contract stays the same.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # w, x, y, z


@dataclass(frozen=True)
class Box:
    name: str
    pose: Sequence[float]  # xyz + quat(wxyz)
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

    @property
    def quat(self) -> Quat:
        if len(self.pose) >= 7:
            return (
                float(self.pose[3]),
                float(self.pose[4]),
                float(self.pose[5]),
                float(self.pose[6]),
            )
        return (1.0, 0.0, 0.0, 0.0)


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    w, x, y, z = q
    vx, vy, vz = v
    # v' = v + 2 * cross(q_vec, cross(q_vec, v) + w * v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def quat_inverse(q: Quat) -> Quat:
    w, x, y, z = q
    return (w, -x, -y, -z)


def transform_point(pose: Sequence[float], local: Vec3) -> Vec3:
    q = (float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])) if len(pose) >= 7 else (1.0, 0.0, 0.0, 0.0)
    rotated = quat_rotate(q, local)
    return (
        float(pose[0]) + rotated[0],
        float(pose[1]) + rotated[1],
        float(pose[2]) + rotated[2],
    )


def ray_box_t(origin: Vec3, direction: Vec3, box: Box) -> float | None:
    """Ray/box intersection in box-local space (slab method).

    ``direction`` must be normalized.  Returns the near hit distance, or None.
    """
    inv = quat_inverse(box.quat)
    rel = (
        origin[0] - box.center[0],
        origin[1] - box.center[1],
        origin[2] - box.center[2],
    )
    o = quat_rotate(inv, rel)
    d = quat_rotate(inv, direction)  # rotation preserves length
    hx, hy, hz = box.half

    t_near = -math.inf
    t_far = math.inf
    for oi, di, hi in ((o[0], d[0], hx), (o[1], d[1], hy), (o[2], d[2], hz)):
        if abs(di) < 1e-12:
            if oi < -hi or oi > hi:
                return None
            continue
        t1 = (-hi - oi) / di
        t2 = (hi - oi) / di
        if t1 > t2:
            t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)
        if t_near > t_far:
            return None
    if t_far < 0:
        return None
    return max(t_near, 0.0)


def _sample_points(box: Box, per_axis: int = 3) -> list[Vec3]:
    """Grid of points in the box volume, in world coordinates."""
    hx, hy, hz = box.half
    points: list[Vec3] = []
    for ix in range(per_axis):
        for iy in range(per_axis):
            for iz in range(per_axis):
                fx = -hx + (2 * hx) * (ix / (per_axis - 1) if per_axis > 1 else 0.5)
                fy = -hy + (2 * hy) * (iy / (per_axis - 1) if per_axis > 1 else 0.5)
                fz = -hz + (2 * hz) * (iz / (per_axis - 1) if per_axis > 1 else 0.5)
                points.append(transform_point(box.pose, (fx, fy, fz)))
    return points


def in_frustum(
    direction: Vec3,
    camera_pose: Sequence[float],
    fov_deg: float,
    aspect: float,
) -> bool:
    """Whether a world-space ray direction falls inside the camera frustum."""
    q = (
        (float(camera_pose[3]), float(camera_pose[4]), float(camera_pose[5]), float(camera_pose[6]))
        if len(camera_pose) >= 7
        else (1.0, 0.0, 0.0, 0.0)
    )
    local = quat_rotate(quat_inverse(q), direction)
    # USD camera looks down -Z; horizontal is X, vertical is Y.
    if local[2] >= -1e-9:
        return False
    half_v = math.radians(fov_deg) / 2.0
    half_h = math.atan(math.tan(half_v) * aspect)
    angle_h = math.atan2(abs(local[0]), -local[2])
    angle_v = math.atan2(abs(local[1]), -local[2])
    return angle_h <= half_h and angle_v <= half_v


def compute_visibility(
    objects: Iterable[Box],
    camera_pose: Sequence[float],
    *,
    fov_deg: float = 60.0,
    aspect: float = 4.0 / 3.0,
    per_axis: int = 3,
) -> dict[str, dict[str, object]]:
    """Return ``{instance_id: {visibility_ratio, occluded_by, occluder_of}}``."""
    obj_list = list(objects)
    cam_pos = (
        (float(camera_pose[0]), float(camera_pose[1]), float(camera_pose[2]))
        if len(camera_pose) >= 3
        else (0.0, 0.0, 0.0)
    )
    result: dict[str, dict[str, object]] = {}
    occluder_of: dict[str, set[str]] = {o.name: set() for o in obj_list}

    for target in obj_list:
        points = _sample_points(target, per_axis=per_axis)
        visible = 0
        blocked_by: set[str] = set()
        for p in points:
            delta = (p[0] - cam_pos[0], p[1] - cam_pos[1], p[2] - cam_pos[2])
            dist = math.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)
            if dist < 1e-9:
                continue
            direction = (delta[0] / dist, delta[1] / dist, delta[2] / dist)
            if not in_frustum(direction, camera_pose, fov_deg, aspect):
                continue
            blocked = False
            for other in obj_list:
                if other.name == target.name:
                    continue
                t_hit = ray_box_t(cam_pos, direction, other)
                if t_hit is not None and t_hit < dist - 1e-6:
                    blocked = True
                    blocked_by.add(other.name)
            if not blocked:
                visible += 1
        count = len(points)
        result[target.name] = {
            "visibility_ratio": visible / count if count else 0.0,
            "occluded_by": sorted(blocked_by),
            "occluder_of": [],
        }
        for name in blocked_by:
            occluder_of[name].add(target.name)

    for name, targets in occluder_of.items():
        result[name]["occluder_of"] = sorted(targets)
    return result
