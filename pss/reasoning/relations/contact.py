"""Collision-geometry contact tests for the Phase 7 relation layer.

The old relation layer used a bounding sphere per object, which misses
contacts between elongated or flat shapes (capsules, cloth, sheets). This
module uses the object's actual MuJoCo geom type and half-extents:

- box: oriented bounding box (SAT);
- sphere: sphere;
- cylinder: capsule with the body's local z as its axis;
- capsule: capsule with the body's local x as its axis (the scene XML lays
  capsules along x);
- ellipsoid: sphere with the largest half-extent (coarse approximation).

The API is pure numpy so it can run on stored samples without MuJoCo.
"""

from __future__ import annotations

import numpy as np


def _quat_matrix(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n == 0:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _point_box_distance(p, center, rot, half, samples: int = 1) -> float:
    local = rot.T @ (np.asarray(p, dtype=float) - np.asarray(center, dtype=float))
    half = np.asarray(half, dtype=float)
    clamped = np.clip(local, -half, half)
    return float(np.linalg.norm(local - clamped))


def _segment_box_distance(p0, p1, center, rot, half, samples: int = 9) -> float:
    ts = np.linspace(0.0, 1.0, max(2, int(samples)))
    points = np.asarray(p0)[None, :] + ts[:, None] * (np.asarray(p1) - np.asarray(p0))[None, :]
    return float(min(_point_box_distance(p, center, rot, half) for p in points))


def _segment_segment_distance(p0, p1, q0, q1) -> float:
    """Closest distance between two 3D segments (Ericson, Real-Time Collision)."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p0 - q0))
    if a <= eps:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    c1 = p0 + s * d1
    c2 = q0 + t * d2
    return float(np.linalg.norm(c1 - c2))


def _obb_overlap(a_c, a_rot, a_half, b_c, b_rot, b_half, margin: float) -> bool:
    d = np.asarray(b_c) - np.asarray(a_c)
    axes = [a_rot[:, i] for i in range(3)] + [b_rot[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            axis = np.cross(a_rot[:, i], b_rot[:, j])
            norm = float(np.linalg.norm(axis))
            if norm > 1e-8:
                axes.append(axis / norm)
    for axis in axes:
        ra = sum(float(a_half[k]) * abs(float(np.dot(axis, a_rot[:, k]))) for k in range(3))
        rb = sum(float(b_half[k]) * abs(float(np.dot(axis, b_rot[:, k]))) for k in range(3))
        if abs(float(np.dot(d, axis))) > ra + rb + margin:
            return False
    return True


def _geometry(spec: dict) -> dict:
    shape = str(spec.get("geom_type", "box"))
    size = np.asarray(spec.get("size", [0.05, 0.05, 0.05]), dtype=float)
    if shape == "sphere":
        return {"kind": "sphere", "radius": float(size[0])}
    if shape == "cylinder":
        return {"kind": "capsule", "radius": float(size[0]), "half_length": float(size[1]), "axis": 2}
    if shape == "capsule":
        return {"kind": "capsule", "radius": float(size[0]), "half_length": float(size[1]), "axis": 0}
    if shape == "ellipsoid":
        # Volume-equivalent sphere. Using max(size) over-approximates and
        # creates contact false positives for elongated eggs.
        return {"kind": "sphere", "radius": float(np.cbrt(np.prod(size[:3])))}
    return {"kind": "box", "half": size[:3].copy()}


def _capsule_segment(center, rot, geom) -> tuple[np.ndarray, np.ndarray]:
    axis = rot[:, int(geom["axis"])]
    offset = axis * float(geom["half_length"])
    center = np.asarray(center, dtype=float)
    return center - offset, center + offset


def _contact(a_geom, a_c, a_rot, b_geom, b_c, b_rot, margin: float) -> bool:
    ka, kb = a_geom["kind"], b_geom["kind"]
    if ka == "box" and kb == "box":
        return _obb_overlap(a_c, a_rot, a_geom["half"], b_c, b_rot, b_geom["half"], margin)

    if ka == "sphere" and kb == "sphere":
        return float(np.linalg.norm(a_c - b_c)) <= a_geom["radius"] + b_geom["radius"] + margin

    if ka == "sphere" and kb == "box":
        return _point_box_distance(a_c, b_c, b_rot, b_geom["half"]) <= a_geom["radius"] + margin
    if ka == "box" and kb == "sphere":
        return _point_box_distance(b_c, a_c, a_rot, a_geom["half"]) <= b_geom["radius"] + margin

    if ka == "capsule" and kb == "capsule":
        a0, a1 = _capsule_segment(a_c, a_rot, a_geom)
        b0, b1 = _capsule_segment(b_c, b_rot, b_geom)
        return _segment_segment_distance(a0, a1, b0, b1) <= (
            a_geom["radius"] + b_geom["radius"] + margin
        )

    if ka == "capsule" and kb == "box":
        a0, a1 = _capsule_segment(a_c, a_rot, a_geom)
        return _segment_box_distance(a0, a1, b_c, b_rot, b_geom["half"]) <= (
            a_geom["radius"] + margin
        )
    if ka == "box" and kb == "capsule":
        b0, b1 = _capsule_segment(b_c, b_rot, b_geom)
        return _segment_box_distance(b0, b1, a_c, a_rot, a_geom["half"]) <= (
            b_geom["radius"] + margin
        )

    if ka == "sphere" and kb == "capsule":
        b0, b1 = _capsule_segment(b_c, b_rot, b_geom)
        return _point_segment_distance(a_c, b0, b1) <= a_geom["radius"] + b_geom["radius"] + margin
    if ka == "capsule" and kb == "sphere":
        a0, a1 = _capsule_segment(a_c, a_rot, a_geom)
        return _point_segment_distance(b_c, a0, a1) <= a_geom["radius"] + b_geom["radius"] + margin

    # Conservative fallback: bounding spheres.
    ra = float(np.max(a_geom.get("half", [a_geom.get("radius", 0.05)])))
    rb = float(np.max(b_geom.get("half", [b_geom.get("radius", 0.05)])))
    return float(np.linalg.norm(a_c - b_c)) <= ra + rb + margin


def _point_segment_distance(p, a, b) -> float:
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def contact_pairs(
    object_specs: list[dict],
    poses,
    margin: float = 1e-5,
) -> list[tuple[int, int]]:
    """Return contact pairs for one frame.

    ``poses`` is an (N, 7) array or a sequence of [x, y, z, qw, qx, qy, qz].
    """
    n = len(object_specs)
    poses = np.asarray(poses, dtype=float)
    geoms = [_geometry(spec) for spec in object_specs]
    rotations = [_quat_matrix(poses[i, 3:7]) for i in range(n)]
    centers = [poses[i, :3] for i in range(n)]
    out: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if _contact(
                geoms[i],
                centers[i],
                rotations[i],
                geoms[j],
                centers[j],
                rotations[j],
                margin,
            ):
                out.append((i, j))
    return out
