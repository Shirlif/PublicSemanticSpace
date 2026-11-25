"""Scripted camera trajectories (ADR-001 / T44).

Camera motion lives in the spec and is hash-frozen.  It is the *only* source of
viewpoint change for the perception/memory suites; no walking policy is
involved.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def _quat_slerp(a: Sequence[float], b: Sequence[float], alpha: float) -> list[float]:
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    dot = aw * bw + ax * bx + ay * by + az * bz
    if dot < 0.0:
        bw, bx, by, bz = -bw, -bx, -by, -bz
        dot = -dot
    if dot > 0.9995:
        out = [aw + alpha * (bw - aw), ax + alpha * (bx - ax),
               ay + alpha * (by - ay), az + alpha * (bz - az)]
    else:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta = math.sin(theta)
        wa = math.sin((1.0 - alpha) * theta) / sin_theta
        wb = math.sin(alpha * theta) / sin_theta
        out = [wa * aw + wb * bw, wa * ax + wb * bx,
               wa * ay + wb * by, wa * az + wb * bz]
    norm = math.sqrt(sum(v * v for v in out))
    return [v / norm for v in out] if norm > 0 else [1.0, 0.0, 0.0, 0.0]


def interpolate_pose(motion: Sequence[Mapping], t_s: float, default_pose: Sequence[float] | None = None) -> list[float]:
    base = list(default_pose or [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    if not motion:
        return base
    keys = sorted(motion, key=lambda k: float(k["t_s"]))
    first = keys[0]
    if t_s <= float(first["t_s"]):
        pos = [float(v) for v in first["pos"]]
        ori = [float(v) for v in first.get("ori", base[3:7])]
        return pos + ori
    last = keys[-1]
    if t_s >= float(last["t_s"]):
        pos = [float(v) for v in last["pos"]]
        ori = [float(v) for v in last.get("ori", base[3:7])]
        return pos + ori
    for a, b in zip(keys, keys[1:]):
        t0, t1 = float(a["t_s"]), float(b["t_s"])
        if t0 <= t_s <= t1:
            alpha = 0.0 if t1 == t0 else (t_s - t0) / (t1 - t0)
            pa = [float(v) for v in a["pos"]]
            pb = [float(v) for v in b["pos"]]
            pos = [pa[i] + (pb[i] - pa[i]) * alpha for i in range(3)]
            ori = _quat_slerp(
                a.get("ori", base[3:7]), b.get("ori", base[3:7]), alpha
            )
            return pos + ori
    return base


def camera_pose(camera: Mapping, t_s: float) -> list[float]:
    base = list(camera.get("pose") or [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    return interpolate_pose(camera.get("motion") or [], t_s, default_pose=base)
