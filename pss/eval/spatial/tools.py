"""Deterministic tool layer for the B3 LLM+tools baseline.

Tool access is tiered so the comparison stays honest:

- ``low``:    pose/camera/vector primitives. The model must compose geometry.
- ``mid``:    low + inside/fits/closer. Direct predicates, still not the
              final answer for most families.
- ``oracle``: mid + relation/count/occluded/ordinal. These return the exact
              answer, so this tier is a tool-assisted upper bound, not a test
              of spatial reasoning.

No tool reads ``query.answer``. Every call is whitelisted and every argument is
validated; failures come back as ``{"ok": false, "error": ...}``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import predicates as P
from .schema import Item, ObjectSpec, Scene

PREDICATES = (
    "left_of",
    "right_of",
    "front_of",
    "behind",
    "above",
    "below",
    "contact",
    "near",
    "inside",
    "on",
)

TOOL_SPECS: dict[str, tuple[str, dict[str, str]]] = {
    "get_object": (
        "Return an object's category, center, half extents, yaw, bottom/top, radius.",
        {"object_id": "str"},
    ),
    "get_camera": (
        "Return a camera's position, look_at, and unit forward/right/up axes.",
        {"camera_id": "str"},
    ),
    "get_axes": (
        "Return the unit right/front/up axes of a frame (world, viewer, object, viewer:cam1).",
        {"frame": "str", "subject": "str (optional, required for object frame)"},
    ),
    "delta": (
        "Return the world-frame vector a.position - b.position.",
        {"a": "str", "b": "str"},
    ),
    "project": (
        "Return dot(a.position - b.position, axis) for an axis of a frame.",
        {
            "a": "str",
            "b": "str",
            "axis": "str (right,left,front,back,up,down,x,y,z)",
            "frame": "str (default world)",
            "subject": "str (optional, for object frame)",
        },
    ),
    "distance": ("Return the center distance between two objects.", {"a": "str", "b": "str"}),
    "surface_gap": (
        "Return center distance minus both radii; <= 0 means surfaces touch or overlap.",
        {"a": "str", "b": "str"},
    ),
    "horizontal_distance": (
        "Return the xy-plane center distance between two objects.",
        {"a": "str", "b": "str"},
    ),
    "z_difference": (
        "Return a.position.z - b.position.z.",
        {"a": "str", "b": "str"},
    ),
    "inside": ("Return whether a is inside container b.", {"a": "str", "b": "str"}),
    "fits": ("Return whether a would fit inside container b.", {"a": "str", "b": "str"}),
    "closer": (
        "Return whether a is closer to the camera than b.",
        {"a": "str", "b": "str", "camera": "str (default cam0)"},
    ),
    "occluded": (
        "Return whether a is occluded by b from the camera's viewpoint.",
        {"a": "str", "b": "str", "camera": "str (default cam0)"},
    ),
    "count": (
        "Count objects satisfying a predicate relative to a reference object.",
        {
            "predicate": "str",
            "reference": "str",
            "frame": "str (default world)",
        },
    ),
    "relation": (
        "Return whether a predicate holds between two objects in a frame.",
        {
            "a": "str",
            "b": "str",
            "predicate": "str",
            "frame": "str (default world)",
        },
    ),
    "ordinal": (
        "Return the k-th object of a category ordered from the left in a frame.",
        {"category": "str", "k": "int", "frame": "str (default viewer)"},
    ),
}

TOOL_PROFILES: dict[str, tuple[str, ...]] = {
    "low": (
        "get_object",
        "get_camera",
        "get_axes",
        "delta",
        "project",
        "distance",
        "surface_gap",
        "horizontal_distance",
        "z_difference",
    ),
    "mid": (
        "get_object",
        "get_camera",
        "get_axes",
        "delta",
        "project",
        "distance",
        "surface_gap",
        "horizontal_distance",
        "z_difference",
        "inside",
        "fits",
        "closer",
    ),
    "oracle": tuple(TOOL_SPECS),
}


class ToolError(ValueError):
    pass


def _num(value: float) -> float:
    return round(float(value), 4)


def _vec(values) -> list[float]:
    return [round(float(v), 4) for v in values]


def _obj(scene: Scene, oid: Any) -> ObjectSpec:
    if not isinstance(oid, str):
        raise ToolError(f"object id must be a string, got {type(oid).__name__}")
    try:
        return scene.obj(oid)
    except KeyError as exc:
        raise ToolError(str(exc)) from exc


def _camera(scene: Scene, name: Any) -> str:
    if not isinstance(name, str):
        raise ToolError("camera id must be a string")
    try:
        scene.camera(name)
    except KeyError as exc:
        raise ToolError(str(exc)) from exc
    return name


def _frame(scene: Scene, frame: Any, subject: ObjectSpec | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(frame, str):
        raise ToolError("frame must be a string")
    return P.frame_axes(scene, frame, subject)


def _axis_vector(scene: Scene, axis: str, frame: str, subject: ObjectSpec | None) -> np.ndarray:
    if axis in {"x", "y", "z"}:
        return np.array([1.0, 0.0, 0.0]) if axis == "x" else (
            np.array([0.0, 1.0, 0.0]) if axis == "y" else np.array([0.0, 0.0, 1.0])
        )
    right, front, up = _frame(scene, frame, subject)
    table = {
        "right": right,
        "left": -right,
        "front": front,
        "back": -front,
        "up": up,
        "down": -up,
    }
    if axis not in table:
        raise ToolError(f"unknown axis: {axis!r}")
    return table[axis]


def _predicate(value: Any) -> str:
    if value not in PREDICATES:
        raise ToolError(f"unknown predicate: {value!r}")
    return str(value)


def execute_tool(item: Item, name: str, args: dict[str, Any], profile: str = "mid") -> dict[str, Any]:
    """Run one whitelisted tool. Never reads ``item.query.answer``."""
    start = time.perf_counter()
    if profile not in TOOL_PROFILES:
        return {"ok": False, "error": f"unknown tool profile: {profile!r}"}
    if name not in TOOL_PROFILES[profile]:
        return {"ok": False, "error": f"tool {name!r} is not available in profile {profile!r}"}
    if not isinstance(args, dict):
        return {"ok": False, "error": "args must be an object"}
    scene = item.scene
    try:
        if name == "get_object":
            obj = _obj(scene, args.get("object_id"))
            result = {
                "id": obj.id,
                "category": obj.category,
                "position": _vec(obj.pos),
                "half_extents": _vec(obj.half),
                "yaw_deg": _num(np.degrees(obj.yaw)),
                "is_container": bool(obj.container),
                "inner_radius": _num(obj.inner_radius) if obj.container else None,
                "bottom": _num(obj.bottom),
                "top": _num(obj.top),
                "radius": _num(obj.radius),
                "height": _num(obj.height),
            }
        elif name == "get_camera":
            cam = scene.camera(_camera(scene, args.get("camera_id")))
            forward, right, up = cam.basis()
            result = {
                "id": cam.name,
                "position": _vec(cam.pos),
                "look_at": _vec(cam.target),
                "forward": _vec(forward),
                "right": _vec(right),
                "up": _vec(up),
            }
        elif name == "get_axes":
            frame = str(args.get("frame", "world"))
            subject = _obj(scene, args["subject"]) if args.get("subject") else None
            right, front, up = _frame(scene, frame, subject)
            result = {"frame": frame, "right": _vec(right), "front": _vec(front), "up": _vec(up)}
        elif name == "delta":
            a = _obj(scene, args.get("a"))
            b = _obj(scene, args.get("b"))
            result = {"delta": _vec(np.asarray(a.pos) - np.asarray(b.pos))}
        elif name == "project":
            a = _obj(scene, args.get("a"))
            b = _obj(scene, args.get("b"))
            frame = str(args.get("frame", "world"))
            subject = _obj(scene, args["subject"]) if args.get("subject") else a
            axis = str(args.get("axis", "right"))
            vector = _axis_vector(scene, axis, frame, subject)
            result = {"value": _num(np.dot(np.asarray(a.pos) - np.asarray(b.pos), vector))}
        elif name == "distance":
            result = {"distance": _num(P.center_distance(_obj(scene, args.get("a")), _obj(scene, args.get("b"))))}
        elif name == "surface_gap":
            a = _obj(scene, args.get("a"))
            b = _obj(scene, args.get("b"))
            result = {"surface_gap": _num(P.center_distance(a, b) - a.radius - b.radius)}
        elif name == "horizontal_distance":
            a = _obj(scene, args.get("a"))
            b = _obj(scene, args.get("b"))
            result = {"horizontal_distance": _num(np.linalg.norm(np.asarray(a.pos)[:2] - np.asarray(b.pos)[:2]))}
        elif name == "z_difference":
            a = _obj(scene, args.get("a"))
            b = _obj(scene, args.get("b"))
            result = {"z_difference": _num(a.pos[2] - b.pos[2])}
        elif name == "inside":
            result = {"inside": P.inside(_obj(scene, args.get("a")), _obj(scene, args.get("b")))}
        elif name == "fits":
            result = {"fits": P.fits(_obj(scene, args.get("a")), _obj(scene, args.get("b")))}
        elif name == "closer":
            result = {
                "closer": P.closer_than(
                    scene,
                    _obj(scene, args.get("a")),
                    _obj(scene, args.get("b")),
                    _camera(scene, args.get("camera", "cam0")),
                )
            }
        elif name == "occluded":
            result = {
                "occluded": P.occluded(
                    scene,
                    _obj(scene, args.get("a")),
                    _obj(scene, args.get("b")),
                    _camera(scene, args.get("camera", "cam0")),
                )
            }
        elif name == "count":
            reference = _obj(scene, args.get("reference"))
            predicate = _predicate(args.get("predicate"))
            frame = str(args.get("frame", "world"))
            if frame == "object":
                raise ToolError("count does not support the object frame; use world or viewer")
            result = {
                "count": int(P.count_relation(scene, reference, predicate, frame)),
                "reference": reference.id,
                "predicate": predicate,
                "frame": frame,
            }
        elif name == "relation":
            a = _obj(scene, args.get("a"))
            b = _obj(scene, args.get("b"))
            predicate = _predicate(args.get("predicate"))
            frame = str(args.get("frame", "world"))
            result = {
                "value": bool(P.relation(scene, a, b, predicate, frame)),
                "a": a.id,
                "b": b.id,
                "predicate": predicate,
                "frame": frame,
            }
        elif name == "ordinal":
            category = str(args.get("category"))
            k = int(args.get("k", 1))
            frame = str(args.get("frame", "viewer"))
            value = P.ordinal_from_left(scene, category, k, frame)
            result = {"object_id": value, "category": category, "k": k, "frame": frame}
        else:
            raise ToolError(f"unhandled tool: {name!r}")
    except (ToolError, KeyError, ValueError, TypeError) as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "tool": name,
            "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
        }
    return {
        "ok": True,
        "tool": name,
        "result": result,
        "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
    }


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


def parse_tool_call(text: str) -> ToolCall | None:
    """Parse the first ``TOOL: {json}`` line, tolerating markdown fences."""
    if not text:
        return None
    lower = text.lower()
    idx = lower.find("tool:")
    if idx == -1:
        return None
    rest = text[idx + len("tool:") :].strip()
    start = rest.find("{")
    if start == -1:
        return None
    try:
        payload, _end = json.JSONDecoder().raw_decode(rest[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    args = payload.get("args", {})
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return ToolCall(name=name, args=args)


def tool_manifest(profile: str) -> str:
    if profile not in TOOL_PROFILES:
        raise ValueError(f"unknown tool profile: {profile!r}")
    lines = []
    for name in TOOL_PROFILES[profile]:
        description, args = TOOL_SPECS[name]
        arg_text = ", ".join(f"{key}: {value}" for key, value in args.items())
        lines.append(f"- {name}({arg_text}): {description}")
    return "\n".join(lines)
