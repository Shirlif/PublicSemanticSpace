"""Minimal 3D scene model for the scene-model benchmark.

The model is deliberately small: object centres, yaw, a radius for the
support/collision queries, and an optional covariance.  It is the latent state
that all observations constrain; relation queries are evaluated on this state,
not on a projected pair feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIZE_RADIUS = {"small": 0.35, "large": 0.70}


@dataclass
class Camera:
    """Pinhole camera with a world->camera transform."""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int = 320
    height: int = 240
    name: str = "cam"

    @classmethod
    def look_at(
        cls,
        eye,
        target,
        up=(0.0, 0.0, 1.0),
        focal: float = 400.0,
        width: int = 320,
        height: int = 240,
        name: str = "cam",
    ) -> "Camera":
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)
        forward = target - eye
        forward /= max(np.linalg.norm(forward), 1e-12)
        right = np.cross(forward, up)
        right /= max(np.linalg.norm(right), 1e-12)
        true_up = np.cross(right, forward)
        R = np.stack([right, true_up, forward], axis=0)
        t = -R @ eye
        K = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return cls(K=K, R=R, t=t, width=width, height=height, name=name)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        cam = points @ self.R.T + self.t
        z = cam[:, 2]
        uv = (cam @ self.K.T)[:, :2] / np.maximum(z[:, None], 1e-12)
        return uv, z

    def to_dict(self) -> dict:
        return {
            "K": self.K.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "width": int(self.width),
            "height": int(self.height),
            "name": self.name,
        }


@dataclass
class ObjectState:
    id: str
    pos: np.ndarray
    yaw: float = 0.0
    shape: str = "sphere"
    size: str = "small"
    radius: float = 0.35
    covariance: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)

    @classmethod
    def from_clevr(cls, obj: dict, obj_id: str) -> "ObjectState":
        size = str(obj.get("size", "small"))
        radius = SIZE_RADIUS.get(size, 0.35)
        return cls(
            id=obj_id,
            pos=np.asarray(obj.get("3d_coords", [0.0, 0.0, 0.0]), dtype=np.float64),
            yaw=float(obj.get("rotation", 0.0)) * np.pi / 180.0,
            shape=str(obj.get("shape", "sphere")),
            size=size,
            radius=radius,
            provenance={"source": "clevr_scene", "shape": obj.get("shape")},
        )


@dataclass
class SceneModel:
    scene_id: str
    objects: dict[str, ObjectState]
    table_z: float = 0.0

    def positions(self) -> dict[str, np.ndarray]:
        return {oid: np.asarray(obj.pos, dtype=np.float64) for oid, obj in self.objects.items()}

    def radii(self) -> dict[str, float]:
        return {oid: float(obj.radius) for oid, obj in self.objects.items()}

    def center(self) -> np.ndarray:
        if not self.objects:
            return np.zeros(3)
        return np.mean([obj.pos for obj in self.objects.values()], axis=0)

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "table_z": float(self.table_z),
            "objects": {
                oid: {
                    "pos": [float(x) for x in obj.pos],
                    "yaw": float(obj.yaw),
                    "shape": obj.shape,
                    "size": obj.size,
                    "radius": float(obj.radius),
                }
                for oid, obj in self.objects.items()
            },
        }


@dataclass
class Observation:
    object_id: str
    camera: Camera
    uv: np.ndarray
    depth: float
    weight: float = 1.0
    outlier: bool = False


__all__ = ["Camera", "ObjectState", "Observation", "SceneModel", "SIZE_RADIUS"]
