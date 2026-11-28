"""Multi-view observation simulator for the scene benchmark.

The simulator is deliberately controlled: a ring of virtual pinhole cameras
around the scene, Gaussian pixel/depth noise, optional camera-pose noise and
outliers.  It is a perception-front-end proxy, not a renderer; the CLEVR
``3d_coords`` are the ground truth the estimator must recover.
"""

from __future__ import annotations

import numpy as np

from .model import Camera, Observation, SceneModel


def make_cameras(
    scene: SceneModel,
    n_views: int,
    *,
    radius: float = 10.0,
    height: float = 6.0,
    focal: float = 400.0,
    width: int = 320,
    height_px: int = 240,
    start_angle: float = 0.0,
) -> list[Camera]:
    center = scene.center()
    target = np.array([center[0], center[1], scene.table_z + 0.5], dtype=np.float64)
    cameras = []
    for k in range(int(n_views)):
        theta = start_angle + 2.0 * np.pi * k / max(int(n_views), 1)
        eye = np.array(
            [
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
                scene.table_z + height,
            ],
            dtype=np.float64,
        )
        cameras.append(
            Camera.look_at(
                eye, target, focal=focal, width=width, height=height_px,
                name=f"view{k}",
            )
        )
    return cameras


def _rotation(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return Ry @ Rx


def perturb_camera(
    camera: Camera,
    *,
    yaw_sigma: float,
    pitch_sigma: float,
    rng: np.random.Generator,
) -> Camera:
    if yaw_sigma <= 0 and pitch_sigma <= 0:
        return camera
    yaw = rng.normal(0.0, yaw_sigma)
    pitch = rng.normal(0.0, pitch_sigma)
    R = _rotation(yaw, pitch) @ camera.R
    return Camera(
        K=camera.K.copy(), R=R, t=camera.t.copy(),
        width=camera.width, height=camera.height, name=camera.name,
    )


def observe(
    scene: SceneModel,
    cameras: list[Camera],
    *,
    sigma_px: float = 2.0,
    sigma_depth: float = 0.01,
    sigma_pose_yaw: float = 0.0,
    sigma_pose_pitch: float = 0.0,
    outlier_frac: float = 0.0,
    rng: np.random.Generator | None = None,
    margin: float = 8.0,
) -> list[Observation]:
    rng = np.random.default_rng(0) if rng is None else rng
    observations: list[Observation] = []
    ids = list(scene.objects)
    points = np.stack([scene.objects[oid].pos for oid in ids], axis=0)
    for camera in cameras:
        noisy_cam = perturb_camera(
            camera,
            yaw_sigma=sigma_pose_yaw,
            pitch_sigma=sigma_pose_pitch,
            rng=rng,
        )
        uv, z = noisy_cam.project(points)
        for oid, (u, v), depth in zip(ids, uv, z):
            if depth <= 0.1:
                continue
            if not (
                -margin <= u <= noisy_cam.width + margin
                and -margin <= v <= noisy_cam.height + margin
            ):
                continue
            outlier = bool(rng.random() < outlier_frac)
            u_obs = float(u + rng.normal(0.0, sigma_px))
            v_obs = float(v + rng.normal(0.0, sigma_px))
            d_obs = float(depth * (1.0 + rng.normal(0.0, sigma_depth)))
            if outlier:
                # Large pixel offset and depth bias; the robust loss should
                # down-weight it.
                u_obs += float(rng.choice([-1.0, 1.0]) * rng.uniform(20.0, 40.0))
                v_obs += float(rng.choice([-1.0, 1.0]) * rng.uniform(20.0, 40.0))
                d_obs *= float(rng.uniform(0.8, 1.2))
            observations.append(
                Observation(
                    object_id=oid,
                    camera=noisy_cam,
                    uv=np.array([u_obs, v_obs], dtype=np.float64),
                    depth=d_obs,
                    weight=0.2 if outlier else 1.0,
                    outlier=outlier,
                )
            )
    return observations


def backproject(camera: Camera, uv: np.ndarray, depth: float) -> np.ndarray:
    """Backproject a pixel at a given camera-frame depth to world coordinates."""

    uv = np.asarray(uv, dtype=np.float64)
    ray_cam = np.linalg.solve(camera.K, np.array([uv[0], uv[1], 1.0]))
    point_cam = ray_cam * float(depth)
    return camera.R.T @ (point_cam - camera.t)


__all__ = ["backproject", "make_cameras", "observe", "perturb_camera"]
