"""Bridge the MuJoCo tabletop simulator to the scene benchmark.

The previous scene benchmark projected known CLEVR ``3d_coords`` through
virtual pinholes.  That tests an estimator, but it is not a 3D environment:
there are no objects, no rendering, no occlusion, no physics.  This module uses
the repo's existing MuJoCo tabletop scene (``pss/sim``) and exposes:

- a true camera model (intrinsics + extrinsics) extracted from the renderer;
- RGB / depth / instance segmentation;
- per-object masks, centroids, bbox extents and world point clouds;
- ground-truth 3D poses from the physics state.

The benchmark's 3D arms can then use the **same rendered observations** as the
1D arms, instead of a synthetic projection of the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pss.sim.scene import Scene, SceneConfig, build_scene
from pss.sim.sensors import SensorRig

from .model import Camera, ObjectState, SceneModel
from .observation import backproject


def _radius_for(geom: str, size: tuple) -> float:
    size = tuple(float(v) for v in size)
    if geom == "sphere":
        return float(size[0])
    if geom == "box":
        return float(max(size))
    if geom == "cylinder":
        return float(max(size[0], 2.0 * size[1]))
    if geom == "capsule":
        return float(size[0] + size[1])
    if geom == "ellipsoid":
        return float(max(size))
    return float(max(size))


def _camera_from_rig(rig: SensorRig) -> Camera:
    """Extract a pinhole camera from the MuJoCo free camera.

    ``mjv_updateCamera`` is called by ``update_scene``; after that
    ``renderer.scene.camera[0]`` holds the camera position/forward/up used for
    the render.  Image y is down, so the camera's world->image R row 1 is the
    negative of the renderer's "up".
    """

    rig.renderer.update_scene(rig.scene.data, camera=rig.cam)
    cam = rig.renderer.scene.camera[0]
    pos = np.asarray(cam.pos, dtype=np.float64)
    forward = np.asarray(cam.forward, dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1e-12)
    up = np.asarray(cam.up, dtype=np.float64)
    up /= max(np.linalg.norm(up), 1e-12)
    right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-12)
    up2 = np.cross(right, forward)
    fovy = float(np.deg2rad(rig.scene.model.vis.global_.fovy))
    f = float(rig.height) / (2.0 * max(np.tan(fovy / 2.0), 1e-6))
    K = np.array(
        [[f, 0.0, rig.width / 2.0], [0.0, f, rig.height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    R = np.stack([right, -up2, forward], axis=0)
    t = -R @ pos
    return Camera(K=K, R=R, t=t, width=rig.width, height=rig.height, name="mujoco")


def _perturb_camera(camera: Camera, yaw: float, pitch: float, rng) -> Camera:
    if yaw <= 0 and pitch <= 0:
        return camera
    y = rng.normal(0.0, yaw)
    p = rng.normal(0.0, pitch)
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    R = Ry @ Rx @ camera.R
    return Camera(
        K=camera.K.copy(), R=R, t=camera.t.copy(),
        width=camera.width, height=camera.height, name=camera.name,
    )


@dataclass
class SimObjectObservation:
    object_id: str
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    uv: np.ndarray
    depth: float
    points_world: np.ndarray
    visible_pixels: int


@dataclass
class SimFrame:
    camera: Camera
    rgb: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray
    objects: dict[str, SimObjectObservation]
    true_poses: dict[str, np.ndarray]


@dataclass
class Sim3DScene:
    n_objects: int = 5
    seed: int = 0
    width: int = 160
    height: int = 120
    occlusion: bool = False
    scene: Scene = field(init=False)
    rig: SensorRig = field(init=False)

    def __post_init__(self):
        cfg = SceneConfig(
            n_objects=int(self.n_objects),
            seed=int(self.seed),
            width=int(self.width),
            height=int(self.height),
            bins=0,
            occlusion=bool(self.occlusion),
        )
        self.scene = build_scene(cfg)
        self.scene.step(20)
        self.rig = SensorRig(self.scene, self.width, self.height)

    def set_view(
        self,
        azimuth: float,
        elevation: float = -35.0,
        distance: float = 1.10,
        lookat=(0.0, 0.0, 0.05),
    ) -> None:
        self.rig.set_camera(
            azimuth=float(azimuth),
            elevation=float(elevation),
            distance=float(distance),
            lookat=lookat,
        )

    def ground_truth(self) -> SceneModel:
        poses = self.rig.object_poses()
        objects = {}
        for obj in self.scene.objects:
            pos = np.asarray(poses[obj.instance_id]["pos"], dtype=np.float64)
            radius = _radius_for(obj.geom_type, obj.size)
            objects[obj.instance_id] = ObjectState(
                id=obj.instance_id,
                pos=pos,
                shape=obj.geom_type,
                size=str(obj.size),
                radius=radius,
                provenance={"source": "mujoco", "category": str(obj.category)},
            )
        return SceneModel(scene_id=f"sim{self.seed}", objects=objects)

    def capture(
        self,
        *,
        sigma_px: float = 0.0,
        sigma_depth: float = 0.0,
        sigma_pose_yaw: float = 0.0,
        sigma_pose_pitch: float = 0.0,
        min_pixels: int = 8,
        max_points: int = 64,
        rng: np.random.Generator | None = None,
    ) -> SimFrame:
        rng = np.random.default_rng(0) if rng is None else rng
        rgb = self.rig.render_rgb()
        depth = self.rig.render_depth()
        seg = self.rig.render_segmentation()
        cam_true = _camera_from_rig(self.rig)
        cam_obs = _perturb_camera(
            cam_true, sigma_pose_yaw, sigma_pose_pitch, rng
        )
        poses = self.rig.object_poses()
        objects: dict[str, SimObjectObservation] = {}
        for obj in self.scene.objects:
            geom_id = int(self.scene.model.geom(obj.geom_name).id)
            mask = seg[..., 0] == geom_id
            n_pixels = int(mask.sum())
            if n_pixels < min_pixels:
                continue
            vv, uu = np.where(mask)
            d = np.asarray(depth[vv, uu], dtype=np.float64)
            # Pixel/depth noise on the observation; the render itself is clean.
            uu_obs = uu + rng.normal(0.0, sigma_px, size=uu.shape)
            vv_obs = vv + rng.normal(0.0, sigma_px, size=vv.shape)
            d_obs = d * (1.0 + rng.normal(0.0, sigma_depth, size=d.shape))
            uv = np.array([float(uu_obs.mean()), float(vv_obs.mean())])
            depth_centroid = float(np.median(d_obs))
            if max_points and n_pixels > max_points:
                idx = rng.choice(n_pixels, size=max_points, replace=False)
            else:
                idx = np.arange(n_pixels)
            points = np.stack(
                [
                    backproject(
                        cam_obs,
                        np.array([uu_obs[i], vv_obs[i]]),
                        float(d_obs[i]),
                    )
                    for i in idx
                ],
                axis=0,
            )
            objects[obj.instance_id] = SimObjectObservation(
                object_id=obj.instance_id,
                mask=mask,
                bbox=(
                    int(uu.min()), int(uu.max()),
                    int(vv.min()), int(vv.max()),
                ),
                uv=uv,
                depth=depth_centroid,
                points_world=points,
                visible_pixels=n_pixels,
            )
        return SimFrame(
            camera=cam_obs,
            rgb=rgb,
            depth=depth,
            segmentation=seg,
            objects=objects,
            true_poses={
                oid: np.asarray(p["pos"], dtype=np.float64)
                for oid, p in poses.items()
            },
        )

    def close(self) -> None:
        self.rig.close()


__all__ = ["Sim3DScene", "SimFrame", "SimObjectObservation"]
