"""Sensor rig for the MuJoCo tabletop scene.

Provides RGB, depth, segmentation, object poses, contact-derived synthetic
tactile maps, and coarse spatial relations. All sensors are simulated; the
tactile channel is a **proxy**, not a calibrated tactile sensor.
"""

from __future__ import annotations

import hashlib

import numpy as np
import mujoco

from .scene import BIN_HALF, BIN_HEIGHT, Scene, SceneObject


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    qv = np.array([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


class SensorRig:
    def __init__(self, scene: Scene, width: int | None = None, height: int | None = None):
        self.scene = scene
        self.width = int(width or scene.config.width)
        self.height = int(height or scene.config.height)
        self.renderer = mujoco.Renderer(scene.model, self.height, self.width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(scene.model, self.cam)
        self.cam.lookat[:] = [0.0, 0.0, 0.05]
        self.cam.distance = 1.10
        self.cam.azimuth = 0.0
        self.cam.elevation = -35.0
        self.geom_to_instance = self._build_geom_map()

    def _build_geom_map(self) -> dict[int, str]:
        model = self.scene.model
        mapping: dict[int, str] = {}
        for obj in self.scene.objects:
            mapping[int(model.geom(obj.geom_name).id)] = obj.instance_id
        for b in self.scene.bins:
            for suffix in ("bottom", "xp", "xn", "yp", "yn"):
                name = f"{b['name']}_{suffix}"
                try:
                    mapping[int(model.geom(name).id)] = b["name"]
                except KeyError:
                    pass
        if self.scene.occluder is not None:
            mapping[int(model.geom(self.scene.occluder.geom_name).id)] = "occluder"
        try:
            mapping[int(model.geom("table").id)] = "table"
        except KeyError:
            pass
        return mapping

    def set_camera(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
        lookat=(0.0, 0.0, 0.05),
    ) -> None:
        self.cam.azimuth = float(azimuth)
        self.cam.elevation = float(elevation)
        self.cam.distance = float(distance)
        self.cam.lookat[:] = lookat

    def render_rgb(self) -> np.ndarray:
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.scene.data, camera=self.cam)
        return self.renderer.render()

    def render_depth(self) -> np.ndarray:
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.scene.data, camera=self.cam)
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return np.asarray(depth, dtype=np.float32)

    def render_segmentation(self) -> np.ndarray:
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.scene.data, camera=self.cam)
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        return np.asarray(seg, dtype=np.int32)

    def object_poses(self) -> dict[str, dict]:
        out = {}
        for obj in self.scene.objects:
            pos, quat = self.scene.get_object_pose(obj.index)
            out[obj.instance_id] = {"pos": pos, "quat": quat}
        return out

    def contacts(self) -> list[dict]:
        model, data = self.scene.model, self.scene.data
        out = []
        for i in range(data.ncon):
            c = data.contact[i]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, force)
            g1, g2 = int(c.geom1), int(c.geom2)
            out.append(
                {
                    "geom1": g1,
                    "geom2": g2,
                    "instance1": self.geom_to_instance.get(g1, "unknown"),
                    "instance2": self.geom_to_instance.get(g2, "unknown"),
                    "dist": float(c.dist),
                    "pos": np.asarray(c.pos, dtype=np.float64).copy(),
                    "force": force.copy(),
                    "normal_force": float(abs(force[0])),
                }
            )
        return out

    @staticmethod
    def instance_texture_pattern(instance_id: str, size: int = 8) -> np.ndarray:
        """Deterministic sparse surface pattern for one instance.

        A sparse set of hot spots gives different instances clearly different
        spatial signatures; a smooth low-frequency pattern was too similar
        across instances under cosine similarity.
        """
        seed = int(hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        pattern = np.zeros((size, size), dtype=np.float64)
        n_hot = int(rng.integers(size, 2 * size))
        flat = rng.choice(size * size, n_hot, replace=False)
        pattern.flat[flat] = rng.uniform(0.6, 1.0, n_hot)
        neighbor = (
            np.roll(pattern, 1, axis=0)
            + np.roll(pattern, -1, axis=0)
            + np.roll(pattern, 1, axis=1)
            + np.roll(pattern, -1, axis=1)
        ) / 4.0
        pattern = 0.6 * pattern + 0.4 * neighbor
        pattern = pattern / (pattern.max() + 1e-9)
        return pattern.astype(np.float32)

    @staticmethod
    def tactile_from_material(
        obj: SceneObject,
        rng: np.random.Generator,
        size: int = 8,
        pattern: np.ndarray | None = None,
        shift: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Synthetic tactile proxy derived from material + instance texture.

        The material term sets the overall hardness/thermal response; the
        instance pattern provides the identity signal. This is still not a
        contact simulation, but it is no longer material-only.
        """
        hardness = float(obj.material["hardness"])
        roughness = float(obj.material["roughness"])
        thermal = float(obj.material["thermal"])
        yy, xx = np.mgrid[0:size, 0:size]
        c = (size - 1) / 2.0
        bump = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * (size / 4.0) ** 2))
        if pattern is None:
            pattern = SensorRig.instance_texture_pattern(obj.instance_id, size)
        if shift is not None:
            pattern = np.roll(pattern, shift, axis=(0, 1))
        texture = rng.normal(size=(size, size)) * roughness * 0.15
        material_term = (0.75 * hardness + 0.25 * thermal) * bump
        arr = 0.5 * material_term + 0.5 * pattern + texture
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def tactile_maps(self, size: int = 8) -> dict[str, np.ndarray]:
        """Contact-derived tactile maps, one per object.

        Contact points are projected into each object's local frame and
        accumulated into a coarse grid. Empty grids mean "no contact", not
        "no tactile reading".
        """
        maps = {
            obj.instance_id: np.zeros((size, size), dtype=np.float32)
            for obj in self.scene.objects
        }
        radii = {
            obj.instance_id: max(float(np.max(obj.size)) * 1.5, 1e-3)
            for obj in self.scene.objects
        }
        for c in self.contacts():
            # Support contacts with the table/floor are not tactile events.
            if c["instance1"] in ("table", "floor") or c["instance2"] in ("table", "floor"):
                continue
            for inst in (c["instance1"], c["instance2"]):
                if inst not in maps:
                    continue
                obj = next(o for o in self.scene.objects if o.instance_id == inst)
                pos, quat = self.scene.get_object_pose(obj.index)
                local = _quat_rotate(_quat_inv(quat), c["pos"] - pos)
                r = radii[inst]
                u = int(np.clip((local[0] + r) / (2.0 * r) * (size - 1), 0, size - 1))
                v = int(np.clip((local[1] + r) / (2.0 * r) * (size - 1), 0, size - 1))
                maps[inst][v, u] += c["normal_force"]
        for inst, arr in maps.items():
            mx = float(arr.max())
            if mx > 0:
                maps[inst] = arr / mx
        return maps

    def relations(self) -> list[tuple]:
        rels: set[tuple] = set()
        poses = self.object_poses()
        for obj in self.scene.objects:
            p = poses[obj.instance_id]["pos"]
            for b in self.scene.bins:
                if (
                    abs(p[0] - b["pos"][0]) < BIN_HALF - 0.01
                    and abs(p[1] - b["pos"][1]) < BIN_HALF - 0.01
                    and p[2] < BIN_HEIGHT + 0.08
                ):
                    rels.add((obj.instance_id, "in", b["name"]))
        ids = [o.instance_id for o in self.scene.objects]
        for c in self.contacts():
            a, b = c["instance1"], c["instance2"]
            if a in ids and b in ids and a != b:
                rels.add((a, "contact", b))
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                d = np.linalg.norm(poses[a]["pos"] - poses[b]["pos"])
                if d < 0.12:
                    rels.add((a, "near", b))
        return sorted(rels)

    def close(self) -> None:
        self.renderer.close()
