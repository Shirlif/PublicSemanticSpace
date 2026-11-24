"""MuJoCo tabletop scene builder (Phase 2, primitive proxy assets).

This module intentionally uses **primitive geoms** (sphere/box/cylinder/capsule)
instead of downloaded meshes:

- no external asset licensing or download step;
- deterministic, fast, and reproducible;
- enough to exercise object-centric slots, relations, occlusion, and action
  transitions.

It is a proxy for the real asset pipeline, not a photorealistic simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

BIN_HALF = 0.06
BIN_WALL = 0.005
BIN_HEIGHT = 0.06

# category -> primitive proxy and physical/material hints
CATEGORY_SPECS = {
    # ---- tableware / containers (cylinders) ----
    "cup": {
        "geom": "cylinder",
        "size": (0.040, 0.050),
        "mass": 0.08,
        "colors": [(0.15, 0.35, 0.80, 1.0), (0.90, 0.90, 0.90, 1.0)],
        "materials": [
            {"hardness": 0.85, "roughness": 0.15, "thermal": 0.2},
            {"hardness": 0.80, "roughness": 0.25, "thermal": 0.2},
        ],
    },
    "mug": {
        "geom": "cylinder",
        "size": (0.045, 0.045),
        "mass": 0.09,
        "colors": [(0.80, 0.15, 0.15, 1.0), (0.15, 0.15, 0.15, 1.0)],
        "materials": [
            {"hardness": 0.82, "roughness": 0.20, "thermal": 0.2},
            {"hardness": 0.78, "roughness": 0.30, "thermal": 0.2},
        ],
    },
    "bowl": {
        "geom": "cylinder",
        "size": (0.070, 0.030),
        "mass": 0.12,
        "colors": [(0.90, 0.90, 0.88, 1.0), (0.15, 0.55, 0.55, 1.0)],
        "materials": [
            {"hardness": 0.88, "roughness": 0.18, "thermal": 0.2},
            {"hardness": 0.84, "roughness": 0.26, "thermal": 0.2},
        ],
    },
    "plate": {
        "geom": "cylinder",
        "size": (0.090, 0.008),
        "mass": 0.12,
        "colors": [(0.92, 0.92, 0.92, 1.0), (0.85, 0.80, 0.70, 1.0)],
        "materials": [
            {"hardness": 0.95, "roughness": 0.10, "thermal": 0.3},
            {"hardness": 0.90, "roughness": 0.20, "thermal": 0.3},
        ],
    },
    "saucer": {
        "geom": "cylinder",
        "size": (0.060, 0.006),
        "mass": 0.08,
        "colors": [(0.94, 0.94, 0.94, 1.0), (0.92, 0.65, 0.70, 1.0)],
        "materials": [
            {"hardness": 0.93, "roughness": 0.12, "thermal": 0.3},
            {"hardness": 0.88, "roughness": 0.22, "thermal": 0.3},
        ],
    },
    "bottle": {
        "geom": "cylinder",
        "size": (0.030, 0.080),
        "mass": 0.20,
        "colors": [(0.20, 0.55, 0.30, 1.0), (0.90, 0.70, 0.20, 1.0)],
        "materials": [
            {"hardness": 0.80, "roughness": 0.20, "thermal": 0.2},
            {"hardness": 0.75, "roughness": 0.30, "thermal": 0.2},
        ],
    },
    "jar": {
        "geom": "cylinder",
        "size": (0.050, 0.070),
        "mass": 0.22,
        "colors": [(0.75, 0.85, 0.90, 1.0), (0.90, 0.85, 0.35, 1.0)],
        "materials": [
            {"hardness": 0.82, "roughness": 0.18, "thermal": 0.2},
            {"hardness": 0.78, "roughness": 0.28, "thermal": 0.2},
        ],
    },
    "can": {
        "geom": "cylinder",
        "size": (0.035, 0.060),
        "mass": 0.15,
        "colors": [(0.70, 0.70, 0.72, 1.0), (0.80, 0.20, 0.15, 1.0)],
        "materials": [
            {"hardness": 0.95, "roughness": 0.25, "thermal": 0.1},
            {"hardness": 0.90, "roughness": 0.35, "thermal": 0.1},
        ],
    },
    # ---- boxes ----
    "box": {
        "geom": "box",
        "size": (0.060, 0.045, 0.035),
        "mass": 0.15,
        "colors": [(0.55, 0.35, 0.20, 1.0), (0.40, 0.45, 0.30, 1.0)],
        "materials": [
            {"hardness": 0.75, "roughness": 0.55, "thermal": 0.2},
            {"hardness": 0.70, "roughness": 0.65, "thermal": 0.2},
        ],
    },
    "carton": {
        "geom": "box",
        "size": (0.050, 0.050, 0.090),
        "mass": 0.18,
        "colors": [(0.90, 0.90, 0.90, 1.0), (0.20, 0.40, 0.75, 1.0)],
        "materials": [
            {"hardness": 0.72, "roughness": 0.60, "thermal": 0.2},
            {"hardness": 0.68, "roughness": 0.70, "thermal": 0.2},
        ],
    },
    "brick": {
        "geom": "box",
        "size": (0.080, 0.040, 0.025),
        "mass": 0.25,
        "colors": [(0.70, 0.25, 0.18, 1.0), (0.55, 0.55, 0.55, 1.0)],
        "materials": [
            {"hardness": 0.90, "roughness": 0.60, "thermal": 0.2},
            {"hardness": 0.85, "roughness": 0.70, "thermal": 0.2},
        ],
    },
    "block": {
        "geom": "box",
        "size": (0.035, 0.035, 0.035),
        "mass": 0.10,
        "colors": [(0.90, 0.80, 0.20, 1.0), (0.80, 0.20, 0.20, 1.0)],
        "materials": [
            {"hardness": 0.90, "roughness": 0.35, "thermal": 0.2},
            {"hardness": 0.85, "roughness": 0.45, "thermal": 0.2},
        ],
    },
    # ---- round ----
    "ball": {
        "geom": "sphere",
        "size": (0.050,),
        "mass": 0.12,
        "colors": [(0.85, 0.15, 0.15, 1.0), (0.25, 0.65, 0.25, 1.0)],
        "materials": [
            {"hardness": 0.55, "roughness": 0.30, "thermal": 0.4},
            {"hardness": 0.50, "roughness": 0.40, "thermal": 0.4},
        ],
    },
    "fruit": {
        "geom": "sphere",
        "size": (0.040,),
        "mass": 0.09,
        "colors": [(0.95, 0.55, 0.15, 1.0), (0.85, 0.20, 0.15, 1.0)],
        "materials": [
            {"hardness": 0.50, "roughness": 0.35, "thermal": 0.4},
            {"hardness": 0.45, "roughness": 0.45, "thermal": 0.4},
        ],
    },
    "egg": {
        "geom": "ellipsoid",
        "size": (0.040, 0.030, 0.050),
        "mass": 0.10,
        "colors": [(0.95, 0.92, 0.85, 1.0), (0.70, 0.55, 0.40, 1.0)],
        "materials": [
            {"hardness": 0.70, "roughness": 0.25, "thermal": 0.3},
            {"hardness": 0.65, "roughness": 0.35, "thermal": 0.3},
        ],
    },
    # ---- elongated ----
    "rod": {
        "geom": "cylinder",
        "size": (0.008, 0.100),
        "mass": 0.08,
        "colors": [(0.55, 0.55, 0.58, 1.0), (0.65, 0.45, 0.25, 1.0)],
        "materials": [
            {"hardness": 0.95, "roughness": 0.30, "thermal": 0.2},
            {"hardness": 0.90, "roughness": 0.40, "thermal": 0.2},
        ],
    },
    # ---- held out for open-set ----
    "cloth": {
        "geom": "box",
        "size": (0.100, 0.080, 0.004),
        "mass": 0.05,
        "colors": [(0.80, 0.20, 0.20, 1.0), (0.20, 0.30, 0.70, 1.0)],
        "materials": [
            {"hardness": 0.15, "roughness": 0.80, "thermal": 0.5},
            {"hardness": 0.10, "roughness": 0.90, "thermal": 0.5},
        ],
    },
    "tool": {
        "geom": "capsule",
        "size": (0.015, 0.060),
        "mass": 0.12,
        "colors": [(0.55, 0.55, 0.58, 1.0), (0.70, 0.55, 0.25, 1.0)],
        "materials": [
            {"hardness": 0.95, "roughness": 0.30, "thermal": 0.2},
            {"hardness": 0.90, "roughness": 0.40, "thermal": 0.2},
        ],
    },
    "sponge": {
        "geom": "box",
        "size": (0.070, 0.050, 0.020),
        "mass": 0.04,
        "colors": [(0.95, 0.85, 0.25, 1.0), (0.90, 0.55, 0.65, 1.0)],
        "materials": [
            {"hardness": 0.20, "roughness": 0.75, "thermal": 0.5},
            {"hardness": 0.15, "roughness": 0.85, "thermal": 0.5},
        ],
    },
    "sheet": {
        "geom": "box",
        "size": (0.120, 0.090, 0.002),
        "mass": 0.03,
        "colors": [(0.92, 0.92, 0.92, 1.0), (0.60, 0.60, 0.60, 1.0)],
        "materials": [
            {"hardness": 0.30, "roughness": 0.60, "thermal": 0.5},
            {"hardness": 0.25, "roughness": 0.70, "thermal": 0.5},
        ],
    },
}

HELD_OUT_CATEGORIES = ("cloth", "tool", "sponge", "sheet")


def _fmt(values) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def _rest_height(geom: str, size: tuple) -> float:
    if geom == "sphere":
        return float(size[0])
    if geom == "box":
        return float(size[2])
    if geom == "cylinder":
        return float(size[1])
    if geom == "capsule":
        return float(size[0])
    if geom == "ellipsoid":
        return float(size[2])
    raise ValueError(geom)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


@dataclass
class SceneConfig:
    n_objects: int = 4
    seed: int = 0
    width: int = 128
    height: int = 128
    occlusion: bool = False
    bins: int = 2
    workspace_x: tuple[float, float] = (-0.32, 0.18)
    workspace_y: tuple[float, float] = (-0.22, 0.22)
    min_separation: float = 0.12
    settle_steps: int = 30
    variation_specs: bool = False
    near_identical: bool = False


@dataclass
class SceneObject:
    index: int
    instance_id: str
    category: str
    body_name: str
    geom_name: str
    geom_type: str
    size: tuple
    rgba: tuple
    mass: float
    material: dict
    rest_height: float
    start_pos: np.ndarray
    start_quat: np.ndarray


@dataclass
class Occluder:
    body_name: str = "occluder"
    geom_name: str = "occluder_geom"
    size: tuple = (0.10, 0.02, 0.12)


@dataclass
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData
    objects: list[SceneObject]
    bins: list[dict]
    config: SceneConfig
    occluder: Occluder | None = None

    def step(self, n: int = 1) -> None:
        for _ in range(max(0, n)):
            mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _free_joint_adr(self, body_name: str) -> tuple[int, int]:
        body_id = self.model.body(body_name).id
        jnt = int(self.model.body_jntadr[body_id])
        return int(self.model.jnt_qposadr[jnt]), int(self.model.jnt_dofadr[jnt])

    def set_object_pose(self, index: int, pos, quat=None) -> None:
        obj = self.objects[index]
        qpos_adr, qvel_adr = self._free_joint_adr(obj.body_name)
        self.data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(pos, dtype=float)
        if quat is not None:
            self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.asarray(quat, dtype=float)
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_object_pose(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        body_id = self.model.body(self.objects[index].body_name).id
        return self.data.xpos[body_id].copy(), self.data.xquat[body_id].copy()

    def set_occluder_pose(self, pos) -> None:
        if self.occluder is None:
            raise RuntimeError("scene has no occluder")
        qpos_adr, qvel_adr = self._free_joint_adr(self.occluder.body_name)
        self.data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(pos, dtype=float)
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_occluder_size(self, size) -> None:
        if self.occluder is None:
            raise RuntimeError("scene has no occluder")
        geom_id = int(self.model.geom(self.occluder.geom_name).id)
        self.model.geom_size[geom_id] = np.asarray(size, dtype=float)
        mujoco.mj_forward(self.model, self.data)


def _instance_spec(category: str, instance_idx: int, rng: np.random.Generator) -> dict:
    spec = CATEGORY_SPECS[category]
    scale = (0.90, 1.00, 1.10)[instance_idx % 3]
    size = tuple(float(s) * scale for s in spec["size"])
    color = spec["colors"][instance_idx % len(spec["colors"])]
    material = dict(spec["materials"][instance_idx % len(spec["materials"])])
    material = {
        "hardness": float(np.clip(material["hardness"] + rng.uniform(-0.05, 0.05), 0.0, 1.0)),
        "roughness": float(np.clip(material["roughness"] + rng.uniform(-0.05, 0.05), 0.0, 1.0)),
        "thermal": float(np.clip(material["thermal"] + rng.uniform(-0.05, 0.05), 0.0, 1.0)),
    }
    return {
        "geom": spec["geom"],
        "size": size,
        "mass": float(spec["mass"] * scale ** 3),
        "rgba": color,
        "material": material,
        "instance_id": f"{category}_{instance_idx:02d}",
    }


def _variation_spec(category: str, variation_idx: int, rng: np.random.Generator) -> dict:
    """Category-preserving variation for pretraining.

    Unlike ``_instance_spec`` (3 fixed variants), this samples a continuous
    scale/color/material distribution so each category has many distinct
    appearances. The category stays the same; only nuisance factors move.
    """
    spec = CATEGORY_SPECS[category]
    scale = float(rng.uniform(0.85, 1.15))
    base = np.asarray(spec["size"], dtype=float)
    # Independent per-axis jitter gives geometry a per-instance shape signal;
    # a single uniform scale only produces three near-identical silhouettes.
    axis_jitter = rng.uniform(0.88, 1.12, size=base.shape)
    size = tuple(float(v) for v in base * scale * axis_jitter)
    base = np.asarray(spec["colors"][variation_idx % len(spec["colors"])][:3], dtype=float)
    jitter = rng.uniform(-0.10, 0.10, size=3)
    rgba = tuple(float(v) for v in np.clip(base + jitter, 0.0, 1.0)) + (1.0,)
    material = dict(spec["materials"][variation_idx % len(spec["materials"])])
    material = {
        key: float(np.clip(value + rng.uniform(-0.15, 0.15), 0.0, 1.0))
        for key, value in material.items()
    }
    return {
        "geom": spec["geom"],
        "size": size,
        "mass": float(spec["mass"] * scale ** 3),
        "rgba": rgba,
        "material": material,
        "instance_id": f"{category}_v{variation_idx:03d}",
    }


def _near_identical_spec(category: str, variation_idx: int, rng: np.random.Generator) -> dict:
    """Same-category instances with identical size, color, and material.

    The only difference is the instance id, so appearance cannot tell them
    apart unless the renderer leaks a position-dependent cue. This is the
    stress test for instance-level identity.
    """
    spec = CATEGORY_SPECS[category]
    return {
        "geom": spec["geom"],
        "size": tuple(float(v) for v in spec["size"]),
        "mass": float(spec["mass"]),
        "rgba": tuple(float(v) for v in spec["colors"][0]),
        "material": dict(spec["materials"][0]),
        "instance_id": f"{category}_n{variation_idx:03d}",
    }


def _sample_positions(config: SceneConfig, rng: np.random.Generator, n: int) -> list[np.ndarray]:
    positions: list[np.ndarray] = []
    x0, x1 = config.workspace_x
    y0, y1 = config.workspace_y
    for _ in range(n):
        for _attempt in range(2000):
            candidate = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1), 0.0])
            if all(np.linalg.norm(candidate[:2] - p[:2]) >= config.min_separation for p in positions):
                positions.append(candidate)
                break
        else:
            # deterministic fallback grid
            k = len(positions)
            positions.append(np.array([x0 + 0.1 * (k % 4), y0 + 0.1 * (k // 4), 0.0]))
    return positions


def _bin_xml(name: str, x: float, y: float) -> str:
    h = BIN_HALF
    t = BIN_WALL
    zh = BIN_HEIGHT / 2.0
    zc = zh + 0.01
    return (
        f'<body name="{name}" pos="{x:.4f} {y:.4f} 0">\n'
        f'  <geom name="{name}_bottom" type="box" size="{h} {h} 0.005" pos="0 0 0.005" '
        f'rgba="0.25 0.25 0.28 1"/>\n'
        f'  <geom name="{name}_xp" type="box" size="{t} {h} {zh}" pos="{h} 0 {zc}" '
        f'rgba="0.30 0.30 0.33 1"/>\n'
        f'  <geom name="{name}_xn" type="box" size="{t} {h} {zh}" pos="{-h} 0 {zc}" '
        f'rgba="0.30 0.30 0.33 1"/>\n'
        f'  <geom name="{name}_yp" type="box" size="{h} {t} {zh}" pos="0 {h} {zc}" '
        f'rgba="0.30 0.30 0.33 1"/>\n'
        f'  <geom name="{name}_yn" type="box" size="{h} {t} {zh}" pos="0 {-h} {zc}" '
        f'rgba="0.30 0.30 0.33 1"/>\n'
        f'</body>'
    )


def build_scene(
    config: SceneConfig,
    categories: list[str] | None = None,
    instance_indices: list[int] | None = None,
    occlusion: bool | None = None,
) -> Scene:
    rng = np.random.default_rng(config.seed)
    n = min(config.n_objects, len(CATEGORY_SPECS))
    if categories is None:
        categories = list(rng.choice(list(CATEGORY_SPECS), size=n, replace=False))
    if instance_indices is None:
        instance_indices = [int(rng.integers(0, 3)) for _ in categories]
    if len(categories) != len(instance_indices):
        raise ValueError("categories and instance_indices must have the same length")
    use_occluder = config.occlusion if occlusion is None else occlusion

    positions = _sample_positions(config, rng, len(categories))
    spec_fn = _variation_spec if config.variation_specs else _instance_spec
    if config.near_identical:
        spec_fn = _near_identical_spec
    objects: list[SceneObject] = []
    for i, (category, inst_idx) in enumerate(zip(categories, instance_indices)):
        spec = spec_fn(category, inst_idx, rng)
        rest = _rest_height(spec["geom"], spec["size"])
        yaw = float(rng.uniform(0.0, np.pi))
        start_pos = positions[i].copy()
        start_pos[2] = rest + 0.002
        objects.append(
            SceneObject(
                index=i,
                instance_id=spec["instance_id"],
                category=category,
                body_name=f"obj_{i}",
                geom_name=f"obj_{i}_geom",
                geom_type=spec["geom"],
                size=spec["size"],
                rgba=tuple(spec["rgba"]),
                mass=spec["mass"],
                material=spec["material"],
                rest_height=rest,
                start_pos=start_pos,
                start_quat=_yaw_quat(yaw),
            )
        )

    bins = []
    if config.bins >= 1:
        bins.append({"name": "bin0", "pos": np.array([0.38, 0.20, 0.0])})
    if config.bins >= 2:
        bins.append({"name": "bin1", "pos": np.array([0.38, -0.20, 0.0])})

    light_pos = (float(rng.uniform(-0.3, 0.3)), float(rng.uniform(-0.5, -0.1)), 1.2)
    light_diffuse = float(rng.uniform(0.7, 1.0))

    parts = [
        '<mujoco model="tabletop">',
        '<compiler angle="degree" autolimits="true"/>',
        '<option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>',
        f'<visual><global offwidth="{config.width}" offheight="{config.height}"/>'
        f'<quality shadowsize="1024"/><map znear="0.01" zfar="10"/></visual>',
        "<worldbody>",
        f'<light name="key" pos="{_fmt(light_pos)}" dir="0 0 -1" directional="true" '
        f'ambient="0.20 0.20 0.20" '
        f'diffuse="{light_diffuse:.3f} {light_diffuse:.3f} {light_diffuse:.3f}" '
        f'specular="0.2 0.2 0.2"/>',
        '<geom name="table" type="box" size="0.60 0.45 0.02" pos="0 0 -0.02" '
        'rgba="0.55 0.55 0.55 1" friction="1.0 0.01 0.001"/>',
        '<geom name="floor" type="plane" size="2 2 0.1" pos="0 0 -0.06" '
        'rgba="0.18 0.18 0.18 1"/>',
    ]
    for b in bins:
        parts.append(_bin_xml(b["name"], float(b["pos"][0]), float(b["pos"][1])))

    for obj in objects:
        quat_attr = ""
        if obj.geom_type == "capsule":
            # lay the capsule along local x; body yaw still applies
            quat_attr = ' quat="0.707107 0 0.707107 0"'
        parts.append(
            f'<body name="{obj.body_name}" pos="{_fmt(obj.start_pos)}" '
            f'quat="{_fmt(obj.start_quat)}">'
        )
        parts.append(f'<freejoint name="{obj.body_name}_joint"/>')
        parts.append(
            f'<geom name="{obj.geom_name}" type="{obj.geom_type}" '
            f'size="{_fmt(obj.size)}" rgba="{_fmt(obj.rgba)}" mass="{obj.mass:.6f}" '
            f'friction="1.0 0.01 0.001"{quat_attr}/>'
        )
        parts.append("</body>")

    if use_occluder:
        parts.append(
            '<body name="occluder" pos="0 -0.30 0.22">'
            '<freejoint name="occluder_joint"/>'
            '<geom name="occluder_geom" type="box" size="0.10 0.02 0.12" '
            'rgba="0.15 0.15 0.15 1" mass="1.0"/>'
            "</body>"
        )

    parts += ["</worldbody>", "</mujoco>"]
    xml = "\n".join(parts)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    scene = Scene(
        model=model,
        data=data,
        objects=objects,
        bins=bins,
        config=config,
        occluder=Occluder() if use_occluder else None,
    )
    mujoco.mj_forward(model, data)
    scene.step(config.settle_steps)
    return scene
