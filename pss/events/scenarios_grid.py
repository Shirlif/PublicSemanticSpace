"""Parameterised contact-event grid scenarios.

The coverage study needs the *same physical scene* under many sensor-noise
settings.  Building MuJoCo frames is the expensive part; force sigma, dropout
and delay only change the ``NoisyForceExtractor`` post-processing.  This module
therefore exposes ``GridSpec`` with a ``physical_key`` that excludes those
noise factors, so a caller can build/capture/GT once and sweep noise settings
on the cached frames.

Physical factors
----------------
``kind``
    ``drop`` / ``collide`` / ``near_miss``.
``n_objects``
    2 or 4.  For 4 objects the two extra instances are parked behind the
    interacting pair and do not touch anything; they exercise multi-pair
    enumeration and geometry false-alarm suppression.
``occlusion``
    Add the C3 occluder panel and pin it every tick.  There is no vision arm,
    so this factor is design-complete but expected to be null for
    force/geometry metrics.
``geometry_mismatch``
    Multiplier of ``EPS = 0.02 m``.  Only ``frame.scene_model`` is displaced;
    contacts, poses and GT stay real.  The moving object is displaced along the
    pair axis in the error-amplifying direction: away during real contact
    (drop: +z, collide: -x) and toward the other body in near-miss (-x).  This
    makes geometry miss real contacts and, separately, false-alarm on
    near-misses by almost exactly the displacement magnitude.
``seed``
    Scene and sensor seed.

Sampling defaults match the coverage runner (200 capture steps every 4 sim
steps, i.e. 0.008 s/frame) and stop early after an object-object contact has
been quiet for a cooldown window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pss.sim.scene import SceneConfig, build_scene
from pss.sim.sensors import SensorRig

from .bench_api import ContactConfig, ContactScenario
from .gt import events_from_contacts
from .model import ContactFrame
from .scenarios import _capture

EPS = 0.02
KINDS = ("drop", "collide", "near_miss")
N_OBJECTS = (2, 4)

# Matches ``scenarios_conflict``: a thin x-facing panel in front of the
# SensorRig camera, pinned so gravity cannot move it.
_OCCLUDER_POS = (-0.40, 0.0, 0.33)
_OCCLUDER_SIZE = (0.02, 0.25, 0.15)

# Near-miss centre separation.  At 0.115 m the box/ball surface gap is just
# below EPS, so the clean geometry channel false-alarms (C3 near_miss_hard).
_NEAR_MISS_SEPARATION = 0.115

# Static escorts for n_objects=4; behind the pair and out of the motion path.
_STATIC_POSITIONS = ((-0.30, 0.20), (-0.30, -0.20))


@dataclass(frozen=True)
class GridSpec:
    kind: str
    n_objects: int
    occlusion: bool
    geometry_mismatch: float  # multiplier of EPS
    seed: int

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; known: {KINDS}")
        if self.n_objects not in N_OBJECTS:
            raise ValueError(f"n_objects must be one of {N_OBJECTS}")
        if self.geometry_mismatch < 0:
            raise ValueError("geometry_mismatch must be non-negative")

    @property
    def physical_key(self) -> tuple:
        """Everything that changes the MuJoCo frames or GT.

        Force sigma, delay and dropout are deliberately absent.
        """
        return (
            self.kind,
            int(self.n_objects),
            bool(self.occlusion),
            float(self.geometry_mismatch),
            int(self.seed),
        )

    @property
    def mismatch_m(self) -> float:
        return float(self.geometry_mismatch) * EPS

    @property
    def mismatch_direction(self) -> str:
        if self.kind == "near_miss":
            return "axial_toward"
        if self.kind == "drop":
            return "vertical_away"
        return "axial_away"


def _categories(n_objects: int) -> list[str]:
    if n_objects == 2:
        return ["box", "ball"]
    return ["box", "ball", "block", "fruit"]


def _place_statics(scene) -> None:
    for offset, (x, y) in enumerate(_STATIC_POSITIONS, start=2):
        if offset >= len(scene.objects):
            break
        obj = scene.objects[offset]
        scene.set_object_pose(offset, [x, y, obj.rest_height + 0.002])


def _apply_geometry_mismatch(frames: list[ContactFrame], spec: GridSpec) -> None:
    """Displace only the scene-model pose of the moving object.

    ``frame.poses``, ``frame.contacts`` and the GT built from them are left
    untouched.  Object index 1 is the dropped/moving ball in every kind.
    """
    m = spec.mismatch_m
    if m <= 0.0:
        return
    obj_id = "ball_00"
    if spec.mismatch_direction == "axial_toward":
        delta = np.array([-m, 0.0, 0.0], dtype=np.float64)
    elif spec.mismatch_direction == "vertical_away":
        # The drop pair axis is vertical; a lateral offset barely changes the
        # 3-D gap, while +z changes it by almost exactly m.
        delta = np.array([0.0, 0.0, m], dtype=np.float64)
    else:
        # Collide: the ball starts at -x of the box and moves +x, so -x is away.
        delta = np.array([-m, 0.0, 0.0], dtype=np.float64)
    for frame in frames:
        entry = frame.scene_model.objects.get(obj_id)
        if entry is not None:
            entry.pos = np.asarray(entry.pos, dtype=np.float64) + delta


def _object_ids(scene) -> set[str]:
    return {str(obj.instance_id) for obj in scene.objects}


def _has_object_contact(frame: ContactFrame, object_ids: set[str], threshold: float) -> bool:
    for c in frame.contacts:
        a = str(c.get("instance1"))
        b = str(c.get("instance2"))
        if a in object_ids and b in object_ids and a != b:
            if float(c.get("normal_force", 0.0)) >= float(threshold):
                return True
    return False


def _run_grid(
    scene,
    rig: SensorRig,
    scene_id: str,
    config: ContactConfig,
    spec: GridSpec,
    *,
    external_force: tuple[int, np.ndarray, int] | None = None,
    cooldown_frames: int = 25,
    min_frames: int = 16,
) -> list[ContactFrame]:
    """Capture frames, optionally pinning the occluder, with a cooldown stop.

    The stop fires only after at least one real object-object contact has been
    seen and it has been absent for ``cooldown_frames`` captured frames.  A
    near-miss never contacts, so it runs to ``config.max_steps``.
    """
    dt = float(scene.model.opt.timestep)
    every = max(1, int(config.capture_every))
    steps = max(1, int(config.max_steps))
    object_ids = _object_ids(scene)
    gt_threshold = float(config.gt_threshold)

    body_id = None
    if external_force is not None:
        body_index, _, _ = external_force
        body_id = int(scene.model.body(scene.objects[body_index].body_name).id)

    frames: list[ContactFrame] = []
    seen_contact = False
    quiet = 0
    for i in range(steps):
        if external_force is not None:
            _, force, until = external_force
            scene.data.xfrc_applied[body_id, :3] = force if i < until else 0.0
        if spec.occlusion:
            scene.set_occluder_pose(_OCCLUDER_POS)
        scene.step(every)
        if spec.occlusion:
            # Counter gravity inside one tick; drift is < 0.1 mm.
            scene.set_occluder_pose(_OCCLUDER_POS)
        frame = _capture(scene, rig, scene_id, (i + 1) * every * dt, config.render)
        frames.append(frame)

        if _has_object_contact(frame, object_ids, gt_threshold):
            seen_contact = True
            quiet = 0
        elif seen_contact:
            quiet += 1
            if quiet >= cooldown_frames and len(frames) >= min_frames:
                break
    return frames


def build_grid_scenario(spec: GridSpec, config: ContactConfig) -> ContactScenario:
    """Build one parameterised grid scenario.

    The caller is responsible for supplying a per-physical-key cache; the
    function itself does not memoise.
    """
    scene = build_scene(
        SceneConfig(
            n_objects=int(spec.n_objects),
            seed=int(spec.seed),
            width=int(config.width),
            height=int(config.height),
            bins=0,
            occlusion=bool(spec.occlusion),
        ),
        categories=_categories(int(spec.n_objects)),
        instance_indices=[0] * int(spec.n_objects),
        occlusion=bool(spec.occlusion),
    )
    rig = SensorRig(scene, config.width, config.height)

    target, mover = scene.objects[0], scene.objects[1]
    external_force = None
    if spec.kind == "drop":
        scene.set_object_pose(0, [0.0, 0.0, target.rest_height + 0.002])
        scene.set_object_pose(1, [0.0, 0.0, mover.rest_height + 0.35])
    elif spec.kind == "collide":
        scene.set_object_pose(0, [0.0, 0.0, target.rest_height + 0.002])
        scene.set_object_pose(1, [-0.18, 0.0, mover.rest_height + 0.002])
        external_force = (1, np.array([2.5, 0.0, 0.0]), 60)
    else:  # near_miss
        half = _NEAR_MISS_SEPARATION / 2.0
        scene.set_object_pose(0, [-half, 0.0, target.rest_height + 0.002])
        scene.set_object_pose(1, [half, 0.0, mover.rest_height + 0.002])

    _place_statics(scene)
    if spec.occlusion:
        scene.set_occluder_size(_OCCLUDER_SIZE)
        scene.set_occluder_pose(_OCCLUDER_POS)
    scene.forward()

    scene_id = (
        f"grid_{spec.kind}_{spec.n_objects}obj_"
        f"{'occ' if spec.occlusion else 'clear'}_m{spec.geometry_mismatch:g}_s{spec.seed}"
    )
    frames = _run_grid(
        scene, rig, scene_id, config, spec, external_force=external_force
    )
    gt = events_from_contacts(
        frames,
        threshold=config.gt_threshold,
        min_duration=config.min_duration,
        max_gap=config.max_gap,
    )
    _apply_geometry_mismatch(frames, spec)
    rig.close()

    return ContactScenario(
        scenario_id=scene_id,
        frames=frames,
        gt_events=gt,
        metadata={
            "kind": spec.kind,
            "seed": int(spec.seed),
            "n_objects": int(spec.n_objects),
            "occlusion": bool(spec.occlusion),
            "geometry_mismatch": float(spec.geometry_mismatch),
            "geometry_mismatch_m": float(spec.mismatch_m),
            "geometry_mismatch_object": "ball_00",
            "geometry_mismatch_direction": spec.mismatch_direction,
            "near_miss_separation_m": _NEAR_MISS_SEPARATION if spec.kind == "near_miss" else None,
            "frames": int(len(frames)),
            "gt_events": int(len(gt)),
            "gt_pairs": sorted({e.pair for e in gt}),
            "cooldown_frames": 25,
        },
    )


__all__ = [
    "EPS",
    "GridSpec",
    "KINDS",
    "N_OBJECTS",
    "build_grid_scenario",
]
