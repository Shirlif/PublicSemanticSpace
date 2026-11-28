"""C3 conflict scenarios: occlusion, force-only / geometry-only conflicts, hard near-miss.

These are **MuJoCo proxy** scenes.  Two of them deliberately corrupt the
*observed* scene model (never the physics) to create a channel conflict:

- ``conflict_force_only``: physics contact happens, but the per-frame scene
  model pose of the moving object is offset, so geometry predicts no contact;
- ``conflict_geometry_only``: objects rest with a real surface gap, but the
  scene-model radii are inflated, so geometry predicts contact while force
  stays at zero.

GT is always derived from MuJoCo contact forces, so force is a near-oracle
against this GT; the report must not read conflict wins as independent
evidence.

The occluder is a free body in ``build_scene``; a small hold loop re-applies
its pose every tick so it does not fall onto the table.  Occlusion is verified
through the segmentation render (visible object pixels), not assumed.
"""

from __future__ import annotations

import numpy as np

from pss.sim.scene import SceneConfig, build_scene
from pss.sim.sensors import SensorRig

from .bench_api import ContactConfig, ContactScenario, register_scenario
from .gt import events_from_contacts
from .scenarios import _capture, _run

# Camera of SensorRig is on the -x side, elevated (lookat ~origin).  A thin
# x-facing panel at x=-0.40 spans the view of the tabletop objects.
_OCCLUDER_POS = (-0.40, 0.0, 0.33)
_OCCLUDER_SIZE = (0.02, 0.25, 0.15)

# Geometry model corruption magnitudes (metres).  Both are observation-model
# errors, not physical changes.
_FORCE_ONLY_POSE_OFFSET = 0.50
_GEOMETRY_ONLY_RADIUS_INFLATION = 0.05

# Static placements (metres, centre-to-centre along x).
_GEOMETRY_ONLY_SEPARATION = 0.125
_NEAR_MISS_HARD_SEPARATION = 0.115
_NEAR_MISS_CLEAR_SEPARATION = 0.140


def _gt(frames, config: ContactConfig):
    return events_from_contacts(
        frames,
        threshold=config.gt_threshold,
        min_duration=config.min_duration,
        max_gap=config.max_gap,
    )


def _place_near(scene, separation: float) -> None:
    a, b = scene.objects[0], scene.objects[1]
    half = float(separation) / 2.0
    scene.set_object_pose(0, [-half, 0.0, a.rest_height + 0.002])
    scene.set_object_pose(1, [half, 0.0, b.rest_height + 0.002])
    scene.forward()


def _run_holding_occluder(
    scene,
    rig: SensorRig,
    scene_id: str,
    config: ContactConfig,
    *,
    external_force: tuple[int, np.ndarray, int] | None = None,
) -> list:
    """Like ``scenarios._run`` but pins the free-joint occluder every tick."""
    dt = float(scene.model.opt.timestep)
    every = max(1, int(config.capture_every))
    steps = max(1, int(config.max_steps))
    frames = []
    body_id = None
    if external_force is not None:
        body_index, _, _ = external_force
        body_id = int(scene.model.body(scene.objects[body_index].body_name).id)
    for i in range(steps):
        if external_force is not None:
            _, force, until = external_force
            scene.data.xfrc_applied[body_id, :3] = force if i < until else 0.0
        scene.set_occluder_pose(_OCCLUDER_POS)
        scene.step(every)
        # Counter gravity for the capture; the drift inside one tick is <0.1 mm.
        scene.set_occluder_pose(_OCCLUDER_POS)
        frames.append(_capture(scene, rig, scene_id, (i + 1) * every * dt, config.render))
    return frames


def _visibility_stats(rig: SensorRig, frames, occ_id: int, gt_events) -> dict:
    """Segmentation-based check that the occluder actually hides the objects."""
    if not frames or frames[0].segmentation is None:
        return {"measured": False, "reason": "render disabled"}
    obj_ids = {
        obj.instance_id: int(rig.scene.model.geom(obj.geom_name).id)
        for obj in rig.scene.objects
    }
    visible_obj = []
    visible_occ = []
    for frame in frames:
        seg = frame.segmentation[..., 0]
        visible_obj.append(sum(int((seg == gid).sum()) for gid in obj_ids.values()))
        visible_occ.append(int((seg == occ_id).sum()))

    def _window_stats(indices: list[int]) -> dict | None:
        if not indices:
            return None
        vals = [visible_obj[i] for i in indices]
        return {
            "frames": len(vals),
            "object_pixels_min": int(min(vals)),
            "object_pixels_median": float(np.median(vals)),
            "object_pixels_max": int(max(vals)),
            "frames_fully_hidden": int(sum(1 for n in vals if n == 0)),
        }

    contact_idx = [
        i for i, frame in enumerate(frames)
        if any(e.onset - 1e-9 <= frame.t <= (e.offset or e.onset) + 1e-9
               for e in gt_events)
    ]
    return {
        "measured": True,
        "object_pixels_min": int(min(visible_obj)),
        "object_pixels_median": float(np.median(visible_obj)),
        "object_pixels_max": int(max(visible_obj)),
        "occluder_pixels_median": float(np.median(visible_occ)),
        "frames_fully_hidden": int(sum(1 for n in visible_obj if n == 0)),
        "frames": int(len(visible_obj)),
        "during_gt_contact": _window_stats(contact_idx),
    }


@register_scenario("occlusion")
def occlusion(config: ContactConfig) -> ContactScenario:
    """Collision happens while the camera view of both objects is blocked."""
    scene = build_scene(
        SceneConfig(
            n_objects=2, seed=config.seed, width=config.width,
            height=config.height, bins=0, occlusion=True,
        ),
        categories=["box", "ball"],
        instance_indices=[0, 0],
        occlusion=True,
    )
    rig = SensorRig(scene, config.width, config.height)
    target, mover = scene.objects[0], scene.objects[1]
    scene.set_object_pose(0, [0.0, 0.0, target.rest_height + 0.002])
    scene.set_object_pose(1, [-0.18, 0.0, mover.rest_height + 0.002])
    scene.set_occluder_size(_OCCLUDER_SIZE)
    scene.set_occluder_pose(_OCCLUDER_POS)
    scene.forward()
    frames = _run_holding_occluder(
        scene, rig, f"occ{config.seed}", config,
        external_force=(1, np.array([2.5, 0.0, 0.0]), 60),
    )
    gt = _gt(frames, config)
    occ_id = int(scene.model.geom(scene.occluder.geom_name).id)
    visibility = _visibility_stats(rig, frames, occ_id, gt)
    rig.close()
    return ContactScenario(
        scenario_id=f"occ{config.seed}",
        frames=frames,
        gt_events=gt,
        metadata={
            "kind": "occlusion",
            "seed": config.seed,
            "conflict": "vision_occluded; force/geometry unobstructed",
            "occluder": {"pos": _OCCLUDER_POS, "size": _OCCLUDER_SIZE},
            "visibility": visibility,
        },
    )


@register_scenario("conflict_force_only")
def conflict_force_only(config: ContactConfig) -> ContactScenario:
    """Real contact; geometry is misled by an offset scene-model pose."""
    scene = build_scene(
        SceneConfig(
            n_objects=2, seed=config.seed, width=config.width,
            height=config.height, bins=0,
        ),
        categories=["box", "ball"],
        instance_indices=[0, 0],
    )
    rig = SensorRig(scene, config.width, config.height)
    target, mover = scene.objects[0], scene.objects[1]
    scene.set_object_pose(0, [0.0, 0.0, target.rest_height + 0.002])
    scene.set_object_pose(1, [-0.18, 0.0, mover.rest_height + 0.002])
    scene.forward()
    frames = _run(
        scene, rig, f"forceonly{config.seed}", config,
        external_force=(1, np.array([2.5, 0.0, 0.0]), 60),
    )
    gt = _gt(frames, config)
    # Observation-model error: the tracked pose of the mover is offset.  The
    # physics/contact stream is untouched.
    for frame in frames:
        entry = frame.scene_model.objects[mover.instance_id]
        entry.pos = np.asarray(entry.pos, dtype=np.float64) + np.array(
            [0.0, _FORCE_ONLY_POSE_OFFSET, 0.0]
        )
    rig.close()
    return ContactScenario(
        scenario_id=f"forceonly{config.seed}",
        frames=frames,
        gt_events=gt,
        metadata={
            "kind": "conflict_force_only",
            "seed": config.seed,
            "conflict": "force/tactile contact, geometry no-contact",
            "corruption": {
                "type": "scene_model_pose_offset",
                "object": mover.instance_id,
                "offset_m": [0.0, _FORCE_ONLY_POSE_OFFSET, 0.0],
            },
        },
    )


@register_scenario("conflict_geometry_only")
def conflict_geometry_only(config: ContactConfig) -> ContactScenario:
    """Geometry predicts contact; there is a real surface gap and no force."""
    scene = build_scene(
        SceneConfig(
            n_objects=2, seed=config.seed, width=config.width,
            height=config.height, bins=0,
        ),
        categories=["box", "ball"],
        instance_indices=[0, 0],
    )
    rig = SensorRig(scene, config.width, config.height)
    _place_near(scene, _GEOMETRY_ONLY_SEPARATION)
    frames = _run(scene, rig, f"geomonly{config.seed}", config)
    gt = _gt(frames, config)
    # Observation-model error: inflated radii (e.g. an over-large eps / radius
    # calibration), not inflated physical objects.
    for frame in frames:
        for oid in (scene.objects[0].instance_id, scene.objects[1].instance_id):
            entry = frame.scene_model.objects[oid]
            entry.radius = float(entry.radius) + _GEOMETRY_ONLY_RADIUS_INFLATION
    rig.close()
    return ContactScenario(
        scenario_id=f"geomonly{config.seed}",
        frames=frames,
        gt_events=gt,
        metadata={
            "kind": "conflict_geometry_only",
            "seed": config.seed,
            "conflict": "geometry contact, force/tactile no-contact",
            "corruption": {
                "type": "scene_model_radius_inflation",
                "inflation_m": _GEOMETRY_ONLY_RADIUS_INFLATION,
            },
            "placement_separation_m": _GEOMETRY_ONLY_SEPARATION,
        },
    )


@register_scenario("near_miss_hard")
def near_miss_hard(config: ContactConfig) -> ContactScenario:
    """Objects closer than ``near_miss`` but still not touching; false-alarm test."""
    scene = build_scene(
        SceneConfig(
            n_objects=2, seed=config.seed, width=config.width,
            height=config.height, bins=0,
        ),
        categories=["box", "ball"],
        instance_indices=[0, 0],
    )
    rig = SensorRig(scene, config.width, config.height)
    _place_near(scene, _NEAR_MISS_HARD_SEPARATION)
    frames = _run(scene, rig, f"nearhard{config.seed}", config)
    rig.close()
    return ContactScenario(
        scenario_id=f"nearhard{config.seed}",
        frames=frames,
        gt_events=[],
        metadata={
            "kind": "near_miss_hard",
            "seed": config.seed,
            "conflict": "none; no-contact false-alarm test",
            "placement_separation_m": _NEAR_MISS_HARD_SEPARATION,
        },
    )


@register_scenario("near_miss_clear")
def near_miss_clear(config: ContactConfig) -> ContactScenario:
    """Objects separated by more than ``eps``; unambiguously no contact.

    This is the clean control for ``near_miss_hard``: geometry's surface-gap
    model should stay below threshold, so any event here is a true false alarm.
    """
    scene = build_scene(
        SceneConfig(
            n_objects=2, seed=config.seed, width=config.width,
            height=config.height, bins=0,
        ),
        categories=["box", "ball"],
        instance_indices=[0, 0],
    )
    rig = SensorRig(scene, config.width, config.height)
    _place_near(scene, _NEAR_MISS_CLEAR_SEPARATION)
    frames = _run(scene, rig, f"nearclear{config.seed}", config)
    rig.close()
    return ContactScenario(
        scenario_id=f"nearclear{config.seed}",
        frames=frames,
        gt_events=[],
        metadata={
            "kind": "near_miss_clear",
            "seed": config.seed,
            "conflict": "none; clear no-contact control",
            "placement_separation_m": _NEAR_MISS_CLEAR_SEPARATION,
        },
    )


__all__ = [
    "conflict_force_only",
    "conflict_geometry_only",
    "near_miss_clear",
    "near_miss_hard",
    "occlusion",
]
