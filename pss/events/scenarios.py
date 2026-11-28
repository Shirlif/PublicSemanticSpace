"""MuJoCo contact-event scenarios: drop, collide, near-miss."""

from __future__ import annotations

import numpy as np

from pss.scene.model import ObjectState, SceneModel
from pss.scene.sim3d import _camera_from_rig
from pss.sim.scene import SceneConfig, build_scene
from pss.sim.sensors import SensorRig

from .bench_api import ContactConfig, ContactScenario, register_scenario
from .gt import events_from_contacts
from .model import ContactFrame


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


def _scene_model(scene, rig: SensorRig, scene_id: str) -> SceneModel:
    poses = rig.object_poses()
    objects = {}
    for obj in scene.objects:
        pos = np.asarray(poses[obj.instance_id]["pos"], dtype=np.float64)
        objects[obj.instance_id] = ObjectState(
            id=obj.instance_id,
            pos=pos,
            shape=obj.geom_type,
            size=str(obj.size),
            radius=_radius_for(obj.geom_type, obj.size),
            provenance={"source": "mujoco", "category": str(obj.category)},
        )
    return SceneModel(scene_id=scene_id, objects=objects)


def _capture(
    scene, rig: SensorRig, scene_id: str, t: float, render: bool
) -> ContactFrame:
    camera = None
    rgb = depth = seg = None
    if render:
        rgb = rig.render_rgb()
        depth = rig.render_depth()
        seg = rig.render_segmentation()
        camera = _camera_from_rig(rig)
    return ContactFrame(
        t=float(t),
        contacts=rig.contacts(),
        tactile=rig.tactile_maps(),
        poses={
            oid: np.asarray(p["pos"], dtype=np.float64)
            for oid, p in rig.object_poses().items()
        },
        scene_model=_scene_model(scene, rig, scene_id),
        rgb=rgb,
        depth=depth,
        segmentation=seg,
        camera=camera,
    )


def _run(
    scene,
    rig: SensorRig,
    scene_id: str,
    config: ContactConfig,
    *,
    external_force: tuple[int, np.ndarray, int] | None = None,
) -> list[ContactFrame]:
    dt = float(scene.model.opt.timestep)
    frames: list[ContactFrame] = []
    steps = max(1, int(config.max_steps))
    every = max(1, int(config.capture_every))
    for i in range(steps):
        if external_force is not None:
            body_index, force, until = external_force
            body_id = int(scene.model.body(scene.objects[body_index].body_name).id)
            scene.data.xfrc_applied[body_id, :3] = force if i < until else 0.0
        scene.step(every)
        t = (i + 1) * every * dt
        frames.append(_capture(scene, rig, scene_id, t, config.render))
    return frames


@register_scenario("drop")
def drop(config: ContactConfig) -> ContactScenario:
    scene = build_scene(
        SceneConfig(
            n_objects=2, seed=config.seed, width=config.width,
            height=config.height, bins=0,
        ),
        categories=["box", "ball"],
        instance_indices=[0, 0],
    )
    rig = SensorRig(scene, config.width, config.height)
    target, dropped = scene.objects[0], scene.objects[1]
    scene.set_object_pose(0, [0.0, 0.0, target.rest_height + 0.002])
    scene.set_object_pose(1, [0.0, 0.0, dropped.rest_height + 0.35])
    scene.forward()
    frames = _run(scene, rig, f"drop{config.seed}", config)
    gt = events_from_contacts(
        frames,
        threshold=config.gt_threshold,
        min_duration=config.min_duration,
        max_gap=config.max_gap,
    )
    rig.close()
    return ContactScenario(
        scenario_id=f"drop{config.seed}",
        frames=frames,
        gt_events=gt,
        metadata={"kind": "drop", "seed": config.seed},
    )


@register_scenario("collide")
def collide(config: ContactConfig) -> ContactScenario:
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
        scene, rig, f"collide{config.seed}", config,
        external_force=(1, np.array([2.5, 0.0, 0.0]), 60),
    )
    gt = events_from_contacts(
        frames,
        threshold=config.gt_threshold,
        min_duration=config.min_duration,
        max_gap=config.max_gap,
    )
    rig.close()
    return ContactScenario(
        scenario_id=f"collide{config.seed}",
        frames=frames,
        gt_events=gt,
        metadata={"kind": "collide", "seed": config.seed},
    )


@register_scenario("near_miss")
def near_miss(config: ContactConfig) -> ContactScenario:
    scene = build_scene(
        SceneConfig(
            n_objects=2, seed=config.seed, width=config.width,
            height=config.height, bins=0,
        ),
        categories=["box", "ball"],
        instance_indices=[0, 0],
    )
    rig = SensorRig(scene, config.width, config.height)
    a, b = scene.objects[0], scene.objects[1]
    scene.set_object_pose(0, [-0.15, 0.15, a.rest_height + 0.002])
    scene.set_object_pose(1, [0.05, -0.15, b.rest_height + 0.002])
    scene.forward()
    frames = _run(scene, rig, f"near{config.seed}", config)
    rig.close()
    return ContactScenario(
        scenario_id=f"near{config.seed}",
        frames=frames,
        gt_events=[],
        metadata={"kind": "near_miss", "seed": config.seed},
    )


__all__ = ["collide", "drop", "near_miss"]
