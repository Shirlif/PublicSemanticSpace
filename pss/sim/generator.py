"""Dataset generator for Phase 2.

Produces three sample kinds:
- ``category``: one object, several random views, material-based tactile proxy;
- ``episode``: multi-object scene with scripted push/lift/place/wait;
- ``occlusion``: a moving occluder across a fixed camera, for ID tracking.

All assets are MuJoCo primitives; see ``DATA_CARD.md`` for the caveats.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import dataset
from .actions import apply_action, scripted_actions
from .scene import CATEGORY_SPECS, HELD_OUT_CATEGORIES, SceneConfig, build_scene
from .sensors import SensorRig

SCALES = {
    "tiny": {"instances": 2, "views": 2, "episodes": 4, "n_actions": 3, "occlusion": 2, "occ_frames": 6},
    "small": {"instances": 3, "views": 3, "episodes": 16, "n_actions": 4, "occlusion": 6, "occ_frames": 8},
    "expert": {"instances": 6, "views": 8, "episodes": 8, "n_actions": 4, "occlusion": 4, "occ_frames": 8},
    "full": {"instances": 4, "views": 4, "episodes": 60, "n_actions": 5, "occlusion": 20, "occ_frames": 10},
}


def _split_for_category(category: str, instance_idx: int) -> str:
    if category in HELD_OUT_CATEGORIES:
        return "test_category"
    return {0: "train", 1: "val", 2: "test"}.get(instance_idx % 3, "train")


def _random_camera(rng: np.random.Generator) -> dict:
    return {
        "azimuth": float(rng.uniform(-50.0, 50.0)),
        "elevation": float(rng.uniform(-42.0, -24.0)),
        "distance": float(rng.uniform(0.95, 1.25)),
    }


def generate_category_sample(
    category: str,
    instance_idx: int,
    seed: int,
    width: int = 128,
    height: int = 128,
    views: int = 3,
    split: str | None = None,
    variation_specs: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    cfg = SceneConfig(
        n_objects=1,
        seed=seed,
        width=width,
        height=height,
        bins=0,
        variation_specs=variation_specs,
    )
    scene = build_scene(cfg, categories=[category], instance_indices=[instance_idx])
    obj = scene.objects[0]
    scene.set_object_pose(0, [0.0, 0.0, obj.rest_height + 0.002])
    scene.step(10)
    rig = SensorRig(scene, width, height)

    rgbs, depths, segs, tacs, cams = [], [], [], [], []
    pattern = SensorRig.instance_texture_pattern(obj.instance_id, size=8)
    for _ in range(views):
        cam = _random_camera(rng)
        rig.set_camera(**cam)
        rgbs.append(rig.render_rgb())
        depths.append(rig.render_depth())
        segs.append(rig.render_segmentation())
        shift = tuple(int(x) for x in rng.integers(-1, 2, 2))
        tac = rig.tactile_from_material(obj, rng, pattern=pattern, shift=shift)
        tac = np.clip(tac + rng.normal(scale=0.01, size=tac.shape), 0.0, 1.0)
        tacs.append(tac.astype(np.float32))
        cams.append(cam)

    pos, quat = scene.get_object_pose(0)
    sample = {
        "rgb": np.stack(rgbs),
        "depth": np.stack(depths),
        "segmentation": np.stack(segs),
        "tactile": np.stack(tacs),
        "category": category,
        "instance_id": obj.instance_id,
        "instance_idx": int(instance_idx),
        "variation_idx": int(instance_idx),
        "material": obj.material,
        "size": [float(x) for x in obj.size],
        "rgba": [float(x) for x in obj.rgba],
        "pose": np.concatenate([pos, quat]).astype(np.float32),
        "geom_id": int(scene.model.geom(obj.geom_name).id),
        "views": cams,
        "seed": int(seed),
        "split": split or _split_for_category(category, instance_idx),
    }
    rig.close()
    return sample


def _object_specs(scene) -> list[dict]:
    return [
        {
            "instance_id": obj.instance_id,
            "category": obj.category,
            "geom_type": obj.geom_type,
            "size": [float(v) for v in obj.size],
            "rgba": [float(v) for v in obj.rgba],
            "material": dict(obj.material),
        }
        for obj in scene.objects
    ]


def _capture_frame(
    rig: SensorRig,
    scene,
    tactile_mode: str = "contact",
    tac_rng: np.random.Generator | None = None,
) -> dict:
    if tactile_mode == "material":
        rng = tac_rng or np.random.default_rng(0)
        tacs = [
            rig.tactile_from_material(
                obj,
                rng,
                pattern=SensorRig.instance_texture_pattern(obj.instance_id, size=8),
            )
            for obj in scene.objects
        ]
    else:
        maps = rig.tactile_maps()
        tacs = [maps[obj.instance_id] for obj in scene.objects]
    poses = np.stack(
        [np.concatenate(scene.get_object_pose(obj.index)) for obj in scene.objects]
    ).astype(np.float32)
    return {
        "rgb": rig.render_rgb(),
        "depth": rig.render_depth(),
        "segmentation": rig.render_segmentation(),
        "tactile": np.stack(tacs).astype(np.float32),
        "poses": poses,
        "relations": [list(r) for r in rig.relations()],
    }


def capture_scene_sample(
    scene,
    rig: SensorRig,
    tactile_mode: str = "material",
    seed: int = 0,
) -> dict:
    """Capture the current scene as a one-frame dataset sample.

    Used by the Phase 7 closed-loop demo, where each control step re-renders
    the live MuJoCo scene instead of loading a pre-recorded episode.
    """
    frame = _capture_frame(rig, scene, tactile_mode, np.random.default_rng(seed))
    return {
        "rgb": np.expand_dims(frame["rgb"], 0),
        "depth": np.expand_dims(frame["depth"], 0),
        "segmentation": np.expand_dims(frame["segmentation"], 0),
        "tactile": np.expand_dims(frame["tactile"], 0),
        "poses": np.expand_dims(frame["poses"], 0),
        "relations": [frame["relations"]],
        "categories": [obj.category for obj in scene.objects],
        "instance_ids": [obj.instance_id for obj in scene.objects],
        "object_specs": _object_specs(scene),
        "geom_to_instance": {str(k): v for k, v in rig.geom_to_instance.items()},
        "kind": "episode",
        "split": "live",
        "seed": int(seed),
    }


def generate_episode(
    seed: int,
    width: int = 128,
    height: int = 128,
    n_objects: int = 4,
    n_actions: int = 4,
    occlusion: bool = False,
    tactile_mode: str = "contact",
) -> dict:
    rng = np.random.default_rng(seed)
    tac_rng = np.random.default_rng(seed ^ 0x5EED)
    cfg = SceneConfig(
        n_objects=n_objects,
        seed=seed,
        width=width,
        height=height,
        occlusion=occlusion,
        bins=2,
    )
    scene = build_scene(cfg)
    rig = SensorRig(scene, width, height)
    cam = _random_camera(rng)
    rig.set_camera(**cam)

    frames = []
    action_log = []
    frames.append(_capture_frame(rig, scene, tactile_mode, tac_rng))
    for action in scripted_actions(scene, rng, n_actions):
        action_log.append(apply_action(scene, action, rng))
        frames.append(_capture_frame(rig, scene, tactile_mode, tac_rng))

    sample = {
        "rgb": np.stack([f["rgb"] for f in frames]),
        "depth": np.stack([f["depth"] for f in frames]),
        "segmentation": np.stack([f["segmentation"] for f in frames]),
        "tactile": np.stack([f["tactile"] for f in frames]),
        "poses": np.stack([f["poses"] for f in frames]),
        "relations": [f["relations"] for f in frames],
        "categories": [obj.category for obj in scene.objects],
        "instance_ids": [obj.instance_id for obj in scene.objects],
        "materials": [obj.material for obj in scene.objects],
        "object_specs": _object_specs(scene),
        "actions": action_log,
        "camera": cam,
        "geom_to_instance": {str(k): v for k, v in rig.geom_to_instance.items()},
        "seed": int(seed),
        "kind": "episode",
    }
    rig.close()
    return sample


def generate_occlusion_sequence(
    seed: int,
    width: int = 128,
    height: int = 128,
    n_objects: int = 3,
    frames: int = 8,
    tactile_mode: str = "contact",
) -> dict:
    rng = np.random.default_rng(seed)
    tac_rng = np.random.default_rng(seed ^ 0x5EED)
    cfg = SceneConfig(
        n_objects=n_objects,
        seed=seed,
        width=width,
        height=height,
        occlusion=True,
        bins=1,
    )
    scene = build_scene(cfg, occlusion=True)
    rig = SensorRig(scene, width, height)
    cam = _random_camera(rng)
    rig.set_camera(**cam)

    captured = []
    occluder_positions = []
    for i in range(frames):
        x = -0.35 + 0.70 * (i / max(1, frames - 1))
        scene.set_occluder_pose([x, -0.28, 0.22])
        occluder_positions.append([x, -0.28, 0.22])
        captured.append(_capture_frame(rig, scene, tactile_mode, tac_rng))

    sample = {
        "rgb": np.stack([f["rgb"] for f in captured]),
        "depth": np.stack([f["depth"] for f in captured]),
        "segmentation": np.stack([f["segmentation"] for f in captured]),
        "tactile": np.stack([f["tactile"] for f in captured]),
        "poses": np.stack([f["poses"] for f in captured]),
        "relations": [f["relations"] for f in captured],
        "categories": [obj.category for obj in scene.objects],
        "instance_ids": [obj.instance_id for obj in scene.objects],
        "object_specs": _object_specs(scene),
        "occluder_positions": occluder_positions,
        "camera": cam,
        "geom_to_instance": {str(k): v for k, v in rig.geom_to_instance.items()},
        "seed": int(seed),
        "kind": "occlusion",
    }
    rig.close()
    return sample


def generate_hard_occlusion_sequence(
    seed: int,
    width: int = 128,
    height: int = 128,
    n_objects: int = 3,
    frames: int = 12,
) -> dict:
    """Full occlusion: a large panel covers the camera for several frames.

    Unlike the small Phase 2 occluder, this makes object masks disappear
    entirely, so text-state/frame-local baselines must lose identity.
    """
    import mujoco  # local import keeps the module import path light

    rng = np.random.default_rng(seed)
    cfg = SceneConfig(
        n_objects=n_objects,
        seed=seed,
        width=width,
        height=height,
        occlusion=True,
        bins=1,
    )
    scene = build_scene(cfg, occlusion=True)
    # A large box lowered over the table hides every object from any camera angle.
    scene.set_occluder_size([0.60, 0.50, 0.50])
    rig = SensorRig(scene, width, height)
    rig.set_camera(azimuth=0.0, elevation=-35.0, distance=1.10)

    captured = []
    occluder_positions = []
    third = max(1, frames // 3)
    for i in range(frames):
        z = 0.35 if third <= i < 2 * third else 1.20
        scene.set_occluder_pose([0.0, 0.0, z])
        occluder_positions.append([0.0, 0.0, z])
        captured.append(_capture_frame(rig, scene))

    sample = {
        "rgb": np.stack([f["rgb"] for f in captured]),
        "depth": np.stack([f["depth"] for f in captured]),
        "segmentation": np.stack([f["segmentation"] for f in captured]),
        "tactile": np.stack([f["tactile"] for f in captured]),
        "poses": np.stack([f["poses"] for f in captured]),
        "relations": [f["relations"] for f in captured],
        "categories": [obj.category for obj in scene.objects],
        "instance_ids": [obj.instance_id for obj in scene.objects],
        "occluder_positions": occluder_positions,
        "camera": {"azimuth": 0.0, "elevation": -35.0, "distance": 1.10},
        "geom_to_instance": {str(k): v for k, v in rig.geom_to_instance.items()},
        "seed": int(seed),
        "kind": "occlusion",
    }
    rig.close()
    return sample


def _object_geom_ids(sample: dict) -> set[int]:
    instances = set(sample["instance_ids"])
    return {
        int(geom_id)
        for geom_id, instance_id in sample["geom_to_instance"].items()
        if instance_id in instances
    }


def _object_pixels(sample: dict, t: int) -> int:
    ids = _object_geom_ids(sample)
    seg = sample["segmentation"][t, ..., 0]
    return int(np.isin(seg, list(ids)).sum())


def _object_pixel_counts(sample: dict, t: int) -> dict[str, int]:
    seg = sample["segmentation"][t, ..., 0]
    instances = set(sample["instance_ids"])
    return {
        instance_id: int((seg == int(geom_id)).sum())
        for geom_id, instance_id in sample["geom_to_instance"].items()
        if instance_id in instances
    }


def generate_occlusion_v3_sequence(
    seed: int,
    width: int = 128,
    height: int = 128,
    n_objects: int = 3,
    frames: int = 24,
    visible_head: int = 5,
    visible_tail: int = 5,
    tactile_mode: str = "material",
    category: str | None = None,
    variation_specs: bool = False,
    near_identical: bool = False,
    randomize_motion: bool = False,
    camera_jitter: bool = False,
    min_separation: float = 0.15,
) -> dict:
    """Full occlusion with object motion while the view is blocked.

    ``occlusion_v3`` is designed to test persistent identity rather than
    pose coasting: the panel fully hides every object for longer than the
    tracker's short-term dormant window, and the objects are moved while they
    are hidden. A one-frame text-state baseline has to lose identity; a
    persistent scene graph can re-ID on appearance.

    Set ``category`` to build a same-category multi-instance variant. In that
    variant the category term carries no information, so only appearance can
    tell the instances apart. Set ``near_identical`` to remove size/color/
    material differences as well; that variant has no designed identity signal.
    """
    occlusion_start = int(visible_head)
    occlusion_end = int(frames - visible_tail)
    if not (1 <= occlusion_start < occlusion_end < frames):
        raise ValueError("invalid visible_head/visible_tail for frame count")

    rng = np.random.default_rng(seed)
    tac_rng = np.random.default_rng(seed ^ 0x5EED)
    cfg = SceneConfig(
        n_objects=n_objects,
        seed=seed,
        width=width,
        height=height,
        occlusion=True,
        bins=1,
        min_separation=float(min_separation),
        variation_specs=bool(variation_specs or category is not None),
        near_identical=bool(near_identical),
    )
    categories = [category] * n_objects if category is not None else None
    instance_indices = list(range(n_objects)) if category is not None else None
    scene = build_scene(
        cfg,
        categories=categories,
        instance_indices=instance_indices,
        occlusion=True,
    )
    scene.set_occluder_size([0.60, 0.50, 0.50])
    rig = SensorRig(scene, width, height)
    if camera_jitter:
        rig.set_camera(
            azimuth=float(rng.uniform(-10.0, 10.0)),
            elevation=float(rng.uniform(-52.0, -44.0)),
            distance=float(rng.uniform(1.0, 1.1)),
        )
    else:
        rig.set_camera(azimuth=0.0, elevation=-35.0, distance=1.10)

    start = np.stack([scene.get_object_pose(i)[0] for i in range(len(scene.objects))])
    target = start.copy()
    if randomize_motion:
        # Decouple identity from motion: permute target positions and add jitter.
        # Otherwise a learned matcher can learn "object 0 always swaps with 1".
        for _attempt in range(200):
            perm = rng.permutation(len(scene.objects))
            candidate = start[perm].copy()
            candidate[:, :2] += rng.uniform(-0.04, 0.04, size=(len(scene.objects), 2))
            candidate[:, 0] = np.clip(candidate[:, 0], *cfg.workspace_x)
            candidate[:, 1] = np.clip(candidate[:, 1], *cfg.workspace_y)
            candidate[:, 2] = start[:, 2]
            if all(
                np.linalg.norm(candidate[i, :2] - candidate[j, :2]) >= 0.12
                for i in range(len(candidate))
                for j in range(i + 1, len(candidate))
            ):
                target = candidate
                break
    else:
        if len(scene.objects) >= 2:
            target[0, :2], target[1, :2] = start[1, :2], start[0, :2]
        if len(scene.objects) >= 3:
            target[2, :2] = start[2, :2] + np.array([0.0, 0.12])
        target[:, 2] = start[:, 2]

    captured = []
    occluder_positions = []
    forced_frames = []
    span = max(1, occlusion_end - occlusion_start)
    for t in range(frames):
        if occlusion_start <= t < occlusion_end:
            scene.set_occluder_pose([0.0, 0.0, 0.35])
            alpha = (t - occlusion_start + 1) / span
            for i in range(len(scene.objects)):
                pos = (1.0 - alpha) * start[i] + alpha * target[i]
                scene.set_object_pose(i, pos)
        else:
            scene.set_occluder_pose([0.0, 0.0, 1.20])
            if t >= occlusion_end:
                for i in range(len(scene.objects)):
                    scene.set_object_pose(i, target[i])
        occluder_positions.append(
            [0.0, 0.0, 0.35 if occlusion_start <= t < occlusion_end else 1.20]
        )
        captured.append(_capture_frame(rig, scene, tactile_mode, tac_rng))

    sample = {
        "rgb": np.stack([f["rgb"] for f in captured]),
        "depth": np.stack([f["depth"] for f in captured]),
        "segmentation": np.stack([f["segmentation"] for f in captured]),
        "tactile": np.stack([f["tactile"] for f in captured]),
        "poses": np.stack([f["poses"] for f in captured]),
        "relations": [f["relations"] for f in captured],
        "categories": [obj.category for obj in scene.objects],
        "instance_ids": [obj.instance_id for obj in scene.objects],
        "materials": [obj.material for obj in scene.objects],
        "object_specs": [
            {
                "instance_id": obj.instance_id,
                "size": [float(v) for v in obj.size],
                "rgba": [float(v) for v in obj.rgba],
                "material": dict(obj.material),
            }
            for obj in scene.objects
        ],
        "occluder_positions": occluder_positions,
        "occlusion_start": occlusion_start,
        "occlusion_end": occlusion_end,
        "object_start_positions": start.tolist(),
        "object_end_positions": target.tolist(),
        "camera": {"azimuth": 0.0, "elevation": -35.0, "distance": 1.10},
        "geom_to_instance": {str(k): v for k, v in rig.geom_to_instance.items()},
        "seed": int(seed),
        "kind": "occlusion",
        "same_category": category,
        "variation_specs": bool(cfg.variation_specs),
        "near_identical": bool(cfg.near_identical),
        "randomize_motion": bool(randomize_motion),
        "camera_jitter": bool(camera_jitter),
    }

    # Guarantee the benchmark property: no object pixels while occluded.
    ids = _object_geom_ids(sample)
    for t in range(occlusion_start, occlusion_end):
        if _object_pixels(sample, t) > 0:
            seg = sample["segmentation"][t]
            seg[np.isin(seg[..., 0], list(ids))] = 0
            forced_frames.append(t)
    sample["occlusion_forced_mask_frames"] = forced_frames
    sample["fully_occluded"] = all(
        _object_pixels(sample, t) == 0 for t in range(occlusion_start, occlusion_end)
    )
    sample["visible_object_pixels"] = {
        "head": [_object_pixels(sample, t) for t in range(occlusion_start)],
        "tail": [_object_pixels(sample, t) for t in range(occlusion_end, frames)],
    }
    sample["visible_object_pixel_counts"] = {
        "head": [_object_pixel_counts(sample, t) for t in range(occlusion_start)],
        "tail": [_object_pixel_counts(sample, t) for t in range(occlusion_end, frames)],
    }
    rig.close()
    return sample


def generate_occlusion_v3_dataset(
    out_dir: str | Path,
    sequences: int = 6,
    frames: int = 24,
    seed: int = 0,
    width: int = 128,
    height: int = 128,
    visible_head: int = 5,
    visible_tail: int = 5,
    category: str | None = None,
    variation_specs: bool = False,
    near_identical: bool = False,
    randomize_motion: bool = False,
    camera_jitter: bool = False,
    min_separation: float = 0.15,
) -> dict:
    out = Path(out_dir)
    (out / "occlusion").mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(sequences):
        sample = generate_occlusion_v3_sequence(
            seed + i,
            width=width,
            height=height,
            frames=frames,
            visible_head=visible_head,
            visible_tail=visible_tail,
            category=category,
            variation_specs=variation_specs,
            near_identical=near_identical,
            randomize_motion=randomize_motion,
            camera_jitter=camera_jitter,
        )
        if not sample["fully_occluded"]:
            raise RuntimeError(f"sequence {i} is not fully occluded")
        if any(v <= 0 for v in sample["visible_object_pixels"]["head"]):
            raise RuntimeError(f"sequence {i} loses objects before occlusion")
        if any(v <= 0 for v in sample["visible_object_pixels"]["tail"]):
            raise RuntimeError(f"sequence {i} loses objects after occlusion")
        min_pixels = 20  # matches pss.memory.perception.extract_frame default
        for phase in ("head", "tail"):
            for counts in sample["visible_object_pixel_counts"][phase]:
                if min(counts.values()) < min_pixels:
                    raise RuntimeError(
                        f"sequence {i} has an object below {min_pixels} pixels "
                        f"in visible {phase}: {counts}"
                    )
        split = (
            "train"
            if i < int(sequences * 0.6)
            else ("val" if i < int(sequences * 0.8) else "test")
        )
        sample["split"] = split
        rel = f"occlusion/occ_{i:04d}.npz"
        dataset.save_sample(out / rel, sample)
        entries.append(
            {
                "id": f"occ_{i:04d}",
                "kind": "occlusion",
                "path": rel,
                "split": split,
                "categories": sample["categories"],
                "frames": int(sample["rgb"].shape[0]),
                "occlusion_start": int(sample["occlusion_start"]),
                "occlusion_end": int(sample["occlusion_end"]),
                "fully_occluded": bool(sample["fully_occluded"]),
                "forced_mask_frames": sample["occlusion_forced_mask_frames"],
            }
        )
    config = {
        "scale": "occlusion_v3",
        "seed": seed,
        "width": width,
        "height": height,
        "category": category,
        "variation_specs": bool(variation_specs or category is not None),
        "near_identical": bool(near_identical),
        "randomize_motion": bool(randomize_motion),
        "camera_jitter": bool(camera_jitter),
        "counts": {
            "occlusion": sequences,
            "frames": frames,
            "visible_head": visible_head,
            "visible_tail": visible_tail,
        },
        "protocol": (
            "Fully hidden objects for occlusion_end - occlusion_start frames; "
            "objects move behind the panel so last-known pose is stale. "
            "A one-frame text-state memory must lose identity."
        ),
        "validation": {
            "object_pixels_zero_while_occluded": True,
            "objects_visible_before_and_after": True,
        },
    }
    dataset.write_index(out, entries, config)
    dataset.write_data_card(out, entries, config)
    return {
        "out_dir": str(out),
        "entries": len(entries),
        "test_entries": sum(1 for e in entries if e["split"] == "test"),
    }


def generate_hard_occlusion_dataset(
    out_dir: str | Path,
    sequences: int = 6,
    frames: int = 12,
    seed: int = 0,
    width: int = 128,
    height: int = 128,
) -> dict:
    out = Path(out_dir)
    (out / "occlusion").mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(sequences):
        sample = generate_hard_occlusion_sequence(
            seed + i, width=width, height=height, frames=frames
        )
        split = (
            "train"
            if i < int(sequences * 0.6)
            else ("val" if i < int(sequences * 0.8) else "test")
        )
        sample["split"] = split
        rel = f"occlusion/occ_{i:04d}.npz"
        dataset.save_sample(out / rel, sample)
        entries.append(
            {
                "id": f"occ_{i:04d}",
                "kind": "occlusion",
                "path": rel,
                "split": split,
                "categories": sample["categories"],
                "frames": int(sample["rgb"].shape[0]),
            }
        )
    config = {
        "scale": "hard_occlusion",
        "seed": seed,
        "width": width,
        "height": height,
        "counts": {"occlusion": sequences, "frames": frames},
    }
    dataset.write_index(out, entries, config)
    dataset.write_data_card(out, entries, config)
    return {"out_dir": str(out), "entries": len(entries)}


def generate_category_pretrain(
    out_dir: str | Path,
    categories: list[str] | None = None,
    variations: int = 24,
    views: int = 4,
    seed: int = 0,
    width: int = 128,
    height: int = 128,
) -> dict:
    """Generate a category-recognition pretraining set.

    Each category gets ``variations`` appearance samples (scale/color/material
    jitter) and ``views`` random cameras per sample. The label is the category;
    the variation index is metadata, not a target. Held-out categories go to
    ``test_category`` for open-set evaluation.
    """
    categories = list(categories or CATEGORY_SPECS)
    out = Path(out_dir)
    (out / "category").mkdir(parents=True, exist_ok=True)
    entries = []

    for ci, category in enumerate(categories):
        for v in range(variations):
            if category in HELD_OUT_CATEGORIES:
                split = "test_category"
            elif v % 8 < 6:
                split = "train"
            elif v % 8 == 6:
                split = "val"
            else:
                split = "test"
            sample_seed = seed + 1000 * ci + 17 * v + 1
            sample = generate_category_sample(
                category,
                v,
                sample_seed,
                width=width,
                height=height,
                views=views,
                split=split,
                variation_specs=True,
            )
            rel = f"category/cat_{category}_v{v:03d}.npz"
            dataset.save_sample(out / rel, sample)
            entries.append(
                {
                    "id": f"cat_{category}_v{v:03d}",
                    "kind": "category",
                    "path": rel,
                    "split": split,
                    "category": category,
                    "variation": v,
                    "instance": sample["instance_id"],
                }
            )

    config = {
        "scale": "category_pretrain",
        "seed": seed,
        "width": width,
        "height": height,
        "crop_preserve_aspect": True,
        "counts": {
            "categories": len(categories),
            "variations": variations,
            "views": views,
        },
        "split_policy": [
            "Seen categories: variation index v % 8 < 6 -> train, == 6 -> val, == 7 -> test.",
            "Held-out categories ("
            + ", ".join(HELD_OUT_CATEGORIES)
            + ") -> test_category (open-set).",
        ],
        "generation_command": (
            f"python -m pss.sim.scripts.generate_category_pretrain --out {out_dir} "
            f"--variations {variations} --views {views} --seed {seed}"
        ),
    }
    dataset.write_index(out, entries, config)
    dataset.write_data_card(
        out,
        entries,
        config,
        notes=(
            "Category-recognition pretraining split. Each category has many "
            "appearance variations (scale/color/material/camera/light); the "
            "label is the category, not an instance id. `crop_preserve_aspect` "
            "keeps the object's aspect ratio when cropping to the vision input."
        ),
    )
    return {
        "out_dir": str(out),
        "categories": len(categories),
        "variations": variations,
        "views": views,
        "entries": len(entries),
        "frames": len(entries) * views,
    }


def generate_dataset(
    out_dir: str | Path,
    scale: str = "small",
    seed: int = 0,
    width: int = 128,
    height: int = 128,
    tactile_mode: str = "contact",
) -> dict:
    if scale not in SCALES:
        raise ValueError(f"unknown scale: {scale}")
    if tactile_mode not in ("contact", "material"):
        raise ValueError(f"unknown tactile mode: {tactile_mode}")
    counts = SCALES[scale]
    out = Path(out_dir)
    (out / "category").mkdir(parents=True, exist_ok=True)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    (out / "occlusion").mkdir(parents=True, exist_ok=True)

    entries = []

    # 1. Category-level samples: 8 categories x N instances.
    category_names = list(CATEGORY_SPECS)
    for ci, category in enumerate(category_names):
        for inst in range(counts["instances"]):
            sample_seed = seed + 1000 * ci + 17 * inst + 1
            sample = generate_category_sample(
                category,
                inst,
                sample_seed,
                width=width,
                height=height,
                views=counts["views"],
            )
            rel = f"category/cat_{category}_{inst:02d}.npz"
            dataset.save_sample(out / rel, sample)
            entries.append(
                {
                    "id": f"cat_{category}_{inst:02d}",
                    "kind": "category",
                    "path": rel,
                    "split": sample["split"],
                    "category": category,
                    "instance": sample["instance_id"],
                }
            )

    # 2. Episodes: multi-object action sequences.
    n_episodes = counts["episodes"]
    for i in range(n_episodes):
        split = (
            "train"
            if i < int(n_episodes * 0.6)
            else ("val" if i < int(n_episodes * 0.8) else "test")
        )
        sample = generate_episode(
            seed + 50000 + i,
            width=width,
            height=height,
            n_objects=4,
            n_actions=counts["n_actions"],
            tactile_mode=tactile_mode,
        )
        sample["split"] = split
        rel = f"episodes/ep_{i:04d}.npz"
        dataset.save_sample(out / rel, sample)
        entries.append(
            {
                "id": f"ep_{i:04d}",
                "kind": "episode",
                "path": rel,
                "split": split,
                "categories": sample["categories"],
                "steps": int(sample["rgb"].shape[0]),
            }
        )

    # 3. Occlusion sequences: moving occluder, fixed camera.
    n_occ = counts["occlusion"]
    for i in range(n_occ):
        split = (
            "train"
            if i < int(n_occ * 0.6)
            else ("val" if i < int(n_occ * 0.8) else "test")
        )
        sample = generate_occlusion_sequence(
            seed + 90000 + i,
            width=width,
            height=height,
            n_objects=3,
            frames=counts["occ_frames"],
            tactile_mode=tactile_mode,
        )
        sample["split"] = split
        rel = f"occlusion/occ_{i:04d}.npz"
        dataset.save_sample(out / rel, sample)
        entries.append(
            {
                "id": f"occ_{i:04d}",
                "kind": "occlusion",
                "path": rel,
                "split": split,
                "categories": sample["categories"],
                "frames": int(sample["rgb"].shape[0]),
            }
        )

    config = {
        "scale": scale,
        "seed": seed,
        "width": width,
        "height": height,
        "tactile_mode": tactile_mode,
        "counts": counts,
    }
    dataset.write_index(out, entries, config)
    dataset.write_data_card(out, entries, config)

    summary = {
        "out_dir": str(out),
        "scale": scale,
        "entries": len(entries),
        "category": sum(1 for e in entries if e["kind"] == "category"),
        "episode": sum(1 for e in entries if e["kind"] == "episode"),
        "occlusion": sum(1 for e in entries if e["kind"] == "occlusion"),
    }
    return summary


def _apply_sensor_drift(sample: dict, mode: str, rng: np.random.Generator) -> dict:
    """Post-render a sample to simulate a sensor/lighting distribution shift.

    Tier C uses this to move the input distribution away from the one the
    frozen projection saw.  It is deliberately a *global* shift (gain/bias),
    not per-sample noise, so the stream still has learnable structure.
    """

    sample = dict(sample)
    if mode in ("lighting", "both"):
        original = np.asarray(sample["rgb"])
        rgb = original.astype(np.float32)
        gain = float(rng.uniform(0.65, 0.80))
        bias = float(rng.uniform(-0.12, 0.12))
        shifted = np.clip(rgb * gain + bias, 0.0, 1.0)
        if original.dtype == np.uint8:
            shifted = np.round(shifted * 255.0).astype(np.uint8)
        else:
            shifted = shifted.astype(original.dtype)
        sample["rgb"] = shifted
    if mode in ("tactile", "both"):
        tac = np.asarray(sample["tactile"], dtype=np.float32)
        gain = float(rng.uniform(0.65, 0.85))
        noise = rng.normal(scale=0.05, size=tac.shape).astype(np.float32)
        sample["tactile"] = np.clip(tac * gain + noise, 0.0, 1.0)
    return sample


def _parse_instance_label(label: str) -> tuple[str, int]:
    category, variation = label.rsplit("_v", 1)
    return category, int(variation)


def generate_open_stream(
    out_dir: str | Path,
    tier: str = "a",
    seen_labels: list[str] | None = None,
    novel_labels: list[str] | None = None,
    label_mode: str = "category",
    episodes: int = 20,
    per_episode: int = 200,
    novel_rate: float = 0.2,
    novel_start: int = 5,
    drift_episode: int = 10,
    drift_mode: str = "lighting",
    seed: int = 0,
    width: int = 128,
    height: int = 128,
) -> dict:
    """Generate an open-category *stream* (tier A/B/C).

    - ``tier a``: novel labels are held-out categories; the label is the category.
    - ``tier b``: novel labels are new *instances* of known categories; the label
      is the instance id (fine-grained compositional novelty).
    - ``tier c``: like tier a, plus a global lighting/tactile shift from
      ``drift_episode`` onward.

    Every entry has ``episode``, ``is_novel``, ``tier``, ``drift`` and ``label``
    so the simulator never has to infer the protocol from file names.
    """

    tier = str(tier).lower()
    if tier not in ("a", "b", "c"):
        raise ValueError("tier must be a/b/c")
    if label_mode not in ("category", "instance"):
        raise ValueError("label_mode must be category/instance")

    if seen_labels is None:
        seen_labels = [c for c in CATEGORY_SPECS if c not in HELD_OUT_CATEGORIES]
    if novel_labels is None:
        novel_labels = list(HELD_OUT_CATEGORIES)
    seen_labels = list(seen_labels)
    novel_labels = list(novel_labels)
    if not seen_labels or not novel_labels:
        raise ValueError("seen_labels and novel_labels must be non-empty")

    out = Path(out_dir)
    (out / "stream").mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    novel_set = set(novel_labels)
    rng = np.random.default_rng(seed)
    counter = 0

    for episode in range(int(episodes)):
        n_novel = int(round(float(per_episode) * float(novel_rate))) if episode >= novel_start else 0
        n_seen = int(per_episode) - n_novel
        labels = list(rng.choice(novel_labels, size=n_novel, replace=True)) if n_novel else []
        labels += list(rng.choice(seen_labels, size=n_seen, replace=True))
        rng.shuffle(labels)

        for j, label in enumerate(labels):
            if label_mode == "instance":
                category, variation = _parse_instance_label(str(label))
            else:
                category = str(label)
                variation = 100 + episode * int(per_episode) + j
            sample_seed = int(seed) + 100_000 * episode + j
            sample = generate_category_sample(
                category,
                int(variation),
                sample_seed,
                width=width,
                height=height,
                views=1,
                split="stream",
                variation_specs=True,
            )
            drift = bool(tier == "c" and episode >= drift_episode)
            if drift:
                sample = _apply_sensor_drift(sample, drift_mode, rng)

            rel = f"stream/ep{episode:03d}_{j:04d}.npz"
            dataset.save_sample(out / rel, sample)
            entries.append(
                {
                    "id": f"stream_{episode:03d}_{j:04d}",
                    "kind": "category",
                    "path": rel,
                    "split": "stream",
                    "category": category,
                    "instance": sample["instance_id"],
                    "label": str(label),
                    "episode": int(episode),
                    "is_novel": bool(str(label) in novel_set),
                    "tier": tier,
                    "drift": drift,
                    "variation": int(variation),
                }
            )
            counter += 1

    config = {
        "scale": "open_stream",
        "tier": tier,
        "label_mode": label_mode,
        "seed": int(seed),
        "width": int(width),
        "height": int(height),
        "crop_preserve_aspect": True,
        "episodes": int(episodes),
        "per_episode": int(per_episode),
        "novel_rate": float(novel_rate),
        "novel_start": int(novel_start),
        "drift_episode": int(drift_episode),
        "drift_mode": drift_mode,
        "seen_labels": seen_labels,
        "novel_labels": novel_labels,
        "counts": {
            "entries": counter,
            "episodes": int(episodes),
            "per_episode": int(per_episode),
            "seen_labels": len(seen_labels),
            "novel_labels": len(novel_labels),
        },
        "split_policy": (
            "All entries have split='stream'; episode/is_novel/drift/label carry "
            "the protocol. Novel labels are absent before novel_start."
        ),
    }
    dataset.write_index(out, entries, config)
    dataset.write_data_card(
        out,
        entries,
        config,
        notes=(
            "Open-category stream. Tier A = held-out categories; tier B = new "
            "instances of known categories; tier C = tier A plus a global "
            "lighting/tactile shift after drift_episode. The label field is the "
            "category (A/C) or the instance id (B)."
        ),
    )
    return {
        "out_dir": str(out),
        "tier": tier,
        "label_mode": label_mode,
        "entries": counter,
        "episodes": int(episodes),
        "seen_labels": len(seen_labels),
        "novel_labels": len(novel_labels),
    }


def build_open_stream_from_index(
    out_dir: str | Path,
    source_index: str | Path,
    seen_labels: list[str],
    novel_labels: list[str],
    *,
    tier: str = "a",
    label_mode: str = "category",
    episodes: int = 20,
    per_episode: int = 200,
    novel_rate: float = 0.2,
    novel_start: int = 5,
    drift_episode: int = 10,
    drift_mode: str = "both",
    seed: int = 0,
) -> dict:
    """Build a stream by re-sampling an existing dataset index.

    This avoids re-rendering MuJoCo scenes: the base ``.npz`` files are copied
    into the stream directory, and tier C applies the sensor shift to the
    copied arrays.  Each source sample keeps its original number of views; the
    stream metadata (episode / is_novel / drift / label) is per entry and is
    repeated for every view by ``load_stream``.
    """

    import shutil
    from collections import defaultdict

    source_index = Path(source_index)
    source_dir = source_index.parent
    out = Path(out_dir)
    (out / "stream").mkdir(parents=True, exist_ok=True)
    index = json.loads(source_index.read_text(encoding="utf-8"))
    entries = [e for e in index["entries"] if e.get("kind") == "category"]
    by_label: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        label = (
            str(entry["category"])
            if label_mode == "category"
            else str(entry["instance"])
        )
        by_label[label].append(entry)

    seen_labels = [str(x) for x in seen_labels]
    novel_labels = [str(x) for x in novel_labels]
    missing = [x for x in seen_labels + novel_labels if x not in by_label]
    if missing:
        raise ValueError(f"labels missing from source index: {missing[:5]}")
    if not seen_labels or not novel_labels:
        raise ValueError("seen_labels and novel_labels must be non-empty")

    rng = np.random.default_rng(int(seed))
    novel_set = set(novel_labels)
    new_entries: list[dict] = []
    for episode in range(int(episodes)):
        n_novel = int(round(float(per_episode) * float(novel_rate))) if episode >= novel_start else 0
        labels = list(rng.choice(novel_labels, size=n_novel)) if n_novel else []
        labels += list(rng.choice(seen_labels, size=int(per_episode) - n_novel))
        rng.shuffle(labels)
        for j, label in enumerate(labels):
            source = by_label[str(label)][int(rng.integers(len(by_label[str(label)])))]
            drift = bool(tier == "c" and episode >= drift_episode)
            rel = f"stream/ep{episode:03d}_{j:04d}.npz"
            if drift:
                sample = dataset.load_sample(source_dir / source["path"])
                sample = _apply_sensor_drift(sample, drift_mode, rng)
                dataset.save_sample(out / rel, sample)
            else:
                shutil.copyfile(source_dir / source["path"], out / rel)
            new_entries.append(
                {
                    "id": f"stream_{episode:03d}_{j:04d}",
                    "kind": "category",
                    "path": rel,
                    "split": "stream",
                    "category": source["category"],
                    "instance": source["instance"],
                    "label": str(label),
                    "episode": int(episode),
                    "is_novel": bool(str(label) in novel_set),
                    "tier": tier,
                    "drift": drift,
                    "source_id": source["id"],
                }
            )

    config = {
        "scale": "open_stream_from_index",
        "source_index": str(source_index),
        "tier": tier,
        "label_mode": label_mode,
        "seed": int(seed),
        "crop_preserve_aspect": bool(index.get("config", {}).get("crop_preserve_aspect", False)),
        "episodes": int(episodes),
        "per_episode": int(per_episode),
        "novel_rate": float(novel_rate),
        "novel_start": int(novel_start),
        "drift_episode": int(drift_episode),
        "drift_mode": drift_mode,
        "seen_labels": seen_labels,
        "novel_labels": novel_labels,
        "counts": {
            "entries": len(new_entries),
            "episodes": int(episodes),
            "seen_labels": len(seen_labels),
            "novel_labels": len(novel_labels),
        },
    }
    dataset.write_index(out, new_entries, config)
    dataset.write_data_card(
        out,
        new_entries,
        config,
        notes=(
            "Open-category stream built by re-sampling an existing dataset. "
            "No new MuJoCo renders; tier C applies a global lighting/tactile "
            "shift to the copied arrays."
        ),
    )
    return {
        "out_dir": str(out),
        "tier": tier,
        "label_mode": label_mode,
        "entries": len(new_entries),
        "episodes": int(episodes),
        "seen_labels": len(seen_labels),
        "novel_labels": len(novel_labels),
    }
