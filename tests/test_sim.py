import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from pss.sim.actions import Action, apply_action
from pss.sim.generator import (
    generate_category_sample,
    generate_episode,
    generate_occlusion_sequence,
)
from pss.sim.scene import SceneConfig, build_scene
from pss.sim.sensors import SensorRig


def test_build_scene_and_render():
    cfg = SceneConfig(n_objects=3, seed=1, width=64, height=64)
    scene = build_scene(cfg)
    assert len(scene.objects) == 3
    rig = SensorRig(scene, 64, 64)
    rgb = rig.render_rgb()
    depth = rig.render_depth()
    seg = rig.render_segmentation()
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8
    assert depth.shape == (64, 64) and depth.dtype == np.float32
    assert seg.shape == (64, 64, 2) and seg.dtype == np.int32
    poses = rig.object_poses()
    assert len(poses) == 3
    tacs = rig.tactile_maps()
    assert set(tacs) == set(poses)
    rig.close()


def test_push_moves_object():
    cfg = SceneConfig(n_objects=1, seed=2, width=64, height=64, bins=0)
    scene = build_scene(cfg, categories=["block"], instance_indices=[0])
    before, _ = scene.get_object_pose(0)
    apply_action(
        scene,
        Action("push", 0, {"force": 6.0, "steps": 20}),
        np.random.default_rng(0),
    )
    after, _ = scene.get_object_pose(0)
    assert np.linalg.norm(after[:2] - before[:2]) > 0.005


def test_category_sample_shapes():
    sample = generate_category_sample("cup", 0, seed=1, width=64, height=64, views=2)
    assert sample["rgb"].shape == (2, 64, 64, 3)
    assert sample["depth"].shape == (2, 64, 64)
    assert sample["segmentation"].shape == (2, 64, 64, 2)
    assert sample["tactile"].shape == (2, 8, 8)
    assert sample["split"] == "train"


def test_episode_shapes_and_actions():
    sample = generate_episode(seed=1, width=64, height=64, n_objects=3, n_actions=3)
    assert sample["rgb"].shape[0] == 4
    assert sample["poses"].shape == (4, 3, 7)
    assert sample["tactile"].shape == (4, 3, 8, 8)
    assert len(sample["actions"]) == 3
    assert sample["kind"] == "episode"


def test_occlusion_sequence_shapes():
    sample = generate_occlusion_sequence(
        seed=1, width=64, height=64, n_objects=2, frames=4
    )
    assert sample["rgb"].shape[0] == 4
    assert sample["poses"].shape == (4, 2, 7)
    assert len(sample["occluder_positions"]) == 4
    assert sample["kind"] == "occlusion"
