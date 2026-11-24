from __future__ import annotations

import pyarrow.parquet as pq

from pss.isaac.bundle import TABLES, read_manifest, write_bundle
from pss.isaac.verify import verify_bundle


def _gt_row(step: int, instance: str, visibility: float) -> dict:
    return {
        "physics_step": step,
        "t_s": step / 120.0,
        "instance_id": instance,
        "class": "cube",
        "twin_group": None,
        "pose_xyz": [0.0, 0.0, 0.0],
        "pose_quat": [1.0, 0.0, 0.0, 0.0],
        "bbox_min": [-0.02, -0.02, -0.02],
        "bbox_max": [0.02, 0.02, 0.02],
        "bbox_3d": [0.0] * 24,
        "visibility_ratio": visibility,
        "occluded_by": [],
        "occluder_of": [],
        "contact_points": [],
        "net_force_n": 0.0,
        "in_contact_with": [],
        "grasped_by": None,
        "q": [],
        "qd": [],
        "tau": [],
        "relations": [],
    }


def _obs_row(step: int) -> dict:
    return {
        "step": step,
        "t_s": step / 10.0,
        "camera": "overhead",
        "image_ref": None,
        "depth_ref": None,
        "robot_q": None,
        "ee_pose": None,
        "camera_pose": [0.0, 0.0, 1.3, 1.0, 0.0, 0.0, 0.0],
    }


def test_write_bundle_creates_all_tables(tmp_path):
    manifest = write_bundle(
        tmp_path,
        episode_id="s1__seed00__oracle__full",
        spec_id="s1_occlusion_v1",
        spec_hash="sha256:" + "0" * 64,
        device_class="dev",
        seed=0,
        stack={"isaac_sim": "5.1.0.0"},
        tables={
            "obs_frames": [_obs_row(0)],
            "gt": [_gt_row(0, "cube_a", 1.0), _gt_row(0, "cube_b", 0.5)],
        },
        events=[],
    )
    assert manifest["counts"]["obs_frames"] == 1
    assert manifest["counts"]["gt_frames"] == 2
    assert set(manifest["files"]) == set(TABLES)
    for filename, _ in TABLES.values():
        assert (tmp_path / filename).exists()
    assert read_manifest(tmp_path)["episode_id"].endswith("full")


def test_empty_events_table_keeps_schema(tmp_path):
    write_bundle(
        tmp_path,
        episode_id="x",
        spec_id="s",
        spec_hash="sha256:" + "0" * 64,
        device_class="dev",
        seed=0,
        stack={},
    )
    schema = pq.read_schema(tmp_path / "events.parquet")
    assert schema.names == ["t_s", "kind", "source", "detail_json"]
    assert pq.read_table(tmp_path / "events.parquet").num_rows == 0


def test_verify_bundle_passes_on_minimal_fixture(tmp_path):
    write_bundle(
        tmp_path,
        episode_id="x",
        spec_id="s1",
        spec_hash="sha256:" + "0" * 64,
        device_class="dev",
        seed=0,
        stack={},
        tables={
            "obs_frames": [_obs_row(0), _obs_row(1)],
            "gt": [
                _gt_row(0, "cube_a", 1.0),
                _gt_row(0, "cube_b", 0.5),
            ],
        },
    )
    report = verify_bundle(
        tmp_path,
        expect_instances=2,
        expect_obs_frames=2,
        expect_physics_steps=1,
    )
    assert report["ok"], report["checks"]
