from __future__ import annotations

from pathlib import Path

import pytest

from pss.isaac.camera_motion import camera_pose, interpolate_pose
from pss.spec import load_spec, validate
from pss.spec.loader import load_yaml

REPO = Path(__file__).resolve().parents[1]


def test_camera_motion_interpolates_and_clamps():
    motion = [
        {"t_s": 0.0, "pos": [0.0, 0.0, 1.3]},
        {"t_s": 5.0, "pos": [0.25, -0.2, 1.1]},
        {"t_s": 10.0, "pos": [0.0, 0.0, 1.3]},
    ]
    assert interpolate_pose(motion, -1)[:3] == [0.0, 0.0, 1.3]
    assert interpolate_pose(motion, 2.5)[:3] == pytest.approx([0.125, -0.1, 1.2])
    assert interpolate_pose(motion, 5.0)[:3] == pytest.approx([0.25, -0.2, 1.1])
    assert interpolate_pose(motion, 99)[:3] == [0.0, 0.0, 1.3]


def test_static_camera_uses_pose_fallback():
    cam = {"pose": [0.1, 0.2, 1.0, 1, 0, 0, 0]}
    assert camera_pose(cam, 3.0) == [0.1, 0.2, 1.0, 1, 0, 0, 0]


def test_schema_accepts_camera_motion():
    spec = load_spec(REPO / "specs" / "s1_camera_motion.yaml")
    assert spec.cameras[0]["motion"][1]["pos"] == [0.25, -0.2, 1.1]


def test_schema_rejects_single_keyframe():
    raw = load_yaml(REPO / "specs" / "s1_camera_motion.yaml")
    raw["cameras"][0]["motion"] = [{"t_s": 0.0, "pos": [0, 0, 1]}]
    result = validate(raw)
    assert not result.ok
    assert any("at least 2 keyframes" in e for e in result.errors)
