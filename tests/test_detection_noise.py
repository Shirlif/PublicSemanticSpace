from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pss.detect.groundingdino import _backproject
from pss.detect.noise import apply_noise


def test_backproject_center_returns_point_below_camera():
    cam = [0.0, 0.0, 1.3, 1.0, 0.0, 0.0, 0.0]
    point = _backproject(320, 240, 0.5, cam, 640, 480, 60.0)
    assert point[0] == 0.0
    assert point[1] == 0.0
    assert point[2] == 0.8


def test_noise_is_deterministic_and_can_drop():
    rows = [
        {"step": 0, "t_s": 0.0, "camera": "c", "det_id": "d0", "bbox_xyxy": [0, 0, 1, 1],
         "score": 1.0, "class_pred": "cube", "pose_xyz": [0.0, 0.0, 0.79],
         "pose_quat": [1.0, 0, 0, 0], "emb": None, "emb_ref": "a"},
        {"step": 0, "t_s": 0.0, "camera": "c", "det_id": "d1", "bbox_xyxy": [0, 0, 1, 1],
         "score": 1.0, "class_pred": "cube", "pose_xyz": [0.1, 0.0, 0.79],
         "pose_quat": [1.0, 0, 0, 0], "emb": None, "emb_ref": "b"},
    ]
    gt = pd.DataFrame(
        [
            {"physics_step": 0, "instance_id": "a", "visibility_ratio": 0.0},
            {"physics_step": 0, "instance_id": "b", "visibility_ratio": 1.0},
        ]
    )
    out1 = apply_noise(rows, gt, profile={"miss_rate": {"base": 0.0, "per_visibility_drop": 1.0},
                                         "false_positive_per_frame": 0.0}, seed=0)
    out2 = apply_noise(rows, gt, profile={"miss_rate": {"base": 0.0, "per_visibility_drop": 1.0},
                                         "false_positive_per_frame": 0.0}, seed=0)
    assert [r["det_id"] for r in out1] == [r["det_id"] for r in out2]
    assert all(r["emb_ref"] != "a" for r in out1)  # fully occluded -> always missed


def test_noise_adds_false_positives_when_configured():
    rows = [
        {"step": 0, "t_s": 0.0, "camera": "c", "det_id": "d0", "bbox_xyxy": [0, 0, 1, 1],
         "score": 1.0, "class_pred": "cube", "pose_xyz": [0.0, 0.0, 0.79],
         "pose_quat": [1.0, 0, 0, 0], "emb": None, "emb_ref": "a"},
    ]
    gt = pd.DataFrame([{"physics_step": 0, "instance_id": "a", "visibility_ratio": 1.0}])
    out = apply_noise(
        rows, gt,
        profile={"miss_rate": {"base": 0.0, "per_visibility_drop": 0.0},
                 "false_positive_per_frame": 5.0},
        seed=1,
    )
    assert any(r["det_id"].startswith("noise_fp") for r in out)
