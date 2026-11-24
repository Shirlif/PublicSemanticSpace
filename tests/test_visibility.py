from __future__ import annotations

import math

from pss.isaac.visibility import Box, compute_visibility, in_frustum, ray_box_t

CAM = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_ray_box_hit_distance():
    box = Box("b", [0.0, 0.0, 0.0, 1, 0, 0, 0], [1.0, 1.0, 1.0])
    t = ray_box_t((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), box)
    assert t is not None
    assert math.isclose(t, 0.5, abs_tol=1e-6)


def test_ray_box_miss():
    box = Box("b", [10.0, 0.0, 0.0, 1, 0, 0, 0], [1.0, 1.0, 1.0])
    assert ray_box_t((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), box) is None


def test_object_fully_visible_without_occluder():
    obj = Box("cube", [0.0, 0.0, 0.0, 1, 0, 0, 0], [0.1, 0.1, 0.1])
    vis = compute_visibility([obj], CAM, fov_deg=80, aspect=4 / 3)
    assert vis["cube"]["visibility_ratio"] == 1.0


def test_occluder_between_camera_and_object_blocks_it():
    obj = Box("cube", [0.0, 0.0, 0.0, 1, 0, 0, 0], [0.1, 0.1, 0.1])
    board = Box("board", [0.0, 0.0, 0.5, 1, 0, 0, 0], [1.0, 1.0, 0.01])
    vis = compute_visibility([obj, board], CAM, fov_deg=80, aspect=4 / 3)
    assert vis["cube"]["visibility_ratio"] == 0.0
    assert "board" in vis["cube"]["occluded_by"]
    assert "cube" in vis["board"]["occluder_of"]


def test_partial_occlusion_is_between_zero_and_one():
    obj = Box("cube", [0.0, 0.0, 0.0, 1, 0, 0, 0], [0.4, 0.4, 0.1])
    board = Box("board", [0.1, 0.0, 0.5, 1, 0, 0, 0], [0.2, 1.0, 0.01])
    vis = compute_visibility([obj, board], CAM, fov_deg=80, aspect=4 / 3)
    ratio = vis["cube"]["visibility_ratio"]
    assert 0.0 < ratio < 1.0


def test_frustum_rejects_side_ray():
    assert in_frustum((0.0, 0.0, -1.0), CAM, 60.0, 4 / 3)
    assert not in_frustum((1.0, 0.0, -1.0), CAM, 60.0, 4 / 3)


def test_sweeping_occluder_changes_visibility():
    obj = Box("cube", [0.0, 0.0, 0.0, 1, 0, 0, 0], [0.2, 0.2, 0.1])
    ratios = []
    for x in (-0.6, 0.0, 0.6):
        board = Box("board", [x, 0.0, 0.5, 1, 0, 0, 0], [0.12, 0.5, 0.01])
        vis = compute_visibility([obj, board], CAM, fov_deg=80, aspect=4 / 3)
        ratios.append(vis["cube"]["visibility_ratio"])
    assert ratios[1] < 1.0
    assert ratios[0] == 1.0
