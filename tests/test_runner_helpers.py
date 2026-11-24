from __future__ import annotations

import pytest

from pss.isaac.runner import _bbox, _interp_motion


def test_motion_interpolates_and_clamps():
    motion = [
        {"t_s": 0.0, "pos": [-0.6, 0.0, 0.9]},
        {"t_s": 5.0, "pos": [0.0, 0.0, 0.9]},
        {"t_s": 10.0, "pos": [0.6, 0.0, 0.9]},
    ]
    assert _interp_motion(motion, -1.0) == [-0.6, 0.0, 0.9]
    assert _interp_motion(motion, 2.5) == pytest.approx([-0.3, 0.0, 0.9])
    assert _interp_motion(motion, 7.5) == pytest.approx([0.3, 0.0, 0.9])
    assert _interp_motion(motion, 99.0) == [0.6, 0.0, 0.9]


def test_bbox_of_axis_aligned_cube():
    mins, maxs, flat = _bbox([0.0, 0.0, 0.0, 1, 0, 0, 0], [0.04, 0.06, 0.08])
    assert mins == pytest.approx([-0.02, -0.03, -0.04])
    assert maxs == pytest.approx([0.02, 0.03, 0.04])
    assert len(flat) == 24


def test_bbox_of_rotated_cube_swaps_extents():
    # 90 degrees about Z: x and y extents swap.
    s = 2 ** -0.5
    mins, maxs, _ = _bbox([0.0, 0.0, 0.0, s, 0, 0, s], [0.04, 0.08, 0.02])
    assert maxs[0] - mins[0] == pytest.approx(0.08, abs=1e-6)
    assert maxs[1] - mins[1] == pytest.approx(0.04, abs=1e-6)
