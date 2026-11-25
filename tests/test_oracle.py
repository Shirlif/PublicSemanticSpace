from __future__ import annotations

import math

from pss.detect.oracle import project_point


def test_projection_center_and_edges():
    cam = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    center = project_point([0.0, 0.0, 0.0], cam, 640, 480, 60.0)
    assert center is not None
    assert abs(center[0] - 320.0) < 1e-6
    assert abs(center[1] - 240.0) < 1e-6

    # A point on the horizontal FOV boundary lands at the image edge.
    half = math.tan(math.radians(30.0))
    edge = project_point([half, 0.0, 0.0], cam, 640, 480, 60.0)
    assert edge is not None
    assert abs(edge[0] - 640.0) < 1e-6


def test_projection_rejects_points_behind_camera():
    cam = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert project_point([0.0, 0.0, 2.0], cam, 640, 480, 60.0) is None
