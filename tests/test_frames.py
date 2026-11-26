from __future__ import annotations

import numpy as np
import pytest

from pss.binding.frames import (
    FrameOp,
    Pose,
    coords_in_frame,
    relation_in_frame,
    relation_in_frame_bool,
)


def _yaw(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0.0:  # pragma: no cover - rare sign combination
        q[:, 0] *= -1.0
    return q


def test_identity_residual_is_zero_and_apply_is_noop():
    op = FrameOp.identity()
    origin = Pose.identity()
    assert op.residual(origin, origin) == 0.0

    point = np.array([1.0, -2.0, 3.0])
    assert np.allclose(op.apply(point), point)

    pose = Pose(_yaw(0.3), np.array([4.0, 5.0, 6.0]))
    applied = op.apply(pose)
    assert np.allclose(applied.R, pose.R)
    assert np.allclose(applied.t, pose.t)
    assert op.residual(pose, pose) == 0.0


def test_inverse_roundtrip_for_random_pose():
    rng = np.random.default_rng(0)
    R = _random_rotation(rng)
    t = rng.normal(size=3)
    op = FrameOp(R, t)
    inv = op.inverse()

    point = rng.normal(size=3)
    assert np.allclose(inv.apply(op.apply(point)), point, atol=1e-9)
    assert np.allclose(op.apply(inv.apply(point)), point, atol=1e-9)

    pose = Pose(_random_rotation(rng), rng.normal(size=3))
    roundtrip = inv.apply(op.apply(pose))
    assert op.residual(pose, roundtrip) < 1e-9


def test_pose_compose_applies_other_first():
    a = Pose(_yaw(0.4), np.array([1.0, 2.0, 3.0]))
    b = Pose(_yaw(-0.9), np.array([0.5, -1.0, 2.0]))
    point = np.array([0.2, 0.3, -0.4])

    composed = a.compose(b)
    assert np.allclose(composed.apply_point(point), a.apply_point(b.apply_point(point)))
    assert np.allclose(a.inverse().apply_point(a.apply_point(point)), point)


def test_composition_is_non_commutative():
    # Two yaw rotations with different origins do not commute.
    a = FrameOp(_yaw(0.3), np.array([1.0, 0.0, 0.0]), name="a")
    b = FrameOp(_yaw(-0.7), np.array([0.0, 2.0, 0.0]), name="b")
    ab = a.compose(b)
    ba = b.compose(a)

    assert isinstance(ab, FrameOp)
    assert isinstance(ba, FrameOp)
    point = np.array([0.5, -0.5, 1.0])
    assert not np.allclose(ab.apply(point), ba.apply(point), atol=1e-9)
    assert not np.allclose(ab.t, ba.t, atol=1e-9)


def test_composition_preserves_rotation():
    a = FrameOp(_yaw(0.3), np.array([1.0, 0.0, 0.0]))
    b = FrameOp(_yaw(-0.7), np.array([0.0, 2.0, 0.0]))
    composed = a.compose(b)

    assert np.allclose(composed.R @ composed.R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(composed.R), 1.0, atol=1e-9)


def test_left_in_world_becomes_right_after_half_turn_frame():
    turn = FrameOp(_yaw(np.pi), np.zeros(3), name="turn")
    left = np.array([-1.0, 0.0, 0.0])
    origin = np.zeros(3)

    local = coords_in_frame(left, turn)
    assert np.allclose(local, [1.0, 0.0, 0.0])
    assert relation_in_frame(left, origin, turn) == ["right_of"]
    assert relation_in_frame_bool(left, origin, turn, "right_of")
    assert not relation_in_frame_bool(left, origin, turn, "left_of")


def test_relation_in_frame_known_configuration():
    frame = FrameOp.identity()
    b = np.zeros(3)

    assert relation_in_frame(np.array([1.0, 2.0, 3.0]), b, frame) == [
        "right_of",
        "front_of",
        "above",
    ]
    assert relation_in_frame(np.array([-1.0, -2.0, -3.0]), b, frame) == [
        "left_of",
        "behind",
        "below",
    ]
    assert relation_in_frame(np.array([0.0, 1.0, 0.0]), b, frame) == ["front_of"]
    assert relation_in_frame(b, b, frame) == []


def test_relation_in_frame_respects_margin():
    frame = FrameOp.identity()
    b = np.zeros(3)
    within_margin = np.array([1e-7, -1e-7, 5e-7])
    assert relation_in_frame(within_margin, b, frame) == []
    assert relation_in_frame(within_margin, b, frame, margin=1e-8) == [
        "right_of",
        "behind",
        "above",
    ]


def test_relation_in_frame_uses_translated_frame():
    frame = FrameOp(np.eye(3), np.array([10.0, 0.0, 0.0]))
    a = np.array([11.0, 0.0, 0.0])
    b = np.array([10.0, 0.0, 0.0])

    assert relation_in_frame(a, b, frame) == ["right_of"]
    assert relation_in_frame_bool(a, b, frame, "right_of")
    assert not relation_in_frame_bool(a, b, frame, "above")


def test_coords_in_frame_matches_inverse():
    rng = np.random.default_rng(3)
    frame = FrameOp(_random_rotation(rng), rng.normal(size=3))
    point = rng.normal(size=3)
    assert np.allclose(coords_in_frame(point, frame), frame.inverse().apply_point(point))


def test_from_axes_valid():
    op = FrameOp.from_axes(
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 2.0, 3.0),
        name="axes",
    )
    assert np.allclose(op.R, np.eye(3))
    assert np.allclose(op.t, [1.0, 2.0, 3.0])
    assert op.name == "axes"


@pytest.mark.parametrize(
    "right, front, up",
    [
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # not orthogonal
        ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),  # not unit length
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # left-handed
    ],
)
def test_from_axes_invalid_raises_value_error(right, front, up):
    with pytest.raises(ValueError):
        FrameOp.from_axes(right, front, up, (0.0, 0.0, 0.0))


def test_from_matrix_roundtrip():
    rng = np.random.default_rng(7)
    pose = Pose(_random_rotation(rng), rng.normal(size=3))
    back = Pose.from_matrix(pose.as_matrix())

    assert np.allclose(back.R, pose.R)
    assert np.allclose(back.t, pose.t)

    op = FrameOp(pose.R, pose.t)
    matrix = op.matrix()
    assert np.allclose(matrix[:3, :3], op.R)
    assert np.allclose(matrix[:3, 3], op.t)
    assert np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(Pose.from_matrix(matrix).as_matrix(), matrix)


@pytest.mark.parametrize(
    "R, t",
    [
        (np.eye(2), np.zeros(3)),  # wrong rotation shape
        (np.eye(3), np.zeros(2)),  # wrong translation shape
        (np.diag([1.0, 1.0, 2.0]), np.zeros(3)),  # not orthonormal
        (np.diag([1.0, 1.0, -1.0]), np.zeros(3)),  # reflection (det < 0)
    ],
)
def test_pose_rejects_invalid_inputs(R, t):
    with pytest.raises(ValueError):
        Pose(R, t)


def test_frameop_compose_with_non_frame_returns_none():
    op = FrameOp.identity()
    assert op.compose(Pose.identity()) is None
    assert op.compose(object()) is None


def test_transform_aliases():
    rng = np.random.default_rng(11)
    op = FrameOp(_random_rotation(rng), rng.normal(size=3))
    point = rng.normal(size=3)
    direction = rng.normal(size=3)

    assert np.allclose(op.transform_point(point), op.apply(point))
    assert np.allclose(op.transform_direction(direction), op.R @ direction)
    assert np.allclose(Pose(op.R, op.t).apply_direction(direction), op.R @ direction)


def test_distance_mixes_metres_and_radians():
    identity = Pose.identity()
    assert identity.distance(identity) == 0.0
    assert np.isclose(identity.distance(Pose.from_position([2.0, 0.0, 0.0])), 2.0)
    assert np.isclose(identity.distance(Pose(_yaw(np.pi), np.zeros(3))), np.pi)


def test_residual_detects_different_poses():
    a = Pose(_yaw(0.2), np.array([0.0, 0.0, 0.0]))
    b = Pose(_yaw(0.3), np.array([0.1, 0.0, 0.0]))
    op = FrameOp.identity()
    assert op.residual(a, b) > 0.0
    assert np.isclose(op.residual(a, b), op.residual(b, a))
