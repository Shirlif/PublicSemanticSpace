"""Reference-frame transforms (SE(3)) for the binding operator family.

Axes and signs
--------------
A frame's rotation ``R`` maps local axes to parent (world) axes, and its
translation ``t`` is the frame origin in parent coordinates.  The local axes
are ``x = right``, ``y = front``, ``z = up``.  A point ``p`` in parent
coordinates becomes ``R.T @ (p - t)`` in frame coordinates, which is exactly
``FrameOp.inverse().apply_point(p)``.

Directional predicates compare the local coordinates of ``a`` and ``b``:

- ``a left_of b``   iff ``a.x < b.x - margin``
- ``a right_of b``  iff ``a.x > b.x + margin``
- ``a behind b``    iff ``a.y < b.y - margin``
- ``a front_of b``  iff ``a.y > b.y + margin``
- ``a below b``     iff ``a.z < b.z - margin``
- ``a above b``     iff ``a.z > b.z + margin``

Because ``x`` grows to the right, a *smaller* x is to the left; because ``y``
grows forward, a *smaller* y is behind.

Composition order
-----------------
This module follows the shared operator contract: ``self.compose(other)``
means ``self ∘ other`` -- ``other`` is applied first.  Frame transforms are
not commutative when their rotations or origins differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import Operator

_ATOL = 1e-6


def _rotation(value) -> np.ndarray:
    """Validate a (3, 3) rotation matrix, returning a frozen float64 copy."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {arr.shape}")
    if not np.allclose(arr @ arr.T, np.eye(3), atol=_ATOL, rtol=0.0):
        raise ValueError("rotation matrix is not orthonormal (R @ R.T != I)")
    if float(np.linalg.det(arr)) <= 0.0:
        raise ValueError("rotation matrix must have a positive determinant")
    arr = arr.copy()
    arr.setflags(write=False)
    return arr


def _translation(value) -> np.ndarray:
    """Validate a (3,) translation vector, returning a frozen float64 copy."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"translation must have shape (3,), got {arr.shape}")
    arr = arr.copy()
    arr.setflags(write=False)
    return arr


def _rotation_angle(r: np.ndarray) -> float:
    """Geodesic angle (radians) of a rotation matrix: ``arccos((tr - 1) / 2)``."""
    cosine = (float(np.trace(r)) - 1.0) * 0.5
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


@dataclass(frozen=True, eq=False)
class Pose:
    """Rigid transform ``p_parent = R @ p_local + t``.

    ``eq`` is disabled because the fields are NumPy arrays; compare poses with
    :meth:`distance` instead of ``==``.
    """

    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "R", _rotation(self.R))
        object.__setattr__(self, "t", _translation(self.t))

    @classmethod
    def identity(cls) -> "Pose":
        """Return the identity pose."""
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_position(cls, t) -> "Pose":
        """Return a pose with identity rotation at position ``t``."""
        return cls(np.eye(3), t)

    @classmethod
    def from_matrix(cls, m4) -> "Pose":
        """Build a pose from a homogeneous 4x4 transform."""
        m = np.asarray(m4, dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError(f"matrix must have shape (4, 4), got {m.shape}")
        return cls(m[:3, :3], m[:3, 3])

    def compose(self, other: "Pose") -> "Pose":
        """Return ``self ∘ other``: ``other`` is applied first."""
        if not isinstance(other, Pose):
            raise TypeError(f"Pose.compose expects a Pose, got {type(other).__name__}")
        return Pose(self.R @ other.R, self.R @ other.t + self.t)

    def inverse(self) -> "Pose":
        """Return the inverse pose."""
        rt = self.R.T
        return Pose(rt, -(rt @ self.t))

    def apply_point(self, p) -> np.ndarray:
        """Transform a point: ``R @ p + t``."""
        arr = np.asarray(p, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {arr.shape}")
        return self.R @ arr + self.t

    def apply_direction(self, d) -> np.ndarray:
        """Rotate a direction: ``R @ d`` (translation is ignored)."""
        arr = np.asarray(d, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"direction must have shape (3,), got {arr.shape}")
        return self.R @ arr

    def as_matrix(self) -> np.ndarray:
        """Return the homogeneous 4x4 transform."""
        m = np.eye(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m

    def distance(self, other: "Pose") -> float:
        """Consistency distance between two poses.

        This mixes units: the translation difference is in metres and the
        relative rotation angle is in radians.  ``0.0`` means identical.  Use
        it only to compare poses at a similar scale, never as a metric length.
        """
        if not isinstance(other, Pose):
            raise TypeError(f"Pose.distance expects a Pose, got {type(other).__name__}")
        translation = float(np.linalg.norm(self.t - other.t))
        angle = _rotation_angle(self.R.T @ other.R)
        return translation + angle


@dataclass(frozen=True, eq=False)
class FrameOp(Operator):
    """SE(3) reference-frame operator.

    ``R`` and ``t`` describe the transform from the frame's local coordinates
    to parent coordinates.  The operator applies that transform to points,
    directions and poses, and composes like a rigid transform.
    """

    R: np.ndarray
    t: np.ndarray
    name: str = "frame"
    kind: str = "frame"

    def __post_init__(self) -> None:
        object.__setattr__(self, "R", _rotation(self.R))
        object.__setattr__(self, "t", _translation(self.t))

    def apply(self, x):
        """Apply the frame transform.

        A :class:`Pose` is transformed as ``self ∘ pose``; an array of shape
        ``(3,)`` is treated as a point.
        """
        if isinstance(x, Pose):
            return Pose(self.R @ x.R, self.R @ x.t + self.t)
        return self.apply_point(x)

    def apply_point(self, p) -> np.ndarray:
        """Transform a point: ``R @ p + t``."""
        arr = np.asarray(p, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {arr.shape}")
        return self.R @ arr + self.t

    def apply_direction(self, d) -> np.ndarray:
        """Rotate a direction: ``R @ d`` (translation is ignored)."""
        arr = np.asarray(d, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"direction must have shape (3,), got {arr.shape}")
        return self.R @ arr

    def inverse(self) -> "FrameOp":
        """Return the inverse frame operator."""
        rt = self.R.T
        return FrameOp(rt, -(rt @ self.t), name=f"{self.name}^-1", kind=self.kind)

    def compose(self, other: "Operator") -> "FrameOp | None":
        """Return ``self ∘ other`` (``other`` first), or ``None`` if not a FrameOp."""
        if not isinstance(other, FrameOp):
            return None
        return FrameOp(
            self.R @ other.R,
            self.R @ other.t + self.t,
            name=f"{self.name}_after_{other.name}",
            kind=self.kind,
        )

    def residual(self, a, b) -> float:
        """Pose consistency distance; ``0.0`` means identical poses."""
        if not isinstance(a, Pose) or not isinstance(b, Pose):
            raise TypeError("FrameOp.residual expects two Pose instances")
        return a.distance(b)

    @classmethod
    def identity(cls) -> "FrameOp":
        """Return the identity frame operator."""
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_axes(cls, right, front, up, origin, name: str = "frame") -> "FrameOp":
        """Build a frame from explicit local axes (columns) and origin.

        ``right``, ``front`` and ``up`` become the columns of ``R`` (local to
        parent).  The axes must already be orthonormal and right-handed; this
        method validates within ``1e-6`` and raises ``ValueError`` rather than
        silently orthonormalizing them.
        """
        vectors = [np.asarray(v, dtype=np.float64) for v in (right, front, up)]
        for label, vector in zip(("right", "front", "up"), vectors):
            if vector.shape != (3,):
                raise ValueError(f"{label} axis must have shape (3,), got {vector.shape}")
        r = np.column_stack(vectors)
        if not np.allclose(r @ r.T, np.eye(3), atol=_ATOL, rtol=0.0):
            raise ValueError("axes are not orthonormal; refusing to modify them implicitly")
        if float(np.linalg.det(r)) <= 0.0:
            raise ValueError("axes do not form a right-handed frame")
        return cls(r, origin, name=name)

    def transform_point(self, p) -> np.ndarray:
        """Explicit alias for transforming a point."""
        return self.apply_point(p)

    def transform_direction(self, d) -> np.ndarray:
        """Explicit alias for rotating a direction."""
        return self.apply_direction(d)

    def matrix(self) -> np.ndarray:
        """Return the homogeneous 4x4 transform."""
        m = np.eye(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m


def coords_in_frame(point, frame) -> np.ndarray:
    """Express a parent-frame ``point`` in ``frame`` local coordinates."""
    return frame.inverse().apply_point(point)


def relation_in_frame(a_pos, b_pos, frame, margin: float = 1e-6) -> list[str]:
    """Return every directional predicate that holds for ``a`` relative to ``b``.

    Both points are first expressed in ``frame`` local coordinates (x = right,
    y = front, z = up).  See the module docstring for the sign convention.  The
    result is ordered as ``left_of``, ``right_of``, ``behind``, ``front_of``,
    ``below``, ``above`` and contains only the predicates that hold.
    """
    inv = frame.inverse()
    a = inv.apply_point(a_pos)
    b = inv.apply_point(b_pos)
    predicates: list[str] = []
    if a[0] < b[0] - margin:
        predicates.append("left_of")
    if a[0] > b[0] + margin:
        predicates.append("right_of")
    if a[1] < b[1] - margin:
        predicates.append("behind")
    if a[1] > b[1] + margin:
        predicates.append("front_of")
    if a[2] < b[2] - margin:
        predicates.append("below")
    if a[2] > b[2] + margin:
        predicates.append("above")
    return predicates


def relation_in_frame_bool(
    a_pos,
    b_pos,
    frame,
    predicate: str,
    margin: float = 1e-6,
) -> bool:
    """Return whether ``predicate`` holds for ``a`` relative to ``b`` in ``frame``."""
    return predicate in relation_in_frame(a_pos, b_pos, frame, margin=margin)


def quaternion_matrix(q) -> np.ndarray:
    """Rotation matrix from a quaternion in ``(w, x, y, z)`` order.

    The tracker stores quaternions as ``[w, x, y, z]`` (identity is
    ``[1, 0, 0, 0]``).  A zero-norm quaternion returns the identity rather
    than NaN.
    """
    q = np.asarray(q, dtype=float).ravel()
    if q.shape[0] != 4:
        raise ValueError(f"expected a 4-element quaternion, got {q.shape}")
    w, x, y, z = q
    norm = w * w + x * x + y * y + z * z
    if norm < 1e-12:
        return np.eye(3)
    s = 2.0 / norm
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ]
    )
