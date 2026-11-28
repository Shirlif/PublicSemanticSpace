"""Physical constraints and joint refinement for the 3D scene benchmark (S2).

The shared ``optimizer.solve_scene`` estimates each object independently.
This module adds the S2 factors and a single joint refinement pass over all
objects:

* support:      ``bottom_z - table_z`` with ``bottom_z = z - radius``;
* non-penetration: ``max(0, r_i + r_j - ||p_i - p_j||)``;
* contact (optional, off by default): ``||p_i - p_j|| - (r_i + r_j)``.

The joint refinement starts from the per-object ``solve_scene`` estimate and
minimises reprojection + depth + constraint residuals with
``scipy.optimize.least_squares`` and a robust loss.  ``optimizer.py`` is not
modified; everything S2-specific lives here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .bench_api import ArmOutput, BenchConfig, SceneExample
from .optimizer import SolveOptions, solve_scene


@dataclass
class ConstraintOptions:
    """Weights and switches for the S2 physical factors."""

    use_support: bool = True
    use_nonpenetration: bool = True
    use_contact: bool = False
    sigma_support: float = 0.02
    sigma_penetration: float = 0.01
    sigma_contact: float = 0.02
    robust: bool = True
    f_scale: float = 1.0


def support_residual(
    position: np.ndarray, radius: float, table_z: float
) -> float:
    """Signed height of the object bottom above the table ``z - r - table_z``."""

    p = np.asarray(position, dtype=np.float64)
    return float(p[2] - float(radius) - float(table_z))


def penetration_residual(
    pa: np.ndarray, pb: np.ndarray, ra: float, rb: float
) -> float:
    """Overlap depth ``max(0, r_i + r_j - ||p_i - p_j||)`` (0 when separated)."""

    d = float(np.linalg.norm(np.asarray(pa, dtype=np.float64) - np.asarray(pb, dtype=np.float64)))
    return float(max(0.0, float(ra) + float(rb) - d))


def contact_residual(
    pa: np.ndarray, pb: np.ndarray, ra: float, rb: float
) -> float:
    """Soft contact: absolute surface gap ``| ||p_i-p_j|| - (r_i+r_j) |``."""

    d = float(np.linalg.norm(np.asarray(pa, dtype=np.float64) - np.asarray(pb, dtype=np.float64)))
    return float(abs(d - (float(ra) + float(rb))))


def constraint_residual(
    positions: dict[str, np.ndarray],
    radii: dict[str, float],
    config: BenchConfig,
    *,
    options: ConstraintOptions | None = None,
) -> np.ndarray:
    """Stacked constraint residual vector for a full scene.

    Signature matches the optional ``constraints`` hook of
    ``optimizer.solve_scene``: ``(positions, radii, config) -> residual``.
    """

    options = options or ConstraintOptions()
    ids = [oid for oid in positions]
    res: list[float] = []
    if options.use_support:
        for oid in ids:
            res.append(
                support_residual(
                    positions[oid], radii.get(oid, 0.35), config.table_z
                )
                / max(options.sigma_support, 1e-12)
            )
    if options.use_nonpenetration or options.use_contact:
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                ra, rb = radii.get(a, 0.35), radii.get(b, 0.35)
                if options.use_nonpenetration:
                    res.append(
                        penetration_residual(positions[a], positions[b], ra, rb)
                        / max(options.sigma_penetration, 1e-12)
                    )
                if options.use_contact:
                    res.append(
                        contact_residual(positions[a], positions[b], ra, rb)
                        / max(options.sigma_contact, 1e-12)
                    )
    return np.asarray(res, dtype=np.float64)


def _groups(example: SceneExample):
    groups = defaultdict(list)
    for obs in example.observations:
        groups[obs.object_id].append(obs)
    return groups


def _joint_residuals(
    x: np.ndarray,
    groups,
    ids: list[str],
    radii: np.ndarray,
    config: BenchConfig,
    solve_options: SolveOptions,
    coptions: ConstraintOptions,
) -> np.ndarray:
    """Residual vector over all objects (reprojection + depth + constraints)."""

    pos = np.asarray(x, dtype=np.float64).reshape(len(ids), 3)
    sigma_px = max(config.sigma_px, 1e-6)
    res: list[float] = []
    for i, oid in enumerate(ids):
        p = pos[i]
        for obs in groups[oid]:
            uv, z = obs.camera.project(p[None, :])
            w = float(obs.weight)
            res.append(w * (float(uv[0, 0]) - float(obs.uv[0])) / sigma_px)
            res.append(w * (float(uv[0, 1]) - float(obs.uv[1])) / sigma_px)
            if solve_options.use_depth:
                scale = max(config.sigma_depth * max(float(obs.depth), 1e-6), 1e-6)
                res.append(w * (float(z[0]) - float(obs.depth)) / scale)
        if coptions.use_support:
            res.append(
                (float(p[2]) - float(radii[i]) - float(config.table_z))
                / max(coptions.sigma_support, 1e-12)
            )

    if coptions.use_nonpenetration or coptions.use_contact:
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = float(np.linalg.norm(pos[i] - pos[j]))
                if coptions.use_nonpenetration:
                    pen = max(0.0, float(radii[i]) + float(radii[j]) - d)
                    res.append(pen / max(coptions.sigma_penetration, 1e-12))
                if coptions.use_contact:
                    gap = abs(d - (float(radii[i]) + float(radii[j])))
                    res.append(gap / max(coptions.sigma_contact, 1e-12))
    return np.asarray(res, dtype=np.float64)


def joint_refine(
    example: SceneExample,
    config: BenchConfig,
    *,
    initial: dict[str, np.ndarray] | None = None,
    solve_options: SolveOptions | None = None,
    options: ConstraintOptions | None = None,
) -> ArmOutput:
    """Jointly refine all object positions with constraints.

    ``initial`` defaults to the per-object ``solve_scene`` estimate.  The
    returned ``ArmOutput`` carries the joint positions and a linearised
    covariance block per object.
    """

    solve_options = solve_options or SolveOptions()
    coptions = options or ConstraintOptions()

    if initial is None:
        initial = solve_scene(example, config, options=solve_options).positions
    groups = _groups(example)
    objects = example.true_scene.objects
    ids = [oid for oid in objects if oid in initial and oid in groups]
    if not ids:
        return ArmOutput(positions={})

    radii = np.asarray([float(objects[oid].radius) for oid in ids], dtype=np.float64)
    x0 = np.concatenate([np.asarray(initial[oid], dtype=np.float64) for oid in ids])
    if not np.isfinite(x0).all():
        x0 = np.array(
            [v for oid in ids for v in (0.0, 0.0, float(objects[oid].radius))],
            dtype=np.float64,
        )

    result = least_squares(
        _joint_residuals,
        x0,
        args=(groups, ids, radii, config, solve_options, coptions),
        loss="soft_l1" if coptions.robust else "linear",
        f_scale=coptions.f_scale,
        bounds=(
            np.tile([-np.inf, -np.inf, 0.0], len(ids)),
            np.tile([np.inf, np.inf, np.inf], len(ids)),
        ),
        max_nfev=200,
    )

    pos = np.asarray(result.x, dtype=np.float64).reshape(len(ids), 3)
    positions = {oid: pos[i].copy() for i, oid in enumerate(ids)}

    # Linearised covariance, per-object 3x3 block of sigma^2 (J^T J)^-1.
    covariances: dict[str, np.ndarray] = {}
    jac = np.asarray(result.jac, dtype=np.float64)
    dof = max(jac.shape[0] - jac.shape[1], 1)
    sigma2 = max(float(np.sum(result.fun ** 2) / dof), 1e-8)
    try:
        cov_full = sigma2 * np.linalg.inv(
            jac.T @ jac + 1e-9 * np.eye(jac.shape[1])
        )
    except np.linalg.LinAlgError:
        cov_full = np.eye(jac.shape[1]) * 1e-6
    for i, oid in enumerate(ids):
        covariances[oid] = np.asarray(
            cov_full[3 * i : 3 * i + 3, 3 * i : 3 * i + 3], dtype=np.float64
        )

    return ArmOutput(positions=positions, covariances=covariances)


__all__ = [
    "ConstraintOptions",
    "constraint_residual",
    "contact_residual",
    "joint_refine",
    "penetration_residual",
    "support_residual",
]
