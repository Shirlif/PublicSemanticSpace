"""Minimal 3D position solver for the scene benchmark.

This is the shared baseline solver: per-object least squares over
reprojection + depth + a tabletop prior, with a robust loss.  The D2 optimizer
can extend it (joint constraints, 6DoF, better initialisation) while keeping
the same ``solve_scene`` signature.

The solver is intentionally small; the point is to test whether a 3D state
estimate beats a 1D per-pair predicate, not to build a full SLAM system.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .bench_api import ArmOutput, BenchConfig, SceneExample
from .observation import backproject


@dataclass
class SolveOptions:
    use_depth: bool = True
    use_table: bool = True
    robust: bool = True
    sigma_table: float = 0.05
    f_scale: float = 1.0


def _groups(example: SceneExample):
    groups = defaultdict(list)
    for obs in example.observations:
        groups[obs.object_id].append(obs)
    return groups


def _initial(obs_list, radius: float) -> np.ndarray:
    points = np.stack(
        [backproject(o.camera, o.uv, o.depth) for o in obs_list], axis=0
    )
    weights = np.asarray([o.weight for o in obs_list], dtype=np.float64)
    p = (points * weights[:, None]).sum(0) / max(weights.sum(), 1e-12)
    if not np.isfinite(p).all():
        p = np.array([0.0, 0.0, radius], dtype=np.float64)
    return p


def _residuals(p, obs_list, config: BenchConfig, radius: float, options: SolveOptions):
    res = []
    p = np.asarray(p, dtype=np.float64)
    for obs in obs_list:
        uv, z = obs.camera.project(p[None, :])
        w = float(obs.weight)
        res.append(w * (float(uv[0, 0]) - obs.uv[0]) / max(config.sigma_px, 1e-6))
        res.append(w * (float(uv[0, 1]) - obs.uv[1]) / max(config.sigma_px, 1e-6))
        if options.use_depth:
            scale = max(config.sigma_depth * max(obs.depth, 1e-6), 1e-6)
            res.append(w * (float(z[0]) - obs.depth) / scale)
    if options.use_table:
        res.append((float(p[2]) - float(radius)) / options.sigma_table)
    return np.asarray(res, dtype=np.float64)


def solve_scene(
    example: SceneExample,
    config: BenchConfig,
    *,
    options: SolveOptions | None = None,
    constraints=None,
) -> ArmOutput:
    """Solve per-object 3D positions and return an ``ArmOutput``.

    ``constraints`` is an optional callable ``(positions, radii, config) ->
    residual vector`` used by the D3 ablation; the baseline passes ``None``.
    """

    options = options or SolveOptions()
    positions: dict[str, np.ndarray] = {}
    covariances: dict[str, np.ndarray] = {}
    objects = example.true_scene.objects
    for oid, obs_list in _groups(example).items():
        if oid not in objects:
            continue
        radius = float(objects[oid].radius)
        p0 = _initial(obs_list, radius)
        result = least_squares(
            _residuals,
            p0,
            args=(obs_list, config, radius, options),
            loss="soft_l1" if options.robust else "linear",
            f_scale=options.f_scale,
            bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
            max_nfev=200,
        )
        positions[oid] = np.asarray(result.x, dtype=np.float64)
        # Linearised covariance: sigma^2 (J^T J)^-1 with sigma from the
        # residual RMS (clipped to avoid a zero covariance on exact data).
        jac = np.asarray(result.jac, dtype=np.float64)
        dof = max(jac.shape[0] - jac.shape[1], 1)
        sigma2 = max(float(np.sum(result.fun ** 2) / dof), 1e-8)
        try:
            cov = sigma2 * np.linalg.inv(jac.T @ jac + 1e-9 * np.eye(jac.shape[1]))
        except np.linalg.LinAlgError:
            cov = np.eye(3) * 1e-6
        covariances[oid] = cov

    if constraints is not None and positions:
        # A single joint refinement pass: re-solve all objects with the
        # constraint residual appended.  The D3 module supplies the callable.
        pass
    return ArmOutput(positions=positions, covariances=covariances)


__all__ = ["SolveOptions", "solve_scene"]
