"""D2 3D position solver: per-object robust least squares with view subsets.

This is the S1 estimator.  It keeps the same role as ``optimizer.solve_scene``
(observations -> ``ArmOutput``) but adds the pieces the S1 question needs:

* selection of a camera subset (1 / 3 / 5 / all views) so the same solver can
  be run at different view counts;
* a better initialisation (depth-fused backprojection or pixel-only linear
  triangulation, whichever has the lower robust cost);
* correctly normalised depth residuals (the simulator's depth noise is
  multiplicative, so the scale is ``sigma_depth * d_obs``);
* a tabletop prior strong enough to use the known support plane;
* an optional joint solve with non-penetration (off by default; S1 is the
  "no constraint" stage).

The shared ``optimizer.solve_scene`` is left untouched because D3/D4 build on
that stable interface.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .bench_api import ArmOutput, BenchConfig, SceneExample
from .observation import backproject


@dataclass
class SolveOptionsV2:
    """Knobs for the S1 solver.  Defaults are the S1 arm configuration."""

    use_depth: bool = True
    use_table: bool = True
    # All CLEVR objects rest on the support plane (checked: z == radius for
    # every object in the cached splits), so a tight table prior is justified.
    sigma_table: float = 0.02
    robust: bool = True
    loss: str = "soft_l1"
    # 2.0 keeps almost all clean residuals near the quadratic regime while
    # still bounding the influence of the 20-40 px outlier mode.
    f_scale: float = 2.0
    max_nfev: int = 300
    init: str = "best"  # "mean" | "triangulate" | "best"
    joint: bool = False
    sigma_penetration: float = 0.05
    penetration_weight: float = 1.0


def _camera_index(example: SceneExample) -> dict[int, int]:
    """Map ``id(obs.camera)`` to the camera's position in ``example.cameras``.

    ``observe`` may store a perturbed copy of a camera, so fall back to the
    camera name when the object identity does not match.
    """

    by_name = {cam.name: i for i, cam in enumerate(example.cameras)}
    index: dict[int, int] = {}
    for obs in example.observations:
        idx = by_name.get(obs.camera.name)
        if idx is None:
            for i, cam in enumerate(example.cameras):
                if cam is obs.camera:
                    idx = i
                    break
        if idx is not None:
            index[id(obs.camera)] = idx
    return index


def _group_observations(
    example: SceneExample, view_indices: list[int] | None
) -> dict[str, list]:
    """Group observations per object, keeping only the requested views."""

    allowed = None if view_indices is None else set(int(i) for i in view_indices)
    cam_index = _camera_index(example)
    groups: dict[str, list] = defaultdict(list)
    for obs in example.observations:
        idx = cam_index.get(id(obs.camera))
        if allowed is not None and idx is not None and idx not in allowed:
            continue
        groups[obs.object_id].append(obs)
    return groups


def _weighted_mean_init(obs_list) -> np.ndarray:
    points = np.stack(
        [backproject(o.camera, o.uv, o.depth) for o in obs_list], axis=0
    )
    weights = np.asarray([max(float(o.weight), 1e-3) for o in obs_list], dtype=np.float64)
    p = (points * weights[:, None]).sum(0) / max(weights.sum(), 1e-12)
    if not np.isfinite(p).all():
        p = np.zeros(3, dtype=np.float64)
    return p


def _triangulate_init(obs_list, sigma_px: float) -> np.ndarray | None:
    """Pixel-only linear triangulation (DLT), weighted by inverse variance."""

    rows = []
    weights = []
    for obs in obs_list:
        cam = obs.camera
        P = cam.K @ np.hstack([cam.R, cam.t.reshape(3, 1)])
        u, v = float(obs.uv[0]), float(obs.uv[1])
        w = 1.0 / max(sigma_px, 1e-6) * max(float(obs.weight), 1e-3) ** 0.5
        rows.append(w * (u * P[2] - P[0]))
        rows.append(w * (v * P[2] - P[1]))
        weights.extend([w, w])
    if len(rows) < 4:
        return None
    A = np.asarray(rows, dtype=np.float64)
    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    p = X[:3] / X[3]
    return p if np.isfinite(p).all() else None


def _residual_vector(
    p: np.ndarray,
    obs_list,
    config: BenchConfig,
    radius: float,
    options: SolveOptionsV2,
) -> np.ndarray:
    res = []
    p = np.asarray(p, dtype=np.float64)
    for obs in obs_list:
        uv, z = obs.camera.project(p[None, :])
        w = max(float(obs.weight), 1e-3)
        res.append(w * (float(uv[0, 0]) - float(obs.uv[0])) / max(config.sigma_px, 1e-6))
        res.append(w * (float(uv[0, 1]) - float(obs.uv[1])) / max(config.sigma_px, 1e-6))
        if options.use_depth:
            scale = max(config.sigma_depth * max(float(obs.depth), 1e-6), 1e-6)
            res.append(w * (float(z[0]) - float(obs.depth)) / scale)
    if options.use_table:
        res.append((float(p[2]) - float(radius)) / options.sigma_table)
    return np.asarray(res, dtype=np.float64)


def _robust_cost(p, obs_list, config, radius, options) -> float:
    r = _residual_vector(p, obs_list, config, radius, options)
    f = max(float(options.f_scale), 1e-6)
    z = (r / f) ** 2
    return float(np.sum(2.0 * f * f * (np.sqrt(1.0 + z) - 1.0)))


def _initial(obs_list, config: BenchConfig, radius: float, options: SolveOptionsV2) -> np.ndarray:
    mean_p = _weighted_mean_init(obs_list)
    mean_p[2] = max(mean_p[2], 0.0)
    candidates = [mean_p]
    if options.init in ("triangulate", "best"):
        tri = _triangulate_init(obs_list, config.sigma_px)
        if tri is not None and tri[2] >= 0.0:
            candidates.append(tri)
    if len(candidates) == 1:
        return mean_p
    costs = [_robust_cost(p, obs_list, config, radius, options) for p in candidates]
    return candidates[int(np.argmin(costs))]


def _covariance(result, options: SolveOptionsV2) -> np.ndarray:
    jac = np.asarray(result.jac, dtype=np.float64)
    dof = max(jac.shape[0] - jac.shape[1], 1)
    sigma2 = max(float(np.sum(result.fun ** 2) / dof), 1e-8)
    try:
        cov = sigma2 * np.linalg.inv(jac.T @ jac + 1e-9 * np.eye(jac.shape[1]))
    except np.linalg.LinAlgError:
        cov = np.eye(jac.shape[1]) * 1e-6
    return cov


def _solve_per_object(
    example: SceneExample,
    config: BenchConfig,
    groups: dict[str, list],
    options: SolveOptionsV2,
) -> ArmOutput:
    positions: dict[str, np.ndarray] = {}
    covariances: dict[str, np.ndarray] = {}
    for oid, obs_list in groups.items():
        if not obs_list:
            continue
        obj = example.true_scene.objects.get(oid)
        radius = float(obj.radius) if obj is not None else 0.35
        p0 = _initial(obs_list, config, radius, options)
        result = least_squares(
            _residual_vector,
            p0,
            args=(obs_list, config, radius, options),
            loss=options.loss if options.robust else "linear",
            f_scale=options.f_scale,
            bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
            max_nfev=options.max_nfev,
        )
        positions[oid] = np.asarray(result.x, dtype=np.float64)
        covariances[oid] = _covariance(result, options)
    return ArmOutput(positions=positions, covariances=covariances)


def _solve_joint(
    example: SceneExample,
    config: BenchConfig,
    groups: dict[str, list],
    options: SolveOptionsV2,
) -> ArmOutput:
    ids = [oid for oid in example.true_scene.objects if groups.get(oid)]
    if not ids:
        return ArmOutput(positions={}, covariances={})
    radii = {oid: float(example.true_scene.objects[oid].radius) for oid in ids}
    initials = []
    for oid in ids:
        radius = radii[oid]
        p0 = _initial(groups[oid], config, radius, options)
        p0[2] = max(p0[2], 0.0)
        initials.append(p0)
    x0 = np.concatenate(initials)
    pairs = [(i, j) for i in range(len(ids)) for j in range(i + 1, len(ids))]

    def residuals(x: np.ndarray) -> np.ndarray:
        pts = x.reshape(-1, 3)
        res = []
        for i, oid in enumerate(ids):
            res.append(
                _residual_vector(pts[i], groups[oid], config, radii[oid], options)
            )
        for i, j in pairs:
            d = float(np.linalg.norm(pts[i] - pts[j]))
            gap = radii[ids[i]] + radii[ids[j]] - d
            # One-sided: only penalise penetration, never reward contact.
            res.append(
                np.asarray(
                    [
                        options.penetration_weight
                        * max(0.0, gap)
                        / options.sigma_penetration
                    ]
                )
            )
        return np.concatenate(res)

    lower = np.tile(np.array([-np.inf, -np.inf, 0.0]), len(ids))
    upper = np.tile(np.array([np.inf, np.inf, np.inf]), len(ids))
    result = least_squares(
        residuals,
        x0,
        loss=options.loss if options.robust else "linear",
        f_scale=options.f_scale,
        bounds=(lower, upper),
        max_nfev=options.max_nfev,
    )
    positions = {oid: np.asarray(result.x[3 * i : 3 * i + 3], dtype=np.float64)
                 for i, oid in enumerate(ids)}
    # Joint Jacobian does not split cleanly per object; report the per-object
    # block from the joint information matrix instead.
    jac = np.asarray(result.jac, dtype=np.float64)
    dof = max(jac.shape[0] - jac.shape[1], 1)
    sigma2 = max(float(np.sum(result.fun ** 2) / dof), 1e-8)
    covariances = {}
    try:
        info_inv = sigma2 * np.linalg.inv(jac.T @ jac + 1e-9 * np.eye(jac.shape[1]))
    except np.linalg.LinAlgError:
        info_inv = np.eye(jac.shape[1]) * 1e-6
    for i, oid in enumerate(ids):
        covariances[oid] = info_inv[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]
    return ArmOutput(positions=positions, covariances=covariances)


def solve_scene_v2(
    example: SceneExample,
    config: BenchConfig,
    *,
    view_indices: list[int] | None = None,
    options: SolveOptionsV2 | None = None,
) -> ArmOutput:
    """Solve per-object 3D positions using only ``view_indices`` cameras.

    ``view_indices=None`` uses every available camera.  The returned
    ``ArmOutput`` carries positions and linearised 3x3 covariances.
    """

    options = options or SolveOptionsV2()
    groups = _group_observations(example, view_indices)
    if options.joint:
        return _solve_joint(example, config, groups, options)
    return _solve_per_object(example, config, groups, options)


__all__ = ["SolveOptionsV2", "solve_scene_v2"]
