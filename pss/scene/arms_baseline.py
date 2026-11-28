"""D1 baseline arms: oracle and unoptimized backprojection."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .bench_api import ArmOutput, BenchConfig, SceneExample, register_arm
from .observation import backproject
from .optimizer import solve_scene


def _by_object(example: SceneExample) -> dict[str, list]:
    groups = defaultdict(list)
    for obs in example.observations:
        groups[obs.object_id].append(obs)
    return groups


@register_arm("oracle")
def oracle(example: SceneExample, config: BenchConfig) -> ArmOutput:
    positions = example.true_scene.positions()
    probs = {}
    ids = list(positions)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            probs[(a, b)] = 1.0
    return ArmOutput(positions=positions, probabilities=probs)


@register_arm("1d_hard_single")
def one_view_hard(example: SceneExample, config: BenchConfig) -> ArmOutput:
    positions = {}
    for oid, obs_list in _by_object(example).items():
        obs = max(obs_list, key=lambda o: o.weight)
        positions[oid] = backproject(obs.camera, obs.uv, obs.depth)
    return ArmOutput(positions=positions)


@register_arm("backproject_mean")
def backproject_mean(example: SceneExample, config: BenchConfig) -> ArmOutput:
    positions = {}
    for oid, obs_list in _by_object(example).items():
        points = np.stack(
            [backproject(o.camera, o.uv, o.depth) for o in obs_list], axis=0
        )
        weights = np.asarray([o.weight for o in obs_list], dtype=np.float64)
        positions[oid] = (points * weights[:, None]).sum(0) / weights.sum()
    return ArmOutput(positions=positions)


@register_arm("backproject_median")
def backproject_median(example: SceneExample, config: BenchConfig) -> ArmOutput:
    positions = {}
    for oid, obs_list in _by_object(example).items():
        points = np.stack(
            [backproject(o.camera, o.uv, o.depth) for o in obs_list], axis=0
        )
        positions[oid] = np.median(points, axis=0)
    return ArmOutput(positions=positions)


@register_arm("3d_ls")
def least_squares_arm(example: SceneExample, config: BenchConfig) -> ArmOutput:
    return solve_scene(example, config)


__all__ = [
    "backproject_mean",
    "backproject_median",
    "least_squares_arm",
    "one_view_hard",
    "oracle",
]
