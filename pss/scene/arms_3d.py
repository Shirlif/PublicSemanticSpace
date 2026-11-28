"""S1 arms: 3D position estimation from 1 / 3 / 5 / all views.

The arms share ``optimizer_v2.solve_scene_v2`` and differ only in which
cameras of the example they consume ("the first N cameras" as registered in
the S1 pre-registration).
"""

from __future__ import annotations

from .bench_api import ArmOutput, BenchConfig, SceneExample, register_arm
from .optimizer_v2 import solve_scene_v2


def _solve_views(example: SceneExample, config: BenchConfig, n_views: int | None) -> ArmOutput:
    if n_views is None:
        view_indices = None
    else:
        view_indices = list(range(min(int(n_views), len(example.cameras))))
    return solve_scene_v2(example, config, view_indices=view_indices)


@register_arm("3d_1view")
def three_d_one_view(example: SceneExample, config: BenchConfig) -> ArmOutput:
    return _solve_views(example, config, 1)


@register_arm("3d_3view")
def three_d_three_view(example: SceneExample, config: BenchConfig) -> ArmOutput:
    return _solve_views(example, config, 3)


@register_arm("3d_5view")
def three_d_five_view(example: SceneExample, config: BenchConfig) -> ArmOutput:
    return _solve_views(example, config, 5)


@register_arm("3d_all")
def three_d_all_view(example: SceneExample, config: BenchConfig) -> ArmOutput:
    return _solve_views(example, config, None)


__all__ = [
    "three_d_all_view",
    "three_d_five_view",
    "three_d_one_view",
    "three_d_three_view",
]
