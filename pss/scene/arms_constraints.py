"""S2 arms: support / non-penetration constraints and their ablations.

Registered into ``pss.scene.bench_api.ARMS``:

* ``3d_3view_noconstraint`` -- per-object ``solve_scene`` only (H2 control);
* ``3d_3view_constraints``  -- ``joint_refine`` with support + non-penetration;
* ``3d_3view_nodepth``      -- ``solve_scene`` with ``use_depth=False`` (H1 control);
* ``3d_3view_outliers``     -- same estimator as ``3d_3view_constraints``; the
  outlier condition is applied by the benchmark via ``--outlier-frac 0.1``.
"""

from __future__ import annotations

from .bench_api import ArmOutput, BenchConfig, SceneExample, register_arm
from .constraints import joint_refine
from .optimizer import SolveOptions, solve_scene


@register_arm("3d_3view_noconstraint")
def no_constraint(example: SceneExample, config: BenchConfig) -> ArmOutput:
    return solve_scene(example, config)


@register_arm("3d_3view_constraints")
def with_constraints(example: SceneExample, config: BenchConfig) -> ArmOutput:
    return joint_refine(example, config)


@register_arm("3d_3view_nodepth")
def without_depth(example: SceneExample, config: BenchConfig) -> ArmOutput:
    return solve_scene(example, config, options=SolveOptions(use_depth=False))


@register_arm("3d_3view_outliers")
def with_outliers(example: SceneExample, config: BenchConfig) -> ArmOutput:
    # The arm is identical to ``3d_3view_constraints``; robustness is measured
    # by running the benchmark with ``--outlier-frac 0.1``.
    return joint_refine(example, config)


__all__ = ["no_constraint", "with_constraints", "with_outliers", "without_depth"]
