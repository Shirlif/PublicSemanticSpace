"""Uncertainty arm: 3D least squares + Monte Carlo relation beliefs.

Registers ``3d_belief``.  Positions come from ``solve_scene`` (per-object
least squares with the table prior); probabilities come from sampling the
linearised covariance and evaluating ``axis_aligned`` on every sample.  The
benchmark turns those probabilities into ``belief_ece`` without any change.

No temperature scaling is applied: the arm sees only the test example, so
calibrating on it would leak.  Any calibration belongs on a held-out cal split
and is left to the S3 analysis.
"""

from __future__ import annotations

import numpy as np

from .bench_api import ArmOutput, BenchConfig, SceneExample, register_arm
from .optimizer import solve_scene
from .uncertainty import (
    N_SAMPLES,
    RelationBelief,
    SamplingInfo,
    belief_from_output,
    stable_seed,
)


def three_d_belief_output(
    example: SceneExample,
    config: BenchConfig,
    *,
    n_samples: int = N_SAMPLES,
) -> tuple[ArmOutput, RelationBelief, SamplingInfo]:
    """Run the 3d_belief arm and also return the full relation belief."""

    solved = solve_scene(example, config)
    radii = example.true_scene.radii()
    seed = stable_seed("3d_belief", config.seed, example.scene_id)
    belief, info, _ = belief_from_output(
        solved, radii, config, n_samples=n_samples, seed=seed
    )
    output = ArmOutput(
        positions=solved.positions,
        probabilities=dict(belief.axis_aligned),
        covariances=solved.covariances,
    )
    return output, belief, info


@register_arm("3d_belief")
def three_d_belief(example: SceneExample, config: BenchConfig) -> ArmOutput:
    output, _, _ = three_d_belief_output(example, config)
    return output


__all__ = ["three_d_belief", "three_d_belief_output"]
