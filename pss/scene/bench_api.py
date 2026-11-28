"""Benchmark API for the 3D scene-model experiments.

Arms are small functions ``(SceneExample, BenchConfig) -> ArmOutput``.  The
benchmark owns data, observations and metrics; arms own estimation/query.
This keeps the 1D fallback, the 3D optimizer, constraints, uncertainty and PSS
bridge in separate modules without editing the benchmark each time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .metrics import binary_ece, mean_std, pair_metrics, pose_errors
from .model import Camera, Observation, SceneModel
from .observation import make_cameras, observe


@dataclass
class BenchConfig:
    n_views: int = 3
    sigma_px: float = 2.0
    sigma_depth: float = 0.01
    sigma_pose_yaw: float = 0.0
    sigma_pose_pitch: float = 0.0
    outlier_frac: float = 0.0
    eps_xy: float = 0.12
    eps_z: float = 0.05
    eps_c: float = 0.02
    eps_s: float = 0.05
    table_z: float = 0.0
    camera_radius: float = 10.0
    camera_height: float = 6.0
    focal: float = 400.0
    seed: int = 0


@dataclass
class SceneExample:
    scene_id: str
    true_scene: SceneModel
    cameras: list[Camera]
    observations: list[Observation]


@dataclass
class ArmOutput:
    positions: dict[str, np.ndarray]
    probabilities: dict[tuple[str, str], float] = field(default_factory=dict)
    covariances: dict[str, np.ndarray] = field(default_factory=dict)


ArmFn = Callable[[SceneExample, BenchConfig], ArmOutput]
ARMS: dict[str, ArmFn] = {}

PrepareFn = Callable[[list[SceneModel], BenchConfig], None]
PREPARE: list[PrepareFn] = []


def register_prepare(fn: PrepareFn) -> PrepareFn:
    PREPARE.append(fn)
    return fn


def run_prepare(train_scenes: list[SceneModel], config: BenchConfig) -> None:
    for fn in PREPARE:
        fn(train_scenes, config)


def register_arm(name: str):
    def _wrap(fn: ArmFn) -> ArmFn:
        ARMS[name] = fn
        return fn

    return _wrap


def build_examples(
    scenes: list[SceneModel], config: BenchConfig, *, seed_offset: int = 0
) -> list[SceneExample]:
    examples = []
    for i, scene in enumerate(scenes):
        rng = np.random.default_rng(config.seed + seed_offset + i)
        cameras = make_cameras(
            scene, config.n_views,
            radius=config.camera_radius,
            height=config.camera_height,
            focal=config.focal,
        )
        observations = observe(
            scene, cameras,
            sigma_px=config.sigma_px,
            sigma_depth=config.sigma_depth,
            sigma_pose_yaw=config.sigma_pose_yaw,
            sigma_pose_pitch=config.sigma_pose_pitch,
            outlier_frac=config.outlier_frac,
            rng=rng,
        )
        examples.append(
            SceneExample(
                scene_id=scene.scene_id,
                true_scene=scene,
                cameras=cameras,
                observations=observations,
            )
        )
    return examples


def evaluate_arm(name: str, examples: list[SceneExample], config: BenchConfig) -> dict:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {sorted(ARMS)}")
    rows = []
    for example in examples:
        output = ARMS[name](example, config)
        true = example.true_scene.positions()
        radii = example.true_scene.radii()
        metrics = pair_metrics(
            output.positions, true, radii,
            eps_xy=config.eps_xy,
            eps_z=config.eps_z,
            eps_c=config.eps_c,
            table_z=config.table_z,
            eps_s=config.eps_s,
        )
        metrics.update(pose_errors(output.positions, true))
        if output.probabilities:
            probs, labels = [], []
            for (a, b), prob in output.probabilities.items():
                if a not in true or b not in true:
                    continue
                from .metrics import axis_aligned
                probs.append(float(prob))
                labels.append(float(axis_aligned(true[a], true[b], config.eps_xy)))
            if probs:
                metrics["belief_ece"] = binary_ece(np.asarray(probs), np.asarray(labels))
        rows.append(metrics)

    keys = (
        "overall", "same_level", "contact", "collision",
        "support", "median", "p90", "mean", "belief_ece",
    )
    out = {key: mean_std(rows, key) for key in keys}
    # Scenes with no near-boundary pair must not contribute a 0 to the near
    # metric; they carry no evidence for it.
    near_rows = [r for r in rows if r.get("near_support", 0) > 0]
    out["near_aligned"] = mean_std(near_rows, "near_aligned")
    out["near_support"] = int(np.sum([r.get("near_support", 0) for r in rows]))
    out["n_pairs"] = int(np.sum([r.get("n_pairs", 0) for r in rows]))
    out["n_scenes"] = len(rows)
    out["n_near_scenes"] = len(near_rows)
    return out


def load_optional_arms() -> None:
    """Import arm modules if they exist; each registers itself."""

    for module in (
        "pss.scene.arms_baseline",
        "pss.scene.arms_3d",
        "pss.scene.arms_constraints",
        "pss.scene.arms_uncertainty",
    ):
        try:
            __import__(module)
        except ModuleNotFoundError:
            continue


__all__ = [
    "ARMS",
    "ArmOutput",
    "BenchConfig",
    "SceneExample",
    "build_examples",
    "evaluate_arm",
    "load_optional_arms",
    "register_arm",
    "register_prepare",
    "run_prepare",
]
