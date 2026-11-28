"""Tests for the D1 3D scene benchmark foundation."""

from __future__ import annotations

import numpy as np

from pss.scene import arms_baseline  # noqa: F401  (registers the baseline arms)
from pss.scene.bench_api import (
    ARMS,
    ArmOutput,
    BenchConfig,
    build_examples,
    evaluate_arm,
    register_arm,
)
from pss.scene.metrics import (
    axis_aligned,
    binary_ece,
    near_boundary,
    pair_metrics,
)
from pss.scene.model import Camera, ObjectState, SceneModel
from pss.scene.observation import backproject, make_cameras, observe
from pss.scene.optimizer import solve_scene


def _scene():
    return SceneModel(
        scene_id="s0",
        objects={
            "a": ObjectState("a", np.array([0.0, 0.0, 0.35]), radius=0.35),
            "b": ObjectState("b", np.array([0.10, 0.0, 0.35]), radius=0.35),
            "c": ObjectState("c", np.array([0.0, 1.0, 0.70]), radius=0.70),
        },
    )


def test_camera_projection_and_backprojection_roundtrip():
    cam = Camera.look_at([0.0, -10.0, 5.0], [0.0, 0.0, 0.5], focal=400.0)
    point = np.array([0.2, 0.3, 1.0])
    uv, z = cam.project(point[None, :])
    recovered = backproject(cam, uv[0], float(z[0]))
    assert np.allclose(recovered, point, atol=1e-6)
    assert uv.shape == (1, 2)


def test_make_cameras_and_observe():
    scene = _scene()
    cams = make_cameras(scene, 3)
    assert len(cams) == 3
    obs = observe(scene, cams, sigma_px=0.0, sigma_depth=0.0)
    assert obs
    assert all(o.object_id in scene.objects for o in obs)


def test_axis_aligned_and_near_boundary():
    pa = np.array([0.0, 0.0, 0.35])
    pb = np.array([0.10, 0.5, 0.35])
    assert axis_aligned(pa, pb, 0.12)
    assert near_boundary(pa, pb, 0.12)
    assert not axis_aligned(pa, np.array([0.5, 0.5, 0.35]), 0.12)


def test_pair_metrics_oracle_is_perfect():
    scene = _scene()
    true = scene.positions()
    m = pair_metrics(
        true, true, scene.radii(),
        eps_xy=0.12, eps_z=0.05, eps_c=0.02,
    )
    assert m["overall"] == 1.0
    if m["near_support"]:
        assert m["near_aligned"] == 1.0


def test_binary_ece_perfect_calibration():
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert binary_ece(probs, labels) < 0.06


def test_bench_oracle_and_least_squares_arms():
    scene = _scene()
    config = BenchConfig(n_views=3, sigma_px=0.0, sigma_depth=0.0, seed=0)
    examples = build_examples([scene], config)
    oracle = evaluate_arm("oracle", examples, config)
    assert oracle["near_aligned"]["mean"] == 1.0
    output = solve_scene(examples[0], config)
    assert set(output.positions) == set(scene.objects)
    assert set(output.covariances) == set(scene.objects)
