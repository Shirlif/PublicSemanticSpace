"""Tests for the MuJoCo scene bridge.  Skips if offscreen rendering is unavailable."""

from __future__ import annotations

import numpy as np
import pytest


def _sim():
    try:
        from pss.scene.sim3d import Sim3DScene
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"sim3d import failed: {exc}")
    try:
        sim = Sim3DScene(n_objects=3, seed=0, width=96, height=72)
        sim.set_view(0.0, -35.0, 1.10)
        sim.capture(rng=np.random.default_rng(0))
    except Exception as exc:  # pragma: no cover - rendering backend dependent
        pytest.skip(f"MuJoCo offscreen rendering unavailable: {exc}")
    return sim


def test_sim3d_capture_and_ground_truth():
    sim = _sim()
    try:
        gt = sim.ground_truth()
        frame = sim.capture(
            sigma_px=1.0, sigma_depth=0.01,
            rng=np.random.default_rng(0), min_pixels=4, max_points=16,
        )
        assert len(gt.objects) >= 1
        assert len(frame.objects) >= 1
        for oid, obs in frame.objects.items():
            assert oid in gt.objects
            assert obs.points_world.ndim == 2
            assert obs.points_world.shape[1] == 3
            assert np.isfinite(obs.points_world).all()
            uv, z = frame.camera.project(gt.objects[oid].pos[None, :])
            assert np.isfinite(uv).all()
            assert z[0] > 0
    finally:
        sim.close()


def test_sim3d_camera_projection_matches_mask_centroid_roughly():
    sim = _sim()
    try:
        gt = sim.ground_truth()
        frame = sim.capture(
            sigma_px=0.0, sigma_depth=0.0,
            rng=np.random.default_rng(0), min_pixels=4, max_points=16,
        )
        for oid, obs in frame.objects.items():
            uv, _ = frame.camera.project(gt.objects[oid].pos[None, :])
            # The visible-surface centroid is offset from the body centre, so
            # this is a loose sanity bound, not an equality.
            assert np.linalg.norm(uv[0] - obs.uv) < 25.0
    finally:
        sim.close()
