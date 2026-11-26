from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pss.detect.replay import replay_bundle
from pss.eval.scripts.run_gap_perception_swap import (
    identity_retention_against,
    post_rollback_consistency,
    score_pose_delta,
)
from pss.experts.contract import ExpertHealth, validate_expert
from pss.experts.vision_modules import (
    FixedEmbeddingExpert,
    LinearProjection,
    VisionExpertModule,
)
from pss.metrics.property import rollback_exactness
from pss.property.swap import ModuleRegistry


class _LinearExpert:
    """Minimal contract-compliant expert over native feature vectors."""

    modality = "vision"

    def __init__(self, version: str, native_dim: int, dim: int = 128):
        self.version = version
        self.dim = int(dim)
        self.native_dim = int(native_dim)
        self.weights = np.eye(self.native_dim, self.dim)

    def encode(self, obs) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != self.native_dim:
            raise ValueError(
                f"expected native dim {self.native_dim}, got {arr.shape[1]}"
            )
        z = arr @ self.weights
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        return (z / np.maximum(norms, 1e-12)).astype(np.float32)

    def health(self) -> ExpertHealth:
        return ExpertHealth()


class _ImageEncoder:
    out_dim = 8
    image_size = 8

    def encode(self, images):
        return images.mean(dim=(1, 2, 3)).unsqueeze(1).repeat(1, self.out_dim)


def _module(native_dim: int, version: str = "m-v1") -> VisionExpertModule:
    return VisionExpertModule(
        name="test",
        version=version,
        native_dim=native_dim,
        projection=LinearProjection(np.eye(native_dim, 128)),
    )


def test_vision_module_satisfies_contract():
    module = _module(6)
    probe = np.arange(24, dtype=np.float64).reshape(4, 6) + 1.0
    assert validate_expert(module, 128, probe=probe)
    out = module.encode(probe)
    assert out.shape == (4, 128)
    assert np.all(np.isfinite(out))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    assert isinstance(module.health(), ExpertHealth)


def test_vision_module_rejects_wrong_native_dim():
    module = _module(4)
    with pytest.raises(ValueError):
        module.encode(np.ones((2, 6)))


def test_vision_module_image_path():
    module = VisionExpertModule(
        name="image-test",
        version="img-v1",
        native_dim=8,
        projection=LinearProjection(np.eye(8, 128)),
        encoder=_ImageEncoder(),
        image_size=8,
    )
    import torch

    out = module.encode(torch.randn(2, 3, 8, 8))
    assert out.shape == (2, 128)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_positive_swap_and_rollback_are_exact():
    small = _LinearExpert("small-v1", 4)
    incoming = _LinearExpert("incoming-v1", 6)
    small_probe = np.ones((2, 4))
    incoming_probe = np.ones((2, 6))

    registry = ModuleRegistry(public_dim=128)
    registry.register(small, probe=small_probe)
    before = registry.behavior_hash("vision", probe=small_probe)

    swap = registry.swap(incoming, probe=incoming_probe)
    assert swap.success
    assert registry.active_versions()["vision"] == incoming.version

    rollback = registry.rollback()
    assert rollback.success
    assert registry.active_versions()["vision"] == small.version
    after = registry.behavior_hash("vision", probe=small_probe)
    assert rollback_exactness(before, after) == 1.0


def test_naive_swap_cannot_roll_back():
    small = _LinearExpert("small-v1", 4)
    incoming = _LinearExpert("incoming-v1", 6)
    small_probe = np.ones((2, 4))

    registry = ModuleRegistry(public_dim=128)
    registry.register(small, probe=small_probe)
    before = registry.behavior_hash("vision", probe=small_probe)

    # The negative control: mutate the module dict, bypassing validate/stack.
    registry._modules["vision"] = incoming
    rollback = registry.rollback()
    assert rollback.success is False
    assert registry.active_versions()["vision"] == incoming.version

    with pytest.raises(ValueError):
        registry.behavior_hash("vision", probe=small_probe)
    assert rollback_exactness(before, "error") == 0.0


def test_unrelated_module_is_not_interfered():
    vision = _LinearExpert("vision-v1", 4)
    geometry = FixedEmbeddingExpert(
        modality="geometry", version="geometry-v1", input_dim=4, seed=0
    )
    vision_probe = np.ones((2, 4))
    geometry_probe = np.ones((2, 4))
    registry = ModuleRegistry(public_dim=128)
    registry.register(vision, probe=vision_probe)
    registry.register(geometry, probe=geometry_probe)
    before = registry.behavior_hash("geometry", probe=geometry_probe)
    registry.swap(_LinearExpert("vision-v2", 6), probe=np.ones((2, 6)))
    after = registry.behavior_hash("geometry", probe=geometry_probe)
    assert before == after


def _tiny_bundle(root):
    rows = []
    for step in range(5):
        for j, pos in enumerate(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])):
            rows.append(
                {
                    "step": step,
                    "t_s": step / 10.0,
                    "camera": "overhead",
                    "det_id": f"{step}_{j}",
                    "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                    "score": 0.9,
                    "class_pred": "cube",
                    "pose_xyz": pos,
                    "pose_quat": [1.0, 0.0, 0.0, 0.0],
                    "emb": [1.0, 0.0] if j == 0 else [0.0, 1.0],
                    "emb_ref": f"{step}_{j}",
                }
            )
    pd.DataFrame(rows).to_parquet(root / "obs_detections.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps({"spec": {"id": "test", "hash": "sha256:test"}}),
        encoding="utf-8",
    )
    return root


def test_replay_accepts_appearance_provider_and_frame_hook(tmp_path):
    root = _tiny_bundle(tmp_path)
    frames: list[int] = []
    seen: list[int] = []

    def hook(frame_idx: int, step: int) -> None:
        frames.append(frame_idx)

    def provider(frame_idx: int, step: int, group: pd.DataFrame):
        seen.append(frame_idx)
        return [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    trace, pss = replay_bundle(
        root,
        write=False,
        appearance_provider=provider,
        frame_hook=hook,
    )
    assert frames == [0, 1, 2, 3, 4]
    assert seen == [0, 1, 2, 3, 4]
    assert len(pss) == 5
    assert not trace.empty
    assert not (root / "gt.parquet").exists()


def test_replay_rejects_provider_row_count_mismatch(tmp_path):
    root = _tiny_bundle(tmp_path)

    def bad_provider(frame_idx, step, group):
        return [np.array([1.0, 0.0])]

    with pytest.raises(ValueError, match="appearance_provider"):
        replay_bundle(root, write=False, appearance_provider=bad_provider)


def test_identity_retention_against_reference():
    reference = pd.DataFrame(
        {
            "step": [0, 0, 1, 1],
            "gt_id": ["a", "b", "a", "b"],
            "pred_id": ["t1", "t2", "t1", "t2"],
        }
    )
    assert identity_retention_against(reference, reference.copy()) == 1.0
    split = pd.DataFrame(
        {
            "step": [0, 0, 1, 1],
            "gt_id": ["a", "b", "a", "b"],
            "pred_id": ["t1", "t2", "t9", "t2"],
        }
    )
    # Group t1 keeps 1 of 2 frames, group t2 keeps both -> mean 0.75.
    assert identity_retention_against(reference, split) == pytest.approx(0.75)


def test_post_rollback_consistency_discriminates():
    baseline = SimpleNamespace(
        used={
            3: {"version": "small", "Z": np.array([[1.0, 0.0]])},
            4: {"version": "small", "Z": np.array([[1.0, 0.0]])},
        }
    )
    positive = SimpleNamespace(
        used={
            3: {"version": "small", "Z": np.array([[1.0, 0.0]])},
            4: {"version": "small", "Z": np.array([[1.0, 0.0]])},
        }
    )
    naive = SimpleNamespace(
        used={
            3: {"version": "incoming", "Z": np.array([[0.0, 1.0]])},
            4: {"version": "incoming", "Z": np.array([[0.0, 1.0]])},
        }
    )
    assert post_rollback_consistency(baseline, positive, 3, "small") == 1.0
    assert post_rollback_consistency(baseline, naive, 3, "small") == 0.0


def test_pose_delta_scoring_reads_gt_only_for_scoring(tmp_path):
    detections = pd.DataFrame(
        {
            "step": [0, 1],
            "t_s": [0.0, 0.1],
            "pose_xyz": [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
            "pose_quat": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        }
    )
    gt = pd.DataFrame(
        {
            "t_s": [0.0, 0.1],
            "instance_id": ["a", "a"],
            "class": ["cube", "cube"],
            "pose_xyz": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "pose_quat": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        }
    )
    gt.to_parquet(tmp_path / "gt.parquet", index=False)
    result = score_pose_delta(tmp_path, detections)
    assert result["matched"] == 2
    assert result["match_rate"] == 1.0
    assert result["pos_mean_m"] == pytest.approx(0.1, abs=1e-9)
    assert result["quat_mean_deg"] == pytest.approx(0.0, abs=1e-9)
