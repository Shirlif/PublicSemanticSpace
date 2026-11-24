import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pss.rd2.public import PUBLIC_DATASETS, LeRobotDataset, temporal_consistency, train_autoencoder


def _write_fixture(root: Path, n: int = 16) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "toy",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": n,
        "features": {
            "observation.tactile.primary": {"dtype": "float32", "shape": [4, 4]},
            "observation.state": {"dtype": "float32", "shape": [3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        tactile = rng.random((4, 4)).astype(np.float32).tolist()
        rows.append(
            {
                "observation.tactile.primary": tactile,
                "observation.state": rng.normal(size=3).astype(np.float32).tolist(),
                "episode_index": 0,
                "frame_index": i,
                "timestamp": i / 30.0,
            }
        )
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")


def test_lerobot_adapter_fixture(tmp_path):
    _write_fixture(tmp_path)
    ds = LeRobotDataset(tmp_path, repo_id="toy/fixture")
    summary = ds.summary(max_rows=10)
    assert summary["robot_type"] == "toy"
    assert summary["tactile_shape"] == [4, 4]
    assert summary["loaded_frames"] == 10
    tactile, df = ds.tactile_matrix(max_rows=10)
    assert tactile.shape == (10, 4, 4)
    assert "episode_index" in df.columns


def test_smoke_functions(tmp_path):
    pytest.importorskip("torch")
    _write_fixture(tmp_path, n=24)
    ds = LeRobotDataset(tmp_path)
    tactile, _ = ds.tactile_matrix()
    ae = train_autoencoder(tactile, epochs=1, latent=8)
    assert ae["input_dim"] == 16
    assert np.isfinite(ae["val_mse"])
    temporal = temporal_consistency(tactile)
    assert np.isfinite(temporal["gap"])


def test_registry_licenses():
    assert PUBLIC_DATASETS["wt_mm"].license == "apache-2.0"
    assert PUBLIC_DATASETS["wt_mm"].commercial_ok is True
    assert PUBLIC_DATASETS["daimon_infinity"].commercial_ok is False
    assert PUBLIC_DATASETS["visuotactile_gs_oxt"].sync_tier == "pose_paired"
