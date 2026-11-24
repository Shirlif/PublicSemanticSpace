from pathlib import Path

import numpy as np
import pytest

from pss.rd2 import (
    RDEpisode,
    RDFrame,
    episode_to_sample,
    load_episode,
    save_episode,
    validate_episode,
)


def _episode() -> RDEpisode:
    frames = []
    for i in range(3):
        frames.append(
            RDFrame(
                timestamp_ns=i * 33_333_333,
                rgb=np.zeros((32, 32, 3), dtype=np.uint8),
                depth=np.ones((32, 32), dtype=np.float32),
                mask=np.zeros((32, 32), dtype=bool),
                tactile=np.zeros((8, 8), dtype=np.float32),
                sensor_pose=np.zeros(7),
                object_pose=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                pose_source="mocap",
                sync_offset_ns=0,
                environment={"lighting": "normal"},
            )
        )
    return RDEpisode(
        episode_id="ep_0000",
        instance_id="cup_00",
        category="cup",
        frames=frames,
        calibration={"camera": "mock", "hand_eye": "mock"},
        consent=True,
        privacy_scrubbed=True,
        source="real",
    )


def test_validate_episode_ok():
    assert validate_episode(_episode()).ok


def test_validate_episode_rejects_bad_sync():
    episode = _episode()
    episode.frames[0].sync_offset_ns = 10_000_000
    assert not validate_episode(episode).ok


def test_validate_episode_rejects_missing_consent():
    episode = _episode()
    episode.consent = False
    assert not validate_episode(episode).ok


def test_save_load_and_convert(tmp_path):
    episode = _episode()
    save_episode(tmp_path / "ep", episode)
    loaded = load_episode(tmp_path / "ep")
    assert len(loaded.frames) == 3
    sample = episode_to_sample(loaded)
    assert sample["rgb"].shape == (3, 32, 32, 3)
    assert sample["segmentation"].shape == (3, 32, 32, 2)
    assert sample["poses"].shape == (3, 1, 7)
    assert sample["tactile"].shape == (3, 1, 8, 8)


@pytest.mark.skipif(
    not Path("data/expert_v1").exists(), reason="run dataset generation first"
)
def test_mock_dataset_validates(tmp_path):
    pytest.importorskip("mujoco")
    from pss.rd2.mock import make_mock_dataset
    from pss.rd2.validate import validate_dataset

    summary = make_mock_dataset(tmp_path, "data/expert_v1", n_episodes=2, seed=0)
    assert summary["episodes"] == 2
    result = validate_dataset(tmp_path)
    assert result["episodes"] == 2
    assert result["ok"] == 2
