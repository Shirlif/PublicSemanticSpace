"""Public embodied dataset adapters (LeRobot v3.0) and smoke tests."""

from .lerobot import LeRobotDataset, LeRobotInfo
from .registry import PUBLIC_DATASETS, PublicDatasetSpec
from .smoke import (
    decode_video,
    run_smoke,
    tactile_stats,
    temporal_consistency,
    train_autoencoder,
    video_tactile_alignment,
)

__all__ = [
    "LeRobotDataset",
    "LeRobotInfo",
    "PUBLIC_DATASETS",
    "PublicDatasetSpec",
    "decode_video",
    "run_smoke",
    "tactile_stats",
    "temporal_consistency",
    "train_autoencoder",
    "video_tactile_alignment",
]
