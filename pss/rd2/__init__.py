"""RD2 real-data schema and adapter (Phase RD)."""

from .adapter import episode_to_sample, load_episode, save_episode
from .schema import POSE_SOURCES, RDEpisode, RDFrame
from .validate import ValidationResult, validate_dataset, validate_episode

__all__ = [
    "POSE_SOURCES",
    "RDEpisode",
    "RDFrame",
    "ValidationResult",
    "episode_to_sample",
    "load_episode",
    "save_episode",
    "validate_dataset",
    "validate_episode",
]
