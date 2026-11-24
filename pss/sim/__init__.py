"""MuJoCo tabletop simulation and dataset pipeline (Phase 2)."""

from .scene import CATEGORY_SPECS, SceneConfig, build_scene
from .sensors import SensorRig
from .generator import generate_dataset, generate_episode, generate_category_sample

__all__ = [
    "CATEGORY_SPECS",
    "SceneConfig",
    "SensorRig",
    "build_scene",
    "generate_category_sample",
    "generate_dataset",
    "generate_episode",
]
