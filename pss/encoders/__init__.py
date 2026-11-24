"""Sensory experts (Phase 3).

Each expert is an independent, frozen feature extractor:
- vision: small CNN (real-pretrained backbone is the production path);
- tactile: MLP on a coarse contact grid;
- geometry: handcrafted features from depth + segmentation.
"""

from .geometry import GEOMETRY_DIM, geometry_features
from .registry import build_tactile_encoder, build_vision_encoder
from .tactile import (
    TactileConvEncoder,
    TactileConvExpert,
    TactileMLP,
    TactileExpert,
    load_tactile_expert,
    save_tactile_expert,
)
from .vision import SmallCNNEncoder, VisionExpert, load_vision_expert, save_vision_expert

__all__ = [
    "GEOMETRY_DIM",
    "SmallCNNEncoder",
    "TactileConvEncoder",
    "TactileConvExpert",
    "TactileExpert",
    "TactileMLP",
    "VisionExpert",
    "build_tactile_encoder",
    "build_vision_encoder",
    "geometry_features",
    "load_tactile_expert",
    "load_vision_expert",
    "save_tactile_expert",
    "save_vision_expert",
]
