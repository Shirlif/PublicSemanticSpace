"""Cross-modal projection heads (Phase 4)."""

from .calibration import (
    ModalityCalibration,
    calibrate_fusion,
    calibrate_modality,
    calibrated_confidence,
    calibrated_scores,
    reliability_weights,
)
from .features import PairedFeatures, load_paired_features
from .heads import PUBLIC_DIM, ProjectionHead, WhiteningFrontend, load_projection, save_projection
from .losses import supcon_cross

__all__ = [
    "PUBLIC_DIM",
    "ModalityCalibration",
    "PairedFeatures",
    "ProjectionHead",
    "WhiteningFrontend",
    "calibrate_fusion",
    "calibrate_modality",
    "calibrated_confidence",
    "calibrated_scores",
    "load_paired_features",
    "load_projection",
    "reliability_weights",
    "save_projection",
    "supcon_cross",
]
