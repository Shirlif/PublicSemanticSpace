"""Cross-modal projection heads (Phase 4)."""

from .calibration import (
    ModalityCalibration,
    calibrate_fusion,
    calibrate_modality,
    calibrated_confidence,
    calibrated_scores,
    expected_calibration_error,
    reliability_weights,
)
from .features import PairedFeatures, load_paired_features
from .heads import PUBLIC_DIM, ProjectionHead, WhiteningFrontend, load_projection, save_projection
from .losses import supcon_cross
from .online import (
    DriftThresholds,
    PoisonThresholds,
    StreamingCalibrator,
    StreamingModalityCalibration,
    audit_calibration_set,
    calibrated_test_metrics,
    select_balanced_subset,
)

__all__ = [
    "PUBLIC_DIM",
    "DriftThresholds",
    "ModalityCalibration",
    "PairedFeatures",
    "PoisonThresholds",
    "ProjectionHead",
    "StreamingCalibrator",
    "StreamingModalityCalibration",
    "WhiteningFrontend",
    "audit_calibration_set",
    "calibrate_fusion",
    "calibrate_modality",
    "calibrated_confidence",
    "calibrated_scores",
    "calibrated_test_metrics",
    "expected_calibration_error",
    "load_paired_features",
    "load_projection",
    "reliability_weights",
    "save_projection",
    "select_balanced_subset",
    "supcon_cross",
]
