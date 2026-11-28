"""Prototype bank, hierarchical tree, and recognition (Phase 5)."""

from .admission import AdmissionResult, pre_review
from .bank import PrototypeBank, build_bank
from .build import run_phase5
from .conflict import ConflictDecision, conflict_aware_recognize
from .joint_readout import JointReadout, recognize_joint
from .recognize import RecognitionResult, recognize
from .scorers import (
    CosineProtoScorer,
    KNNScorer,
    MahalanobisScorer,
    NoveltyMap,
    fit_novelty_map,
    fuse_scores,
    predict_classes,
)
from .tree import build_tree, load_tree, save_tree, tree_candidates

__all__ = [
    "AdmissionResult",
    "ConflictDecision",
    "CosineProtoScorer",
    "JointReadout",
    "KNNScorer",
    "MahalanobisScorer",
    "NoveltyMap",
    "PrototypeBank",
    "RecognitionResult",
    "build_bank",
    "build_tree",
    "conflict_aware_recognize",
    "fit_novelty_map",
    "fuse_scores",
    "load_tree",
    "pre_review",
    "predict_classes",
    "recognize",
    "recognize_joint",
    "run_phase5",
    "save_tree",
    "tree_candidates",
]
