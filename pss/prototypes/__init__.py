"""Prototype bank, hierarchical tree, and recognition (Phase 5)."""

from .admission import AdmissionResult, pre_review
from .bank import PrototypeBank, build_bank
from .build import run_phase5
from .conflict import ConflictDecision, conflict_aware_recognize
from .recognize import RecognitionResult, recognize
from .tree import build_tree, load_tree, save_tree, tree_candidates

__all__ = [
    "AdmissionResult",
    "ConflictDecision",
    "PrototypeBank",
    "RecognitionResult",
    "build_bank",
    "build_tree",
    "conflict_aware_recognize",
    "load_tree",
    "pre_review",
    "recognize",
    "run_phase5",
    "save_tree",
    "tree_candidates",
]
