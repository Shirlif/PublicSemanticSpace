"""Hot-swappable expert contract and synthetic stubs (property validation)."""

from .adapter import EncoderExpert, admit_encoder, default_probe, resolve_out_dim
from .contract import ExpertContract, ExpertHealth, ExpertValidationError, validate_expert
from .quality import (
    QUALITY_LEVELS,
    NoisyEmbeddingWrapper,
    QualityLevel,
    make_quality_expert,
    manipulation_check,
)
from .stubs import (
    AdversarialExpert,
    CategoryCodebook,
    ConstantExpert,
    DelayedExpert,
    DropoutExpert,
    NaNExpert,
    NoisyOracleExpert,
    NonNormalizedExpert,
    OODExpert,
    OracleExpert,
    RandomExpert,
    WrongDimExpert,
)

__all__ = [
    "ExpertContract",
    "ExpertHealth",
    "ExpertValidationError",
    "validate_expert",
    "EncoderExpert",
    "admit_encoder",
    "default_probe",
    "resolve_out_dim",
    "QUALITY_LEVELS",
    "QualityLevel",
    "make_quality_expert",
    "manipulation_check",
    "NoisyEmbeddingWrapper",
    "CategoryCodebook",
    "OracleExpert",
    "NoisyOracleExpert",
    "RandomExpert",
    "ConstantExpert",
    "AdversarialExpert",
    "OODExpert",
    "DelayedExpert",
    "DropoutExpert",
    "WrongDimExpert",
    "NonNormalizedExpert",
    "NaNExpert",
]
