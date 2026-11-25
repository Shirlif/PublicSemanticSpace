"""Expert contract for hot-swappable sensory modules.

This is the P0 interface used by the property-level validation suite.  It is
deliberately narrower than the backbone registry: a module only has to expose
``modality``, ``version``, ``dim``, ``encode`` and ``health``.  The runtime
never looks inside; a module that satisfies the contract is replaceable.

The contract is intentionally compatible with the existing encoders'
``encode(x) -> (B, d)`` convention.  The difference is that stubs may ignore
the input, which is what lets the L0 sandbox remove perception from the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


class ExpertValidationError(ValueError):
    """Raised when a module does not satisfy the expert contract."""


@dataclass
class ExpertHealth:
    """Self-reported health of an expert module."""

    latency_p50: float = 0.0
    latency_p99: float = 0.0
    calibration_ece: float = 0.0
    ood_score: float = 0.0
    failure_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "latency_p50": float(self.latency_p50),
            "latency_p99": float(self.latency_p99),
            "calibration_ece": float(self.calibration_ece),
            "ood_score": float(self.ood_score),
            "failure_flags": list(self.failure_flags),
        }


@runtime_checkable
class ExpertContract(Protocol):
    """Minimal interface a hot-swappable expert must implement."""

    modality: str
    version: str
    dim: int

    def encode(self, obs) -> np.ndarray:
        """Return ``(B, dim)`` embeddings, L2-normalized, finite."""

    def health(self) -> ExpertHealth:
        """Return latency, calibration, OOD and failure information."""


def _as_batch(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def validate_expert(
    expert,
    public_dim: int = 128,
    tol: float = 1e-6,
    probe=None,
) -> bool:
    """Validate an expert against the contract.

    Parameters
    ----------
    expert:
        Object implementing :class:`ExpertContract`.
    public_dim:
        Required embedding dimensionality (``PUBLIC_DIM`` by default).
    tol:
        Tolerance for the unit-norm check.
    probe:
        Optional input passed to ``encode``.  When omitted only the static
        attributes and ``health()`` are checked; pass a real observation (or a
        list of observations) to check output shape, finiteness and norm.

    Returns
    -------
    bool
        ``True`` if valid.  Raises :class:`ExpertValidationError` otherwise.
    """

    for attr in ("modality", "version", "dim", "encode", "health"):
        if not hasattr(expert, attr):
            raise ExpertValidationError(f"expert missing required attribute {attr!r}")

    if not isinstance(expert.modality, str) or not expert.modality:
        raise ExpertValidationError("expert.modality must be a non-empty string")
    if not isinstance(expert.version, str) or not expert.version:
        raise ExpertValidationError("expert.version must be a non-empty string")
    if int(expert.dim) != int(public_dim):
        raise ExpertValidationError(
            f"expert.dim={expert.dim} does not match public_dim={public_dim}"
        )

    health = expert.health()
    if not isinstance(health, ExpertHealth):
        raise ExpertValidationError("expert.health() must return ExpertHealth")

    if probe is not None:
        try:
            out = expert.encode(probe)
        except Exception as exc:  # pragma: no cover - defensive
            raise ExpertValidationError(f"expert.encode() failed: {exc}") from exc
        arr = _as_batch(out)
        if arr.ndim != 2:
            raise ExpertValidationError(
                f"expert.encode() must return (B, dim), got shape {arr.shape}"
            )
        if arr.shape[1] != int(public_dim):
            raise ExpertValidationError(
                f"expert.encode() returned dim {arr.shape[1]}, expected {public_dim}"
            )
        if not np.all(np.isfinite(arr)):
            raise ExpertValidationError("expert.encode() returned NaN/Inf")
        norms = np.linalg.norm(arr, axis=1)
        if not np.allclose(norms, 1.0, atol=tol):
            raise ExpertValidationError(
                f"expert.encode() returned non-unit vectors: {norms.tolist()}"
            )

    return True
