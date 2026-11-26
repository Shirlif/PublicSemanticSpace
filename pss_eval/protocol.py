"""The public ``SystemUnderTest`` protocol for PSS property evaluation.

This module is deliberately standalone: it imports only the Python standard
library and NumPy.  It must never import ``pss`` or any adapter package, so an
external system can vendor ``pss_eval.protocol`` without pulling in the PSS
runtime, its experts, or its state types.

Semantics
---------
A system is driven as an online stream:

1. ``reset(seed=...)`` returns the system to its initial state.
2. ``ingest(observation)`` appends one observation to the stream.
3. ``step()`` returns a :class:`Prediction` for the most recently ingested
   observation.
4. ``state()`` returns a snapshot of the system's persistent state.

``state()`` values must be composed of standard Python scalars/containers and
NumPy arrays.  They are compared with :func:`state_digest`, so two runs that
produce the same logical state must produce the same digest.

Optional capabilities
---------------------
Contract-level properties need operations that a plain predictor does not have.
They are expressed as separate runtime-checkable protocols rather than being
forced into the core interface:

* :class:`SnapshotCapable` -- snapshot/rollback of persistent state;
* :class:`SwapCapable` -- install a module descriptor, rejecting invalid ones;
* :class:`AuditCapable` -- return the system's audit log;
* :class:`IdentityCapable` -- expose per-detection entity assignments;
* :class:`BudgetCapable` -- expose an anytime deadline budget and its stats.

A system that does not implement a capability is reported as *skipped* (not
applicable) by the corresponding property, never as an implicit pass.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "AuditCapable",
    "BudgetCapable",
    "IdentityCapable",
    "Observation",
    "Prediction",
    "SnapshotCapable",
    "StateSnapshot",
    "SwapCapable",
    "SystemUnderTest",
    "capabilities_of",
    "iter_arrays",
    "prediction_digest",
    "predictions_equal",
    "state_digest",
    "state_size_bytes",
]


# ── value objects ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Observation:
    """One stream item.  Carries no label; labels live in the eval harness.

    ``metadata`` is an opaque, adapter-specific side channel (for example a
    real bundle frame's per-detection poses and embeddings).  The protocol
    never inspects it, and it must never carry ground-truth labels: those stay
    in :class:`pss_eval.fixtures.LabeledSample` / ``Stream.identity_labels`` so
    the system cannot read them through the protocol.
    """

    observation_id: str
    modality: str
    features: np.ndarray
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        arr = np.array(self.features, dtype=np.float64, copy=True)
        if arr.ndim != 1:
            raise ValueError(f"features must be 1-D, got shape {arr.shape}")
        object.__setattr__(self, "features", arr)
        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class Prediction:
    """A system's output for the most recently ingested observation.

    ``label`` is optional: unsupervised state models may only emit an
    ``embedding``.  ``scores`` is an optional label -> score mapping.
    """

    observation_id: str
    embedding: np.ndarray
    label: str | None = None
    scores: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        arr = np.array(self.embedding, dtype=np.float64, copy=True)
        if arr.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {arr.shape}")
        object.__setattr__(self, "embedding", arr)
        if self.scores is not None:
            object.__setattr__(
                self,
                "scores",
                {str(k): float(v) for k, v in self.scores.items()},
            )


# A state snapshot is a JSON-like mapping whose leaves are standard scalars,
# containers, or NumPy arrays.  The alias documents the contract; it is not
# enforced at runtime beyond ``state_digest`` / ``state_size_bytes``.
StateSnapshot = dict[str, Any]


# ── core protocol ──────────────────────────────────────────────────────────


class SystemUnderTest(Protocol):
    """Minimal interface every evaluated system must implement."""

    name: str

    def reset(self, *, seed: int = 0) -> None:
        """Return to the initial state for a deterministic run."""

    def ingest(self, observation: Observation) -> None:
        """Append one observation to the stream."""

    def step(self) -> Prediction:
        """Predict for the most recently ingested observation."""

    def state(self) -> StateSnapshot:
        """Return a snapshot of persistent state (stdlib + NumPy only)."""


# ── optional capability protocols ──────────────────────────────────────────


@runtime_checkable
class SnapshotCapable(Protocol):
    """State snapshot / rollback used by exactness properties."""

    def snapshot(self) -> str:
        """Return an opaque token identifying the current state."""

    def rollback(self, token: str) -> bool:
        """Restore the state identified by ``token``; return success."""


@runtime_checkable
class SwapCapable(Protocol):
    """Module installation used by swap / admission properties."""

    def swap(self, module: Mapping[str, Any]) -> bool:
        """Install a module descriptor; return ``True`` iff it was accepted.

        Descriptors use plain values, for example::

            {"modality": "vision", "kind": "quality", "quality": "Q2",
             "valid": True, "version": "probe-v1"}

        A conforming system validates the descriptor before mutating state.
        """


@runtime_checkable
class AuditCapable(Protocol):
    """Audit log used by justification / traceability properties."""

    def audit(self) -> list[dict[str, Any]]:
        """Return the audit entries recorded so far."""


@runtime_checkable
class IdentityCapable(Protocol):
    """Per-detection entity assignments used by identity-retention properties."""

    def identity_assignments(self) -> Mapping[str, str]:
        """Map a detection reference to the persistent entity/track id it got.

        The harness supplies the ground-truth grouping separately (via
        ``Stream.identity_labels``); the system only reports its own assignment.
        """


@runtime_checkable
class BudgetCapable(Protocol):
    """Anytime deadline budget and the statistics of its last run."""

    def budget(self) -> Mapping[str, Any]:
        """Return ``{"budget_ms": float, "reserve_ms": float}`` or an empty map."""

    def budget_stats(self) -> Mapping[str, Any]:
        """Return the deadline guard's observed statistics (p50/p99, timeouts)."""


_CAPABILITIES: dict[str, type] = {
    "snapshot": SnapshotCapable,
    "swap": SwapCapable,
    "audit": AuditCapable,
    "identity": IdentityCapable,
    "budget": BudgetCapable,
}


def capabilities_of(system: object) -> frozenset[str]:
    """Return the optional capability names implemented by ``system``."""

    return frozenset(
        name for name, proto in _CAPABILITIES.items() if isinstance(system, proto)
    )


# ── canonicalisation / digests ─────────────────────────────────────────────


def _canonical(value: Any) -> Any:
    """Convert a state/prediction value to a JSON-safe canonical structure."""

    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        if arr.dtype.kind in "fc" and not np.all(np.isfinite(arr)):
            raise ValueError("state contains non-finite array values")
        return {
            "__ndarray__": True,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_canonical(item) for item in sorted(value, key=repr)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Unknown objects are represented by their repr; adapters should avoid
    # them, but a digest is still better than a crash.
    return repr(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def state_digest(state: StateSnapshot) -> str:
    """Stable SHA-256 over a state snapshot.

    Raises ``ValueError`` if the state contains NaN/Infinity, which makes
    non-finite state a loud failure rather than a silent digest difference.
    """

    payload = _canonical_json(state).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_size_bytes(state: StateSnapshot) -> int:
    """Size of the canonical JSON encoding of ``state`` in bytes."""

    return len(_canonical_json(state).encode("utf-8"))


def prediction_digest(prediction: Prediction) -> str:
    """Stable SHA-256 over a prediction (id, embedding, label, scores)."""

    payload = {
        "observation_id": prediction.observation_id,
        "embedding": prediction.embedding,
        "label": prediction.label,
        "scores": prediction.scores,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def predictions_equal(left: Prediction, right: Prediction) -> bool:
    """Exact equality on ids, labels, scores, and embedding bytes."""

    if left.observation_id != right.observation_id or left.label != right.label:
        return False
    if (left.scores is None) != (right.scores is None):
        return False
    if left.scores is not None and left.scores != right.scores:
        return False
    if left.embedding.shape != right.embedding.shape:
        return False
    return bool(np.array_equal(left.embedding, right.embedding))


def iter_arrays(value: Any) -> Iterator[np.ndarray]:
    """Yield every NumPy array nested inside ``value``."""

    if isinstance(value, np.ndarray):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from iter_arrays(item)
        return
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from iter_arrays(item)
