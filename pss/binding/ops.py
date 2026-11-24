"""Structured binding operators for the public semantic space (contract v0).

Design decisions (see `contracts/README.md`):
- ``bind`` is element-wise multiplication (Hadamard product).
- ``unbind`` is element-wise division with a **sign-preserving** near-zero guard.
- Slot basis vectors are unit-norm and, by default, mutually orthogonal
  (rows of a normalized Hadamard matrix). With Hadamard/Rademacher slots every
  component has magnitude ``1/sqrt(D)``, so division never amplifies near-zero
  noise -- this is the fix for the demo's original sign-flipping guard.
- Cross-modality features are NEVER summed into one vector before storage.
  Summation is only allowed for single-modality similarity scoring (9.2.3).
"""

from __future__ import annotations

import numpy as np

SLOT_EPS = 1e-6


def _sign_preserving(slot, eps: float = SLOT_EPS) -> np.ndarray:
    """Clamp slot components away from zero without flipping their sign.

    The original demo used ``np.where(abs(slot) < eps, eps, slot)``, which turns
    every tiny negative component into a positive ``eps`` and silently flips
    the sign. Here the magnitude is clamped and the sign is kept.
    """
    slot = np.asarray(slot, dtype=np.float64)
    sign = np.where(slot < 0.0, -1.0, 1.0)
    return sign * np.maximum(np.abs(slot), eps)


def bind(feature, slot) -> np.ndarray:
    """Bind a feature vector to a slot: ``feature * slot``."""
    return np.asarray(feature, dtype=np.float64) * np.asarray(slot, dtype=np.float64)


def unbind(bound, slot, eps: float = SLOT_EPS) -> np.ndarray:
    """Unbind: ``bound / slot`` with a sign-preserving near-zero guard."""
    return np.asarray(bound, dtype=np.float64) / _sign_preserving(slot, eps)


def hadamard(n: int) -> np.ndarray:
    """Sylvester Hadamard matrix of order ``n`` (``n`` must be a power of two)."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard order must be a power of two, got {n}")
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def make_slots(scheme: str, d: int, k: int, seed: int = 0) -> np.ndarray:
    """Return ``k`` unit-norm slot basis vectors of dimension ``d``.

    Schemes:
    - ``hadamard``: rows of a normalized Hadamard matrix (orthogonal, ``|s_i|=1/sqrt(d)``)
    - ``rademacher``: random ``+/-1/sqrt(d)`` vectors (``|s_i|=1/sqrt(d)``, near-orthogonal)
    - ``gaussian``: random unit vectors (some components can be near zero)
    """
    if k < 1 or d < 1:
        raise ValueError("k and d must be positive")
    rng = np.random.default_rng(seed)
    if scheme == "hadamard":
        if d & (d - 1):
            raise ValueError("hadamard slots require a power-of-two dimension")
        if k > d:
            raise ValueError(f"k={k} exceeds dimension d={d}")
        rows = hadamard(d) / np.sqrt(d)
        # Skip the all-ones row so slots have varied sign patterns.
        idx = np.arange(1, k + 1) % d
        return rows[idx].copy()
    if scheme == "rademacher":
        return rng.choice([-1.0, 1.0], size=(k, d)) / np.sqrt(d)
    if scheme == "gaussian":
        slots = rng.normal(size=(k, d))
        slots /= np.linalg.norm(slots, axis=1, keepdims=True)
        return slots
    raise ValueError(f"unknown slot scheme: {scheme}")


def cosine(a, b) -> float:
    """Cosine similarity, safe for zero vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def slot_orthogonality(slots) -> float:
    """Max absolute off-diagonal entry of the slot Gram matrix."""
    s = np.asarray(slots, dtype=np.float64)
    g = s @ s.T
    k = g.shape[0]
    if k < 2:
        return 0.0
    off = g - np.eye(k)
    return float(np.max(np.abs(off)))


def bind_sum(features, slot) -> np.ndarray:
    """Sum several single-modality features bound to one slot.

    WARNING: this collapses the individual features (9.2.3). Use it only for
    similarity scoring within one modality, never to fuse modalities.
    """
    features = list(features)
    if not features:
        raise ValueError("bind_sum requires at least one feature")
    out = np.zeros_like(np.asarray(features[0], dtype=np.float64))
    for f in features:
        out = out + bind(f, slot)
    return out
