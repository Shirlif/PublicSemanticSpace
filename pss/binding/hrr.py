"""Holographic Reduced Representations (HRR) as a comparison binding scheme.

Circular convolution binding:

    bind(a, b)[k] = sum_i a[i] * b[(k - i) mod D]

Circular correlation unbinding:

    unbind(c, b)[k] = sum_i c[i] * b[(i - k) mod D]

HRR is only **approximately** invertible: correlation with the same role vector
recovers the filler up to the autocorrelation sidelobes. The Phase 1 benchmark
measures the trade-off against Hadamard binding; the result is that HRR is not
adopted (no capacity gain, much worse round-trip error). It stays in the
codebase as an audited alternative.
"""

from __future__ import annotations

import numpy as np


def hrr_bind(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=a.shape[0])


def hrr_unbind(bound, b) -> np.ndarray:
    bound = np.asarray(bound, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if bound.shape != b.shape:
        raise ValueError(f"shape mismatch: {bound.shape} vs {b.shape}")
    return np.fft.irfft(
        np.fft.rfft(bound) * np.conj(np.fft.rfft(b)), n=bound.shape[0]
    )


def make_roles(d: int, k: int, seed: int = 0, kind: str = "gaussian") -> np.ndarray:
    """Return ``k`` unit-norm role vectors of dimension ``d``."""
    if k < 1 or d < 1:
        raise ValueError("k and d must be positive")
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        roles = rng.normal(size=(k, d))
    elif kind == "rademacher":
        roles = rng.choice([-1.0, 1.0], size=(k, d))
    else:
        raise ValueError(f"unknown role kind: {kind}")
    roles /= np.linalg.norm(roles, axis=1, keepdims=True)
    return roles


def naive_convolution(a, b) -> np.ndarray:
    """Reference implementation, used only in tests."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a.shape[0]
    out = np.zeros(d)
    for k in range(d):
        for i in range(d):
            out[k] += a[i] * b[(k - i) % d]
    return out


def naive_correlation(c, b) -> np.ndarray:
    """Reference implementation, used only in tests."""
    c = np.asarray(c, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = c.shape[0]
    out = np.zeros(d)
    for k in range(d):
        for i in range(d):
            out[k] += c[i] * b[(i - k) % d]
    return out
