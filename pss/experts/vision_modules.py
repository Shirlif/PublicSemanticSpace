"""Perception-level vision experts for the L2 mid-stream swap experiment.

These wrappers turn a frozen backbone plus a projection/calibration layer into
a module that satisfies :class:`pss.experts.contract.ExpertContract`.  The
public interface is ``encode(x) -> (B, PUBLIC_DIM)`` with L2-normalized rows.

Two input forms are supported:

* native feature vectors ``(B, native_dim)`` -- used by offline replay, where
  the detector has already run the backbone; and
* image batches ``(B, 3, H, W)`` (ImageNet-normalized) -- used to validate the
  full image-to-public-space path.

Keeping both forms lets the replay reuse a frozen backbone's cached features
for every control run while the contract probe still exercises the real
encoder path.

The projection is deliberately an existing component, not a new architecture:

* a frozen :class:`pss.projection.heads.ProjectionHead` for the incumbent
  expert, and
* a ridge calibration fitted into the incumbent's public space for an
  incoming expert (the same ``_ridge_map`` used by
  ``run_l1_family_calibration``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pss.experts.contract import ExpertHealth

PUBLIC_DIM = 128


def _l2(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows stay zero."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.divide(arr, norms, out=np.zeros_like(arr), where=norms > 1e-12)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


class LinearProjection:
    """A fixed linear map ``native @ W`` into the public space."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.weights.ndim != 2:
            raise ValueError("linear projection weights must be (native_dim, public_dim)")

    def __call__(self, native) -> np.ndarray:
        return np.asarray(native, dtype=np.float64) @ self.weights


class TorchProjection:
    """Adapter around a frozen :class:`ProjectionHead`."""

    def __init__(self, head, device: str = "cpu"):
        self.head = head
        self.device = device

    def __call__(self, native) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = torch.as_tensor(np.asarray(native, dtype=np.float32), device=self.device)
            return self.head(x).cpu().numpy()


@dataclass
class VisionExpertModule:
    """A frozen vision backbone + projection exposed as an expert module."""

    name: str
    version: str
    native_dim: int
    projection: Any
    encoder: Any = None
    image_size: int | None = None
    device: str = "cpu"
    dim: int = PUBLIC_DIM
    modality: str = "vision"
    _latencies_ms: list[float] = field(default_factory=list, repr=False)
    _failure_flags: list[str] = field(default_factory=list, repr=False)

    # ── input handling ─────────────────────────────────────────────────────

    def native(self, obs) -> np.ndarray:
        """Return native backbone features for any supported input form."""
        import torch

        if isinstance(obs, torch.Tensor):
            if obs.ndim == 4:
                if self.encoder is None:
                    raise ValueError(
                        f"expert {self.version!r} has no image encoder; pass native features"
                    )
                with torch.no_grad():
                    out = self.encoder.encode(obs.to(self.device))
                return np.asarray(out.detach().cpu(), dtype=np.float32)
            return np.asarray(obs.detach().cpu(), dtype=np.float32)

        if isinstance(obs, np.ndarray):
            arr = np.asarray(obs, dtype=np.float32)
            return arr[None, :] if arr.ndim == 1 else arr

        items = list(obs) if isinstance(obs, (list, tuple)) else [obs]
        native: list[np.ndarray] = []
        images: list[Any] = []
        for item in items:
            if isinstance(item, np.ndarray):
                native.append(np.asarray(item, dtype=np.float32).reshape(-1))
            elif hasattr(item, "raw"):
                raw = item.raw
                if "vision_native" in raw:
                    native.append(np.asarray(raw["vision_native"], dtype=np.float32).reshape(-1))
                elif "vision_image" in raw:
                    images.append(raw["vision_image"])
                else:
                    raise ValueError(
                        "Observation.raw needs 'vision_native' or 'vision_image'"
                    )
            else:
                native.append(np.asarray(item, dtype=np.float32).reshape(-1))
        if native and images:
            raise ValueError("cannot mix native features and images in one call")
        if images:
            return self.native(torch.stack([torch.as_tensor(x) for x in images]))
        arr = np.stack(native)
        return arr

    # ── contract surface ───────────────────────────────────────────────────

    def encode(self, obs) -> np.ndarray:
        started = time.perf_counter()
        native = self.native(obs)
        if native.ndim != 2 or native.shape[1] != int(self.native_dim):
            raise ValueError(
                f"expert {self.version!r} expected native dim {self.native_dim}, "
                f"got shape {tuple(native.shape)}"
            )
        out = _l2(np.asarray(self.projection(native), dtype=np.float64))
        if out.shape[1] != int(self.dim):
            raise ValueError(
                f"expert {self.version!r} produced dim {out.shape[1]}, expected {self.dim}"
            )
        self._latencies_ms.append((time.perf_counter() - started) * 1000.0)
        return out.astype(np.float32)

    def health(self) -> ExpertHealth:
        return ExpertHealth(
            latency_p50=_percentile(self._latencies_ms, 50),
            latency_p99=_percentile(self._latencies_ms, 99),
            failure_flags=list(self._failure_flags),
        )


@dataclass
class FixedEmbeddingExpert:
    """Deterministic linear expert used as an unrelated-module control."""

    modality: str
    version: str
    input_dim: int = 4
    dim: int = PUBLIC_DIM
    seed: int = 0
    _weights: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(int(self.seed))
        self._weights = rng.normal(size=(int(self.input_dim), int(self.dim))) / np.sqrt(
            int(self.input_dim)
        )

    def encode(self, obs) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != int(self.input_dim):
            raise ValueError(
                f"expected input dim {self.input_dim}, got shape {tuple(arr.shape)}"
            )
        return _l2(arr @ self._weights).astype(np.float32)

    def health(self) -> ExpertHealth:
        return ExpertHealth()


# ── calibration ────────────────────────────────────────────────────────────


def fit_linear_calibration(native, target, ridge: float = 1e-2) -> np.ndarray:
    """Fit ``native @ W ~= target`` with the repository's ridge helper.

    This is the same solver as ``run_l1_family_calibration``: a small,
    label-free linear map from an incoming expert's native features into the
    incumbent expert's public space.  Only pre-swap detections are used; no
    ground truth is involved.
    """
    from pss.property.l1 import _ridge_map

    X = np.asarray(native, dtype=np.float64)
    Y = np.asarray(target, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
        raise ValueError("native and target must be aligned 2-D arrays")
    if len(X) < 2:
        raise ValueError("need at least two calibration samples")
    return _ridge_map(X, Y, ridge=float(ridge))


def calibration_stats(native, target, weights) -> dict[str, float]:
    """Cosine agreement between a calibrated expert and the incumbent."""
    z = _l2(np.asarray(native, dtype=np.float64) @ np.asarray(weights, dtype=np.float64))
    t = _l2(np.asarray(target, dtype=np.float64))
    if len(z) == 0:
        return {"n": 0, "cos_mean": 0.0, "cos_median": 0.0, "cos_min": 0.0}
    cos = np.sum(z * t, axis=1)
    return {
        "n": int(len(cos)),
        "cos_mean": float(cos.mean()),
        "cos_median": float(np.median(cos)),
        "cos_min": float(cos.min()),
        "cos_p10": float(np.percentile(cos, 10)),
    }
