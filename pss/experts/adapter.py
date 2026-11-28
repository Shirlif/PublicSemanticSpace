"""Production admission: raw registry encoders -> expert contract.

The property suite validates full :class:`~pss.experts.contract.ExpertContract`
objects (stubs that emit 128-d public-space vectors).  The registry builds raw
feature extractors at their **native** width (SmallCNN 256, DINOv2 384, CLIP
768, tactile MLP 64, ...), which a projection head later maps into the public
space.  This module bridges the two without changing encoder numerics:

* :func:`resolve_out_dim` reads the declared native width;
* :class:`EncoderExpert` is a thin contract adapter around a raw encoder;
* :func:`admit_encoder` runs the same validation the property suite runs and
  rejects a module that violates it.

The adapter does **not** normalize by default.  Raw encoders keep their own
output convention (DINOv2/CLIP already L2-normalize; SmallCNN/tactile do not)
and the projection head owns public-space normalization.  Set
``normalize=True`` only when the adapter itself is the thing being admitted
into the public space.

Admission checks two layers:

1. static — ``encode`` is callable and the declared width is resolvable and
   positive;
2. dynamic — when ``probe`` is supplied, ``encode(probe)`` returns a finite
   ``(B, dim)`` array, and (only if ``normalize=True``) unit-norm rows.

No probe means static-only.  The registry therefore never silently accepts a
broken module: a caller that wants the dynamic check passes ``probe=...``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .contract import ExpertHealth, ExpertValidationError, validate_expert

_DIM_ATTRS = ("out_dim", "dim", "embed_dim")


def resolve_out_dim(encoder: Any) -> int:
    """Return the encoder's declared native output width.

    Handles both flat objects (``encoder.out_dim``) and the registry's expert
    dataclasses, which wrap the network (``encoder.encoder.out_dim``).
    """
    for target in (encoder, getattr(encoder, "encoder", None)):
        if target is None:
            continue
        for attr in _DIM_ATTRS:
            value = getattr(target, attr, None)
            if value is not None:
                dim = int(value)
                if dim <= 0:
                    raise ExpertValidationError(
                        f"{type(encoder).__name__}.{attr} must be positive, got {dim}"
                    )
                return dim
    raise ExpertValidationError(
        f"encoder {type(encoder).__name__} declares no output width "
        f"(looked for {', '.join(_DIM_ATTRS)} on the encoder and its .encoder)"
    )


def _as_numpy(out) -> np.ndarray:
    if hasattr(out, "detach"):
        out = out.detach()
    if hasattr(out, "cpu"):
        out = out.cpu()
    return np.asarray(out, dtype=np.float64)


def default_probe(encoder, *, modality: str):
    """Build a zero-input probe matched to the encoder's declared geometry.

    Returns ``None`` when torch is unavailable or the modality is unknown, in
    which case admission falls back to the static checks.  The probe is small
    (batch 2) and lives on CPU; encoders move it to their own device.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch is an optional dependency
        return None

    if modality == "vision":
        size = int(getattr(encoder, "image_size", 224))
        return torch.zeros(2, 3, size, size)

    if modality == "tactile":
        input_dim = getattr(encoder, "input_dim", None)
        if input_dim:
            return torch.zeros(2, int(input_dim))
        shape = tuple(int(v) for v in getattr(encoder, "out_shape", (8, 8)))
        return torch.zeros(2, *shape)

    return None


class EncoderExpert:
    """``ExpertContract`` adapter around a raw registry encoder.

    Delegates unknown attributes to the wrapped encoder, so code that expects
    ``classes``, ``image_size``, ``input_dim`` or ``eval()`` keeps working.
    """

    def __init__(
        self,
        encoder,
        *,
        modality: str,
        version: str,
        dim: int | None = None,
        normalize: bool = False,
    ) -> None:
        if not isinstance(modality, str) or not modality:
            raise ExpertValidationError("modality must be a non-empty string")
        if not isinstance(version, str) or not version:
            raise ExpertValidationError("version must be a non-empty string")
        if not callable(getattr(encoder, "encode", None)):
            raise ExpertValidationError(
                f"encoder {type(encoder).__name__} has no callable encode()"
            )

        resolved = resolve_out_dim(encoder)
        if dim is not None and int(dim) != resolved:
            raise ExpertValidationError(
                f"declared dim={dim} does not match encoder width={resolved}"
            )

        self.encoder = encoder
        self.modality = modality
        self.version = version
        self.dim = resolved
        self.normalize = bool(normalize)
        self._latencies: list[float] = []
        self._last_error = ""

    def __getattr__(self, name: str):
        # Only reached for attributes not set on the adapter itself.
        if name == "encoder":
            raise AttributeError(name)
        return getattr(self.encoder, name)

    def encode(self, obs):
        start = time.perf_counter()
        try:
            out = self.encoder.encode(obs)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise
        self._latencies.append((time.perf_counter() - start) * 1000.0)
        if not self.normalize:
            return out
        arr = _as_numpy(out)
        if arr.ndim == 1:
            arr = arr[None, :]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return (arr / np.clip(norms, 1e-12, None)).astype(np.float32)

    def health(self) -> ExpertHealth:
        if self._latencies:
            p50 = float(np.percentile(self._latencies, 50))
            p99 = float(np.percentile(self._latencies, 99))
        else:
            p50 = p99 = 0.0
        return ExpertHealth(
            latency_p50=p50,
            latency_p99=p99,
            failure_flags=[] if not self._last_error else [self._last_error],
        )


def admit_encoder(
    encoder,
    *,
    modality: str,
    version: str,
    probe=None,
    expected_dim: int | None = None,
    normalize: bool = False,
    auto_probe: bool = True,
) -> EncoderExpert:
    """Validate a raw encoder and return its contract adapter.

    Raises :class:`ExpertValidationError` when the module violates the
    contract.  When ``probe`` is omitted and ``auto_probe`` is true, a
    zero-input probe matched to the encoder geometry is built automatically,
    so production paths get the dynamic (shape/finiteness) check without
    every caller constructing one.
    """
    expert = EncoderExpert(
        encoder,
        modality=modality,
        version=version,
        dim=expected_dim,
        normalize=normalize,
    )
    if probe is None and auto_probe:
        probe = default_probe(expert, modality=modality)
    validate_expert(
        expert,
        public_dim=expert.dim,
        probe=probe,
        require_unit_norm=expert.normalize,
    )
    return expert
