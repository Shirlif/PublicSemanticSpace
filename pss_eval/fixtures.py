"""Synthetic labeled streams for ``pss_eval``.

The streams are generated with NumPy only.  Labels are kept next to the
observation in :class:`LabeledSample`; the runner passes only
``sample.observation`` to the system under test, so a system cannot read the
ground truth through the protocol.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np

from .protocol import Observation

__all__ = ["LabeledSample", "Stream", "make_gaussian_stream"]


@dataclass(frozen=True)
class LabeledSample:
    observation: Observation
    label: str | None


@dataclass(frozen=True)
class Stream:
    samples: tuple[LabeledSample, ...]
    categories: tuple[str, ...]
    dim: int
    seed: int
    # Harness-side ground truth for identity retention: detection reference ->
    # physical instance id.  Never placed on an Observation, so the system
    # cannot read it.  Empty for streams without identity labels.
    identity_labels: Mapping[str, str] = field(default_factory=dict)

    def __iter__(self) -> Iterator[LabeledSample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


def make_gaussian_stream(
    n: int = 32,
    dim: int = 32,
    n_classes: int = 4,
    seed: int = 0,
    noise: float = 0.35,
    modalities: tuple[str, ...] = ("vision",),
) -> Stream:
    """Balanced class-conditional Gaussian stream with unit-norm features.

    The stream is deterministic for a given ``(n, dim, n_classes, seed,
    noise)``.  It is a fixture for protocol-level smoke tests, not a capability
    benchmark: no property in this package scores classification accuracy.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    if not modalities:
        raise ValueError("modalities must be non-empty")

    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    categories = tuple(f"cat{k}" for k in range(n_classes))

    samples: list[LabeledSample] = []
    for i in range(int(n)):
        klass = i % n_classes
        vector = centers[klass] + float(noise) * rng.normal(size=dim)
        norm = float(np.linalg.norm(vector))
        vector = vector / max(norm, 1e-12)
        observation = Observation(
            observation_id=f"s{i:04d}",
            modality=modalities[i % len(modalities)],
            features=vector.astype(np.float64),
        )
        samples.append(LabeledSample(observation=observation, label=categories[klass]))

    return Stream(
        samples=tuple(samples),
        categories=categories,
        dim=int(dim),
        seed=int(seed),
    )
