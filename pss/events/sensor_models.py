"""Sensor degradation models for the contact-event benchmark.

The clean MuJoCo force channel is an oracle for the GT event definition: the
GT is ``normal_force >= 1 N`` and the channel score is a monotone function of
the same force.  ``NoisyForceExtractor`` wraps ``ForceExtractor`` and applies
the degradations a real force sensor would have, while the GT remains clean:

* relative Gaussian noise (default sigma = 10% of the reading),
* a 1-2 frame delay (the sensor reports a stale value),
* quantization (default 0.1 N),
* Bernoulli frame dropout (default 5%).

The noise is seeded deterministically from ``(seed, pair)`` so a benchmark
run is reproducible and the calibration/evaluation see the same sensor.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .extractors import ChannelExtractor, ForceExtractor


def _stable_seed(seed: int, pair: tuple[str, str]) -> int:
    key = f"{int(seed)}:{pair[0]}:{pair[1]}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


class NoisyForceExtractor(ChannelExtractor):
    """Force evidence degraded to a non-oracle sensor.

    Parameters
    ----------
    seed:
        Scenario seed; also selects the delay when ``delay`` is ``None``.
    sigma:
        Relative Gaussian noise standard deviation (fraction of reading).
    delay:
        Frame delay.  ``None`` means ``1 + seed % 2`` (1 or 2 frames).
    quantization:
        Force quantization step in newtons (0 disables).
    dropout:
        Frame dropout probability (0 disables).
    """

    name = "force_noisy"

    def __init__(
        self,
        *,
        seed: int = 0,
        sigma: float = 0.10,
        delay: int | None = None,
        quantization: float = 0.1,
        dropout: float = 0.05,
    ):
        self.seed = int(seed)
        self.sigma = float(sigma)
        self.delay = int(1 + self.seed % 2) if delay is None else int(delay)
        self.quantization = float(quantization)
        self.dropout = float(dropout)
        self._clean = ForceExtractor()

    def _series(self, frames, pair) -> tuple[np.ndarray, np.ndarray]:
        """Return (noisy force, keep-mask) with delay/noise/quantization/dropout."""
        raw = np.asarray(self._clean.pair_intensity(frames, pair), dtype=np.float64)
        n = len(raw)

        # Stale reading: shift the signal to the right by ``delay`` frames.
        if self.delay > 0 and n > 0:
            shifted = np.zeros_like(raw)
            d = min(self.delay, n)
            shifted[d:] = raw[: n - d]
        else:
            shifted = raw.copy()

        rng = np.random.default_rng(_stable_seed(self.seed, pair))
        noisy = shifted * (1.0 + rng.normal(0.0, self.sigma, size=n))
        noisy = np.clip(noisy, 0.0, None)

        if self.quantization > 0:
            q = self.quantization
            noisy = np.round(noisy / q) * q

        if self.dropout > 0:
            keep = (rng.random(size=n) >= self.dropout).astype(np.float64)
        else:
            keep = np.ones(n, dtype=np.float64)
        noisy = noisy * keep
        return noisy, keep

    def pair_score(self, frames, pair):
        noisy, _ = self._series(frames, pair)
        return noisy / (noisy + 1.0)

    def pair_intensity(self, frames, pair):
        noisy, _ = self._series(frames, pair)
        return noisy

    def pair_location(self, frames, pair):
        clean = self._clean.pair_location(frames, pair)
        n = len(clean)
        if self.delay > 0 and n > 0:
            d = min(self.delay, n)
            clean = [None] * d + clean[: n - d]
        _, keep = self._series(frames, pair)
        return [loc if keep[i] > 0 else None for i, loc in enumerate(clean)]

    def describe(self) -> dict:
        return {
            "model": "noisy_force",
            "sigma": self.sigma,
            "delay_frames": self.delay,
            "quantization_n": self.quantization,
            "dropout": self.dropout,
            "seed": self.seed,
        }


__all__ = ["NoisyForceExtractor"]
