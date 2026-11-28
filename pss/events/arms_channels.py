"""C1 single-channel arms: tuned force/tactile, vision, synthetic audio.

These arms keep the C0 channel interface: each returns ``list[ContactEvent]``
via ``run_channel`` with a channel extractor.  The tuned arms only change the
per-arm threshold/extractor parameters, so their events remain comparable to
the untuned ``force`` / ``tactile`` / ``geometry`` channels.

Important boundaries (C1, simulation only)
------------------------------------------
- ``force_tuned`` / ``tactile_tuned`` are near-oracle in MuJoCo; real sensor
  noise/bandwidth is not modelled.
- ``tactile_tuned`` uses the *mutual* tactile peak (min over the pair): both
  endpoints must register contact.  This removes false alarms where only one
  object touches something else.  The tactile maps are normalized per object,
  so magnitude information is already lost upstream; ``intensity`` is binary.
- ``vision_motion`` is a rendered RGB-D/segmentation proxy, not a real camera
  pipeline; its intensity is a kinematic proxy (m/s), not force.
- ``audio_proxy`` is synthesized from contact force.  It is **not** real audio
  and its intensity reuses the force peak, so its intensity error is
  optimistic and circular.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .bench_api import register_arm, run_channel
from .extractors import ChannelExtractor, ForceExtractor
from .extractors_vision import VisionMotionExtractor

# Tuned operating points.  Selected from a sweep over the C1 benchmark
# (drop/collide/near_miss, seeds 0-2); see docs/experiments/
# PSS-ContactEvent-C1-Channels.md for the plateau and caveats.
FORCE_F0 = 1.0
FORCE_THRESHOLD = 0.5
TACTILE_T0 = 0.5
TACTILE_THRESHOLD = 0.5
AUDIO_D0 = 0.5
AUDIO_DECAY = 0.9
AUDIO_THRESHOLD = 0.5
VISION_EPS = 0.05
VISION_TAU = 0.008
VISION_THRESHOLD = 0.5


class MutualTactileExtractor(ChannelExtractor):
    """Tactile evidence that requires both endpoints of the pair.

    The per-object tactile maps are normalized to a peak of one whenever any
    non-table contact exists, so a single normalizer loses magnitude.  Using
    ``min(peak_a, peak_b)`` at least rejects one-sided contacts (an object
    resting on something that is not its pair), which the plain ``max``
    aggregation cannot.
    """

    name = "tactile_tuned"

    def __init__(self, t0: float = 1.0):
        self.t0 = float(t0)

    @staticmethod
    def _peaks(frame, pair: tuple[str, str]) -> tuple[float, float]:
        a, b = pair
        va = (
            float(np.max(frame.tactile[a]))
            if frame.tactile.get(a) is not None
            else 0.0
        )
        vb = (
            float(np.max(frame.tactile[b]))
            if frame.tactile.get(b) is not None
            else 0.0
        )
        return va, vb

    def pair_score(self, frames, pair):
        out = np.zeros(len(frames), dtype=np.float64)
        for i, frame in enumerate(frames):
            va, vb = self._peaks(frame, pair)
            best = min(va, vb)
            out[i] = best / (best + self.t0) if best > 0.0 else 0.0
        return out

    def pair_intensity(self, frames, pair):
        out = np.zeros(len(frames), dtype=np.float64)
        for i, frame in enumerate(frames):
            va, vb = self._peaks(frame, pair)
            out[i] = min(va, vb)
        return out


class AudioProxyExtractor(ChannelExtractor):
    """Synthetic impact-sound proxy derived from contact force.

    A real impact sound is a short transient.  We model it as the positive
    force derivative (the impulse) followed by an exponential ring-down, so a
    one-frame impact still spans the segmenter's minimum duration.  This is an
    interface proxy for an audio channel; it is not real audio.

    ``intensity`` returns the force magnitude in the window (the same signal
    used to synthesize the proxy), so the reported intensity MAE is
    circularly optimistic.
    """

    name = "audio_proxy"

    def __init__(self, d0: float = 0.5, decay: float = 0.9, f0: float = 1.0):
        self.d0 = float(d0)
        self.decay = float(decay)
        self._force = ForceExtractor(f0=f0)

    def _series(self, frames, pair) -> tuple[np.ndarray, np.ndarray]:
        force = self._force.pair_intensity(frames, pair)
        impulse = np.maximum(np.diff(force, prepend=force[:1]), 0.0)
        raw = impulse / (impulse + self.d0) if self.d0 > 0.0 else (impulse > 0.0)
        envelope = np.zeros(len(raw), dtype=np.float64)
        prev = 0.0
        for i, value in enumerate(raw):
            prev = max(float(value), self.decay * prev)
            envelope[i] = prev
        return envelope, force

    def pair_score(self, frames, pair):
        return self._series(frames, pair)[0]

    def pair_intensity(self, frames, pair):
        return self._series(frames, pair)[1]


def _run(scenario, config, extractor: ChannelExtractor, threshold: float):
    cfg = replace(config, threshold=float(threshold))
    return run_channel(scenario, cfg, channel=extractor)


@register_arm("force_tuned")
def force_tuned(scenario, config):
    """Force channel at the GT-consistent operating point.

    The C1 sweep found a broad F1=1.0 plateau; ``f0=1 N`` with threshold 0.5
    places the score crossing exactly at the GT force threshold (1 N), giving
    zero onset latency.  There is no headroom to improve on this benchmark.
    """
    return _run(
        scenario,
        config,
        ForceExtractor(f0=FORCE_F0),
        FORCE_THRESHOLD,
    )


@register_arm("tactile_tuned")
def tactile_tuned(scenario, config):
    """Mutual (two-endpoint) tactile aggregation with a tuned score knee."""
    return _run(
        scenario,
        config,
        MutualTactileExtractor(t0=TACTILE_T0),
        TACTILE_THRESHOLD,
    )


@register_arm("vision_motion")
def vision_motion(scenario, config):
    """Rendered RGB-D/segmentation channel; requires ``--render``."""
    return _run(
        scenario,
        config,
        VisionMotionExtractor(eps=VISION_EPS, tau=VISION_TAU),
        VISION_THRESHOLD,
    )


@register_arm("audio_proxy")
def audio_proxy(scenario, config):
    """Synthetic impact transient derived from force (not real audio)."""
    return _run(
        scenario,
        config,
        AudioProxyExtractor(d0=AUDIO_D0, decay=AUDIO_DECAY),
        AUDIO_THRESHOLD,
    )


__all__ = [
    "AudioProxyExtractor",
    "MutualTactileExtractor",
    "audio_proxy",
    "force_tuned",
    "tactile_tuned",
    "vision_motion",
]
