"""Single-channel evidence extractors for contact events.

Each extractor produces a per-pair score time series over the same frame grid.
The score is a *contact evidence* in [0, 1], not a calibrated probability; the
fusion layer calibrates it.  Pair-specific channels (force/geometry) use the
pair identity; tactile uses the two objects' tactile maps as a proxy.
"""

from __future__ import annotations

import numpy as np

from .model import ContactFrame

EPS = 0.02


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


class ChannelExtractor:
    name = "base"

    def pair_score(self, frames: list[ContactFrame], pair: tuple[str, str]) -> np.ndarray:
        raise NotImplementedError

    def pair_intensity(
        self, frames: list[ContactFrame], pair: tuple[str, str]
    ) -> np.ndarray:
        return np.zeros(len(frames), dtype=np.float64)

    def pair_location(
        self, frames: list[ContactFrame], pair: tuple[str, str]
    ) -> list[np.ndarray | None]:
        return [None] * len(frames)


class ForceExtractor(ChannelExtractor):
    name = "force"

    def __init__(self, f0: float = 1.0):
        self.f0 = float(f0)

    def pair_score(self, frames, pair):
        out = np.zeros(len(frames), dtype=np.float64)
        for i, frame in enumerate(frames):
            best = 0.0
            for c in frame.contacts:
                if tuple(sorted((str(c.get("instance1")), str(c.get("instance2"))))) != pair:
                    continue
                best = max(best, float(c.get("normal_force", 0.0)))
            out[i] = best / (best + self.f0) if best > 0 else 0.0
        return out

    def pair_intensity(self, frames, pair):
        out = np.zeros(len(frames), dtype=np.float64)
        for i, frame in enumerate(frames):
            best = 0.0
            for c in frame.contacts:
                if tuple(sorted((str(c.get("instance1")), str(c.get("instance2"))))) != pair:
                    continue
                best = max(best, float(c.get("normal_force", 0.0)))
            out[i] = best
        return out

    def pair_location(self, frames, pair):
        out = []
        for frame in frames:
            best = None
            for c in frame.contacts:
                if tuple(sorted((str(c.get("instance1")), str(c.get("instance2"))))) != pair:
                    continue
                if best is None or float(c.get("normal_force", 0.0)) > best[0]:
                    best = (float(c.get("normal_force", 0.0)), np.asarray(c.get("pos")))
            out.append(None if best is None else best[1])
        return out


class TactileExtractor(ChannelExtractor):
    name = "tactile"

    def __init__(self, t0: float = 1.0):
        self.t0 = float(t0)

    def pair_score(self, frames, pair):
        out = np.zeros(len(frames), dtype=np.float64)
        a, b = pair
        for i, frame in enumerate(frames):
            va = float(np.max(frame.tactile.get(a, 0.0))) if frame.tactile.get(a) is not None else 0.0
            vb = float(np.max(frame.tactile.get(b, 0.0))) if frame.tactile.get(b) is not None else 0.0
            best = max(va, vb)
            out[i] = best / (best + self.t0) if best > 0 else 0.0
        return out

    def pair_intensity(self, frames, pair):
        out = np.zeros(len(frames), dtype=np.float64)
        a, b = pair
        for i, frame in enumerate(frames):
            va = float(np.max(frame.tactile.get(a, 0.0))) if frame.tactile.get(a) is not None else 0.0
            vb = float(np.max(frame.tactile.get(b, 0.0))) if frame.tactile.get(b) is not None else 0.0
            out[i] = max(va, vb)
        return out


class GeometryExtractor(ChannelExtractor):
    name = "geometry"

    def __init__(self, eps: float = EPS, tau: float = 0.01):
        self.eps = float(eps)
        self.tau = float(tau)

    def _gap(self, frame: ContactFrame, pair: tuple[str, str]) -> float | None:
        scene = frame.scene_model
        if scene is None or pair[0] not in scene.objects or pair[1] not in scene.objects:
            return None
        a = scene.objects[pair[0]]
        b = scene.objects[pair[1]]
        dist = float(np.linalg.norm(np.asarray(a.pos) - np.asarray(b.pos)))
        return dist - float(a.radius) - float(b.radius)

    def pair_score(self, frames, pair):
        out = np.zeros(len(frames), dtype=np.float64)
        for i, frame in enumerate(frames):
            gap = self._gap(frame, pair)
            if gap is not None:
                out[i] = float(_sigmoid((self.eps - gap) / self.tau))
        return out

    def pair_intensity(self, frames, pair):
        out = np.zeros(len(frames), dtype=np.float64)
        for i, frame in enumerate(frames):
            gap = self._gap(frame, pair)
            if gap is not None:
                out[i] = float(np.clip(1.0 - gap / max(self.eps, 1e-9), 0.0, 1.0))
        return out


class VisionExtractor(ChannelExtractor):
    """Placeholder: a real vision channel needs masks/motion (C1)."""

    name = "vision"

    def pair_score(self, frames, pair):
        return np.zeros(len(frames), dtype=np.float64)


class AudioExtractor(ChannelExtractor):
    """Synthetic audio proxy from contact force; clearly not real audio."""

    name = "audio"

    def __init__(self, f0: float = 1.0):
        self._force = ForceExtractor(f0=f0)

    def pair_score(self, frames, pair):
        # Impact sound is more peak-like than force; emphasize the derivative.
        force = self._force.pair_score(frames, pair)
        if len(force) < 2:
            return force
        diff = np.abs(np.diff(force, prepend=force[:1]))
        scale = max(float(np.max(diff)), 1e-9)
        return diff / scale


CHANNELS = {
    "force": ForceExtractor,
    "tactile": TactileExtractor,
    "geometry": GeometryExtractor,
    "vision": VisionExtractor,
    "audio": AudioExtractor,
}


def make_channel(name: str, **kwargs) -> ChannelExtractor:
    if name not in CHANNELS:
        raise KeyError(f"unknown channel {name!r}; known: {sorted(CHANNELS)}")
    return CHANNELS[name](**kwargs)


__all__ = [
    "AudioExtractor",
    "CHANNELS",
    "ChannelExtractor",
    "ForceExtractor",
    "GeometryExtractor",
    "TactileExtractor",
    "VisionExtractor",
    "make_channel",
]
