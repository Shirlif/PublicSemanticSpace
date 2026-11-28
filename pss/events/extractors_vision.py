"""Vision channel for contact events: rendered RGB-D + segmentation.

This is a **simulation proxy**, not a calibrated real-camera pipeline.  It uses
only the observations stored on a ``ContactFrame`` when ``--render`` is set:

- ``segmentation[..., 0]``: MuJoCo geom ids (instance segmentation);
- ``depth``: per-pixel depth in metres;
- ``camera``: pinhole intrinsics/extrinsics (world -> camera ``R, t``).

Two simple cues are combined:

1. **Mask proximity**: back-project each object's mask pixels to 3D world
   points and take the minimum point-to-point distance between the two point
   clouds.  Objects that touch have a small cloud distance even when their 2D
   silhouettes overlap or are occluded.
2. **Motion discontinuity**: pair-consistent centroid speed change
   ``min(|dv_a|, |dv_b|)`` between consecutive frames, gated by proximity and
   smoothed with a short decay envelope so a one-frame impact survives the
   segmenter's minimum duration.

The per-instance label is associated once per scenario by matching the
depth-derived 3D centroid of each segmentation label to the known object
positions (``frame.poses``).  Positions are used **only** for identity, never
for the contact score: the score itself is computed from rendered depth and
segmentation.

``intensity`` is a kinematic proxy (metres/second), not a force: a camera
cannot measure normal force.  Its intensity MAE against force is therefore not
a meaningful force error and is reported only for completeness.
"""

from __future__ import annotations

import numpy as np

from .extractors import ChannelExtractor
from .model import ContactFrame


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


class VisionMotionExtractor(ChannelExtractor):
    """Position/motion evidence from rendered RGB-D + instance segmentation."""

    name = "vision_motion"

    def __init__(
        self,
        *,
        eps: float = 0.05,
        tau: float = 0.008,
        max_points: int = 128,
        dv_min: float = 0.8,
        dv_scale: float = 2.0,
        decay: float = 0.85,
        proximity: float = 0.12,
        assoc_tol: float = 0.15,
    ):
        self.eps = float(eps)
        self.tau = float(tau)
        self.max_points = int(max_points)
        self.dv_min = float(dv_min)
        self.dv_scale = float(dv_scale)
        self.decay = float(decay)
        self.proximity = float(proximity)
        self.assoc_tol = float(assoc_tol)
        self._maps: dict[int, dict[int, str]] = {}
        self._series: dict[tuple, tuple] = {}

    # -- observation helpers -------------------------------------------------

    @staticmethod
    def _seg(frame: ContactFrame) -> np.ndarray | None:
        if frame.segmentation is None:
            return None
        seg = np.asarray(frame.segmentation)
        return seg[..., 0] if seg.ndim == 3 else seg

    def _cloud(
        self, frame: ContactFrame, label: int, *, subsample: bool = True
    ) -> np.ndarray | None:
        """World points for one segmentation label (None if not visible)."""
        seg = self._seg(frame)
        if (
            seg is None
            or frame.depth is None
            or frame.camera is None
        ):
            return None
        vv, uu = np.where(seg == int(label))
        if len(vv) == 0:
            return None
        depth = np.asarray(frame.depth, dtype=np.float64)[vv, uu]
        ok = np.isfinite(depth) & (depth > 0.0) & (depth < 10.0)
        vv, uu, depth = vv[ok], uu[ok], depth[ok]
        if len(vv) == 0:
            return None
        if subsample and len(vv) > self.max_points:
            idx = np.linspace(0, len(vv) - 1, self.max_points).astype(int)
            vv, uu, depth = vv[idx], uu[idx], depth[idx]
        rays = np.stack([uu, vv, np.ones_like(uu)], axis=1)
        points_cam = (np.linalg.solve(frame.camera.K, rays.T) * depth[None, :]).T
        # world = R^T (cam - t); (row @ R) == R^T @ row
        return (points_cam - np.asarray(frame.camera.t)) @ np.asarray(frame.camera.R)

    def _label_mapping(self, frames: list[ContactFrame]) -> dict[int, str]:
        key = id(frames)
        cached = self._maps.get(key)
        if cached is not None:
            return cached
        ids = sorted(frames[0].poses)
        mapping: dict[int, str] = {}
        used: set[str] = set()
        for frame in frames:
            seg = self._seg(frame)
            if seg is None:
                continue
            centroids: dict[int, np.ndarray] = {}
            for label in np.unique(seg):
                label = int(label)
                if label < 0:
                    continue
                cloud = self._cloud(frame, label, subsample=False)
                if cloud is not None:
                    centroids[label] = cloud.mean(axis=0)
            candidates = sorted(
                (
                    float(np.linalg.norm(cloud - np.asarray(frame.poses[oid]))),
                    label,
                    oid,
                )
                for label, cloud in centroids.items()
                for oid in ids
            )
            for dist, label, oid in candidates:
                if dist > self.assoc_tol or label in mapping or oid in used:
                    continue
                mapping[label] = oid
                used.add(oid)
            if len(used) == len(ids):
                break
        self._maps[key] = mapping
        return mapping

    # -- per-pair series -----------------------------------------------------

    def _compute(
        self, frames: list[ContactFrame], pair: tuple[str, str]
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray | None]]:
        key = (id(frames), pair)
        cached = self._series.get(key)
        if cached is not None:
            return cached

        n = len(frames)
        times = np.asarray([float(f.t) for f in frames], dtype=np.float64)
        dt = float(times[1] - times[0]) if n > 1 else 0.004
        mapping = self._label_mapping(frames)
        label_of = {oid: label for label, oid in mapping.items()}
        label_a, label_b = label_of.get(pair[0]), label_of.get(pair[1])

        if label_a is None or label_b is None:
            empty = np.zeros(n, dtype=np.float64)
            self._series[key] = (empty, empty, [None] * n)
            return self._series[key]

        gaps = np.full(n, np.nan, dtype=np.float64)
        centroids: dict[str, list[np.ndarray | None]] = {pair[0]: [], pair[1]: []}
        locations: list[np.ndarray | None] = []
        for frame in frames:
            cloud_a = self._cloud(frame, label_a)
            cloud_b = self._cloud(frame, label_b)
            centroids[pair[0]].append(None if cloud_a is None else cloud_a.mean(0))
            centroids[pair[1]].append(None if cloud_b is None else cloud_b.mean(0))
            if cloud_a is None or cloud_b is None:
                locations.append(None)
                continue
            dist = np.sqrt(
                ((cloud_a[:, None, :] - cloud_b[None, :, :]) ** 2).sum(-1)
            )
            index = np.unravel_index(int(np.argmin(dist)), dist.shape)
            gaps[len(locations)] = float(dist[index])
            locations.append((cloud_a[index[0]] + cloud_b[index[1]]) / 2.0)

        def speed_change(series: list[np.ndarray | None]) -> np.ndarray:
            out = np.zeros(n, dtype=np.float64)
            prev = None
            prev_v = np.zeros(3)
            for i, centroid in enumerate(series):
                if centroid is None:
                    continue
                if prev is not None:
                    velocity = (centroid - prev) / dt
                    if i >= 2:
                        out[i] = float(np.linalg.norm(velocity - prev_v))
                    prev_v = velocity
                prev = centroid
            return out

        dv_a = speed_change(centroids[pair[0]])
        dv_b = speed_change(centroids[pair[1]])
        # Require both endpoints to move: a single object's fall/roll is not a
        # pairwise contact.  This is what keeps ball-off-table impacts out.
        dv_pair = np.minimum(dv_a, dv_b)

        finite = np.isfinite(gaps)
        gap_score = np.where(
            finite, _sigmoid((self.eps - gaps) / self.tau), 0.0
        )
        gate = np.where(
            finite, np.clip((self.proximity - gaps) / self.proximity, 0.0, 1.0), 0.0
        )
        motion = np.clip((dv_pair - self.dv_min) / self.dv_scale, 0.0, 1.0) * gate
        envelope = np.zeros(n, dtype=np.float64)
        prev = 0.0
        for i, value in enumerate(motion):
            prev = max(float(value), self.decay * prev)
            envelope[i] = prev

        # ``bench_api._event_from_interval`` takes the intensity at the frame
        # with the maximum score.  Persistent proximity saturates the score at
        # many frames, so add a small kinematic tie-break: the peak frame is
        # then the impact frame and carries the impact-speed proxy, not an
        # arbitrary resting frame.
        motion_evidence = np.clip(dv_pair / self.dv_scale, 0.0, 1.0)
        score = 0.98 * np.maximum(gap_score, envelope) + 0.02 * motion_evidence
        intensity = dv_pair
        self._series[key] = (score, intensity, locations)
        return self._series[key]

    # -- ChannelExtractor API ------------------------------------------------

    def pair_score(self, frames, pair):
        return self._compute(frames, pair)[0]

    def pair_intensity(self, frames, pair):
        return self._compute(frames, pair)[1]

    def pair_location(self, frames, pair):
        return self._compute(frames, pair)[2]


__all__ = ["VisionMotionExtractor"]
