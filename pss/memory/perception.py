"""Per-frame observations from simulated data (Phase 6).

This module turns an episode/occlusion frame into tracker observations:
- segmentation-based object crops -> frozen vision expert -> projection;
- contact tactile map -> tactile expert -> projection;
- depth + segmentation -> handcrafted geometry -> projection;
- GT pose + simulated SLAM noise.

The GT instance id is carried only for evaluation; the tracker never uses it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from pss.contracts.schema import SPACE_VERSION
from pss.encoders.data import crop_object, normalize_image
from pss.encoders.geometry import geometry_features
from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision import load_vision_expert
from pss.experts.adapter import admit_encoder
from pss.projection.heads import load_projection
from pss.prototypes.bank import PrototypeBank
from pss.prototypes.recognize import recognize


@dataclass
class ModalityObservation:
    """One modality's vector plus the timing metadata needed for fusion.

    ``t_capture`` and ``t_arrival`` are in seconds on the same ``clock``.  A
    stream is stale when ``|t_capture - t_ref|`` exceeds the modality's
    ``max_latency_ms`` budget, or when the producer marked it invalid.
    """

    modality: str
    vector: np.ndarray
    t_capture: float
    t_arrival: float
    clock: str = "monotonic"
    sensor_id: str = ""
    valid: bool = True
    stale: bool = False
    quality: float = 1.0

    def validate(self) -> None:
        if not self.modality:
            raise ValueError("modality must be non-empty")
        if self.t_arrival < self.t_capture:
            raise ValueError(
                f"modality {self.modality!r}: t_arrival < t_capture "
                f"({self.t_arrival} < {self.t_capture}); check the clock domain"
            )
        if not 0.0 <= float(self.quality) <= 1.0:
            raise ValueError(f"quality must be in [0,1], got {self.quality}")


@dataclass
class Observation:
    gt_id: str
    pos: np.ndarray
    quat: np.ndarray
    appearance: np.ndarray
    raw: dict[str, np.ndarray]
    confidence: float
    category: str | None
    frame: int = 0
    gt_category: str | None = None
    category_score: float = 0.0
    unknown: bool = False
    conflict: bool = False
    modality_reliability: dict[str, float] = field(default_factory=dict)
    # Temporal contract (W5).  Defaults keep every existing constructor valid.
    t_ref: float = 0.0
    streams: dict[str, ModalityObservation] = field(default_factory=dict)
    time_skew_ms: dict[str, float] = field(default_factory=dict)
    stale_modalities: list[str] = field(default_factory=list)

    def add_stream(self, stream: ModalityObservation, t_ref: float | None = None) -> None:
        """Register a timed modality stream and recompute staleness."""
        stream.validate()
        if t_ref is not None:
            self.t_ref = float(t_ref)
        self.streams[stream.modality] = stream
        self.time_skew_ms[stream.modality] = round(
            (float(stream.t_capture) - float(self.t_ref)) * 1000.0, 6
        )
        self.refresh_staleness()

    def stamp_streams(
        self,
        timings: dict[str, tuple[float, float]] | None = None,
        *,
        t_ref: float | None = None,
        max_latency_ms: dict[str, float] | float | None = None,
    ) -> list[str]:
        """Attach producer-supplied capture/arrival times and refresh staleness.

        ``timings`` maps ``modality -> (t_capture, t_arrival)`` in seconds,
        measured on the same clock.  Modalities present in ``raw`` but absent
        from ``timings`` are treated as synchronous with ``t_ref`` (which
        defaults to the current ``t_ref``).  Existing streams are updated in
        place; missing ones are created from ``raw``.

        Real drivers should call this after filling ``raw`` and before handing
        the observation to the tracker or runtime.
        """
        if t_ref is not None:
            self.t_ref = float(t_ref)
        timings = timings or {}
        for modality, vector in self.raw.items():
            capture, arrival = timings.get(modality, (self.t_ref, self.t_ref))
            stream = self.streams.get(modality)
            if stream is None:
                stream = ModalityObservation(
                    modality=modality,
                    vector=np.asarray(vector, dtype=np.float32),
                    t_capture=float(capture),
                    t_arrival=float(arrival),
                )
                self.streams[modality] = stream
            else:
                stream.vector = np.asarray(vector, dtype=np.float32)
                stream.t_capture = float(capture)
                stream.t_arrival = float(arrival)
            stream.validate()
            self.time_skew_ms[modality] = round(
                (float(stream.t_capture) - float(self.t_ref)) * 1000.0, 6
            )
        self.refresh_staleness(max_latency_ms)
        return list(self.stale_modalities)

    def refresh_staleness(
        self, max_latency_ms: dict[str, float] | float | None = None
    ) -> list[str]:
        """Mark streams stale against the registry budget; return stale names."""
        from pss.contracts.modality import ModalityContractError, get_spec

        stale: list[str] = []
        for modality, stream in self.streams.items():
            if isinstance(max_latency_ms, dict):
                limit = max_latency_ms.get(modality)
            else:
                limit = max_latency_ms
            if limit is None:
                try:
                    limit = get_spec(modality).max_latency_ms
                except ModalityContractError:
                    limit = 50.0
            skew = abs(float(stream.t_capture) - float(self.t_ref)) * 1000.0
            stream.stale = bool(not stream.valid or skew > float(limit))
            if stream.stale:
                stale.append(modality)
        self.stale_modalities = sorted(stale)
        return list(self.stale_modalities)

    def valid_raw(self) -> dict[str, np.ndarray]:
        """Vectors safe to fuse: non-stale streams, else the legacy ``raw``."""
        if self.streams:
            usable = {
                modality: np.asarray(stream.vector)
                for modality, stream in self.streams.items()
                if stream.valid and not stream.stale
            }
            return usable
        return dict(self.raw)

    def temporal_summary(self) -> dict:
        return {
            "t_ref": float(self.t_ref),
            "time_skew_ms": dict(self.time_skew_ms),
            "stale_modalities": list(self.stale_modalities),
            "streams": {
                modality: {
                    "t_capture": float(stream.t_capture),
                    "t_arrival": float(stream.t_arrival),
                    "clock": stream.clock,
                    "sensor_id": stream.sensor_id,
                    "valid": bool(stream.valid),
                    "stale": bool(stream.stale),
                    "quality": float(stream.quality),
                }
                for modality, stream in self.streams.items()
            },
        }


class PoseNoise:
    """Simulated SLAM noise: per-instance random-walk drift + white noise."""

    def __init__(
        self,
        seed: int = 0,
        enabled: bool = True,
        sigma_pos: float = 0.01,
        sigma_yaw: float = 0.02,
        drift: float = 0.002,
    ):
        self.rng = np.random.default_rng(seed)
        self.enabled = enabled
        self.sigma_pos = float(sigma_pos)
        self.sigma_yaw = float(sigma_yaw)
        self.drift = float(drift)
        self._drift: dict[str, np.ndarray] = {}

    def apply(self, gt_id: str, pos, quat, t: int = 0):
        pos = np.asarray(pos, dtype=float).copy()
        quat = np.asarray(quat, dtype=float).copy()
        if not self.enabled:
            return pos, quat
        d = self._drift.setdefault(gt_id, np.zeros(3))
        d += self.rng.normal(scale=self.drift, size=3)
        pos = pos + d + self.rng.normal(scale=self.sigma_pos, size=3)
        yaw = 2.0 * np.arctan2(quat[3], quat[0]) if abs(quat[0]) + abs(quat[3]) > 1e-9 else 0.0
        yaw += self.rng.normal(scale=self.sigma_yaw)
        quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        return pos, quat


@dataclass
class PerceptionModels:
    vision: object
    tactile: object
    vision_proj: object
    tactile_proj: object
    geometry_proj: object
    device: str
    image_size: int
    bank: object | None = None
    weights: dict | None = None
    threshold: float | None = None
    preserve_aspect: bool = True
    weights_mode: str = "reliability"
    dino: object | None = None
    dino_image_size: int = 224


def load_perception_models(
    expert_dir: str | Path,
    projection_dir: str | Path,
    prototype_dir: str | Path | None = None,
    device: str | None = None,
    weights_mode: str = "reliability",
    dino: bool = False,
    vision_expert=None,
    tactile_expert=None,
) -> PerceptionModels:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    expert_dir = Path(expert_dir)
    projection_dir = Path(projection_dir)
    vision = vision_expert or admit_encoder(
        load_vision_expert(expert_dir / "vision.pt", device=device),
        modality="vision",
        version=f"smallcnn@{SPACE_VERSION}",
    ).encoder
    dino_model = None
    if dino:
        from pss.encoders.vision.dinov2 import load_dinov2_encoder

        dino_model = admit_encoder(
            load_dinov2_encoder(device=device),
            modality="vision",
            version=f"dinov2@{SPACE_VERSION}",
        ).encoder
    tactile = tactile_expert or admit_encoder(
        load_tactile_expert(expert_dir / "tactile.pt", device=device),
        modality="tactile",
        version=f"mlp@{SPACE_VERSION}",
    ).encoder
    bank = None
    weights = None
    threshold = None
    if prototype_dir is not None:
        prototype_dir = Path(prototype_dir)
        bank = PrototypeBank.load(prototype_dir / "prototypes.npz")
        metrics_path = prototype_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            tuned = dict(metrics.get("weights", {}))
            reliability = {
                modality: float(scores["test_acc"])
                for modality, scores in (metrics.get("single_modality") or {}).items()
                if isinstance(scores, dict) and scores.get("test_acc") is not None
            }
            # Fusion-by-reliability: a modality that cannot classify on its
            # own should not outvote one that can. The tuned weights were
            # optimized on the pretraining domain; scene crops shift the
            # per-modality balance, so the bank's measured test accuracy is a
            # more robust default. ``weights_mode="tuned"`` restores Phase 5.
            if weights_mode == "reliability" and reliability:
                weights = reliability
            elif weights_mode == "tuned":
                weights = tuned
            else:
                weights = tuned or reliability
            threshold = (metrics.get("open_set") or {}).get("threshold")
    return PerceptionModels(
        vision=vision,
        tactile=tactile,
        vision_proj=load_projection(projection_dir / "vision.pt", device=device),
        tactile_proj=load_projection(projection_dir / "tactile.pt", device=device),
        geometry_proj=load_projection(projection_dir / "geometry.pt", device=device),
        device=device,
        image_size=int(vision.image_size),
        bank=bank,
        weights=weights,
        threshold=threshold,
        weights_mode=weights_mode,
        dino=dino_model,
        dino_image_size=int(getattr(dino_model, "image_size", 224)),
    )


def _instance_geoms(sample: dict) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for key, instance_id in sample["geom_to_instance"].items():
        out.setdefault(str(instance_id), []).append(int(key))
    return out


def extract_frame(
    sample: dict,
    frame: int,
    models: PerceptionModels,
    noise: PoseNoise,
    min_pixels: int = 20,
    timings: dict[str, tuple[float, float]] | None = None,
    t_ref: float | None = None,
) -> list[Observation]:
    """Build per-object observations for one frame.

    ``timings`` optionally supplies real ``(t_capture, t_arrival)`` seconds
    per modality (for asynchronous sensors).  Without it every stream is
    synchronous with ``frame``, preserving the historical behaviour.
    """
    rgb = sample["rgb"][frame]
    depth = sample["depth"][frame]
    seg = sample["segmentation"][frame]
    instance_geoms = _instance_geoms(sample)

    crops, tacs, geos, metas = [], [], [], []
    dino_crops = []
    modality_flags: list[dict[str, bool]] = []
    for i, gt_id in enumerate(sample["instance_ids"]):
        geoms = instance_geoms.get(str(gt_id), [])
        if not geoms:
            continue
        mask = np.isin(seg[..., 0], geoms)
        pixels = int(mask.sum())
        if pixels < min_pixels:
            continue
        geom_id = geoms[0]
        crop = crop_object(
            rgb,
            seg,
            geom_id,
            size=models.image_size,
            preserve_aspect=models.preserve_aspect,
        )
        tac = sample["tactile"][frame, i].reshape(-1).astype(np.float32)
        geo = geometry_features(depth, seg, geom_id)
        crops.append(normalize_image(crop))
        tacs.append(tac)
        geos.append(geo)
        flags = {
            "vision": bool(np.std(crop) > 1e-6),
            "tactile": bool(np.std(tac) > 1e-6),
            "geometry": bool(np.linalg.norm(geo) > 1e-6),
        }
        if models.dino is not None:
            dino_crop = crop_object(
                rgb,
                seg,
                geom_id,
                size=models.dino_image_size,
                preserve_aspect=models.preserve_aspect,
            )
            dino_crops.append(normalize_image(dino_crop))
            flags["dino"] = bool(np.std(dino_crop) > 1e-6)
        modality_flags.append(flags)
        pos, quat = noise.apply(
            str(gt_id),
            sample["poses"][frame, i, :3],
            sample["poses"][frame, i, 3:7],
            frame,
        )
        metas.append(
            {
                "gt_id": str(gt_id),
                "pos": pos,
                "quat": quat,
                "category": sample["categories"][i],
                "confidence": min(1.0, pixels / 500.0),
                "frame": frame,
            }
        )

    if not crops:
        return []

    with torch.no_grad():
        zv = models.vision.encode(torch.as_tensor(np.stack(crops), dtype=torch.float32))
        zt = models.tactile.encode(torch.as_tensor(np.stack(tacs), dtype=torch.float32))
        zv = models.vision_proj(
            torch.as_tensor(zv, dtype=torch.float32, device=models.device)
        ).cpu().numpy()
        zt = models.tactile_proj(
            torch.as_tensor(zt, dtype=torch.float32, device=models.device)
        ).cpu().numpy()
        zg = models.geometry_proj(
            torch.as_tensor(np.stack(geos), dtype=torch.float32, device=models.device)
        ).cpu().numpy()
        zd = None
        if models.dino is not None and dino_crops:
            zd = models.dino.encode(
                torch.as_tensor(np.stack(dino_crops), dtype=torch.float32)
            ).cpu().numpy()

    obs: list[Observation] = []
    for k, meta in enumerate(metas):
        flags = modality_flags[k]
        raw = {
            "vision": zv[k].astype(np.float32),
            "tactile": zt[k].astype(np.float32),
            "geometry": zg[k].astype(np.float32),
        }
        if zd is not None:
            raw["dino"] = zd[k].astype(np.float32)
        # A zero-signal channel means "sensor absent", not an observation.
        # Keep it out of recognition so a constant projection bias cannot win.
        for modality, available in flags.items():
            if not available:
                raw.pop(modality, None)
        # Attach the timing contract.  Default is synchronous with ``frame``;
        # a real driver passes ``timings`` to stamp capture/arrival seconds.
        ref = float(frame if t_ref is None else t_ref)
        streams = {
            modality: ModalityObservation(
                modality=modality,
                vector=np.asarray(vector, dtype=np.float32),
                t_capture=ref,
                t_arrival=ref,
            )
            for modality, vector in raw.items()
        }
        observation = Observation(
            gt_id=meta["gt_id"],
            pos=meta["pos"],
            quat=meta["quat"],
            appearance=raw["vision"],
            raw=raw,
            confidence=meta["confidence"],
            category=meta["category"],
            frame=frame,
            gt_category=meta["category"],
            category_score=1.0,
            modality_reliability=dict(models.weights or {}),
            t_ref=ref,
            streams=streams,
        )
        if timings is not None:
            observation.stamp_streams(timings, t_ref=ref)
        valid_raw = observation.valid_raw()

        gt_category = meta["category"]
        category = gt_category
        category_score = 1.0
        unknown = False
        conflict = False
        if models.bank is not None:
            result = recognize(
                valid_raw,
                models.bank,
                weights=models.weights,
                use_tree=False,
                threshold=models.threshold,
            )
            if result.ranked:
                category_score = float(result.ranked[0][1])
                category = None if result.unknown else result.ranked[0][0]
            else:
                category = None
            unknown = bool(result.unknown)
            conflict = bool(result.conflict)
        observation.category = category
        observation.category_score = category_score
        observation.unknown = unknown
        observation.conflict = conflict
        obs.append(observation)
    return obs
