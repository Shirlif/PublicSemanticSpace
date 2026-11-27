"""Perception service for the runtime: frozen encoders -> observations.

The runtime should not care which encoder produced an observation. This
service selects the encoder pair, loads the matching projection heads, and
returns the same ``Observation`` objects used by the tracker/entity layer.

Default (row 1 of the gap table): frozen DINOv2 vision + tactile MLP with the
calibrated projection from the encoder-priority experiment. SmallCNN + the
original projection is the fallback. CLIP/VLM require weights that are not in
the local cache; the service records that rather than pretending.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from pss.encoders.registry import build_tactile_encoder, build_vision_encoder
from pss.memory.perception import PoseNoise, extract_frame, load_perception_models

BASE_EXPERTS = "checkpoints/category_pretrain_experts"
BASE_PROJECTION = "checkpoints/phase45_projection"
DINOV2_PROJECTION = "checkpoints/encoder_priority/dinov2/projection"


@dataclass
class PerceptionConfig:
    vision: str = "dinov2"
    tactile: str = "mlp"
    projection_dir: str = DINOV2_PROJECTION
    allow_fallback: bool = True


class PerceptionService:
    def __init__(
        self,
        config: PerceptionConfig | None = None,
        device: str | None = None,
        noise: PoseNoise | None = None,
        calibration: "CalibrationService | None" = None,
    ):
        self.config = config or PerceptionConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.noise = noise or PoseNoise(seed=0, enabled=False)
        # Opt-in.  With ``None`` the service behaves exactly as before and uses
        # the static checkpoint weights, so existing L2 numbers are unchanged.
        self.calibration = calibration
        self.models = None
        self.fallback = False
        self.error = ""
        self._load()

    def _load(self) -> None:
        try:
            vision = build_vision_encoder(
                self.config.vision,
                expert_dir=BASE_EXPERTS,
                device=self.device,
            )
            tactile = build_tactile_encoder(
                self.config.tactile,
                expert_dir=BASE_EXPERTS,
                device=self.device,
            )
            projection = self.config.projection_dir
            if not Path(projection, "vision.pt").exists():
                raise FileNotFoundError(f"projection not found: {projection}")
            self.models = load_perception_models(
                BASE_EXPERTS,
                projection,
                device=self.device,
                vision_expert=vision,
                tactile_expert=tactile,
            )
            self.models.weights = self._load_weights(projection)
        except Exception as exc:
            if not self.config.allow_fallback:
                raise
            self.fallback = True
            self.error = f"{type(exc).__name__}: {exc}"
            self.models = load_perception_models(
                BASE_EXPERTS, BASE_PROJECTION, device=self.device
            )

    @staticmethod
    def _load_weights(projection: str) -> dict | None:
        import json

        candidates = [
            Path(projection).parent / "prototypes" / "metrics.json",
            Path("checkpoints/phase45_prototypes/metrics.json"),
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                metrics = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            tuned = metrics.get("weights")
            single = metrics.get("single_modality") or {}
            reliability = {
                modality: float(scores["test_acc"])
                for modality, scores in single.items()
                if isinstance(scores, dict) and scores.get("test_acc") is not None
            }
            if tuned:
                return dict(tuned)
            if reliability:
                return reliability
        return None

    def observe(self, sample, frame: int):
        if self.models is None:
            raise RuntimeError("perception service has no models")
        return extract_frame(sample, frame, self.models, self.noise)

    def attach_calibration(self, calibration: "CalibrationService | None") -> None:
        """Enable or disable streaming calibration without rebuilding models."""
        self.calibration = calibration

    def update_calibration(self, window, holdout=None, *, force_gate: bool = False):
        """Feed one labelled window to the streaming calibrator.

        On success the runtime's recognition weights are replaced by the
        pipeline's reliability, so the next ``observe`` call sees them through
        ``Observation.modality_reliability``.  Returns the pipeline record, or
        ``None`` when calibration is not attached.
        """
        if self.calibration is None:
            return None
        record = self.calibration.update(window, holdout, force_gate=force_gate)
        reliability = dict(self.calibration.params().reliability)
        if reliability and self.models is not None:
            self.models.weights = reliability
        return record

    def confidence(self, fused_top1: float) -> float:
        """Calibrated confidence for a fused top-1 score (map channel)."""
        if self.calibration is None:
            return float("nan")
        return self.calibration.confidence(fused_top1)

    def describe(self) -> dict:
        projection = self.config.projection_dir if not self.fallback else BASE_PROJECTION
        calibration = {}
        metrics_path = Path(projection) / "metrics.json"
        if metrics_path.exists():
            try:
                import json

                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                calibration = {
                    "whiten_dim": metrics.get("whiten_dim"),
                    "whiten_modalities": metrics.get("whiten_modalities"),
                    "residual": metrics.get("residual"),
                    "weights": metrics.get("weights"),
                }
            except Exception:
                calibration = {}
        return {
            "vision": self.config.vision if not self.fallback else "smallcnn",
            "tactile": self.config.tactile if not self.fallback else "mlp",
            "projection": projection,
            "image_size": int(getattr(self.models.vision, "image_size", 64)),
            "device": self.device,
            "fallback": self.fallback,
            "error": self.error,
            "calibration": calibration,
            "reliability": dict(getattr(self.models, "weights", None) or {}),
            "calibration_service": (
                self.calibration.describe() if self.calibration is not None else None
            ),
        }
