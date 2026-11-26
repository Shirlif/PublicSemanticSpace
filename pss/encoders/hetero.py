"""Heterogeneous-width expert adapters for the property suite (F4).

The public semantic space is always 128-d, but the frozen experts feeding it
have different native widths (SmallCNN 256, DINOv2 384, CLIP 768, tactile MLP
64, WT-MM 64).  A single frozen projection head cannot serve two widths, so
each expert needs its own head and its own calibration set.

This module only declares the experts and builds them through
:mod:`pss.encoders.registry`; the projection/calibration logic lives in
:mod:`pss.projection.hetero`.  Nothing here touches runtime code.

Widths
------
- ``vision_smallcnn_256``  same-width control;
- ``vision_dinov2_384``    frozen DINOv2-small CLS (384);
- ``vision_clip_768``      frozen CLIP ViT-B/16 vision tower (768);
- ``tactile_mlp_64``       same-width control;
- ``tactile_wtmm_64``      WT-MM tactile CNN (64).

The CLIP entry uses ``openai/clip-vit-base-patch16`` (768-d) rather than the
registry default ViT-L/14 (1024-d) so the suite exercises three distinct
widths.  The checkpoint must be present in the local HF cache; pass
``allow_download=True`` to fetch it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pss.encoders.registry import build_tactile_encoder, build_vision_encoder


@dataclass(frozen=True)
class HeteroExpertSpec:
    """Declarative description of one frozen expert at its native width."""

    name: str
    modality: str  # "vision" | "tactile"
    kind: str  # registry kind
    in_dim: int  # width of the feature vector the projection head receives
    model_id: str | None = None
    pool: str = "cls"
    control: bool = False
    note: str = ""

    def build(
        self,
        *,
        base_expert_dir: str | Path = "checkpoints/expert_v1",
        wtmm_dir: str | Path = "checkpoints/tactile_wtmm",
        device: str = "cpu",
        allow_download: bool = False,
    ):
        if self.modality == "vision":
            expert_dir = Path(base_expert_dir) if self.kind == "smallcnn" else None
            return build_vision_encoder(
                self.kind,
                expert_dir=expert_dir,
                device=device,
                allow_download=allow_download,
                pool=self.pool,
                model_id=self.model_id,
            )
        if self.modality != "tactile":
            raise ValueError(f"unknown modality {self.modality!r}")
        expert_dir = Path(wtmm_dir) if self.kind == "wtmm" else Path(base_expert_dir)
        return build_tactile_encoder(self.kind, expert_dir=expert_dir, device=device)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "modality": self.modality,
            "kind": self.kind,
            "in_dim": int(self.in_dim),
            "model_id": self.model_id,
            "pool": self.pool,
            "control": bool(self.control),
            "note": self.note,
        }


HETERO_EXPERTS: tuple[HeteroExpertSpec, ...] = (
    HeteroExpertSpec(
        "vision_smallcnn_256",
        "vision",
        "smallcnn",
        256,
        control=True,
        note="same-width control (base expert)",
    ),
    HeteroExpertSpec(
        "vision_dinov2_384",
        "vision",
        "dinov2",
        384,
        model_id="facebook/dinov2-small",
        note="frozen DINOv2-small CLS",
    ),
    HeteroExpertSpec(
        "vision_clip_768",
        "vision",
        "clip",
        768,
        model_id="openai/clip-vit-base-patch16",
        note="frozen CLIP ViT-B/16 vision tower",
    ),
    HeteroExpertSpec(
        "tactile_mlp_64",
        "tactile",
        "mlp",
        64,
        control=True,
        note="same-width control (base expert)",
    ),
    HeteroExpertSpec(
        "tactile_wtmm_64",
        "tactile",
        "wtmm",
        64,
        note="WT-MM tactile CNN (domain-bridge adapter)",
    ),
)

HETERO_BY_NAME: dict[str, HeteroExpertSpec] = {spec.name: spec for spec in HETERO_EXPERTS}

# The three widths F4 is required to exercise.
REQUIRED_WIDTHS = ("vision_dinov2_384", "vision_clip_768", "tactile_wtmm_64")


def get_expert_spec(name: str) -> HeteroExpertSpec:
    try:
        return HETERO_BY_NAME[str(name)]
    except KeyError as exc:
        known = ", ".join(sorted(HETERO_BY_NAME))
        raise ValueError(f"unknown heterogeneous expert {name!r}; known: {known}") from exc


def feature_dim(expert) -> int:
    """Return the width of the vector ``expert.encode`` produces."""
    if hasattr(expert, "out_dim"):
        return int(getattr(expert, "out_dim"))
    inner = getattr(expert, "encoder", None)
    if inner is not None and hasattr(inner, "out_dim"):
        return int(inner.out_dim)
    if hasattr(expert, "input_dim"):
        return int(getattr(expert, "input_dim"))
    raise TypeError(f"cannot infer feature width from {type(expert).__name__}")


def build_hetero_expert(
    spec: HeteroExpertSpec | str,
    *,
    base_expert_dir: str | Path = "checkpoints/expert_v1",
    wtmm_dir: str | Path = "checkpoints/tactile_wtmm",
    device: str = "cpu",
    allow_download: bool = False,
):
    """Build a spec's frozen encoder and verify its declared width."""
    spec = get_expert_spec(spec) if isinstance(spec, str) else spec
    expert = spec.build(
        base_expert_dir=base_expert_dir,
        wtmm_dir=wtmm_dir,
        device=device,
        allow_download=allow_download,
    )
    actual = feature_dim(expert)
    if actual != int(spec.in_dim):
        raise ValueError(
            f"{spec.name}: declared in_dim={spec.in_dim}, encoder produces {actual}"
        )
    return expert


__all__ = [
    "HETERO_BY_NAME",
    "HETERO_EXPERTS",
    "REQUIRED_WIDTHS",
    "HeteroExpertSpec",
    "build_hetero_expert",
    "feature_dim",
    "get_expert_spec",
]
