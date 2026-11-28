"""Encoder registry: swap perception backbones without touching PSS interfaces.

The PSS cognitive layer only ever sees ``encode(x) -> (B, d)`` plus an input
size / input dim. This module is the single place where a backbone is chosen,
so experiments can swap vision/tactile encoders by name instead of editing
perception, projection, tracker or memory code.

Kinds
-----
vision:
  ``smallcnn``        task-trained SmallCNN (current baseline);
  ``dinov2``          frozen DINOv2 CLS/patch-mean;
  ``dinov2_vlad``     frozen DINOv2 patch tokens + fitted VLAD codebook;
  ``clip``            frozen CLIP ViT-L/14 vision tower;
  ``handcrafted``     zero-parameter colour/HOG/LBP descriptor (no checkpoint).
tactile:
  ``mlp``             current 8x8 tactile MLP;
  ``conv``            residual conv tactile encoder;
  ``wtmm``            WT-MM pretrained tactile CNN (domain-bridge adapter).
"""

from __future__ import annotations

from pathlib import Path


def _expert_file(expert_dir, name: str) -> Path:
    path = Path(expert_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"encoder checkpoint not found: {path}")
    return path


def build_vision_encoder(
    kind: str = "smallcnn",
    expert_dir: str | Path | None = None,
    device: str = "cpu",
    allow_download: bool = False,
    pool: str = "cls",
    codebook: str | Path | None = None,
    model_id: str | None = None,
    image_size: int = 64,
):
    """Return a frozen vision encoder object with ``encode``/``image_size``.

    ``model_id`` selects an alternative Hugging Face backbone for the
    ``dinov2``/``clip`` kinds (e.g. ``openai/clip-vit-base-patch16`` for a
    768-d CLIP tower).  It is ignored by the task-trained ``smallcnn``.
    """
    kind = str(kind)
    if kind == "handcrafted":
        from pss.encoders.vision.handcrafted import load_handcrafted_encoder

        # Zero parameters, no checkpoint and no device placement: the
        # descriptor runs on NumPy and ignores ``device``/``expert_dir``.
        return load_handcrafted_encoder(size=image_size)
    if kind == "smallcnn":
        from pss.encoders.vision import load_vision_expert

        if expert_dir is None:
            raise ValueError("smallcnn requires expert_dir")
        return load_vision_expert(_expert_file(expert_dir, "vision.pt"), device=device)
    if kind in ("dinov2", "dinov2_patchmean"):
        from pss.encoders.vision.dinov2 import load_dinov2_encoder

        kwargs = {"device": device, "local_files_only": not allow_download, "pool": pool}
        if model_id is not None:
            kwargs["model_id"] = model_id
        return load_dinov2_encoder(**kwargs)
    if kind == "dinov2_vlad":
        from pss.encoders.vision.dinov2 import load_dinov2_encoder
        from pss.encoders.vision.local import load_vlad

        if codebook is None:
            raise ValueError("dinov2_vlad requires a fitted VLAD codebook path")
        kwargs = {"device": device, "local_files_only": not allow_download, "pool": pool}
        if model_id is not None:
            kwargs["model_id"] = model_id
        backbone = load_dinov2_encoder(**kwargs)
        return load_vlad(codebook, backbone, device=device)
    if kind == "clip":
        from pss.encoders.vision.clip import load_clip_vision_encoder

        kwargs = {"device": device, "local_files_only": not allow_download, "pool": pool}
        if model_id is not None:
            kwargs["model_id"] = model_id
        return load_clip_vision_encoder(**kwargs)
    raise ValueError(f"unknown vision encoder {kind!r}")


def build_tactile_encoder(
    kind: str = "mlp",
    expert_dir: str | Path | None = None,
    device: str = "cpu",
):
    """Return a tactile encoder object with ``encode``/``input_dim``."""
    kind = str(kind)
    if kind == "mlp":
        from pss.encoders.tactile import load_tactile_expert

        if expert_dir is None:
            raise ValueError("mlp tactile encoder requires expert_dir")
        return load_tactile_expert(_expert_file(expert_dir, "tactile.pt"), device=device)
    if kind == "conv":
        from pss.encoders.tactile.conv import load_tactile_conv_expert

        if expert_dir is None:
            raise ValueError("conv tactile encoder requires expert_dir")
        return load_tactile_conv_expert(
            _expert_file(expert_dir, "tactile_conv.pt"), device=device
        )
    if kind == "wtmm":
        from pss.encoders.tactile.wtmm import WTMMTactileExpert, load_wtmm_tactile

        if expert_dir is None:
            raise ValueError("wtmm tactile encoder requires expert_dir")
        encoder = load_wtmm_tactile(
            _expert_file(expert_dir, "tactile_wtmm.pt"), device=device
        )
        return WTMMTactileExpert(encoder=encoder, device=device)
    raise ValueError(f"unknown tactile encoder {kind!r}")
