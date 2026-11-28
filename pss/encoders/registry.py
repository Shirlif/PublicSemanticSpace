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
  ``wtmm``            WT-MM pretrained tactile CNN (domain-bridge adapter);
  ``handcrafted``     zero-parameter contact statistics (no checkpoint).
"""

from __future__ import annotations

from pathlib import Path

from pss.contracts.schema import SPACE_VERSION
from pss.experts.adapter import admit_encoder


def _expert_file(expert_dir, name: str) -> Path:
    path = Path(expert_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"encoder checkpoint not found: {path}")
    return path


def _admit(
    encoder,
    *,
    modality: str,
    kind: str,
    probe=None,
    validate: bool = True,
    auto_probe: bool = True,
):
    """Run the expert contract on a freshly built encoder.

    The registry is the single production entry point for perception modules,
    so admission happens here rather than only in the property suite.  The
    encoder is returned unchanged; ``admit_encoder`` builds a temporary
    adapter and validates it.

    By default a zero-input probe is generated from the encoder geometry, so
    the production path checks output shape and finiteness, not just the
    declared width.  ``auto_probe=False`` keeps the static-only check for
    latency-sensitive callers; ``validate=False`` exists for tests and
    ablations that deliberately admit a malformed module.
    """
    if not validate:
        return encoder
    admit_encoder(
        encoder,
        modality=modality,
        version=f"{kind}@{SPACE_VERSION}",
        probe=probe,
        auto_probe=auto_probe,
    )
    return encoder


def build_vision_encoder(
    kind: str = "smallcnn",
    expert_dir: str | Path | None = None,
    device: str = "cpu",
    allow_download: bool = False,
    pool: str = "cls",
    codebook: str | Path | None = None,
    model_id: str | None = None,
    image_size: int = 64,
    probe=None,
    validate: bool = True,
    auto_probe: bool = True,
):
    """Return a frozen vision encoder object with ``encode``/``image_size``.

    ``model_id`` selects an alternative Hugging Face backbone for the
    ``dinov2``/``clip`` kinds (e.g. ``openai/clip-vit-base-patch16`` for a
    768-d CLIP tower).  It is ignored by the task-trained ``smallcnn``.

    Every encoder passes the expert admission contract before it is returned.
    By default a zero-input probe is generated from the encoder geometry, so
    output shape and finiteness are checked too.  Pass ``probe`` to use a real
    observation, ``auto_probe=False`` for the static-only check, or
    ``validate=False`` to bypass admission (tests/ablations only).
    """
    kind = str(kind)
    if kind == "handcrafted":
        from pss.encoders.vision.handcrafted import load_handcrafted_encoder

        # Zero parameters, no checkpoint and no device placement: the
        # descriptor runs on NumPy and ignores ``device``/``expert_dir``.
        encoder = load_handcrafted_encoder(size=image_size)
    elif kind == "smallcnn":
        from pss.encoders.vision import load_vision_expert

        if expert_dir is None:
            raise ValueError("smallcnn requires expert_dir")
        encoder = load_vision_expert(_expert_file(expert_dir, "vision.pt"), device=device)
    elif kind in ("dinov2", "dinov2_patchmean"):
        from pss.encoders.vision.dinov2 import load_dinov2_encoder

        kwargs = {"device": device, "local_files_only": not allow_download, "pool": pool}
        if model_id is not None:
            kwargs["model_id"] = model_id
        encoder = load_dinov2_encoder(**kwargs)
    elif kind == "dinov2_vlad":
        from pss.encoders.vision.dinov2 import load_dinov2_encoder
        from pss.encoders.vision.local import load_vlad

        if codebook is None:
            raise ValueError("dinov2_vlad requires a fitted VLAD codebook path")
        kwargs = {"device": device, "local_files_only": not allow_download, "pool": pool}
        if model_id is not None:
            kwargs["model_id"] = model_id
        backbone = load_dinov2_encoder(**kwargs)
        encoder = load_vlad(codebook, backbone, device=device)
    elif kind == "clip":
        from pss.encoders.vision.clip import load_clip_vision_encoder

        kwargs = {"device": device, "local_files_only": not allow_download, "pool": pool}
        if model_id is not None:
            kwargs["model_id"] = model_id
        encoder = load_clip_vision_encoder(**kwargs)
    else:
        raise ValueError(f"unknown vision encoder {kind!r}")

    return _admit(
        encoder,
        modality="vision",
        kind=kind,
        probe=probe,
        validate=validate,
        auto_probe=auto_probe,
    )


def build_tactile_encoder(
    kind: str = "mlp",
    expert_dir: str | Path | None = None,
    device: str = "cpu",
    out_shape: tuple[int, int] = (8, 8),
    probe=None,
    validate: bool = True,
    auto_probe: bool = True,
):
    """Return a tactile encoder object with ``encode``/``input_dim``.

    Like :func:`build_vision_encoder`, the module is admitted against the
    expert contract before it is returned; the probe is generated from the
    encoder geometry unless one is supplied.
    """
    kind = str(kind)
    if kind == "handcrafted":
        from pss.encoders.tactile.handcrafted import load_handcrafted_tactile

        encoder = load_handcrafted_tactile(out_shape=out_shape)
    elif kind == "mlp":
        from pss.encoders.tactile import load_tactile_expert

        if expert_dir is None:
            raise ValueError("mlp tactile encoder requires expert_dir")
        encoder = load_tactile_expert(_expert_file(expert_dir, "tactile.pt"), device=device)
    elif kind == "conv":
        from pss.encoders.tactile.conv import load_tactile_conv_expert

        if expert_dir is None:
            raise ValueError("conv tactile encoder requires expert_dir")
        encoder = load_tactile_conv_expert(
            _expert_file(expert_dir, "tactile_conv.pt"), device=device
        )
    elif kind == "wtmm":
        from pss.encoders.tactile.wtmm import WTMMTactileExpert, load_wtmm_tactile

        if expert_dir is None:
            raise ValueError("wtmm tactile encoder requires expert_dir")
        encoder = load_wtmm_tactile(
            _expert_file(expert_dir, "tactile_wtmm.pt"), device=device
        )
        encoder = WTMMTactileExpert(encoder=encoder, device=device)
    else:
        raise ValueError(f"unknown tactile encoder {kind!r}")

    return _admit(
        encoder,
        modality="tactile",
        kind=kind,
        probe=probe,
        validate=validate,
        auto_probe=auto_probe,
    )
