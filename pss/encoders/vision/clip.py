"""Frozen CLIP vision tower as a swappable PSS vision encoder.

The public-space contract is unchanged: ``encode(images) -> (B, out_dim)``
with ImageNet-normalized input tensors. CLIP was trained with its own
normalization, so the wrapper converts ImageNet-normalized inputs back to
[0, 1] and then applies CLIP statistics internally. Callers keep using the
same preprocessing path as ``SmallCNNEncoder`` and ``DinoV2Encoder``.

Only the vision tower is loaded: the text tower is irrelevant for PSS and
would double the footprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CLIP_IMAGE_SIZE = 224
CLIP_DIM = 1024

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _imagenet_to_clip(images: torch.Tensor) -> torch.Tensor:
    """Renormalize ImageNet-normalized CHW tensors for CLIP."""
    mean_in = torch.as_tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std_in = torch.as_tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    mean_clip = torch.as_tensor(_CLIP_MEAN, device=images.device).view(1, 3, 1, 1)
    std_clip = torch.as_tensor(_CLIP_STD, device=images.device).view(1, 3, 1, 1)
    unit = images * std_in + mean_in
    return (unit - mean_clip) / std_clip


@dataclass
class CLIPVisionEncoder:
    model: object
    device: str = "cpu"
    image_size: int = CLIP_IMAGE_SIZE
    out_dim: int = CLIP_DIM
    pool: str = "cls"  # "cls" | "patch_mean"
    _preprocess: dict = field(default_factory=dict, repr=False)

    def _features(self, images: torch.Tensor):
        self.model.eval()
        with torch.inference_mode():
            return self.model(pixel_values=_imagenet_to_clip(images.to(self.device)))

    def _batched(self, images: torch.Tensor, batch_size: int):
        for start in range(0, len(images), max(1, batch_size)):
            yield self._features(images[start : start + max(1, batch_size)])

    def encode(self, images: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """Return L2-normalized global features for a batch of images."""
        outs = []
        for out in self._batched(images, batch_size):
            if self.pool == "patch_mean":
                feat = out.last_hidden_state[:, 1:].mean(dim=1)
            else:
                feat = getattr(out, "pooler_output", None)
                if feat is None:
                    feat = out.last_hidden_state[:, 0]
            outs.append(F.normalize(feat.float(), dim=-1))
        return torch.cat(outs, dim=0)

    def encode_patches(self, images: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """Return per-patch tokens, ``(B, P, D)``, excluding the CLS token."""
        return torch.cat(
            [out.last_hidden_state[:, 1:].float() for out in self._batched(images, batch_size)],
            dim=0,
        )


def load_clip_vision_encoder(
    device: str = "cpu",
    model_id: str = CLIP_MODEL_ID,
    local_files_only: bool = True,
    pool: str = "cls",
) -> CLIPVisionEncoder:
    from transformers import CLIPVisionModel

    try:
        model = CLIPVisionModel.from_pretrained(model_id, local_files_only=local_files_only)
    except Exception as exc:  # offline cache miss, corrupt snapshot, version skew
        raise FileNotFoundError(
            f"CLIP vision weights unavailable for {model_id!r} "
            f"(local_files_only={local_files_only}): {type(exc).__name__}: {exc}"
        ) from exc
    model.to(device).eval()
    config = model.config
    size = int(getattr(config, "image_size", CLIP_IMAGE_SIZE))
    dim = int(getattr(config, "hidden_size", CLIP_DIM))
    return CLIPVisionEncoder(model=model, device=device, image_size=size, out_dim=dim, pool=pool)


def clip_available(model_id: str = CLIP_MODEL_ID) -> bool:
    """Cheap offline availability check used by evaluation scripts."""
    try:
        from transformers import CLIPVisionModel

        CLIPVisionModel.from_pretrained(model_id, local_files_only=True)
        return True
    except Exception:
        return False
