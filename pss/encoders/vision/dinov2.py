"""Frozen DINOv2 vision backbone used as an instance-descriptor expert.

The PSS semantic space keeps its own 128-d vision expert for recognition.
DINOv2 is a second, frozen visual channel used only by the re-ID matcher: its
CLS token carries fine-grained instance appearance that the small task-trained
CNN does not. Keeping it out of the prototype bank means category recognition
and open-set behavior do not change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


DINOV2_MODEL_ID = "facebook/dinov2-small"
DINOV2_IMAGE_SIZE = 224
DINOV2_DIM = 384


@dataclass
class DinoV2Encoder:
    model: object
    device: str = "cpu"
    image_size: int = DINOV2_IMAGE_SIZE
    out_dim: int = DINOV2_DIM
    pool: str = "cls"  # "cls" | "patch_mean"

    def _hidden(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.inference_mode():
            out = self.model(pixel_values=images.to(self.device))
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is not None:
                return hidden.float()
            return out.pooler_output.float().unsqueeze(1)

    def _batched(self, images: torch.Tensor, batch_size: int):
        for start in range(0, len(images), max(1, batch_size)):
            yield self._hidden(images[start : start + max(1, batch_size)])

    def encode(self, images: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        """Return L2-normalized global features for a batch of ImageNet tensors."""
        outs = []
        for hidden in self._batched(images, batch_size):
            if self.pool == "patch_mean":
                outs.append(F.normalize(hidden[:, 1:].mean(dim=1), dim=-1))
            else:
                outs.append(F.normalize(hidden[:, 0], dim=-1))
        return torch.cat(outs, dim=0)

    def encode_patches(self, images: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        """Return per-patch tokens, ``(B, P, D)``, excluding the CLS token."""
        return torch.cat(
            [hidden[:, 1:] for hidden in self._batched(images, batch_size)], dim=0
        )


def load_dinov2_encoder(
    device: str = "cpu",
    model_id: str = DINOV2_MODEL_ID,
    local_files_only: bool = True,
    pool: str = "cls",
) -> DinoV2Encoder:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, local_files_only=local_files_only)
    model.to(device).eval()
    dim = int(getattr(model.config, "hidden_size", DINOV2_DIM))
    return DinoV2Encoder(
        model=model,
        device=device,
        image_size=DINOV2_IMAGE_SIZE,
        out_dim=dim,
        pool=pool,
    )
