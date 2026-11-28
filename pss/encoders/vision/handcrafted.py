"""Handcrafted vision encoder: colour, HOG and LBP features, no learned weights.

This is the non-neural control for the expert bake-off.  It answers a concrete
question: how much of the vision expert's accuracy is actually *learned*, and
how much is already carried by classical image statistics?

The descriptor is deliberately simple and dependency-free (NumPy + Pillow):

- HSV histograms (12 / 8 / 8 bins) -- colour;
- HOG over a cell grid with 2x2 block L2 normalisation -- shape/edges;
- uniform LBP histogram over a 2x2 grid -- texture;
- a 4x4 mean-RGB grid -- coarse colour layout.

Nothing is fitted; ``params()`` is zero.  The output is **not** L2-normalised
here because the bake-off probe standardises features itself; the expert
contract still declares the intended downstream normalisation.

Input convention: unlike the CNN/tower encoders this expects RGB in ``[0, 1]``
(``(B, 3, H, W)`` tensors or ``(H, W, 3)`` uint8/float arrays).  It is not
ImageNet-normalised, so callers should not feed it ``normalize_image`` output.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

__all__ = [
    "HandcraftedVisionEncoder",
    "default_out_dim",
    "handcrafted_features",
]


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        scale = 255.0 if float(np.max(arr, initial=0.0)) <= 1.0 + 1e-6 else 1.0
        arr = np.clip(arr.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return arr


def _colour_histograms(img: Image.Image) -> np.ndarray:
    hsv = np.asarray(img.convert("HSV"), dtype=np.uint8)
    out: list[np.ndarray] = []
    for channel, bins in ((hsv[..., 0], 12), (hsv[..., 1], 8), (hsv[..., 2], 8)):
        hist, _ = np.histogram(channel, bins=bins, range=(0, 256))
        out.append(hist.astype(np.float32) / max(channel.size, 1))
    return np.concatenate(out)


def _hog(gray: np.ndarray, cell: int = 16, bins: int = 9) -> np.ndarray:
    """Unsigned-orientation HOG with 2x2 block normalisation (stride 1)."""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    bin_idx = np.minimum((angle / np.pi * bins).astype(np.int32), bins - 1)

    h, w = gray.shape
    ch, cw = max(h // cell, 1), max(w // cell, 1)
    hist = np.zeros((ch, cw, bins), dtype=np.float32)
    for by in range(ch):
        for bx in range(cw):
            m = mag[by * cell : (by + 1) * cell, bx * cell : (bx + 1) * cell]
            b = bin_idx[by * cell : (by + 1) * cell, bx * cell : (bx + 1) * cell]
            np.add.at(hist[by, bx], b.ravel(), m.ravel())

    blocks: list[np.ndarray] = []
    for by in range(max(ch - 1, 1)):
        for bx in range(max(cw - 1, 1)):
            block = hist[by : by + 2, bx : bx + 2].ravel()
            blocks.append(block / (np.linalg.norm(block) + 1e-6))
    return np.concatenate(blocks) if blocks else hist.ravel()


def _lbp(gray_u8: np.ndarray, grid: int = 2, bins: int = 10) -> np.ndarray:
    """8-neighbour LBP, uniform patterns mapped to 0-8, non-uniform to 9."""
    h, w = gray_u8.shape
    if h < 3 or w < 3:
        return np.zeros(grid * grid * bins, dtype=np.float32)
    centre = gray_u8[1:-1, 1:-1].astype(np.int16)
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    code = np.zeros_like(centre, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        neighbour = gray_u8[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx].astype(np.int16)
        code |= ((neighbour >= centre).astype(np.uint8) << bit)

    bits = ((code[..., None] >> np.arange(8, dtype=np.uint8)) & 1).astype(np.int8)
    transitions = (bits != np.roll(bits, -1, axis=2)).sum(axis=2)
    ones = bits.sum(axis=2).astype(np.uint8)
    mapped = np.where(transitions <= 2, ones, bins - 1).astype(np.uint8)

    out: list[np.ndarray] = []
    gh, gw = max(h // grid, 1), max(w // grid, 1)
    for gy in range(grid):
        for gx in range(grid):
            region = mapped[gy * gh : (gy + 1) * gh, gx * gw : (gx + 1) * gw]
            hist, _ = np.histogram(region, bins=bins, range=(0, bins))
            out.append(hist.astype(np.float32) / max(region.size, 1))
    return np.concatenate(out)


def _colour_grid(arr: np.ndarray, grid: int = 4) -> np.ndarray:
    h, w = arr.shape[:2]
    return (
        arr.reshape(grid, h // grid, grid, w // grid, 3).mean(axis=(1, 3)).reshape(-1)
    )


def default_out_dim(size: int = 64, cell: int = 16, hog_bins: int = 9) -> int:
    ch = cw = max(size // cell, 1)
    hog = max(ch - 1, 1) * max(cw - 1, 1) * 4 * hog_bins
    return int(28 + hog + 40 + 48)


def handcrafted_features(
    rgb: np.ndarray,
    size: int = 64,
    cell: int = 16,
    hog_bins: int = 9,
) -> np.ndarray:
    """Return a fixed-length, zero-parameter descriptor for one RGB image."""
    arr = _as_uint8_rgb(rgb)
    img = Image.fromarray(arr, mode="RGB").resize((size, size), Image.BILINEAR)
    small = np.asarray(img, dtype=np.float32) / 255.0
    gray = small @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    gray_u8 = np.asarray(img.convert("L"), dtype=np.uint8)

    feats = np.concatenate(
        [
            _colour_histograms(img),
            _hog(gray, cell=cell, bins=hog_bins),
            _lbp(gray_u8),
            _colour_grid(small),
        ]
    )
    return feats.astype(np.float32)


class HandcraftedVisionEncoder:
    """Registry-compatible wrapper: ``encode(images) -> (B, out_dim)``.

    ``images`` are expected as ``(B, 3, H, W)`` in ``[0, 1]``.  There are no
    parameters and ``eval()`` is a no-op, but the interface mirrors the frozen
    CNN/tower encoders so the registry can swap it in.
    """

    def __init__(self, size: int = 64, cell: int = 16, hog_bins: int = 9) -> None:
        self.image_size = int(size)
        self.cell = int(cell)
        self.hog_bins = int(hog_bins)
        self.out_dim = default_out_dim(self.image_size, self.cell, self.hog_bins)

    def eval(self) -> "HandcraftedVisionEncoder":
        return self

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        arr = images.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[None]
        arr = np.transpose(arr, (0, 2, 3, 1))
        feats = np.stack(
            [
                handcrafted_features(a, self.image_size, self.cell, self.hog_bins)
                for a in arr
            ]
        )
        return torch.as_tensor(feats, dtype=torch.float32)

    def params(self) -> int:
        return 0


def load_handcrafted_encoder(
    size: int = 64, cell: int = 16, hog_bins: int = 9
) -> HandcraftedVisionEncoder:
    return HandcraftedVisionEncoder(size=size, cell=cell, hog_bins=hog_bins)
