"""VLM-style text-state baseline (B1).

A real VLM client must be supplied by the caller; no external API call is made
automatically. ``MockVLMClient`` reads GT structure and is only a test double.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from pss.eval.baselines import TextStateTracker
from pss.eval.metrics import id_retention


class VLMClient(Protocol):
    def describe(self, frame_index: int) -> str: ...


@dataclass
class TextObs:
    gt_id: str
    pos: np.ndarray
    category: str


def parse_vlm_text(text: str) -> list[dict]:
    """Parse lines like ``obj0: category=cup, x=0.12, y=-0.03``."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        ident, rest = line.split(":", 1)
        category = None
        x = y = None
        for part in rest.split(","):
            part = part.strip()
            if part.startswith("category="):
                category = part.split("=", 1)[1].strip()
            elif part.startswith("x="):
                try:
                    x = float(part.split("=", 1)[1])
                except ValueError:
                    pass
            elif part.startswith("y="):
                try:
                    y = float(part.split("=", 1)[1])
                except ValueError:
                    pass
        if ident.strip() and category is not None and x is not None and y is not None:
            out.append({"id": ident.strip(), "category": category, "pos": [x, y]})
    return out


class MockVLMClient:
    def __init__(
        self,
        sample: dict,
        dropout: float = 0.0,
        pos_noise: float = 0.0,
        seed: int = 0,
    ):
        self.sample = sample
        self.dropout = float(dropout)
        self.pos_noise = float(pos_noise)
        self.rng = np.random.default_rng(seed)

    def _visible(self, frame_index: int, gt_id: str) -> bool:
        seg = self.sample.get("segmentation")
        mapping = self.sample.get("geom_to_instance")
        if seg is None or mapping is None:
            return True
        geoms = [int(k) for k, v in mapping.items() if v == gt_id]
        if not geoms:
            return True
        mask = np.isin(seg[frame_index, ..., 0], geoms)
        return int(mask.sum()) >= 20

    def describe(self, frame_index: int) -> str:
        lines = []
        for i, gt_id in enumerate(self.sample["instance_ids"]):
            if not self._visible(frame_index, gt_id):
                continue
            if self.rng.random() < self.dropout:
                continue
            pos = np.asarray(self.sample["poses"][frame_index, i, :3], dtype=float)
            noise = (
                self.rng.normal(scale=self.pos_noise, size=2)
                if self.pos_noise
                else np.zeros(2)
            )
            lines.append(
                f"{gt_id}: category={self.sample['categories'][i]}, "
                f"x={pos[0] + noise[0]:.2f}, y={pos[1] + noise[1]:.2f}"
            )
        return "\n".join(lines)


class CallableVLMClient:
    """Wrap a real VLM callable: ``fn(prompt, image_b64) -> text``.

    The caller owns API keys, data-egress consent, and rate limiting.
    """

    def __init__(self, fn: Callable[[str, str], str], prompt: str, images_b64: list[str]):
        self.fn = fn
        self.prompt = prompt
        self.images_b64 = images_b64

    def describe(self, frame_index: int) -> str:
        return self.fn(self.prompt, self.images_b64[frame_index])


def evaluate_vlm_text_state(sample: dict, client: VLMClient) -> dict:
    tracker = TextStateTracker()
    gt_tracks: dict[str, list[tuple[int, str]]] = {}
    for t in range(sample["rgb"].shape[0]):
        parsed = parse_vlm_text(client.describe(t))
        observations = [
            TextObs(
                gt_id=item["id"],
                pos=np.array([item["pos"][0], item["pos"][1], 0.0]),
                category=item["category"],
            )
            for item in parsed
        ]
        events = tracker.step(observations, t)
        for key in ("matched", "new"):
            for track_id, gt_id in events[key]:
                gt_tracks.setdefault(gt_id, []).append((t, track_id))
    return id_retention(gt_tracks)
