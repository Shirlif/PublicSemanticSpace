"""Transformer re-ID adapter for the PSS tracker (baseline)."""

from __future__ import annotations

import numpy as np
import torch

from pss.memory.learned_metric import pad_observations


class TransformerReID:
    def __init__(
        self,
        model,
        device: str = "cpu",
        max_cost: float = 0.9,
        include_tactile: bool = False,
        include_dino: bool = False,
    ):
        self.model = model
        self.device = device
        self.max_cost = float(max_cost)
        self.include_tactile = bool(include_tactile)
        self.include_dino = bool(include_dino)

    def cost_matrix(self, tracks, observations) -> np.ndarray:
        n = max(3, len(tracks), len(observations))
        pre_tokens, pre_mask = pad_observations(
            tracks,
            n,
            include_tactile=self.include_tactile,
            include_dino=self.include_dino,
        )
        post_tokens, post_mask = pad_observations(
            observations,
            n,
            include_tactile=self.include_tactile,
            include_dino=self.include_dino,
        )
        with torch.no_grad():
            scores = self.model(
                torch.as_tensor(pre_tokens, dtype=torch.float32, device=self.device)[None],
                torch.as_tensor(post_tokens, dtype=torch.float32, device=self.device)[None],
                torch.as_tensor(pre_mask, dtype=torch.float32, device=self.device)[None],
                torch.as_tensor(post_mask, dtype=torch.float32, device=self.device)[None],
            ).cpu().numpy()[0]
        cost = -scores[: len(tracks), : len(observations)]
        invalid = (pre_mask[: len(tracks)] <= 0)[:, None] | (
            post_mask[: len(observations)] <= 0
        )[None, :]
        cost[invalid] = 1e6
        return cost
