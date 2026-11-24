"""Learned active-set and write policies (third priority).

Two decisions are learned instead of hand-tuned:

- **ActiveSetPolicy**: which candidate entities deserve attention for the
  current observation. It is trained with a listwise objective (the correct
  candidate should get the highest score), not pairwise BCE, and selects a
  small top-k set.
- **WritePolicy**: which observations/facts are worth writing to memory. It
  predicts correctness/usefulness and supports a review band for uncertain
  items.

Both expose a calibrated uncertainty gate: when the score margin is below a
threshold calibrated on validation data, the active-set policy falls back to a
larger context, and the write policy routes the item to review instead of
writing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss.memory.learned_metric import token_layout


# ── active set ──────────────────────────────────────────────────────────────


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def tracking_pair_features(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Features for (post observation, pre candidate) pairs.

    ``pre``/``post`` are entity-token arrays ``(N, D)`` / ``(M, D)``. The
    output is ``(M, N, F)``: each post observation is a row, each pre
    candidate a column, matching the listwise selection objective.
    """
    layout = token_layout(True, False)
    vs, ve = layout["vision"]
    gs, ge = layout["geometry"]
    ps, pe = layout["pose"]
    cs, ce = layout["category"]
    ts, te = layout["tactile"]
    pv, qv = _l2(pre[:, vs:ve]), _l2(post[:, vs:ve])
    pg, qg = _l2(pre[:, gs:ge]), _l2(post[:, gs:ge])
    pt, qt = _l2(pre[:, ts:te]), _l2(post[:, ts:te])
    dist = np.linalg.norm(
        post[:, None, ps:pe] - pre[None, :, ps:pe], axis=-1
    )
    vision = qv @ pv.T
    geometry = qg @ pg.T
    tactile = qt @ pt.T
    category = post[:, cs:ce] @ pre[:, cs:ce].T
    pre_tactile_norm = np.linalg.norm(pre[:, ts:te], axis=1)[None, :].repeat(len(post), 0)
    ones = np.ones((len(post), len(pre)), dtype=np.float32)
    return np.stack(
        [-dist, vision, geometry, tactile, category, pre_tactile_norm, -5.0 * dist, ones],
        axis=-1,
    ).astype(np.float32)


def _row_targets(pair) -> list[int]:
    """Argmax GT candidate per post row, ``-1`` for padded rows."""
    targets = []
    for j in range(len(pair.post_mask)):
        if pair.post_mask[j] == 0:
            targets.append(-1)
            continue
        row = pair.labels[j, : int(pair.pre_mask.sum())]
        targets.append(int(np.argmax(row)) if row.max() > 0 else -1)
    return targets


class ActiveSetPolicy(nn.Module):
    """Scores candidate entities; selects top-k with a calibrated fallback."""

    def __init__(self, feature_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    @torch.no_grad()
    def scores(self, features: np.ndarray, device: str = "cpu") -> np.ndarray:
        self.eval()
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        return self(x).cpu().numpy()


@dataclass
class ActiveSelection:
    mask: np.ndarray  # (M, N) boolean
    margins: np.ndarray  # (M,)
    fallback_rows: np.ndarray  # (M,) bool
    mean_active: float

    def to_cost(self, scores: np.ndarray) -> np.ndarray:
        return np.where(self.mask, scores, -1e6)


def select_active(
    scores: np.ndarray,
    k: int,
    margin_threshold: float | None = None,
) -> ActiveSelection:
    """Top-k per row; if the k-th margin is below threshold, keep the full row."""
    m, n = scores.shape
    mask = np.zeros((m, n), dtype=bool)
    margins = np.full(m, np.inf, dtype=np.float32)
    fallback = np.zeros(m, dtype=bool)
    for j in range(m):
        order = np.argsort(-scores[j])
        take = order[: min(k, n)]
        mask[j, take] = True
        if n > k:
            top = scores[j, order]
            margins[j] = float(top[k - 1] - top[k])
            if margin_threshold is not None and margins[j] < margin_threshold:
                mask[j, :] = True
                fallback[j] = True
    return ActiveSelection(
        mask=mask,
        margins=margins,
        fallback_rows=fallback,
        mean_active=float(mask.sum(axis=1).mean()),
    )


def train_active_policy(
    pairs,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    feature_fn=tracking_pair_features,
    corrupt: str | None = None,
):
    """Listwise training: the correct candidate should win each post row.

    ``pairs`` are ``RelationPair``-like objects (``pre``/``post``/masks/labels``).
    """
    from pss.memory.typed_relations import perturb_tokens

    torch.manual_seed(seed)
    policy = ActiveSetPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    data = []
    for pair in pairs:
        pre, post = pair.pre, pair.post
        if corrupt:
            pre = perturb_tokens(pre, corrupt, seed=hash(pair.sample_id) % 1000)
            post = perturb_tokens(post, corrupt, seed=(hash(pair.sample_id) + 1) % 1000)
        data.append(
            (
                feature_fn(pre, post),
                _row_targets(pair),
                (pair.pre_mask[:, None] * pair.post_mask[None, :]).astype(bool),
            )
        )
    for epoch in range(epochs):
        policy.train()
        for i in np.random.default_rng(epoch).permutation(len(data)):
            features, targets, mask = data[i]
            logits = policy(torch.as_tensor(features[None], device=device))[0]
            loss = 0.0
            count = 0
            for j, target in enumerate(targets):
                if target < 0:
                    continue
                valid = torch.as_tensor(mask[j], device=device)
                loss = loss + F.cross_entropy(
                    logits[j][valid][None],
                    torch.as_tensor([target], device=device),
                )
                count += 1
            if count == 0:
                continue
            opt.zero_grad()
            (loss / count).backward()
            opt.step()
    policy.eval()
    return policy


def calibrate_margin(
    policy: ActiveSetPolicy,
    pairs,
    k: int = 1,
    target_fallback: float = 0.10,
    device: str = "cpu",
    feature_fn=tracking_pair_features,
) -> float:
    """Choose a margin threshold so validation fallback ≈ ``target_fallback``."""
    margins = []
    for pair in pairs:
        n = int(pair.pre_mask.sum())
        m = int(pair.post_mask.sum())
        if n <= k or m == 0:
            continue
        scores = policy.scores(feature_fn(pair.pre, pair.post), device)[:m, :n]
        for row in scores:
            order = np.sort(row)[::-1]
            margins.append(float(order[k - 1] - order[k]))
    if not margins:
        return 0.0
    return float(np.percentile(margins, 100.0 * max(0.0, min(1.0, target_fallback))))


# ── write policy ────────────────────────────────────────────────────────────


def write_features(
    reliability: float,
    conflict: float,
    novelty: float,
    key_frequency: float,
    age_ratio: float,
    is_new: float,
    value_ratio: float,
) -> np.ndarray:
    """Default write-policy feature vector (kept explicit for auditability)."""
    return np.array(
        [reliability, conflict, novelty, key_frequency, age_ratio, is_new, value_ratio, 1.0],
        dtype=np.float32,
    )


class WritePolicy(nn.Module):
    """Predicts whether a new observation is worth writing to memory."""

    def __init__(self, feature_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    @torch.no_grad()
    def probability(self, features: np.ndarray, device: str = "cpu") -> float:
        self.eval()
        x = torch.as_tensor(np.asarray(features)[None], dtype=torch.float32, device=device)
        return float(torch.sigmoid(self(x))[0])

    def decide(
        self,
        features: np.ndarray,
        write_threshold: float = 0.6,
        review_low: float = 0.4,
        review_high: float = 0.6,
        device: str = "cpu",
    ) -> str:
        p = self.probability(features, device)
        if p >= write_threshold:
            return "write"
        if review_low <= p < review_high:
            return "review"
        return "skip"


def train_write_policy(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
):
    torch.manual_seed(seed)
    policy = WritePolicy(feature_dim=int(features.shape[1])).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    for _ in range(epochs):
        policy.train()
        perm = torch.randperm(len(x), device=device)
        for start in range(0, len(perm), 128):
            idx = perm[start : start + 128]
            loss = F.binary_cross_entropy_with_logits(policy(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    policy.eval()
    return policy
