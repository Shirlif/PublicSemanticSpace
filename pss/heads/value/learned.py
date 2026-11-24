"""Learned value head (Phase 8 closure).

Synthetic preference data with a nonlinear utility:

    U(d, safety, novelty) = 0.6 * task(d) + 0.3 * safety + 0.1 * novelty
                            - 0.25 * max(0, 0.25 - safety)

The learned head is trained with a Bradley-Terry pairwise loss. This is a
proxy preference set, not human data; it closes the "value head is rule-only"
boundary while keeping the interface identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ValueHead

FEATURE_DIM = 3


def ground_truth_utility(distance: float, safety: float, novelty: float) -> float:
    task = 1.0 / (1.0 + float(distance) / 0.2)
    return (
        0.6 * task
        + 0.3 * float(safety)
        + 0.1 * float(novelty)
        - 0.25 * max(0.0, 0.25 - float(safety))
    )


def _sample_candidate(rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [
            rng.uniform(0.0, 0.5),
            rng.uniform(0.0, 1.0),
            rng.uniform(0.0, 1.0),
        ],
        dtype=np.float32,
    )


def sample_preferences(n_pairs: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Xa, Xb, Y = [], [], []
    while len(Y) < n_pairs:
        a = _sample_candidate(rng)
        b = _sample_candidate(rng)
        ua = ground_truth_utility(*a)
        ub = ground_truth_utility(*b)
        if abs(ua - ub) < 1e-6:
            continue
        Xa.append(a)
        Xb.append(b)
        Y.append(1.0 if ua > ub else 0.0)
    return np.stack(Xa), np.stack(Xb), np.asarray(Y, dtype=np.float32)


class PreferenceMLP(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LearnedValueHead:
    def __init__(self, model: PreferenceMLP, goal_xy=None):
        self.model = model
        self.goal_xy = None if goal_xy is None else np.asarray(goal_xy, dtype=float)

    def score_features(self, distance: float, safety: float = 0.0, novelty: float = 0.0) -> float:
        x = torch.as_tensor(
            [[float(distance), float(safety), float(novelty)]], dtype=torch.float32
        )
        with torch.no_grad():
            logit = float(self.model(x).cpu().numpy()[0])
        return float(1.0 / (1.0 + np.exp(-logit)))

    def score(self, track, safety: float = 0.0, novelty: float = 0.0) -> float:
        if self.goal_xy is None:
            distance = 0.0
        else:
            distance = float(
                np.linalg.norm(np.asarray(track.pos[:2], dtype=float) - self.goal_xy)
            )
        return self.score_features(distance, safety, novelty)

    def score_tracks(self, tracks, safety: dict[str, float] | None = None) -> dict[str, float]:
        safety = safety or {}
        return {
            tr.id: self.score(tr, safety=safety.get(getattr(tr, "category", None), 0.0))
            for tr in tracks
        }


def train_learned_value_head(
    out_dir: str | Path,
    n_pairs: int = 4000,
    epochs: int = 100,
    hidden: int = 64,
    seed: int = 0,
    regression_weight: float = 5.0,
) -> dict:
    torch.manual_seed(seed)
    Xa, Xb, Y = sample_preferences(n_pairs, seed)
    model = PreferenceMLP(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xa = torch.as_tensor(Xa, dtype=torch.float32)
    xb = torch.as_tensor(Xb, dtype=torch.float32)
    y = torch.as_tensor(Y, dtype=torch.float32)
    ua = torch.as_tensor(
        [ground_truth_utility(*x) for x in Xa], dtype=torch.float32
    )
    ub = torch.as_tensor(
        [ground_truth_utility(*x) for x in Xb], dtype=torch.float32
    )
    losses = []
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xa))
        epoch_loss = 0.0
        for start in range(0, len(xa), 128):
            idx = perm[start : start + 128]
            pred_a = model(xa[idx])
            pred_b = model(xb[idx])
            pairwise = F.binary_cross_entropy_with_logits(pred_a - pred_b, y[idx])
            regression = F.mse_loss(pred_a, ua[idx]) + F.mse_loss(pred_b, ub[idx])
            loss = pairwise + float(regression_weight) * regression
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(idx)
        losses.append(epoch_loss / max(1, len(xa)))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "hidden": hidden},
        out / "value_head.pt",
    )
    metrics = {
        "n_pairs": int(n_pairs),
        "epochs": int(epochs),
        "final_loss": float(losses[-1]) if losses else 0.0,
        "regression_weight": float(regression_weight),
        "note": "Bradley-Terry + utility regression on synthetic nonlinear preferences; not human data.",
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def load_learned_value_head(path: str | Path, goal_xy=None, device: str = "cpu") -> LearnedValueHead:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = PreferenceMLP(int(payload["hidden"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return LearnedValueHead(model, goal_xy=goal_xy)


def evaluate_value_scorer(scorer, n_pairs: int = 1000, seed: int = 0) -> dict:
    Xa, Xb, Y = sample_preferences(n_pairs, seed)
    correct = 0
    for a, b, label in zip(Xa, Xb, Y):
        sa = scorer(*a)
        sb = scorer(*b)
        pred = 1.0 if sa > sb else 0.0
        correct += int(pred == float(label))

    rng = np.random.default_rng(seed + 1)
    groups = 200
    top1 = 0
    for _ in range(groups):
        candidates = [_sample_candidate(rng) for _ in range(5)]
        true_best = max(candidates, key=lambda c: ground_truth_utility(*c))
        pred_best = max(candidates, key=lambda c: scorer(*c))
        top1 += int(np.allclose(true_best, pred_best))

    return {
        "n_pairs": int(n_pairs),
        "pairwise_accuracy": correct / max(1, len(Y)),
        "top1_accuracy": top1 / groups,
        "top1_groups": groups,
    }


def evaluate_rule_value(n_pairs: int = 1000, seed: int = 0) -> dict:
    head = ValueHead(goal_xy=np.zeros(2))
    return evaluate_value_scorer(head.score_features, n_pairs, seed)
