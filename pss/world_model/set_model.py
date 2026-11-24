"""Per-entity set world model (Phase 7).

A shared MLP scores each entity independently. Each entity sees its own state,
the action, whether it is the action target, and a summary of the other
entities. This is permutation-equivariant and uses all entities as training
samples instead of fitting one global vector.

Compared with ``multi.py`` (one global MLP over the whole scene), this is the
better inductive bias for a variable number of entities.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ACTION_KINDS, quat_from_yaw, yaw_from_quat
from .multi import (
    DELTA_PER_ENTITY,
    MAX_ENTITIES,
    _analytic_model,
    _analytic_step,
    _split_episodes,
    build_multi_transitions,
)

FEATURE_DIM = 33  # own pos3 + quat4 + vel3 + target1 + kind4 + params6 + other7 + target-rel5


def entity_features(
    poses: np.ndarray,
    velocities: np.ndarray,
    ids: list[str],
    action: dict,
    n_entities: int = MAX_ENTITIES,
) -> np.ndarray:
    kind = action.get("kind", "wait")
    kind_onehot = np.zeros(len(ACTION_KINDS), dtype=np.float32)
    if kind in ACTION_KINDS:
        kind_onehot[ACTION_KINDS.index(kind)] = 1.0
    params = action.get("params", {})
    xy = params.get("target_xy", [0.0, 0.0])
    direction = params.get("direction", [0.0, 0.0])
    params_vec = np.array(
        [
            float(params.get("force", 0.0)),
            float(params.get("height", 0.0)),
            float(xy[0]),
            float(xy[1]),
            float(direction[0]),
            float(direction[1]),
        ],
        dtype=np.float32,
    )
    target = action.get("target")
    present = min(len(ids), n_entities)
    target_pos = (
        poses[int(target), :3]
        if target is not None and int(target) < present
        else None
    )
    out = np.zeros((n_entities, FEATURE_DIM), dtype=np.float32)
    for i in range(present):
        others = [j for j in range(present) if j != i]
        if others:
            mean_pos = poses[others, :3].mean(axis=0)
            mean_vel = velocities[others].mean(axis=0)
            other_count = float(len(others))
        else:
            mean_pos = np.zeros(3)
            mean_vel = np.zeros(3)
            other_count = 0.0
        if target_pos is None or int(target) == i:
            target_rel = np.zeros(3)
            target_distance = 0.0
            target_contact = 0.0
        else:
            target_rel = target_pos - poses[i, :3]
            target_distance = float(np.linalg.norm(target_rel))
            target_contact = 1.0 if target_distance < 0.09 else 0.0
        out[i] = np.concatenate(
            [
                poses[i, :3],
                poses[i, 3:7],
                velocities[i],
                [1.0 if target is not None and int(target) == i else 0.0],
                kind_onehot,
                params_vec,
                mean_pos,
                mean_vel,
                [other_count],
                target_rel,
                [target_distance, target_contact],
            ]
        )
    return out


def build_set_examples(rows: list[dict], n_entities: int = MAX_ENTITIES) -> list[dict]:
    examples: list[dict] = []
    for row in rows:
        feats = entity_features(
            row["poses_t"], row["velocities"], row["ids"], row["action"], n_entities
        )
        for i in range(min(len(row["ids"]), n_entities)):
            examples.append(
                {
                    "episode": row["episode"],
                    "t": row["t"],
                    "entity": i,
                    "feature": feats[i],
                    "delta": row["delta"][i],
                }
            )
    return examples


class SetDeltaMLP(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 128):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, DELTA_PER_ENTITY),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def apply_set_delta(poses: np.ndarray, deltas: np.ndarray, n_entities: int) -> np.ndarray:
    out = np.asarray(poses, dtype=float).copy()
    for i in range(min(n_entities, len(out))):
        out[i, :3] += np.asarray(deltas[i, :3], dtype=float)
        yaw = yaw_from_quat(out[i, 3:7]) + float(deltas[i, 3])
        out[i, 3:7] = quat_from_yaw(yaw)
    return out


def _pos_error(pred: np.ndarray, truth: np.ndarray, n: int) -> float:
    return float(
        np.mean(np.linalg.norm(pred[:n, :3] - truth[:n, :3], axis=1))
    )


def evaluate_set_models(model: SetDeltaMLP, rows: list[dict], device: str = "cpu") -> dict:
    model.eval()
    n = MAX_ENTITIES
    analytic = _analytic_model()
    one_step = {"zero": [], "analytic": [], "learned": []}
    one_step_target = {"zero": [], "analytic": [], "learned": []}
    with torch.no_grad():
        for row in rows:
            poses = row["poses_t"]
            truth = row["poses_next"]
            target = row["target"]
            feats = entity_features(
                poses, row["velocities"], row["ids"], row["action"], n
            )
            pred = model(
                torch.as_tensor(feats, dtype=torch.float32, device=device)
            ).cpu().numpy()
            pred_learned = apply_set_delta(poses, pred, n)
            pred_analytic = _analytic_step(analytic, poses, row["action"], row["ids"], n)

            one_step["zero"].append(_pos_error(poses, truth, n))
            one_step["analytic"].append(_pos_error(pred_analytic, truth, n))
            one_step["learned"].append(_pos_error(pred_learned, truth, n))
            if target is not None:
                t = int(target)
                one_step_target["zero"].append(
                    float(np.linalg.norm(poses[t, :3] - truth[t, :3]))
                )
                one_step_target["analytic"].append(
                    float(np.linalg.norm(pred_analytic[t, :3] - truth[t, :3]))
                )
                one_step_target["learned"].append(
                    float(np.linalg.norm(pred_learned[t, :3] - truth[t, :3]))
                )

    rollout = {"zero": [], "analytic": [], "learned": []}
    rollout_target = {"zero": [], "analytic": [], "learned": []}
    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    for episode_rows in by_episode.values():
        episode_rows = sorted(episode_rows, key=lambda r: r["t"])
        first = episode_rows[0]
        poses = first["poses_t"].copy()
        pred_learned = poses.copy()
        pred_analytic = poses.copy()
        prev = poses.copy()
        with torch.no_grad():
            for row in episode_rows:
                velocity = pred_learned[:, :3] - prev[:, :3]
                feats = entity_features(
                    pred_learned, velocity, row["ids"], row["action"], n
                )
                pred = model(
                    torch.as_tensor(feats, dtype=torch.float32, device=device)
                ).cpu().numpy()
                prev = pred_learned.copy()
                pred_learned = apply_set_delta(pred_learned, pred, n)
                pred_analytic = _analytic_step(
                    analytic, pred_analytic, row["action"], row["ids"], n
                )
        truth = episode_rows[-1]["poses_next"]
        rollout["zero"].append(_pos_error(poses, truth, n))
        rollout["analytic"].append(_pos_error(pred_analytic, truth, n))
        rollout["learned"].append(_pos_error(pred_learned, truth, n))
        target = next(
            (row["target"] for row in episode_rows if row["target"] is not None),
            None,
        )
        if target is not None:
            t = int(target)
            rollout_target["zero"].append(
                float(np.linalg.norm(poses[t, :3] - truth[t, :3]))
            )
            rollout_target["analytic"].append(
                float(np.linalg.norm(pred_analytic[t, :3] - truth[t, :3]))
            )
            rollout_target["learned"].append(
                float(np.linalg.norm(pred_learned[t, :3] - truth[t, :3]))
            )

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "one_step_pos_error": {k: mean(v) for k, v in one_step.items()},
        "one_step_target_pos_error": {k: mean(v) for k, v in one_step_target.items()},
        "rollout_pos_error": {k: mean(v) for k, v in rollout.items()},
        "rollout_target_pos_error": {k: mean(v) for k, v in rollout_target.items()},
    }


def _rollout_loss(model, rows, device):
    """Multi-step loss. State is detached between steps, so each delta is
    trained against the rollout distribution without full BPTT."""
    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    losses = []
    n = MAX_ENTITIES
    for episode_rows in by_episode.values():
        episode_rows = sorted(episode_rows, key=lambda r: r["t"])
        poses = episode_rows[0]["poses_t"].copy()
        prev = poses.copy()
        for row in episode_rows:
            velocity = poses[:, :3] - prev[:, :3]
            feats = entity_features(
                poses, velocity, row["ids"], row["action"], n
            )
            pred = model(
                torch.as_tensor(feats, dtype=torch.float32, device=device)
            )
            truth = torch.as_tensor(
                row["poses_next"][:, :3], dtype=torch.float32, device=device
            )
            base = torch.as_tensor(poses[:, :3], dtype=torch.float32, device=device)
            losses.append(F.mse_loss(pred[:, :3], truth - base))
            prev = poses
            poses = apply_set_delta(poses, pred.detach().cpu().numpy(), n)
    return torch.stack(losses).mean() if losses else torch.zeros((), device=device)


def train_set_world_model(
    data_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 100,
    hidden: int = 256,
    n_entities: int = MAX_ENTITIES,
    seed: int = 0,
    rollout_weight: float = 0.0,
) -> dict:
    torch.manual_seed(seed)
    rows = build_multi_transitions(data_dir, n_entities)
    train_rows, val_rows = _split_episodes(rows, seed)
    examples = build_set_examples(train_rows, n_entities)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SetDeltaMLP(FEATURE_DIM, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.as_tensor(
        np.stack([e["feature"] for e in examples]), dtype=torch.float32
    )
    Y = torch.as_tensor(
        np.stack([e["delta"] for e in examples]), dtype=torch.float32
    )
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for start in range(0, len(X), 64):
            idx = perm[start : start + 64]
            xb, yb = X[idx].to(device), Y[idx].to(device)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if rollout_weight > 0:
            model.train()
            rloss = _rollout_loss(model, train_rows, device)
            opt.zero_grad()
            (rollout_weight * rloss).backward()
            opt.step()

    eval_metrics = evaluate_set_models(model, val_rows, device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": FEATURE_DIM,
            "hidden": hidden,
            "n_entities": n_entities,
            "model_type": "set",
        },
        out / "world_model.pt",
    )
    metrics = {
        "data_dir": str(data_dir),
        "device": device,
        "epochs": int(epochs),
        "n_train": len(examples),
        "n_val": len(val_rows),
        "in_dim": FEATURE_DIM,
        "n_entities": n_entities,
        **eval_metrics,
        "note": "Per-entity shared MLP; each entity sees its own state, action, target flag, and other-entity summary.",
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def load_set_world_model(path: str | Path, device: str = "cpu") -> SetDeltaMLP:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = SetDeltaMLP(int(payload["in_dim"]), int(payload["hidden"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
