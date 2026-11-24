"""Multi-entity world model (Phase 7).

The v1 model in ``model.py`` predicts only the action target's delta. This
module predicts a delta for every entity, conditioned on the full scene state,
so non-target motion (collisions, settling) is represented instead of assumed
away.

Feature layout per slot: pos(3), quat(4), vel(3), present(1).
Action layout: kind one-hot(4), params(6), target one-hot(E).
Output: E x (dx, dy, dz, dyaw).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss.encoders.data import load_index
from pss.sim.dataset import load_sample

from .model import ACTION_KINDS, EntityState, quat_from_yaw, yaw_from_quat

MAX_ENTITIES = 4
FEATURE_PER_ENTITY = 11
DELTA_PER_ENTITY = 4


def multi_transition_feature(
    states: dict[str, EntityState],
    action: dict,
    ids: list[str],
    n_entities: int = MAX_ENTITIES,
) -> np.ndarray:
    slots: list[float] = []
    for i in range(n_entities):
        if i < len(ids) and ids[i] in states:
            state = states[ids[i]]
            slots.extend(float(v) for v in state.pos)
            slots.extend(float(v) for v in state.quat)
            slots.extend(float(v) for v in state.vel)
            slots.append(1.0)
        else:
            slots.extend([0.0] * (FEATURE_PER_ENTITY - 1))
            slots.append(0.0)

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
    target_onehot = np.zeros(n_entities, dtype=np.float32)
    if target is not None and int(target) < n_entities:
        target_onehot[int(target)] = 1.0
    return np.concatenate(
        [np.asarray(slots, dtype=np.float32), kind_onehot, params_vec, target_onehot]
    )


def feature_dim(n_entities: int = MAX_ENTITIES) -> int:
    return n_entities * FEATURE_PER_ENTITY + len(ACTION_KINDS) + 6 + n_entities


def build_multi_transitions(
    data_dir: str | Path,
    n_entities: int = MAX_ENTITIES,
) -> list[dict]:
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    rows: list[dict] = []
    for entry in index["entries"]:
        if entry["kind"] != "episode":
            continue
        sample = load_sample(data_dir / entry["path"])
        poses = np.asarray(sample["poses"], dtype=float)
        ids = [str(x) for x in sample["instance_ids"]]
        velocities = np.zeros_like(poses[..., :3])
        if len(poses) > 1:
            velocities[1:] = poses[1:, :, :3] - poses[:-1, :, :3]
        for t, action in enumerate(sample["actions"]):
            states = {
                ids[i]: EntityState(
                    poses[t, i, :3],
                    velocities[t, i],
                    poses[t, i, 3:7],
                )
                for i in range(len(ids))
            }
            feature = multi_transition_feature(states, action, ids, n_entities)
            delta = np.zeros((n_entities, DELTA_PER_ENTITY), dtype=np.float32)
            mask = np.zeros((n_entities,), dtype=np.float32)
            for i in range(min(len(ids), n_entities)):
                delta[i, :3] = poses[t + 1, i, :3] - poses[t, i, :3]
                delta[i, 3] = yaw_from_quat(poses[t + 1, i, 3:7]) - yaw_from_quat(
                    poses[t, i, 3:7]
                )
                mask[i] = 1.0
            rows.append(
                {
                    "episode": entry["id"],
                    "t": t,
                    "feature": feature,
                    "delta": delta,
                    "mask": mask,
                    "ids": ids,
                    "poses_t": poses[t].copy(),
                    "poses_next": poses[t + 1].copy(),
                    "velocities": velocities[t].copy(),
                    "action": action,
                    "target": action.get("target"),
                }
            )
    return rows


def _split_episodes(rows: list[dict], seed: int = 0) -> tuple[list[dict], list[dict]]:
    episodes = sorted({r["episode"] for r in rows})
    if len(episodes) <= 1:
        return rows, rows
    n_val = max(1, len(episodes) // 5)
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(episodes))
    val_ids = set(shuffled[:n_val])
    train = [r for r in rows if r["episode"] not in val_ids]
    val = [r for r in rows if r["episode"] in val_ids]
    return train, val


class MultiDeltaMLP(nn.Module):
    def __init__(self, in_dim: int, n_entities: int = MAX_ENTITIES, hidden: int = 128):
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_entities = int(n_entities)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.n_entities * DELTA_PER_ENTITY),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.n_entities, DELTA_PER_ENTITY)


def apply_multi_delta(poses: np.ndarray, delta: np.ndarray, n_entities: int) -> np.ndarray:
    out = np.asarray(poses, dtype=float).copy()
    for i in range(min(n_entities, len(out))):
        out[i, :3] += np.asarray(delta[i, :3], dtype=float)
        yaw = yaw_from_quat(out[i, 3:7]) + float(delta[i, 3])
        out[i, 3:7] = quat_from_yaw(yaw)
    return out


def _states_from_poses(
    poses: np.ndarray,
    ids: list[str],
    n_entities: int,
    velocities: np.ndarray | None = None,
) -> dict[str, EntityState]:
    states = {}
    for i in range(min(len(ids), n_entities)):
        vel = np.zeros(3) if velocities is None else velocities[i]
        states[ids[i]] = EntityState(poses[i, :3], vel, poses[i, 3:7])
    return states


def _pos_error(pred: np.ndarray, truth: np.ndarray, n_entities: int) -> float:
    return float(
        np.mean(
            np.linalg.norm(
                pred[:n_entities, :3] - truth[:n_entities, :3], axis=1
            )
        )
    )


def evaluate_multi_models(
    model: MultiDeltaMLP,
    rows: list[dict],
    device: str = "cpu",
) -> dict:
    model.eval()
    one_step = {"zero": [], "analytic": [], "learned": []}
    one_step_target = {"zero": [], "analytic": [], "learned": []}
    analytic = _analytic_model()
    with torch.no_grad():
        for row in rows:
            n = model.n_entities
            poses = row["poses_t"]
            truth = row["poses_next"]
            ids = row["ids"]
            target = row["target"]
            one_step["zero"].append(_pos_error(poses, truth, n))
            if target is not None:
                one_step_target["zero"].append(
                    float(np.linalg.norm(poses[int(target), :3] - truth[int(target), :3]))
                )
            else:
                one_step_target["zero"].append(0.0)

            pred_analytic = _analytic_step(analytic, poses, row["action"], ids, n)
            one_step["analytic"].append(_pos_error(pred_analytic, truth, n))
            if target is not None:
                one_step_target["analytic"].append(
                    float(
                        np.linalg.norm(
                            pred_analytic[int(target), :3] - truth[int(target), :3]
                        )
                    )
                )
            else:
                one_step_target["analytic"].append(0.0)

            feat = torch.as_tensor(row["feature"], dtype=torch.float32, device=device).unsqueeze(0)
            delta = model(feat).cpu().numpy()[0]
            pred_learned = apply_multi_delta(poses, delta, n)
            one_step["learned"].append(_pos_error(pred_learned, truth, n))
            if target is not None:
                one_step_target["learned"].append(
                    float(
                        np.linalg.norm(
                            pred_learned[int(target), :3] - truth[int(target), :3]
                        )
                    )
                )
            else:
                one_step_target["learned"].append(0.0)

    rollout = {"zero": [], "analytic": [], "learned": []}
    rollout_target = {"zero": [], "analytic": [], "learned": []}
    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    for episode_rows in by_episode.values():
        episode_rows = sorted(episode_rows, key=lambda r: r["t"])
        first = episode_rows[0]
        n = model.n_entities
        ids = first["ids"]
        poses = first["poses_t"].copy()
        pred_learned = poses.copy()
        pred_analytic = poses.copy()
        prev_learned = poses.copy()
        with torch.no_grad():
            for row in episode_rows:
                # Rebuild features from the model's own predicted state.
                velocity = pred_learned[:, :3] - prev_learned[:, :3]
                feat = torch.as_tensor(
                    multi_transition_feature(
                        _states_from_poses(pred_learned, ids, n, velocity),
                        row["action"],
                        ids,
                        n,
                    ),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                delta = model(feat).cpu().numpy()[0]
                prev_learned = pred_learned.copy()
                pred_learned = apply_multi_delta(pred_learned, delta, n)
                pred_analytic = _analytic_step(
                    analytic, pred_analytic, row["action"], ids, n
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
            rollout_target["zero"].append(
                float(np.linalg.norm(poses[int(target), :3] - truth[int(target), :3]))
            )
            rollout_target["analytic"].append(
                float(
                    np.linalg.norm(
                        pred_analytic[int(target), :3] - truth[int(target), :3]
                    )
                )
            )
            rollout_target["learned"].append(
                float(
                    np.linalg.norm(
                        pred_learned[int(target), :3] - truth[int(target), :3]
                    )
                )
            )

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "one_step_pos_error": {k: mean(v) for k, v in one_step.items()},
        "one_step_target_pos_error": {k: mean(v) for k, v in one_step_target.items()},
        "rollout_pos_error": {k: mean(v) for k, v in rollout.items()},
        "rollout_target_pos_error": {k: mean(v) for k, v in rollout_target.items()},
    }


def _analytic_model():
    from .model import AnalyticWorldModel

    return AnalyticWorldModel()


def _analytic_step(analytic, poses: np.ndarray, action: dict, ids: list[str], n: int) -> np.ndarray:
    target = action.get("target")
    if target is None or int(target) >= len(ids):
        return poses.copy()
    states = _states_from_poses(poses, ids, n)
    predicted = analytic.predict(states, {**action, "target": ids[int(target)]})
    out = poses.copy()
    out[int(target), :3] = predicted[ids[int(target)]].pos
    return out


def train_multi_world_model(
    data_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 100,
    hidden: int = 128,
    n_entities: int = MAX_ENTITIES,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    rows = build_multi_transitions(data_dir, n_entities)
    train_rows, val_rows = _split_episodes(rows, seed)
    in_dim = feature_dim(n_entities)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiDeltaMLP(in_dim, n_entities, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.as_tensor(np.stack([r["feature"] for r in train_rows]), dtype=torch.float32)
    Y = torch.as_tensor(np.stack([r["delta"] for r in train_rows]), dtype=torch.float32)
    M = torch.as_tensor(np.stack([r["mask"] for r in train_rows]), dtype=torch.float32)
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for start in range(0, len(X), 32):
            idx = perm[start : start + 32]
            xb = X[idx].to(device)
            yb = Y[idx].to(device)
            mb = M[idx].to(device)
            pred = model(xb)
            loss = (((pred - yb) ** 2).mean(dim=-1) * mb).sum() / mb.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()

    eval_metrics = evaluate_multi_models(model, val_rows, device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": in_dim,
            "hidden": hidden,
            "n_entities": n_entities,
            "model_type": "multi",
        },
        out / "world_model.pt",
    )
    metrics = {
        "data_dir": str(data_dir),
        "device": device,
        "epochs": int(epochs),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "in_dim": in_dim,
        "n_entities": n_entities,
        **eval_metrics,
        "note": "All-entity deltas; rollout rebuilds features from predicted state.",
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def load_multi_world_model(path: str | Path, device: str = "cpu") -> MultiDeltaMLP:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = MultiDeltaMLP(
        int(payload["in_dim"]),
        int(payload.get("n_entities", MAX_ENTITIES)),
        int(payload["hidden"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
