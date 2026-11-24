"""Entity-state world models (Phase 7).

Two models share a small interface:
- ``AnalyticWorldModel``: constant-velocity + scripted action effects (baseline);
- ``LearnedWorldModel``: an MLP predicting the target entity's pose delta from
  the current state and action (smoke test on Phase 2 episodes).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss.encoders.data import load_index
from pss.sim.dataset import load_sample

ACTION_KINDS = ("push", "lift", "place", "wait")
MAX_ENTITIES = 4


def yaw_from_quat(quat) -> float:
    q = np.asarray(quat, dtype=float)
    return float(2.0 * np.arctan2(q[3], q[0]))


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


@dataclass
class EntityState:
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    def copy(self) -> "EntityState":
        return EntityState(self.pos.copy(), self.vel.copy(), self.quat.copy())


class AnalyticWorldModel:
    def __init__(self, dt: float = 1.0):
        self.dt = float(dt)

    def predict(self, states: dict[str, EntityState], action: dict) -> dict[str, EntityState]:
        new = {k: v.copy() for k, v in states.items()}
        target = action.get("target")
        if target is None or target not in new:
            return new
        state = new[target]
        kind = action.get("kind")
        params = action.get("params", {})
        if kind == "push":
            direction = np.asarray(params.get("direction", [1.0, 0.0]), dtype=float)
            force = float(params.get("force", 1.0))
            state.vel[:2] += direction * force * 0.02
            state.pos += state.vel * self.dt
        elif kind == "place":
            xy = params.get("target_xy")
            if xy is not None:
                state.pos[0], state.pos[1] = float(xy[0]), float(xy[1])
                state.vel[:] = 0.0
        elif kind == "lift":
            state.pos[2] += float(params.get("height", 0.1))
            state.vel[:] = 0.0
        elif kind == "wait":
            state.pos += state.vel * self.dt
        return new


def transition_feature(states: dict[str, EntityState], action: dict, ids: list[str]) -> np.ndarray:
    target = action.get("target")
    kind = action.get("kind", "wait")
    params = action.get("params", {})
    target_state = states[ids[target]] if target is not None and target < len(ids) else EntityState(np.zeros(3))
    kind_onehot = np.zeros(len(ACTION_KINDS), dtype=np.float32)
    if kind in ACTION_KINDS:
        kind_onehot[ACTION_KINDS.index(kind)] = 1.0
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
    target_onehot = np.zeros(MAX_ENTITIES, dtype=np.float32)
    if target is not None and target < MAX_ENTITIES:
        target_onehot[target] = 1.0
    return np.concatenate(
        [
            target_state.pos.astype(np.float32),
            target_state.quat.astype(np.float32),
            target_state.vel.astype(np.float32),
            kind_onehot,
            params_vec,
            target_onehot,
        ]
    )


class DeltaMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_transitions(data_dir: str | Path) -> list[dict]:
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    rows: list[dict] = []
    for entry in index["entries"]:
        if entry["kind"] != "episode":
            continue
        sample = load_sample(data_dir / entry["path"])
        poses = np.asarray(sample["poses"], dtype=float)
        ids = [str(x) for x in sample["instance_ids"]]
        for t, action in enumerate(sample["actions"]):
            states = {
                ids[i]: EntityState(
                    poses[t, i, :3], np.zeros(3), poses[t, i, 3:7]
                )
                for i in range(len(ids))
            }
            feature = transition_feature(states, action, ids)
            target = action.get("target")
            if target is None:
                delta = np.zeros(4, dtype=np.float32)
            else:
                target = int(target)
                dyaw = yaw_from_quat(poses[t + 1, target, 3:7]) - yaw_from_quat(
                    poses[t, target, 3:7]
                )
                delta = np.concatenate(
                    [poses[t + 1, target, :3] - poses[t, target, :3], [dyaw]]
                ).astype(np.float32)
            rows.append(
                {
                    "episode": entry["id"],
                    "t": t,
                    "feature": feature,
                    "delta": delta,
                    "target": target,
                    "ids": ids,
                    "poses_t": poses[t].copy(),
                    "poses_next": poses[t + 1].copy(),
                    "action": action,
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


def _apply_delta(poses: np.ndarray, target: int, delta) -> np.ndarray:
    out = poses.copy()
    out[target, :3] += np.asarray(delta[:3], dtype=float)
    yaw = yaw_from_quat(out[target, 3:7]) + float(delta[3])
    out[target, 3:7] = quat_from_yaw(yaw)
    return out


def _error(pred: np.ndarray, truth: np.ndarray) -> dict:
    pos = float(np.mean(np.linalg.norm(pred[:, :3] - truth[:, :3], axis=1)))
    yaw = float(
        np.mean(
            [
                abs(np.arctan2(np.sin(a - b), np.cos(a - b)))
                for a, b in zip(
                    [yaw_from_quat(q) for q in pred[:, 3:7]],
                    [yaw_from_quat(q) for q in truth[:, 3:7]],
                )
            ]
        )
    )
    return {"pos": pos, "yaw": yaw}


def evaluate_models(model: DeltaMLP, rows: list[dict], device: str = "cpu") -> dict:
    model.eval()
    one_step = {"zero": [], "analytic": [], "learned": []}
    analytic = AnalyticWorldModel()
    with torch.no_grad():
        for row in rows:
            ids = row["ids"]
            poses = row["poses_t"]
            action = row["action"]
            target = row["target"]
            states = {
                ids[i]: EntityState(poses[i, :3], np.zeros(3), poses[i, 3:7])
                for i in range(len(ids))
            }
            # zero baseline
            one_step["zero"].append(
                float(
                    np.mean(
                        np.linalg.norm(
                            poses[:, :3] - row["poses_next"][:, :3], axis=1
                        )
                    )
                )
            )
            # analytic
            if target is not None:
                pred_states = analytic.predict(
                    {k: v for k, v in states.items()}, {**action, "target": ids[int(target)]}
                )
                pred = poses.copy()
                pred[int(target), :3] = pred_states[ids[int(target)]].pos
                one_step["analytic"].append(_error(pred, row["poses_next"])["pos"])
            else:
                one_step["analytic"].append(_error(poses, row["poses_next"])["pos"])
            # learned
            feat = torch.as_tensor(row["feature"], dtype=torch.float32, device=device).unsqueeze(0)
            delta = model(feat).cpu().numpy()[0]
            pred = poses if target is None else _apply_delta(poses, int(target), delta)
            one_step["learned"].append(_error(pred, row["poses_next"])["pos"])

    rollout = {"zero": [], "analytic": [], "learned": []}
    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    for episode_rows in by_episode.values():
        episode_rows = sorted(episode_rows, key=lambda r: r["t"])
        poses = episode_rows[0]["poses_t"].copy()
        ids = episode_rows[0]["ids"]
        pred_learned = poses.copy()
        pred_analytic = poses.copy()
        with torch.no_grad():
            for row in episode_rows:
                target = row["target"]
                if target is None:
                    continue
                feat = torch.as_tensor(row["feature"], dtype=torch.float32, device=device).unsqueeze(0)
                delta = model(feat).cpu().numpy()[0]
                pred_learned = _apply_delta(pred_learned, int(target), delta)
                states = {
                    ids[i]: EntityState(
                        pred_analytic[i, :3], np.zeros(3), pred_analytic[i, 3:7]
                    )
                    for i in range(len(ids))
                }
                pred_states = analytic.predict(
                    states, {**row["action"], "target": ids[int(target)]}
                )
                pred_analytic[int(target), :3] = pred_states[ids[int(target)]].pos
        truth = episode_rows[-1]["poses_next"]
        rollout["zero"].append(_error(poses, truth)["pos"])
        rollout["analytic"].append(_error(pred_analytic, truth)["pos"])
        rollout["learned"].append(_error(pred_learned, truth)["pos"])

    return {
        "one_step_pos_error": {k: float(np.mean(v)) if v else 0.0 for k, v in one_step.items()},
        "rollout_pos_error": {k: float(np.mean(v)) if v else 0.0 for k, v in rollout.items()},
    }


def train_world_model(
    data_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 100,
    hidden: int = 64,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    rows = build_transitions(data_dir)
    train_rows, val_rows = _split_episodes(rows, seed)
    in_dim = len(rows[0]["feature"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeltaMLP(in_dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.as_tensor(np.stack([r["feature"] for r in train_rows]), dtype=torch.float32)
    Y = torch.as_tensor(np.stack([r["delta"] for r in train_rows]), dtype=torch.float32)
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for start in range(0, len(X), 32):
            idx = perm[start : start + 32]
            xb, yb = X[idx].to(device), Y[idx].to(device)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    eval_metrics = evaluate_models(model, val_rows, device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "in_dim": in_dim, "hidden": hidden},
        out / "world_model.pt",
    )
    metrics = {
        "data_dir": str(data_dir),
        "device": device,
        "epochs": int(epochs),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "in_dim": in_dim,
        **eval_metrics,
        "note": "Tiny episode set; smoke test, not a dynamics claim.",
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def load_world_model(path: str | Path, device: str = "cpu") -> DeltaMLP:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = DeltaMLP(int(payload["in_dim"]), int(payload["hidden"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
