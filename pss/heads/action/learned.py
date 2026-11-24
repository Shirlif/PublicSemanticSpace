"""Learned action head (Phase 8 closure).

Behavior cloning from the rule-based push controller. The learned head keeps
the same ``propose_push_to`` interface, so the closed-loop evaluator can swap
it in without changing the executor.

Feature: [pos_x, pos_y, goal_x, goal_y, dx, dy, distance]
Target:  [direction_x, direction_y, force]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss.sim.scene import SceneConfig, build_scene

from . import ActionHead, ActionProposal

FEATURE_DIM = 7


class PushPolicyMLP(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_scene(start, seed: int, trial: int):
    cfg = SceneConfig(n_objects=1, seed=seed + 100 + trial, width=64, height=64, bins=0)
    scene = build_scene(cfg, categories=["block"], instance_indices=[trial % 3])
    obj = scene.objects[0]
    scene.set_object_pose(0, [start[0], start[1], obj.rest_height + 0.002])
    scene.step(10)
    scene.model.geom_friction[scene.model.geom(obj.geom_name).id] = [0.2, 0.01, 0.001]
    return scene, obj


def run_push_trial(
    head,
    start,
    goal,
    seed: int,
    trial: int,
    collect: bool = False,
    max_pushes: int = 20,
    success_tolerance: float = 0.06,
) -> dict:
    scene, obj = _make_scene(np.asarray(start, dtype=float), seed, trial)
    body_id = scene.model.body(obj.body_name).id
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    pushes = 0
    for _ in range(max_pushes):
        pos, _ = scene.get_object_pose(0)
        if float(np.linalg.norm(pos[:2] - goal)) < success_tolerance:
            break
        proposal = head.propose_push_to(pos[:2], goal, target=0)
        if proposal is None:
            break
        direction = np.asarray(proposal.params["direction"], dtype=float)
        force = float(proposal.params["force"])
        if collect:
            delta = np.asarray(goal, dtype=float) - pos[:2]
            dist = float(np.linalg.norm(delta))
            samples.append(
                (
                    np.array(
                        [pos[0], pos[1], goal[0], goal[1], delta[0], delta[1], dist],
                        dtype=np.float32,
                    ),
                    np.array([direction[0], direction[1], force], dtype=np.float32),
                )
            )
        scene.data.xfrc_applied[body_id, :3] = [
            direction[0] * force,
            direction[1] * force,
            0.0,
        ]
        scene.step(8)
        scene.data.xfrc_applied[body_id, :3] = 0.0
        scene.step(12)
        pushes += 1
    pos, _ = scene.get_object_pose(0)
    return {
        "final_distance": float(np.linalg.norm(pos[:2] - goal)),
        "pushes": pushes,
        "samples": samples,
        "success": float(np.linalg.norm(pos[:2] - goal)) < success_tolerance,
    }


def collect_demonstrations(n_demos: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rule = ActionHead(gain=15.0, max_force=10.0, min_force=1.0, tolerance=0.03)
    X, Y = [], []
    for i in range(n_demos):
        start = np.array([rng.uniform(-0.30, -0.05), rng.uniform(-0.15, 0.15)])
        goal = np.array([0.25, rng.uniform(-0.12, 0.12)])
        result = run_push_trial(rule, start, goal, seed, i, collect=True)
        for x, y in result["samples"]:
            X.append(x)
            Y.append(y)
    return np.stack(X), np.stack(Y)


class LearnedActionHead:
    def __init__(
        self,
        model: PushPolicyMLP,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        min_force: float = 0.5,
        max_force: float = 5.0,
        tolerance: float = 0.03,
    ):
        self.model = model
        self.feat_mean = np.asarray(feat_mean, dtype=np.float32)
        self.feat_std = np.asarray(feat_std, dtype=np.float32)
        self.min_force = float(min_force)
        self.max_force = float(max_force)
        self.tolerance = float(tolerance)

    def propose_push_to(self, pos_xy, goal_xy, target: int = 0) -> ActionProposal | None:
        pos = np.asarray(pos_xy, dtype=float)
        goal = np.asarray(goal_xy, dtype=float)
        delta = goal - pos
        dist = float(np.linalg.norm(delta))
        if dist < self.tolerance:
            return None
        feat = np.array(
            [pos[0], pos[1], goal[0], goal[1], delta[0], delta[1], dist],
            dtype=np.float32,
        )
        x = (feat - self.feat_mean) / np.maximum(self.feat_std, 1e-6)
        with torch.no_grad():
            out = self.model(torch.as_tensor(x).unsqueeze(0)).cpu().numpy()[0]
        direction = out[:2]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = delta / max(dist, 1e-9)
        else:
            direction = direction / norm
        force = float(np.clip(out[2], self.min_force, self.max_force))
        return ActionProposal(
            kind="push",
            target=int(target),
            params={"force": force, "direction": direction.tolist()},
            confidence=float(min(1.0, dist / 0.2)),
        )


def train_learned_action_head(
    out_dir: str | Path,
    n_demos: int = 60,
    epochs: int = 200,
    hidden: int = 64,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    X, Y = collect_demonstrations(n_demos, seed)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std
    model = PushPolicyMLP(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.as_tensor(Xn, dtype=torch.float32)
    yt = torch.as_tensor(Y, dtype=torch.float32)
    losses = []
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        epoch_loss = 0.0
        for start in range(0, len(xt), 64):
            idx = perm[start : start + 64]
            pred = model(xt[idx])
            loss = F.mse_loss(pred, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(idx)
        losses.append(epoch_loss / max(1, len(xt)))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": hidden,
            "feat_mean": mean,
            "feat_std": std,
        },
        out / "action_head.pt",
    )
    metrics = {
        "n_demos": int(n_demos),
        "n_samples": int(len(X)),
        "epochs": int(epochs),
        "final_loss": float(losses[-1]) if losses else 0.0,
        "feat_mean": mean.tolist(),
        "feat_std": std.tolist(),
        "note": "Behavior cloning from the rule controller; same proposal interface.",
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def load_learned_action_head(
    path: str | Path,
    device: str = "cpu",
    min_force: float = 0.5,
    max_force: float = 5.0,
) -> LearnedActionHead:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = PushPolicyMLP(int(payload["hidden"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return LearnedActionHead(
        model,
        np.asarray(payload["feat_mean"], dtype=np.float32),
        np.asarray(payload["feat_std"], dtype=np.float32),
        min_force=min_force,
        max_force=max_force,
    )


def evaluate_head(
    head,
    n_trials: int = 10,
    seed: int = 0,
    success_tolerance: float = 0.06,
) -> dict:
    rng = np.random.default_rng(seed)
    finals, pushes, successes = [], [], 0
    for i in range(n_trials):
        start = np.array([rng.uniform(-0.30, -0.05), rng.uniform(-0.15, 0.15)])
        goal = np.array([0.25, rng.uniform(-0.12, 0.12)])
        result = run_push_trial(
            head,
            start,
            goal,
            seed,
            i,
            max_pushes=20,
            success_tolerance=success_tolerance,
        )
        finals.append(result["final_distance"])
        pushes.append(result["pushes"])
        successes += int(result["success"])
    return {
        "trials": n_trials,
        "success_rate": successes / max(1, n_trials),
        "mean_final_distance": float(np.mean(finals)) if finals else 0.0,
        "mean_pushes": float(np.mean(pushes)) if pushes else 0.0,
        "final_distances": finals,
    }
