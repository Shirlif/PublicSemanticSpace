"""Tactile encoder pretraining on WT-MM (12x32 contact arrays).

Objectives:
1. temporal contrastive: adjacent frames in the same episode are positives;
2. reconstruction: a linear decoder must recover the 12x32 map;
3. optional video alignment: a frozen tactile encoder must retrieve the
   synchronized camera frame.

This is B-class pretraining (no object 6D pose / instance IDs).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss.rd2.public.lerobot import LeRobotDataset
from pss.rd2.public.smoke import decode_video

INPUT_SHAPE = (12, 32)


class TactileCNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if tuple(x.shape[-2:]) != INPUT_SHAPE:
            x = F.interpolate(x, size=INPUT_SHAPE, mode="bilinear", align_corners=False)
        return self.proj(self.net(x).flatten(1))


class TactileDecoder(nn.Module):
    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, INPUT_SHAPE[0] * INPUT_SHAPE[1]),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, *INPUT_SHAPE)


def load_wtmm_frames(pair_root: str | Path, extra_root: str | Path | None = None):
    roots = [Path(pair_root)]
    if extra_root is not None:
        roots.append(Path(extra_root))
    tactiles, episodes, frame_ids = [], [], []
    for root in roots:
        if not (root / "meta" / "info.json").exists():
            continue
        ds = LeRobotDataset(root, repo_id="WT-MM/lerobot_tactile")
        tactile, df = ds.tactile_matrix()
        tactiles.append(tactile)
        episodes.append(df["episode_index"].to_numpy())
        frame_ids.append(df["frame_index"].to_numpy())
    if not tactiles:
        raise FileNotFoundError("no WT-MM parquet data found")
    tactile = np.concatenate(tactiles).astype(np.float32)
    episodes = np.concatenate(episodes).astype(np.int64)
    frame_ids = np.concatenate(frame_ids).astype(np.int64)
    # Deduplicate frames (the pair root may overlap with the extra root).
    seen = set()
    keep = []
    for i, (ep, fr) in enumerate(zip(episodes, frame_ids)):
        key = (int(ep), int(fr))
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return tactile[keep], episodes[keep], frame_ids[keep]


def _episode_index(episodes: np.ndarray, frames: np.ndarray) -> dict[int, dict[int, int]]:
    out: dict[int, dict[int, int]] = {}
    for i, (ep, fr) in enumerate(zip(episodes, frames)):
        out.setdefault(int(ep), {})[int(fr)] = i
    return out


def _temporal_pairs(ep_map: dict[int, dict[int, int]], rng, n: int):
    valid = [ep for ep, frames in ep_map.items() if len(frames) > 1]
    anchors, positives = [], []
    for _ in range(n):
        ep = valid[int(rng.integers(0, len(valid)))]
        frames = sorted(ep_map[ep])
        k = int(rng.integers(0, len(frames) - 1))
        anchors.append(ep_map[ep][frames[k]])
        positives.append(ep_map[ep][frames[k + 1]])
    return np.asarray(anchors), np.asarray(positives)


def temporal_retrieval(features: np.ndarray, episodes, frames, holdout: int) -> dict:
    idx = [i for i in range(len(episodes)) if int(episodes[i]) == holdout]
    fmap = {int(frames[i]): i for i in idx}
    anchors, positives = [], []
    for f, i in fmap.items():
        if f + 1 in fmap:
            anchors.append(i)
            positives.append(fmap[f + 1])
    if len(anchors) < 2:
        return {"r1": 0.0, "chance": 0.0, "pairs": len(anchors)}
    za = features[anchors]
    zp = features[positives]
    za = za / (np.linalg.norm(za, axis=1, keepdims=True) + 1e-9)
    zp = zp / (np.linalg.norm(zp, axis=1, keepdims=True) + 1e-9)
    pred = (za @ zp.T).argmax(axis=1)
    correct = int((pred == np.arange(len(anchors))).sum())
    top5 = np.argsort(-(za @ zp.T), axis=1)[:, :5]
    top5_correct = int(
        sum(int(k in top5[k]) for k in range(len(anchors)))
    )
    return {
        "r1": correct / len(anchors),
        "top5": top5_correct / len(anchors),
        "chance": 1.0 / len(anchors),
        "pairs": len(anchors),
    }


def contact_probe(features, tactile, episodes, holdout: int) -> dict:
    from sklearn.linear_model import LogisticRegression

    train = np.where(episodes != holdout)[0]
    test = np.where(episodes == holdout)[0]
    labels = (tactile.reshape(len(tactile), -1).max(axis=1) > 0).astype(int)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(features[train], labels[train])
    holdout_acc = float(clf.score(features[test], labels[test]))

    # Random balanced split: a feature-quality check not confounded by episode shift.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(features))
    cut = int(len(perm) * 0.8)
    tr, te = perm[:cut], perm[cut:]
    clf2 = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf2.fit(features[tr], labels[tr])
    random_acc = float(clf2.score(features[te], labels[te]))
    return {
        "episode_holdout_acc": holdout_acc,
        "random_balanced_acc": random_acc,
        "test_positive_rate": float(labels[test].mean()),
        "train_frames": int(len(train)),
        "test_frames": int(len(test)),
    }


def video_alignment(features, tactile, video_path, epochs: int = 20) -> dict:
    video = decode_video(video_path, max_frames=len(tactile))
    n = min(len(video), len(tactile))
    if n < 8:
        return {"frames": int(n), "r1": 0.0, "r1_shuffled": 0.0, "chance": 0.0}
    v = video[:n].astype(np.float32)
    t = features[:n].astype(np.float32)
    v = (v - v.mean(0)) / (v.std(0) + 1e-6)
    t = (t - t.mean(0)) / (t.std(0) + 1e-6)
    V = torch.as_tensor(v)
    T = torch.as_tensor(t)
    vp = nn.Linear(v.shape[1], 64)
    tp = nn.Linear(t.shape[1], 64)
    opt = torch.optim.Adam(list(vp.parameters()) + list(tp.parameters()), lr=1e-3)
    labels = torch.arange(n)
    for _epoch in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, 128):
            idx = order[start : start + 128]
            if len(idx) < 2:
                continue
            zv = F.normalize(vp(V[idx]), dim=-1)
            zt = F.normalize(tp(T[idx]), dim=-1)
            logits = zv @ zt.T / 0.1
            lab = torch.arange(len(idx))
            loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()

    def r1(zv, zt):
        with torch.no_grad():
            pred = (F.normalize(zt, dim=-1) @ F.normalize(zv, dim=-1).T).argmax(1)
            return float((pred == labels).float().mean().item())

    with torch.no_grad():
        aligned = r1(vp(V), tp(T))
        order = torch.randperm(n)
        shuffled = r1(vp(V), tp(T[order]))
    return {
        "frames": int(n),
        "r1": aligned,
        "r1_shuffled": shuffled,
        "chance": 1.0 / n,
    }


def train_wtmm_tactile(
    pair_root: str | Path,
    extra_root: str | Path | None,
    out_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 128,
    out_dim: int = 64,
    seed: int = 0,
    video_path: str | Path | None = None,
) -> dict:
    torch.manual_seed(seed)
    tactile, episodes, frames = load_wtmm_frames(pair_root, extra_root)
    holdout = int(max(episodes))
    ep_map = _episode_index(episodes, frames)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = TactileCNNEncoder(out_dim).to(device)
    decoder = TactileDecoder(out_dim).to(device)
    contact_head = nn.Linear(out_dim, 1).to(device)
    opt = torch.optim.Adam(
        list(encoder.parameters())
        + list(decoder.parameters())
        + list(contact_head.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )
    train_idx = np.where(episodes != holdout)[0]
    train_map = {ep: frames for ep, frames in ep_map.items() if ep != holdout}
    rng = np.random.default_rng(seed)
    T = torch.as_tensor(tactile)

    for _epoch in range(epochs):
        encoder.train()
        decoder.train()
        anchors, positives = _temporal_pairs(train_map, rng, max(batch_size, len(train_idx)))
        order = rng.permutation(len(anchors))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            if len(idx) < 2:
                continue
            xa = T[anchors[idx]].to(device)
            xp = T[positives[idx]].to(device)
            za = encoder(xa)
            zp = encoder(xp)
            logits = za @ zp.T / 0.1
            labels = torch.arange(len(idx), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            rec = decoder(za)
            loss = loss + 0.1 * F.mse_loss(rec, xa)
            contact_labels = (xa.flatten(1).max(dim=1).values > 0).float()
            loss = loss + 0.2 * F.binary_cross_entropy_with_logits(
                contact_head(za).squeeze(1), contact_labels
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

    encoder.eval()
    with torch.no_grad():
        feats = encoder(T.to(device)).cpu().numpy()
        rec = decoder(encoder(T[train_idx].to(device))).cpu()
        rec_mse = float(F.mse_loss(rec, T[train_idx]).item())
    metrics = {
        "source": "WT-MM/lerobot_tactile",
        "license": "apache-2.0",
        "frames": int(len(tactile)),
        "episodes": sorted(int(x) for x in set(episodes.tolist())),
        "holdout_episode": holdout,
        "device": device,
        "epochs": int(epochs),
        "out_dim": int(out_dim),
        "input_shape": list(INPUT_SHAPE),
        "reconstruction_mse": rec_mse,
        "contact_probe": contact_probe(feats, tactile, episodes, holdout),
        "temporal_retrieval": temporal_retrieval(feats, episodes, frames, holdout),
    }
    if video_path is not None and Path(video_path).exists():
        pair_ds = LeRobotDataset(pair_root)
        pair_tactile, _ = pair_ds.tactile_matrix()
        pair_feats = feats[: len(pair_tactile)]
        metrics["video_tactile_alignment"] = video_alignment(
            pair_feats, pair_tactile, video_path, epochs=20
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "out_dim": int(out_dim),
            "input_shape": list(INPUT_SHAPE),
            "source": "WT-MM/lerobot_tactile",
            "license": "apache-2.0",
        },
        out / "tactile_wtmm.pt",
    )
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def load_wtmm_tactile(path: str | Path, device: str = "cpu") -> TactileCNNEncoder:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    encoder = TactileCNNEncoder(int(payload["out_dim"])).to(device)
    encoder.load_state_dict(payload["state_dict"])
    encoder.eval()
    return encoder


@dataclass
class WTMMTactileExpert:
    """Adapter so the WT-MM encoder can replace the synthetic tactile expert."""

    encoder: TactileCNNEncoder
    device: str = "cpu"
    input_dim: int = 64

    def encode(self, tactile) -> np.ndarray:
        if isinstance(tactile, torch.Tensor):
            x = tactile.to(self.device, dtype=torch.float32)
        else:
            x = torch.as_tensor(
                np.asarray(tactile, dtype=np.float32), device=self.device
            )
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            if tuple(x.shape) in ((8, 8), (12, 32)):
                x = x.unsqueeze(0)
            elif int(x.shape[1]) == 64:
                x = x.reshape(-1, 8, 8)
            elif int(x.shape[1]) == 384:
                x = x.reshape(-1, 12, 32)
            else:
                raise ValueError(f"unsupported tactile input dim {int(x.shape[1])}")
        with torch.no_grad():
            return self.encoder(x).cpu().numpy()
