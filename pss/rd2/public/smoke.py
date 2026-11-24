"""Smoke tests for public tactile / visuo-tactile data.

These are pipeline checks, not capability claims: a small autoencoder and a
frame-level alignment check on a downloaded shard.
"""

from __future__ import annotations

import numpy as np


def tactile_stats(tactile: np.ndarray) -> dict:
    flat = tactile.reshape(len(tactile), -1)
    contact = flat.max(axis=1) > 0
    return {
        "frames": int(len(tactile)),
        "shape": list(tactile.shape[1:]),
        "min": float(tactile.min()),
        "max": float(tactile.max()),
        "mean": float(tactile.mean()),
        "std": float(tactile.std()),
        "contact_frame_fraction": float(contact.mean()),
        "per_frame_variance_mean": float(flat.var(axis=1).mean()),
    }


def temporal_consistency(tactile: np.ndarray) -> dict:
    x = tactile.reshape(len(tactile), -1).astype(np.float64)
    norms = np.linalg.norm(x, axis=1)
    valid = norms > 0
    if valid.sum() < 3:
        return {"consecutive_cosine": 0.0, "random_cosine": 0.0, "gap": 0.0}
    x = x[valid]
    norms = norms[valid]

    def cos(a, b, nb):
        return float(np.dot(a, b) / (np.linalg.norm(a) * nb + 1e-12))

    consec = [
        cos(x[i], x[i + 1], norms[i + 1]) for i in range(len(x) - 1)
    ]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(x))
    random_pairs = [
        cos(x[i], x[j], norms[j]) for i, j in zip(idx[:-1], idx[1:])
    ]
    return {
        "consecutive_cosine": float(np.mean(consec)),
        "random_cosine": float(np.mean(random_pairs)),
        "gap": float(np.mean(consec) - np.mean(random_pairs)),
    }


def train_autoencoder(tactile: np.ndarray, epochs: int = 30, latent: int = 64, seed: int = 0) -> dict:
    import torch
    import torch.nn as nn

    x = tactile.reshape(len(tactile), -1).astype(np.float32)
    if x.max() > 1.5:
        x = x / x.max()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    n_val = max(1, len(x) // 5)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    model = nn.Sequential(
        nn.Linear(x.shape[1], latent),
        nn.ReLU(),
        nn.Linear(latent, x.shape[1]),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.as_tensor(x)
    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_idx))
        for start in range(0, len(train_idx), 64):
            idx = torch.as_tensor(train_idx[order[start : start + 64].numpy()])
            xb = X[idx]
            recon = model(xb)
            loss = torch.nn.functional.mse_loss(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        val = X[torch.as_tensor(val_idx)]
        val_mse = torch.nn.functional.mse_loss(model(val), val).item()
        z = model[:2](X[torch.as_tensor(train_idx)])
        latent_std = float(z.std(dim=0).mean().item())
    return {
        "input_dim": int(x.shape[1]),
        "latent_dim": int(latent),
        "val_mse": float(val_mse),
        "latent_std": latent_std,
        "epochs": int(epochs),
    }


def decode_video(path, max_frames: int = 2000, size: int = 32) -> np.ndarray:
    import av
    import cv2

    container = av.open(str(path))
    frames = []
    try:
        for frame in container.decode(video=0):
            try:
                image = frame.to_ndarray(format="gray")
            except Exception:
                rgb = frame.to_ndarray(format="rgb24")
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, (size, size)).astype(np.float32) / 255.0
            frames.append(image.reshape(-1))
            if len(frames) >= max_frames:
                break
    finally:
        container.close()
    return np.stack(frames) if frames else np.zeros((0, size * size), dtype=np.float32)


def video_tactile_alignment(video_features: np.ndarray, tactile: np.ndarray, epochs: int = 30) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    n = min(len(video_features), len(tactile))
    if n < 8:
        return {"frames": int(n), "r1": 0.0, "r1_shuffled": 0.0, "chance": 0.0}
    v = video_features[:n].astype(np.float32)
    t = tactile[:n].reshape(n, -1).astype(np.float32)
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
        for start in range(0, n, 64):
            idx = order[start : start + 64]
            if len(idx) < 2:
                continue
            zv = F.normalize(vp(V[idx]), dim=-1)
            zt = F.normalize(tp(T[idx]), dim=-1)
            logits = zv @ zt.T / 0.1
            lab = torch.arange(len(idx))
            loss = (
                F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)
            ) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()

    def r1(video_proj, tactile_proj):
        with torch.no_grad():
            zv = F.normalize(video_proj(V), dim=-1)
            zt = F.normalize(tactile_proj(T), dim=-1)
            pred = (zt @ zv.T).argmax(dim=1)
            return float((pred == labels).float().mean().item())

    aligned = r1(vp, tp)
    shuffled = tactile[:n][torch.randperm(n).numpy()]
    T_shuffled = torch.as_tensor(
        (shuffled.reshape(n, -1).astype(np.float32) - t.mean(0)) / (t.std(0) + 1e-6)
    )
    with torch.no_grad():
        zv = F.normalize(vp(V), dim=-1)
        zt = F.normalize(tp(T_shuffled), dim=-1)
        pred = (zt @ zv.T).argmax(dim=1)
        shuffled_r1 = float((pred == labels).float().mean().item())
    return {
        "frames": int(n),
        "r1": aligned,
        "r1_shuffled": shuffled_r1,
        "chance": 1.0 / n,
    }


def run_smoke(dataset, limit_frames: int = 2000, video_path=None) -> dict:
    summary = dataset.summary(max_rows=limit_frames)
    tactile, df = dataset.tactile_matrix(max_rows=limit_frames)
    result = {
        "summary": summary,
        "tactile_stats": tactile_stats(tactile),
        "temporal": temporal_consistency(tactile),
        "autoencoder": train_autoencoder(tactile, epochs=30),
    }
    if video_path is not None:
        from pathlib import Path

        if Path(video_path).exists():
            video = decode_video(video_path, max_frames=limit_frames)
            result["video_tactile_alignment"] = video_tactile_alignment(video, tactile, epochs=30)
        else:
            result["video_tactile_alignment"] = {"error": f"missing {video_path}"}
    return result
