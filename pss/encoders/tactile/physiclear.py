"""PhysiCLeAR tactile pipeline (R2).

The Octopi repository ships 476 real GelSight ``.mov`` clips under
``dataset/`` plus hardness/roughness/texture ranks in ``utils/constants.py``.
This module:

1. extracts a fixed number of RGB frames per video into a local cache;
2. trains a small 2D CNN with temporal contrastive + reconstruction +
   optional property classification;
3. saves a checkpoint compatible with the ``tactile_real_smallcnn_v0``
   ExpertSpec.

The encoder learns real tactile features, but the D-group probe still runs on
the synthetic 8x8 contact grid. That domain bridge is deliberate and must be
reported as such.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PHYSICLEAR_PREFIX = "physiclear_"
PROPERTY_NAMES = ("hardness", "roughness", "texture")
DEFAULT_FRAMES = 5
DEFAULT_SIZE = 64


# ── data preparation ────────────────────────────────────────────────────────


def object_name_from_video(path: str | Path) -> str:
    return re.sub(r"_\d+$", "", Path(path).stem)


def load_physiclear_constants(repo_root: str | Path) -> dict:
    """Parse Octopi's constants without executing the untrusted file."""
    path = Path(repo_root) / "utils" / "constants.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
    required = [
        "TRAIN_OBJECTS",
        "VAL_OBJECTS",
        "TEST_OBJECTS",
        "HARDNESS_RANK",
        "ROUGHNESS_RANK",
        "TEXTURE_RANK",
    ]
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"PhysiCLeAR constants missing: {missing}")
    return values


def extract_clip(
    video_path: str | Path, num_frames: int = DEFAULT_FRAMES, size: int = DEFAULT_SIZE
) -> np.ndarray | None:
    """Decode a video and return ``(L, size, size, 3)`` uint8 RGB frames."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return None
    indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    out = []
    for idx in indices:
        rgb = cv2.cvtColor(frames[int(idx)], cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        out.append(rgb)
    return np.stack(out, axis=0)


def prepare_physiclear(
    video_dir: str | Path,
    repo_root: str | Path,
    out_dir: str | Path,
    num_frames: int = DEFAULT_FRAMES,
    size: int = DEFAULT_SIZE,
    force: bool = False,
) -> dict:
    """Extract frames and write ``index.json`` + ``DATA_CARD.md``."""
    video_dir = Path(video_dir)
    out_dir = Path(out_dir)
    index_path = out_dir / "index.json"
    if index_path.exists() and not force:
        return json.loads(index_path.read_text(encoding="utf-8"))

    constants = load_physiclear_constants(repo_root)
    train_objects = set(constants["TRAIN_OBJECTS"])
    val_objects = set(constants["VAL_OBJECTS"])
    test_objects = set(constants["TEST_OBJECTS"])
    ranks = {
        "hardness": constants["HARDNESS_RANK"],
        "roughness": constants["ROUGHNESS_RANK"],
        "texture": constants["TEXTURE_RANK"],
    }

    entries: list[dict] = []
    videos = sorted(
        path for path in video_dir.iterdir() if path.suffix.lower() in {".mov", ".avi", ".mp4"}
    )
    for i, video_path in enumerate(videos):
        key = PHYSICLEAR_PREFIX + object_name_from_video(video_path)
        if key in train_objects:
            split = "train"
        elif key in val_objects:
            split = "val"
        elif key in test_objects:
            split = "test"
        else:
            continue  # 13 unlisted objects are not part of the official split
        clip = extract_clip(video_path, num_frames=num_frames, size=size)
        if clip is None:
            continue
        frame_dir = out_dir / "frames" / key
        frame_dir.mkdir(parents=True, exist_ok=True)
        relative_frames = []
        for frame_idx, frame in enumerate(clip):
            relative = Path("frames") / key / f"{video_path.stem}_{frame_idx:02d}.jpg"
            Image.fromarray(frame).save(out_dir / relative, quality=92)
            relative_frames.append(str(relative))
        entries.append(
            {
                "id": f"{key}_{video_path.stem}",
                "object": key,
                "video": str(video_path),
                "split": split,
                "frames": relative_frames,
                "hardness": int(ranks["hardness"].get(key, -1)),
                "roughness": int(ranks["roughness"].get(key, -1)),
                "texture": int(ranks["texture"].get(key, -1)),
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  prepared {i + 1}/{len(videos)} videos")

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(entry["split"] for entry in entries)
    payload = {
        "version": "v1",
        "source": "PhysiCLeAR (Octopi repository dataset/*.mov)",
        "license": "unknown",
        "config": {
            "frames": int(num_frames),
            "size": int(size),
            "repo_root": str(repo_root),
        },
        "counts": dict(sorted(counts.items())),
        "entries": entries,
    }
    index_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "DATA_CARD.md").write_text(
        "\n".join(
            [
                "# PhysiCLeAR R2 cache",
                "",
                f"- source: `{video_dir}`",
                f"- clips: {len(entries)} (train/val/test = "
                f"{counts.get('train', 0)}/{counts.get('val', 0)}/{counts.get('test', 0)})",
                f"- frames per clip: {num_frames}, size: {size}x{size}",
                "- labels: hardness / roughness / texture ranks (0..2)",
                "- caveat: real GelSight frames; D-group probes on synthetic 8x8 "
                "contact grids, so this is a deliberate domain bridge.",
                "- license: unknown; check PhysiCLeAR/Octopi terms before redistribution.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


# ── dataset ─────────────────────────────────────────────────────────────────


class PhysiCLeARClipDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        entries: list[dict],
        random_flip: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.entries = entries
        self.random_flip = random_flip

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        frames = [
            np.asarray(Image.open(self.data_dir / path).convert("RGB"))
            for path in entry["frames"]
        ]
        array = np.stack(frames).astype(np.float32) / 255.0  # (L, H, W, 3)
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
        if self.random_flip and torch.rand(1).item() < 0.5:
            tensor = torch.flip(tensor, dims=[3])
        labels = torch.tensor(
            [int(entry.get(name, -1)) for name in PROPERTY_NAMES], dtype=torch.long
        )
        return tensor, labels


# ── model ───────────────────────────────────────────────────────────────────


class RealTactileEncoder(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x).flatten(1))


class RealTactileDecoder(nn.Module):
    def __init__(self, out_dim: int = 128, size: int = 64) -> None:
        super().__init__()
        base = max(1, int(size) // 4)
        self.net = nn.Sequential(
            nn.Linear(out_dim, 128 * base * base),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, base, base)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _property_heads(out_dim: int, constants: dict) -> nn.ModuleDict:
    ranks = {
        "hardness": constants["HARDNESS_RANK"],
        "roughness": constants["ROUGHNESS_RANK"],
        "texture": constants["TEXTURE_RANK"],
    }
    heads = {}
    for name in PROPERTY_NAMES:
        classes = max(ranks[name].values()) + 1 if ranks[name] else 1
        heads[name] = nn.Linear(out_dim, classes)
    return nn.ModuleDict(heads)


# ── training ────────────────────────────────────────────────────────────────


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _epoch_loss(
    encoder,
    decoder,
    heads,
    loader,
    device,
    optimizer=None,
    temp: float = 0.1,
    recon_weight: float = 0.1,
    prop_weight: float = 0.2,
) -> dict:
    training = optimizer is not None
    encoder.train(training)
    decoder.train(training)
    heads.train(training)
    totals = {"loss": 0.0, "contrastive": 0.0, "reconstruction": 0.0, "property": 0.0}
    correct = {name: 0 for name in PROPERTY_NAMES}
    counted = {name: 0 for name in PROPERTY_NAMES}
    batches = 0

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        batch, length, channels, height, width = frames.shape
        flat = frames.reshape(batch * length, channels, height, width)
        raw = encoder(flat)
        normalized = F.normalize(raw, dim=-1)
        logits = normalized @ normalized.T / temp
        targets = torch.arange(batch, device=device).repeat_interleave(length)
        contrastive = (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
        ) / 2
        reconstruction = F.mse_loss(decoder(raw), flat)
        clip_features = raw.reshape(batch, length, -1).mean(dim=1)
        property_loss = torch.zeros((), device=device)
        for idx, name in enumerate(PROPERTY_NAMES):
            valid = labels[:, idx] >= 0
            if valid.any():
                out = heads[name](clip_features[valid])
                target = labels[valid, idx]
                property_loss = property_loss + F.cross_entropy(out, target)
                if not training:
                    correct[name] += int((out.argmax(1) == target).sum().item())
                    counted[name] += int(valid.sum().item())
        loss = (
            contrastive
            + recon_weight * reconstruction
            + prop_weight * property_loss
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        totals["loss"] += float(loss.item())
        totals["contrastive"] += float(contrastive.item())
        totals["reconstruction"] += float(reconstruction.item())
        totals["property"] += float(property_loss.item())
        batches += 1

    metrics = {key: value / max(1, batches) for key, value in totals.items()}
    metrics["property_acc"] = {
        name: correct[name] / counted[name] if counted[name] else float("nan")
        for name in PROPERTY_NAMES
    }
    return metrics


def train_physiclear_encoder(
    data_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    out_dim: int = 128,
    seed: int = 0,
    device: str = "auto",
    temp: float = 0.1,
    recon_weight: float = 0.1,
    prop_weight: float = 0.2,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    data_dir = Path(data_dir)
    index = json.loads((data_dir / "index.json").read_text(encoding="utf-8"))
    entries = index["entries"]
    train_entries = [e for e in entries if e["split"] == "train"]
    val_entries = [e for e in entries if e["split"] == "val"]
    if not train_entries:
        raise ValueError("no train entries in PhysiCLeAR index")
    if not val_entries:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(train_entries))
        cut = max(1, int(len(train_entries) * 0.1))
        val_entries = [train_entries[i] for i in order[:cut]]

    train_loader = DataLoader(
        PhysiCLeARClipDataset(data_dir, train_entries, random_flip=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        PhysiCLeARClipDataset(data_dir, val_entries, random_flip=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    resolved = _resolve_device(device)
    size = int(index["config"]["size"])
    encoder = RealTactileEncoder(out_dim).to(resolved)
    decoder = RealTactileDecoder(out_dim, size).to(resolved)
    constants = load_physiclear_constants(
        index["config"].get("repo_root", "_external/octopi")
    )
    heads = _property_heads(out_dim, constants).to(resolved)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()) + list(heads.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    best_val = float("inf")
    best_epoch = -1
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch_loss(
            encoder, decoder, heads, train_loader, resolved, optimizer,
            temp=temp, recon_weight=recon_weight, prop_weight=prop_weight,
        )
        with torch.no_grad():
            val_metrics = _epoch_loss(
                encoder, decoder, heads, val_loader, resolved, None,
                temp=temp, recon_weight=recon_weight, prop_weight=prop_weight,
            )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": encoder.state_dict(),
                    "out_dim": int(out_dim),
                    "input_shape": [3, size, size],
                    "source": "PhysiCLeAR (Octopi repository dataset/*.mov)",
                    "license": "unknown",
                    "preprocess": "physiclear_frame_v1",
                    "frames": int(index["config"]["frames"]),
                },
                out / "tactile_real.pt",
            )
        print(
            f"epoch {epoch:02d}/{epochs} "
            f"train={train_metrics['loss']:.4f} val={val_metrics['loss']:.4f} "
            f"prop_acc={val_metrics['property_acc']}"
        )

    metrics = {
        "data": str(data_dir),
        "device": resolved,
        "clips": {"train": len(train_entries), "val": len(val_entries)},
        "frames": int(index["config"]["frames"]),
        "size": int(index["config"]["size"]),
        "out_dim": int(out_dim),
        "epochs": int(epochs),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "source": "PhysiCLeAR",
        "license": "unknown",
        "history": history,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def load_physiclear_encoder(path: str | Path, device: str = "cpu") -> RealTactileEncoder:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    encoder = RealTactileEncoder(int(payload["out_dim"])).to(device)
    encoder.load_state_dict(payload["state_dict"])
    encoder.eval()
    return encoder
