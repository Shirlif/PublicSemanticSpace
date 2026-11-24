"""Train a tactile encoder on ObjectFolder 2.0 and measure object-level R@1.

Vision anchors are frozen DINOv2 embeddings of the rendered object views.
A small CNN is trained on the rendered touch images with:

- object-level InfoNCE between tactile embeddings and the vision anchors;
- object classification as an auxiliary loss.

Two evaluations:
- seen objects: train and test on all objects (alignment upper bound);
- held-out objects: 5-fold object cross-validation (generalization to new objects).

Rendered from ObjectFile implicit representations; object-level, not synchronized
robot interaction. ObjectFolder 2.0 is CC-BY-4.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from pss.projection.losses import supcon_cross
from pss.rd2.objectfolder import _encode_paths, _l2, _load_encoder, _retrieval


class TactileEncoder(nn.Module):
    def __init__(self, out_dim: int = 384) -> None:
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
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(256, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return F.normalize(self.proj(self.dropout(h)), dim=-1)


class TouchDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], train: bool = True) -> None:
        self.paths = paths
        self.labels = labels
        self.train = train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB").resize((128, 96), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        if self.train and torch.rand(1).item() < 0.5:
            tensor = torch.flip(tensor, dims=[2])
        return tensor, self.labels[index]


def load_renders(root: str | Path) -> tuple[list[str], dict[str, list[Path]], dict[str, list[Path]]]:
    root = Path(root)
    objects = []
    vision: dict[str, list[Path]] = {}
    touch: dict[str, list[Path]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        vision_paths = sorted((path / "vision").glob("*.png"))
        touch_paths = sorted((path / "touch").glob("*.png"))
        if vision_paths and touch_paths:
            objects.append(path.name)
            vision[path.name] = vision_paths
            touch[path.name] = touch_paths
    return objects, vision, touch


def build_vision_anchors(
    objects: list[str], vision: dict[str, list[Path]], device: str
) -> np.ndarray:
    model = _load_encoder().to(device)
    anchors = []
    for name in objects:
        features = _encode_paths(model, vision[name], device)
        anchors.append(features.mean(0))
    return _l2(np.stack(anchors))


def _touch_paths(
    objects: list[str],
    touch: dict[str, list[Path]],
    object_indices: list[int],
    local_labels: bool,
) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    for local, obj_idx in enumerate(object_indices):
        label = local if local_labels else obj_idx
        for path in touch[objects[obj_idx]]:
            paths.append(path)
            labels.append(label)
    return paths, labels


def train_encoder(
    paths: list[Path],
    labels: list[int],
    anchors: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    temp: float,
    seed: int,
    device: str,
) -> TactileEncoder:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TactileEncoder(anchors.shape[1]).to(device)
    head = nn.Linear(anchors.shape[1], len(np.unique(labels))).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4
    )
    loader = DataLoader(
        TouchDataset(paths, labels, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    anchor_tensor = torch.as_tensor(anchors, dtype=torch.float32, device=device)
    for _epoch in range(epochs):
        model.train()
        head.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            z_t = model(x)
            z_v = anchor_tensor[y]
            loss = (
                supcon_cross(z_t, z_v, y, temp) + supcon_cross(z_v, z_t, y, temp)
            ) / 2.0
            loss = loss + 0.5 * F.cross_entropy(head(z_t), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def encode_touch(model: TactileEncoder, paths: list[Path], device: str) -> np.ndarray:
    dataset = TouchDataset(paths, [0] * len(paths), train=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    model.eval()
    out = []
    with torch.inference_mode():
        for x, _ in loader:
            out.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate_objects(
    model: TactileEncoder,
    objects: list[str],
    touch: dict[str, list[Path]],
    anchors: np.ndarray,
    object_indices: list[int],
    device: str,
) -> dict:
    tactile = []
    for idx in object_indices:
        features = encode_touch(model, touch[objects[idx]], device)
        tactile.append(features.mean(0))
    T = _l2(np.stack(tactile))
    V = anchors[object_indices]
    return {
        "vision_to_touch": _retrieval(V, T),
        "touch_to_vision": _retrieval(T, V),
    }


def _aggregate(folds: list[dict]) -> dict:
    out = {}
    for direction in ("vision_to_touch", "touch_to_vision"):
        for metric in ("r1", "r5"):
            values = [fold[direction][metric] for fold in folds]
            out[f"{direction}_{metric}"] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
    return out


def run_tactile_encoder(
    renders: str | Path,
    out_dir: str | Path,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    temp: float = 0.1,
    folds: int = 5,
    seed: int = 0,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    renders = Path(renders)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        Path(checkpoint_dir)
        if checkpoint_dir
        else Path("checkpoints/objectfolder_tactile")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    objects, vision, touch = load_renders(renders)
    if len(objects) < 10:
        raise SystemExit(f"need at least 10 rendered objects, found {len(objects)}")
    anchors = build_vision_anchors(objects, vision, device)

    all_indices = list(range(len(objects)))
    seen_paths, seen_labels = _touch_paths(objects, touch, all_indices, local_labels=False)
    seen_model = train_encoder(
        seen_paths, seen_labels, anchors, epochs, batch_size, lr, temp, seed, device
    )
    seen = evaluate_objects(seen_model, objects, touch, anchors, all_indices, device)
    torch.save(
        {
            "state_dict": seen_model.state_dict(),
            "out_dim": int(seen_model.out_dim),
            "input_shape": [3, 96, 128],
            "objects": objects,
        },
        out / "tactile_encoder.pt",
    )
    torch.save(
        {
            "state_dict": seen_model.state_dict(),
            "out_dim": int(seen_model.out_dim),
            "input_shape": [3, 96, 128],
            "objects": objects,
        },
        checkpoint_dir / "tactile_encoder.pt",
    )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(objects))
    fold_indices = np.array_split(order, folds)
    fold_metrics = []
    for fold_id, test_array in enumerate(fold_indices):
        test_idx = [int(i) for i in test_array]
        train_idx = [i for i in all_indices if i not in set(test_idx)]
        train_paths, train_labels = _touch_paths(
            objects, touch, train_idx, local_labels=True
        )
        train_anchors = anchors[train_idx]
        model = train_encoder(
            train_paths,
            train_labels,
            train_anchors,
            epochs,
            batch_size,
            lr,
            temp,
            seed + fold_id + 1,
            device,
        )
        metrics = evaluate_objects(model, objects, touch, anchors, test_idx, device)
        metrics["fold"] = fold_id
        metrics["test_objects"] = test_idx
        fold_metrics.append(metrics)
        print(
            f"[fold {fold_id}] vision->touch R@1={metrics['vision_to_touch']['r1']:.3f} "
            f"touch->vision R@1={metrics['touch_to_vision']['r1']:.3f}"
        )

    results = {
        "renders": str(renders),
        "objects": len(objects),
        "touch_images": sum(len(touch[name]) for name in objects),
        "vision_images": sum(len(vision[name]) for name in objects),
        "device": device,
        "epochs": int(epochs),
        "folds": int(folds),
        "seen": seen,
        "heldout": _aggregate(fold_metrics),
        "folds_raw": fold_metrics,
        "license": "CC-BY-4.0",
        "citation": "Gao et al., ObjectFolder 2.0, CVPR 2022",
        "baseline": None,
        "note": (
            "Seen = train and test on all objects (alignment upper bound). "
            "Held-out = 5-fold object cross-validation. Rendered, object-level, "
            "not synchronized robot interaction."
        ),
    }
    baseline_path = Path("eval/results/objectfolder/objectfolder_r1.json")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        results["baseline"] = {
            "zero_shot": baseline.get("zero_shot"),
            "ridge_half_split": baseline.get("ridge_half_split"),
        }
    (out / "tactile_encoder_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "TACTILE_ENCODER_REPORT.md").write_text(_render_report(results), encoding="utf-8")
    return results


def _fmt(value, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _render_report(results: dict) -> str:
    lines = [
        "# ObjectFolder 2.0 tactile encoder",
        "",
        f"- objects: {results['objects']}",
        f"- touch images: {results['touch_images']}",
        f"- vision images: {results['vision_images']}",
        f"- license: {results['license']}",
        f"- citation: {results['citation']}",
        f"- epochs: {results['epochs']}; folds: {results['folds']}",
        "",
        "## Seen objects (alignment upper bound)",
        "",
        "| direction | R@1 | R@5 |",
        "|---|---:|---:|",
    ]
    for direction in ("vision_to_touch", "touch_to_vision"):
        metric = results["seen"][direction]
        lines.append(f"| {direction} | {_fmt(metric['r1'])} | {_fmt(metric['r5'])} |")
    lines += [
        "",
        "## Held-out objects (5-fold object cross-validation)",
        "",
        "| direction | R@1 mean ± std | R@5 mean ± std |",
        "|---|---:|---:|",
    ]
    for direction in ("vision_to_touch", "touch_to_vision"):
        r1 = results["heldout"][f"{direction}_r1"]
        r5 = results["heldout"][f"{direction}_r5"]
        lines.append(
            f"| {direction} | {r1['mean']:.3f} ± {r1['std']:.3f} "
            f"| {r5['mean']:.3f} ± {r5['std']:.3f} |"
        )
    if results.get("baseline"):
        lines += [
            "",
            "## Baselines (100 objects)",
            "",
            "| protocol | direction | R@1 | R@5 | chance |",
            "|---|---|---:|---:|---:|",
        ]
        for protocol in ("zero_shot", "ridge_half_split"):
            baseline = results["baseline"].get(protocol) or {}
            for direction in ("vision_to_touch", "touch_to_vision"):
                metric = baseline.get(direction)
                if not metric:
                    continue
                lines.append(
                    f"| {protocol} | {direction} | {_fmt(metric['r1'])} "
                    f"| {_fmt(metric['r5'])} | {_fmt(metric['chance'])} |"
                )
    lines += [
        "",
        "## Caveats",
        "",
        "- Rendered from implicit ObjectFiles, not raw sensor captures.",
        "- Object-level pairing, not synchronized robot interaction.",
        "- Seen results are an upper bound; held-out results are new-object generalization.",
        "",
    ]
    return "\n".join(lines)
