"""Symmetric vision/tactile projection on ObjectFolder 2.0.

The tactile-only encoder maps touch into the frozen DINOv2 vision space; that
leaves touch->vision weak because the vision side is never adapted. This module
adds a trainable vision projector on top of the frozen DINOv2 anchors and trains
both sides with symmetric object-level InfoNCE.

Evaluation:
- seen objects: train and test on all objects (alignment upper bound);
- held-out objects: 5-fold object cross-validation (new-object generalization).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pss.projection.losses import supcon_cross
from pss.rd2.objectfolder import _l2, _retrieval
from pss.rd2.objectfolder_tactile import (
    TactileEncoder,
    TouchDataset,
    _aggregate,
    _fmt,
    _touch_paths,
    build_vision_anchors,
    encode_touch,
    load_renders,
)


class VisionProjector(nn.Module):
    def __init__(self, in_dim: int = 384, out_dim: int = 384, hidden: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def train_symmetric(
    paths,
    labels,
    anchors: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    temp: float,
    seed: int,
    device: str,
) -> tuple[TactileEncoder, VisionProjector]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_classes = len(np.unique(labels))
    tactile = TactileEncoder(anchors.shape[1]).to(device)
    vision = VisionProjector(anchors.shape[1], anchors.shape[1]).to(device)
    head_t = nn.Linear(anchors.shape[1], n_classes).to(device)
    head_v = nn.Linear(anchors.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(
        list(tactile.parameters())
        + list(vision.parameters())
        + list(head_t.parameters())
        + list(head_v.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    loader = DataLoader(
        TouchDataset(paths, labels, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    anchor_tensor = torch.as_tensor(anchors, dtype=torch.float32, device=device)
    for _epoch in range(epochs):
        tactile.train()
        vision.train()
        head_t.train()
        head_v.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            z_t = tactile(x)
            z_v = vision(anchor_tensor[y])
            loss = (
                supcon_cross(z_t, z_v, y, temp) + supcon_cross(z_v, z_t, y, temp)
            ) / 2.0
            loss = loss + 0.5 * (
                F.cross_entropy(head_t(z_t), y) + F.cross_entropy(head_v(z_v), y)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return tactile, vision


def evaluate_symmetric(
    tactile: TactileEncoder,
    vision: VisionProjector,
    objects,
    touch,
    anchors: np.ndarray,
    object_indices: list[int],
    device: str,
) -> dict:
    tactile_features = []
    for idx in object_indices:
        features = encode_touch(tactile, touch[objects[idx]], device)
        tactile_features.append(features.mean(0))
    T = _l2(np.stack(tactile_features))
    with torch.inference_mode():
        V = (
            vision(torch.as_tensor(anchors[object_indices], device=device))
            .cpu()
            .numpy()
        )
    return {
        "vision_to_touch": _retrieval(V, T),
        "touch_to_vision": _retrieval(T, V),
    }


def run_symmetric(
    renders: str | Path,
    out_dir: str | Path,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    temp: float = 0.1,
    folds: int = 5,
    seed: int = 0,
    checkpoint_dir: str | Path | None = None,
    tactile_baseline_path: str | Path | None = None,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    renders = Path(renders)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        Path(checkpoint_dir)
        if checkpoint_dir
        else Path("checkpoints/objectfolder_symmetric")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    objects, vision_paths, touch_paths = load_renders(renders)
    if len(objects) < 10:
        raise SystemExit(f"need at least 10 rendered objects, found {len(objects)}")
    anchors = build_vision_anchors(objects, vision_paths, device)

    all_indices = list(range(len(objects)))
    seen_paths, seen_labels = _touch_paths(
        objects, touch_paths, all_indices, local_labels=False
    )
    tactile, vision = train_symmetric(
        seen_paths, seen_labels, anchors, epochs, batch_size, lr, temp, seed, device
    )
    seen = evaluate_symmetric(
        tactile, vision, objects, touch_paths, anchors, all_indices, device
    )
    torch.save(
        {
            "state_dict": tactile.state_dict(),
            "out_dim": int(tactile.out_dim),
            "input_shape": [3, 96, 128],
            "objects": objects,
        },
        checkpoint_dir / "tactile_encoder_symmetric.pt",
    )
    torch.save(
        {
            "state_dict": vision.state_dict(),
            "in_dim": int(anchors.shape[1]),
            "out_dim": int(anchors.shape[1]),
            "objects": objects,
        },
        checkpoint_dir / "vision_projector.pt",
    )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(objects))
    fold_indices = np.array_split(order, folds)
    fold_metrics = []
    for fold_id, test_array in enumerate(fold_indices):
        test_idx = [int(i) for i in test_array]
        train_idx = [i for i in all_indices if i not in set(test_idx)]
        train_paths, train_labels = _touch_paths(
            objects, touch_paths, train_idx, local_labels=True
        )
        train_anchors = anchors[train_idx]
        fold_tactile, fold_vision = train_symmetric(
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
        metrics = evaluate_symmetric(
            fold_tactile,
            fold_vision,
            objects,
            touch_paths,
            anchors,
            test_idx,
            device,
        )
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
        "touch_images": sum(len(touch_paths[name]) for name in objects),
        "vision_images": sum(len(vision_paths[name]) for name in objects),
        "device": device,
        "epochs": int(epochs),
        "folds": int(folds),
        "seen": seen,
        "heldout": _aggregate(fold_metrics),
        "folds_raw": fold_metrics,
        "license": "CC-BY-4.0",
        "citation": "Gao et al., ObjectFolder 2.0, CVPR 2022",
        "tactile_only": None,
        "note": (
            "Symmetric vision projector + tactile encoder. Seen = train/test all "
            "objects; held-out = 5-fold object cross-validation. Rendered, "
            "object-level, not synchronized robot interaction."
        ),
    }
    tactile_only_path = (
        Path(tactile_baseline_path)
        if tactile_baseline_path
        else Path("eval/results/objectfolder_tactile/tactile_encoder_metrics.json")
    )
    if tactile_only_path.exists():
        tactile_only = json.loads(tactile_only_path.read_text(encoding="utf-8"))
        results["tactile_only"] = {
            "seen": tactile_only.get("seen"),
            "heldout": tactile_only.get("heldout"),
        }
    (out / "symmetric_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "SYMMETRIC_REPORT.md").write_text(_render_report(results), encoding="utf-8")
    return results


def _render_report(results: dict) -> str:
    lines = [
        "# ObjectFolder 2.0 symmetric vision/tactile projection",
        "",
        f"- objects: {results['objects']}",
        f"- touch images: {results['touch_images']}; vision images: {results['vision_images']}",
        f"- license: {results['license']}",
        f"- citation: {results['citation']}",
        f"- epochs: {results['epochs']}; folds: {results['folds']}",
        "",
        "## Seen objects (upper bound)",
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
    if results.get("tactile_only"):
        lines += [
            "",
            "## Tactile-only baseline",
            "",
            "| direction | seen R@1 | held-out R@1 | held-out R@5 |",
            "|---|---:|---:|---:|",
        ]
        seen = results["tactile_only"].get("seen") or {}
        heldout = results["tactile_only"].get("heldout") or {}
        for direction in ("vision_to_touch", "touch_to_vision"):
            seen_metric = seen.get(direction) or {}
            held_r1 = heldout.get(f"{direction}_r1") or {}
            held_r5 = heldout.get(f"{direction}_r5") or {}
            lines.append(
                f"| {direction} | {_fmt(seen_metric.get('r1', float('nan')))} "
                f"| {_fmt(held_r1.get('mean', float('nan')))} "
                f"| {_fmt(held_r5.get('mean', float('nan')))} |"
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
