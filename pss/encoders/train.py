"""Train the Phase 3 sensory experts on the category-level dataset.

The vision and tactile heads are trained with cross-entropy for evaluation;
the saved experts are the penultimate feature extractors. This is a **pipeline
smoke test** on primitive-proxy simulation data, not a capability benchmark.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import build_split, load_category_examples, load_geometry_examples
from .geometry import GEOMETRY_DIM
from .tactile import TactileMLP, save_tactile_expert
from .tactile.conv import TactileConvEncoder, save_tactile_conv_expert
from .vision import SmallCNNEncoder, save_vision_expert


def evaluate(
    encoder: nn.Module,
    head: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> float:
    if len(X) == 0:
        return 0.0
    encoder.eval()
    head.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y[i : i + batch_size], dtype=torch.long, device=device)
            pred = head(encoder(xb)).argmax(1)
            correct += int((pred == yb).sum().item())
    return correct / max(1, len(X))


def vision_augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1, device=x.device).item() < 0.5:
        x = torch.flip(x, dims=[3])
    pad = 4
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    h = x.shape[2] - 2 * pad
    w = x.shape[3] - 2 * pad
    i = torch.randint(0, 2 * pad + 1, (1,), device=x.device).item()
    j = torch.randint(0, 2 * pad + 1, (1,), device=x.device).item()
    x = x[:, :, i : i + h, j : j + w]
    scale = 0.8 + 0.4 * torch.rand(x.shape[0], 1, 1, 1, device=x.device)
    return x * scale


def tactile_augment(x: torch.Tensor) -> torch.Tensor:
    return x + 0.02 * torch.randn_like(x)


def train_encoder(
    encoder: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    augment=None,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
):
    torch.manual_seed(seed)
    encoder = encoder.to(device)
    head = nn.Linear(encoder.out_dim, n_classes).to(device)
    opt = torch.optim.Adam(
        [p for p in encoder.parameters() if p.requires_grad] + list(head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    X = torch.as_tensor(X_train, dtype=torch.float32)
    y = torch.as_tensor(y_train, dtype=torch.long)
    best_val = -1.0
    best_state = None
    history: list[float] = []

    for _epoch in range(epochs):
        encoder.train()
        head.train()
        perm = torch.randperm(len(X))
        for start in range(0, len(X), batch_size):
            idx = perm[start : start + batch_size]
            xb = X[idx].to(device)
            yb = y[idx].to(device)
            if augment is not None:
                xb = augment(xb)
            logits = head(encoder(xb))
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_acc = evaluate(encoder, head, X_val, y_val, device)
        history.append(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_state = (
                copy.deepcopy(encoder.state_dict()),
                copy.deepcopy(head.state_dict()),
            )

    if best_state is not None:
        encoder.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    return encoder, head, {"best_val": best_val, "history": history}


def _geo_arrays(examples, split: str, classes: list[str]):
    selected = [e for e in examples if e.split == split and e.category in classes]
    if not selected:
        return (
            np.zeros((0, GEOMETRY_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    X = np.stack([e.features for e in selected]).astype(np.float32)
    y = np.array([classes.index(e.category) for e in selected], dtype=np.int64)
    return X, y


def _write_report(path: Path, metrics: dict) -> None:
    geo = metrics["geometry"]
    lines = [
        "# Phase 3 report: sensory experts (smoke test)",
        "",
        f"- data: `{metrics['data_dir']}`",
        f"- device: `{metrics['device']}`",
        f"- epochs: {metrics['epochs']}, image size: {metrics['image_size']}",
        f"- classes ({len(metrics['classes'])}): {', '.join(metrics['classes'])}",
        "",
        "## Counts",
        "",
        "| split | examples |",
        "|---|---:|",
    ]
    for k, v in metrics["counts"].items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Closed-set top-1 accuracy",
        "",
        "| expert | train | val | test | out dim |",
        "|---|---:|---:|---:|---:|",
        f"| vision | {metrics['vision']['train_acc']:.3f} | "
        f"{metrics['vision']['val_acc']:.3f} | {metrics['vision']['test_acc']:.3f} | "
        f"{metrics['vision']['out_dim']} |",
        f"| tactile | {metrics['tactile']['train_acc']:.3f} | "
        f"{metrics['tactile']['val_acc']:.3f} | {metrics['tactile']['test_acc']:.3f} | "
        f"{metrics['tactile']['out_dim']} |",
        f"| geometry | {geo['train_acc']:.3f} | {geo['val_acc']:.3f} | "
        f"{geo['test_acc']:.3f} | {geo['dim']} |",
        "",
        "Geometry is a 16-d handcrafted feature vector with a standardized "
        "logistic-regression probe (no gradient training).",
        "",
        "## Caveats",
        "",
        "- Primitive-proxy simulation; from-scratch CNN (no torchvision/timm installed).",
        "- Category samples are few (see counts); augmentation is minimal.",
        "- `test_category` samples are unseen classes and are **not** evaluable by a closed-set head;",
        "  open-set/prototype matching is Phase 5 work.",
        "- Tactile is synthetic (material parameters), not a measured sensor.",
        "- Treat these numbers as a pipeline check, not a capability claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def train_all(
    data_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 30,
    image_size: int = 64,
    seed: int = 0,
    lr: float = 1e-3,
    tactile_arch: str = "mlp",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    examples = load_category_examples(data_dir, size=image_size)
    classes = sorted({e.category for e in examples if e.split == "train"})
    if not classes:
        raise RuntimeError("no training categories found")

    Xtr, Xtt, ytr, _ = build_split(examples, "train", classes)
    Xva, Xvt, yva, _ = build_split(examples, "val", classes)
    Xte, Xte_t, yte, _ = build_split(examples, "test", classes)
    unseen = sum(1 for e in examples if e.split == "test_category")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision = SmallCNNEncoder(out_dim=256)
    vision, vhead, _vinfo = train_encoder(
        vision,
        Xtr,
        ytr,
        Xva,
        yva,
        len(classes),
        epochs=epochs,
        lr=lr,
        device=device,
        seed=seed,
        augment=vision_augment,
    )
    tactile = (
        TactileConvEncoder(in_dim=Xtt.shape[1], out_dim=128)
        if tactile_arch == "conv"
        else TactileMLP(in_dim=Xtt.shape[1], out_dim=64)
    )
    tactile, thead, _tinfo = train_encoder(
        tactile,
        Xtt,
        ytr,
        Xvt,
        yva,
        len(classes),
        epochs=epochs,
        lr=lr,
        device=device,
        seed=seed,
        augment=tactile_augment,
    )

    geo_examples = load_geometry_examples(data_dir)
    Xg_tr, yg_tr = _geo_arrays(geo_examples, "train", classes)
    Xg_va, yg_va = _geo_arrays(geo_examples, "val", classes)
    Xg_te, yg_te = _geo_arrays(geo_examples, "test", classes)
    geo_probe = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
    )
    geo_probe.fit(Xg_tr, yg_tr)
    geometry_metrics = {
        "dim": GEOMETRY_DIM,
        "train_acc": float(geo_probe.score(Xg_tr, yg_tr)),
        "val_acc": float(geo_probe.score(Xg_va, yg_va)),
        "test_acc": float(geo_probe.score(Xg_te, yg_te)),
    }

    metrics = {
        "data_dir": str(data_dir),
        "device": device,
        "epochs": int(epochs),
        "image_size": int(image_size),
        "classes": classes,
        "counts": {
            "train": int(len(ytr)),
            "val": int(len(yva)),
            "test": int(len(yte)),
            "test_category_samples": int(unseen),
        },
        "vision": {
            "out_dim": int(vision.out_dim),
            "train_acc": float(evaluate(vision, vhead, Xtr, ytr, device)),
            "val_acc": float(evaluate(vision, vhead, Xva, yva, device)),
            "test_acc": float(evaluate(vision, vhead, Xte, yte, device)),
        },
        "tactile": {
            "out_dim": int(tactile.out_dim),
            "arch": str(tactile_arch),
            "train_acc": float(evaluate(tactile, thead, Xtt, ytr, device)),
            "val_acc": float(evaluate(tactile, thead, Xvt, yva, device)),
            "test_acc": float(evaluate(tactile, thead, Xte_t, yte, device)),
        },
        "geometry": geometry_metrics,
        "note": "Smoke test on primitive-proxy sim data; not a capability claim.",
    }

    save_vision_expert(out / "vision.pt", vision, classes, image_size)
    if tactile_arch == "conv":
        save_tactile_conv_expert(
            out / "tactile_conv.pt", tactile, classes, int(Xtt.shape[1])
        )
    else:
        save_tactile_expert(out / "tactile.pt", tactile, classes, int(Xtt.shape[1]))
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "REPORT.md", metrics)
    return metrics
