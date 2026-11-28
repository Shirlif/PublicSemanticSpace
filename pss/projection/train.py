"""Train the Phase 4 per-modality projection heads with supervised contrastive
alignment on frozen expert features.

This is a **category-level** alignment smoke test: the dataset has multiple
views/instances per category, not synchronized real instance-level
visual-tactile pairs. Instance-level binding is deferred to RD2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss.contracts.schema import SPACE_VERSION
from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision import load_vision_expert
from pss.experts.adapter import admit_encoder

from .features import PairedFeatures, load_paired_features
from .heads import PUBLIC_DIM, ProjectionHead, save_projection
from .losses import material_within_category, supcon_cross

MODALITIES = ("vision", "tactile", "geometry")
# Material separation is meaningful for vision/tactile; geometry carries size
# and shape, not hardness/roughness. Forcing it there distorts the expert.
MATERIAL_MODALITIES = ("vision", "tactile")


def _material_bins(attrs: np.ndarray, train_idx: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """Tertile bins for hardness and roughness, combined into one label."""
    bins = []
    for column in (0, 1):  # hardness, roughness
        edges = np.quantile(
            attrs[train_idx, column], np.linspace(0, 1, n_bins + 1)[1:-1]
        )
        bins.append(np.digitize(attrs[:, column], edges))
    return (bins[0] * n_bins + bins[1]).astype(np.int64)


def _project(head: nn.Module, X: np.ndarray, device: str, batch_size: int = 256) -> torch.Tensor:
    head.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
            outs.append(head(xb).cpu())
    return torch.cat(outs, 0)


def _retrieval(
    z1: torch.Tensor, z2: torch.Tensor, labels: list[str], idx: np.ndarray, mode: str
) -> float:
    if len(idx) == 0:
        return 0.0
    z1n = F.normalize(z1[idx], dim=-1)
    z2n = F.normalize(z2[idx], dim=-1)
    top = (z1n @ z2n.T).argmax(1).tolist()
    if mode == "category":
        return float(np.mean([labels[idx[t]] == labels[idx[k]] for k, t in enumerate(top)]))
    if mode == "pair":
        return float(np.mean([idx[t] == idx[k] for k, t in enumerate(top)]))
    raise ValueError(mode)


def _retrieval_bundle(zv, zt, zg, labels, idx) -> dict:
    pairs = (("vision", zv), ("tactile", zt), ("geometry", zg))
    out = {}
    for i, (n1, z1) in enumerate(pairs):
        for n2, z2 in pairs[i + 1 :]:
            out[f"{n1}->{n2}"] = {
                "category_r1": _retrieval(z1, z2, labels, idx, "category"),
                "pair_r1": _retrieval(z1, z2, labels, idx, "pair"),
            }
            out[f"{n2}->{n1}"] = {
                "category_r1": _retrieval(z2, z1, labels, idx, "category"),
                "pair_r1": _retrieval(z2, z1, labels, idx, "pair"),
            }
    return out


def _write_report(path: Path, metrics: dict) -> None:
    lines = [
        "# Phase 4 report: projection heads (smoke test)",
        "",
        f"- data: `{metrics['data_dir']}`",
        f"- experts: `{metrics['experts']}`",
        f"- level: `{metrics['level']}`; temp: {metrics['temp']}; epochs: {metrics['epochs']}",
        f"- material weight: {metrics.get('material_weight', 0.0)}",
        f"- device: `{metrics['device']}`",
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
        "## Cross-modal retrieval R@1",
        "",
        "`category` = top-1 retrieved sample shares the category; `pair` = top-1 is the exact paired view.",
        "",
        "| split | direction | category R@1 | pair R@1 |",
        "|---|---|---:|---:|",
    ]
    for split in ("val", "test", "test_category"):
        for direction, vals in metrics["retrieval"][split].items():
            lines.append(
                f"| {split} | {direction} | {vals['category_r1']:.3f} | {vals['pair_r1']:.3f} |"
            )
    lines += [
        "",
        "## Attribute reconstruction MSE (test)",
        "",
        "| modality | MSE |",
        "|---|---:|",
    ]
    for m, v in metrics["attribute_mse"].items():
        lines.append(f"| {m} | {v:.4f} |")
    lines += [
        "",
        "## Caveats",
        "",
        "- Category-level pairing only; no synchronized real instance-level visual-tactile pairs (RD2).",
        "- Tactile is synthetic; the tactile expert is weak (see Phase 3).",
        "- `test_category` is unseen; a closed-set interpretation does not apply. Category R@1 there",
        "  only asks whether the top-1 sample shares the same unseen category.",
        "- Small dataset and from-scratch vision backbone; treat as a pipeline check.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def train_projections(
    data_dir: str | Path,
    ckpt_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    temp: float = 0.07,
    level: str = "category",
    seed: int = 0,
    out_dim: int = PUBLIC_DIM,
    vision_expert=None,
    tactile_expert=None,
    material_weight: float = 0.3,
    whiten_dim: int | None = None,
    whiten_modalities: tuple[str, ...] | None = None,
    residual: bool = False,
) -> dict:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(ckpt_dir)
    vision = vision_expert or admit_encoder(
        load_vision_expert(ckpt_dir / "vision.pt", device=device),
        modality="vision",
        version=f"smallcnn@{SPACE_VERSION}",
    ).encoder
    tactile = tactile_expert or admit_encoder(
        load_tactile_expert(ckpt_dir / "tactile.pt", device=device),
        modality="tactile",
        version=f"mlp@{SPACE_VERSION}",
    ).encoder
    feats: PairedFeatures = load_paired_features(data_dir, vision, tactile)

    splits = np.array(feats.splits)
    label_values = feats.categories if level == "category" else feats.instances
    train_idx = np.where(splits == "train")[0]
    val_idx = np.where(splits == "val")[0]
    test_idx = np.where(splits == "test")[0]
    unseen_idx = np.where(splits == "test_category")[0]

    train_labels = sorted({label_values[i] for i in train_idx})
    label_map = {v: i for i, v in enumerate(train_labels)}
    y_all = np.array([label_map.get(v, -1) for v in label_values], dtype=np.int64)
    train_idx = train_idx[y_all[train_idx] >= 0]
    val_idx = val_idx[y_all[val_idx] >= 0]
    test_idx = test_idx[y_all[test_idx] >= 0]

    V = torch.as_tensor(feats.vision)
    T = torch.as_tensor(feats.tactile)
    G = torch.as_tensor(feats.geometry)
    A = torch.as_tensor(feats.attrs)
    material_all = torch.as_tensor(
        _material_bins(feats.attrs, train_idx), dtype=torch.long
    )
    y = torch.as_tensor(y_all, dtype=torch.long)

    heads = {}
    whitening = {}
    for name, X in (("vision", V), ("tactile", T), ("geometry", G)):
        k = None
        if whiten_dim is not None and (
            whiten_modalities is None or name in whiten_modalities
        ):
            k = int(min(whiten_dim, X.shape[1], len(train_idx) - 1))
        head = ProjectionHead(
            X.shape[1], out_dim, whiten_dim=k, residual=residual
        ).to(device)
        if k is not None:
            head.fit_whitening(X[train_idx].numpy())
        heads[name] = head
        whitening[name] = k
    attr_heads = {name: nn.Linear(out_dim, A.shape[1]).to(device) for name in MODALITIES}
    params = [p for h in heads.values() for p in h.parameters()]
    params += [p for h in attr_heads.values() for p in h.parameters()]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    X = {"vision": V, "tactile": T, "geometry": G}
    for _epoch in range(epochs):
        for h in heads.values():
            h.train()
        for h in attr_heads.values():
            h.train()
        perm = torch.randperm(len(train_idx))
        for start in range(0, len(train_idx), batch_size):
            idx = torch.as_tensor(train_idx[perm[start : start + batch_size]], dtype=torch.long)
            yb = y[idx].to(device)
            myb = material_all[idx].to(device)
            z = {m: heads[m](X[m][idx].to(device)) for m in MODALITIES}
            category_loss = (
                supcon_cross(z["vision"], z["tactile"], yb, temp)
                + supcon_cross(z["tactile"], z["vision"], yb, temp)
                + supcon_cross(z["vision"], z["geometry"], yb, temp)
                + supcon_cross(z["geometry"], z["vision"], yb, temp)
                + supcon_cross(z["tactile"], z["geometry"], yb, temp)
                + supcon_cross(z["geometry"], z["tactile"], yb, temp)
            ) / 6.0
            material_loss = sum(
                material_within_category(z[m], yb, myb, temp)
                for m in MATERIAL_MODALITIES
            ) / len(MATERIAL_MODALITIES)
            attr_loss = sum(
                F.mse_loss(attr_heads[m](z[m]), A[idx].to(device)) for m in MODALITIES
            ) / 3.0
            loss = category_loss + material_weight * material_loss + 0.1 * attr_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

    zv = _project(heads["vision"], V.numpy(), device)
    zt = _project(heads["tactile"], T.numpy(), device)
    zg = _project(heads["geometry"], G.numpy(), device)

    retrieval = {
        "val": _retrieval_bundle(zv, zt, zg, feats.categories, val_idx),
        "test": _retrieval_bundle(zv, zt, zg, feats.categories, test_idx),
        "test_category": _retrieval_bundle(zv, zt, zg, feats.categories, unseen_idx),
    }

    attribute_mse = {}
    eval_idx = test_idx if len(test_idx) else unseen_idx
    for name in MODALITIES:
        z = {"vision": zv, "tactile": zt, "geometry": zg}[name]
        pred = attr_heads[name](z[eval_idx].to(device)).cpu()
        attribute_mse[name] = float(F.mse_loss(pred, A[eval_idx]).item())

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, head in heads.items():
        save_projection(out / f"{name}.pt", head)

    metrics = {
        "data_dir": str(data_dir),
        "experts": str(ckpt_dir),
        "device": device,
        "level": level,
        "temp": float(temp),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "out_dim": int(out_dim),
        "material_weight": float(material_weight),
        "whiten_dim": whiten_dim,
        "whiten_modalities": list(whiten_modalities) if whiten_modalities else None,
        "whitening": whitening,
        "residual": bool(residual),
        "counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "test_category": int(len(unseen_idx)),
        },
        "retrieval": retrieval,
        "attribute_mse": attribute_mse,
        "note": "Category-level pairing smoke test; not a capability claim.",
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "REPORT.md", metrics)
    return metrics
