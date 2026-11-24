"""Phase 4 anchor retraining (原文 9.1.3).

Protocol:
1. Replace one modality's expert with a new checkpoint.
2. Freeze every other expert and projection head, and the prototype bank.
3. Train only the new modality's projection head so it aligns to frozen anchors
   (the mean of the other modalities' projected features).
4. Verify the frozen artifacts are byte-identical before and after.
5. Report cross-modal R@1, attribute retention, alignment, and prototype drift.

The new expert's backbone is never trained: features are extracted once, then the
head is fitted on top. Prototypes are read-only; the new modality's centroid
drift against the frozen bank is reported but the bank is not modified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision import load_vision_expert
from pss.projection.features import PairedFeatures, load_paired_features
from pss.projection.heads import ProjectionHead, load_projection, save_projection
from pss.projection.losses import material_within_category, supcon_cross
from pss.prototypes.bank import PrototypeBank

MODALITIES = ("vision", "tactile", "geometry")
MATERIAL_MODALITIES = ("vision", "tactile")
ATTR_NAMES = ("hardness", "roughness", "thermal", "size_x", "size_y")


def _hash_files(paths: list[Path]) -> dict[str, str]:
    out = {}
    for path in paths:
        out[str(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return out


def _material_bins(attrs: np.ndarray, train_idx: np.ndarray, n_bins: int = 3) -> np.ndarray:
    bins = []
    for column in (0, 1):
        edges = np.quantile(
            attrs[train_idx, column], np.linspace(0, 1, n_bins + 1)[1:-1]
        )
        bins.append(np.digitize(attrs[:, column], edges))
    return (bins[0] * n_bins + bins[1]).astype(np.int64)


def _project(head: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        xb = torch.as_tensor(X, dtype=torch.float32, device=device)
        return head(xb).cpu().numpy()


def _retrieval(z1: np.ndarray, z2: np.ndarray, labels, idx, mode: str) -> float:
    if len(idx) == 0:
        return float("nan")
    z1n = z1[idx] / (np.linalg.norm(z1[idx], axis=1, keepdims=True) + 1e-9)
    z2n = z2[idx] / (np.linalg.norm(z2[idx], axis=1, keepdims=True) + 1e-9)
    top = (z1n @ z2n.T).argmax(1)
    if mode == "category":
        return float(np.mean([labels[idx[t]] == labels[idx[k]] for k, t in enumerate(top)]))
    return float(np.mean([idx[t] == idx[k] for k, t in enumerate(top)]))


def _attribute_retention(raw, projected, attrs, train_idx, test_idx) -> dict:
    raw_scores, proj_scores = [], []
    for a in range(attrs.shape[1]):
        raw_reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        proj_reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        raw_reg.fit(raw[train_idx], attrs[train_idx, a])
        proj_reg.fit(projected[train_idx], attrs[train_idx, a])
        raw_scores.append(float(r2_score(attrs[test_idx, a], raw_reg.predict(raw[test_idx]))))
        proj_scores.append(float(r2_score(attrs[test_idx, a], proj_reg.predict(projected[test_idx]))))
    retention = [(p / r) if r > 1e-3 else float("nan") for r, p in zip(raw_scores, proj_scores)]
    return {
        "raw_r2": dict(zip(ATTR_NAMES, raw_scores)),
        "projected_r2": dict(zip(ATTR_NAMES, proj_scores)),
        "retention": dict(zip(ATTR_NAMES, retention)),
        "mean_raw_r2": float(np.nanmean(raw_scores)),
        "mean_projected_r2": float(np.nanmean(proj_scores)),
        "mean_retention": float(np.nanmean(retention)),
    }


def _prototype_drift(
    bank: PrototypeBank,
    modality: str,
    projected: np.ndarray,
    categories,
    train_idx,
    classes: list[str],
) -> dict:
    sims = []
    for i, category in enumerate(classes):
        idx = [int(j) for j in train_idx if categories[j] == category]
        if not idx:
            continue
        centroid = projected[idx].mean(0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims.append(float(np.dot(centroid, bank.protos[modality][i])))
    return {
        "mean_cosine": float(np.mean(sims)) if sims else float("nan"),
        "min_cosine": float(np.min(sims)) if sims else float("nan"),
        "n": len(sims),
    }


def _load_experts(experts: Path, modality: str, new_expert: Path | None, device: str):
    vision_path = (
        new_expert if modality == "vision" and new_expert else experts / "vision.pt"
    )
    tactile_path = (
        new_expert if modality == "tactile" and new_expert else experts / "tactile.pt"
    )
    return (
        load_vision_expert(vision_path, device=device),
        load_tactile_expert(tactile_path, device=device),
    )


def run_anchor_retrain(
    data_dir: str | Path,
    experts: str | Path,
    projections: str | Path,
    prototypes: str | Path,
    modality: str,
    out_dir: str | Path,
    new_expert: str | Path | None = None,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    temp: float = 0.07,
    material_weight: float = 1.0,
    seed: int = 0,
    init_from_old: bool = True,
) -> dict:
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(data_dir)
    experts = Path(experts)
    projections = Path(projections)
    prototypes = Path(prototypes)
    out = Path(out_dir)

    new_expert_path = Path(new_expert) if new_expert else None
    vision, tactile = _load_experts(experts, modality, new_expert_path, device)
    feats: PairedFeatures = load_paired_features(data_dir, vision, tactile)
    splits = np.array(feats.splits)
    categories = np.array(feats.categories)
    attrs = np.asarray(feats.attrs, dtype=np.float32)
    raw = {"vision": feats.vision, "tactile": feats.tactile, "geometry": feats.geometry}

    train_idx = np.where(splits == "train")[0]
    val_idx = np.where(splits == "val")[0]
    test_idx = np.where(splits == "test")[0]
    unseen_idx = np.where(splits == "test_category")[0]
    seen_classes = sorted({categories[i] for i in train_idx})
    class_to_idx = {c: i for i, c in enumerate(seen_classes)}
    y_all = np.array([class_to_idx.get(c, -1) for c in categories], dtype=np.int64)
    keep = y_all >= 0
    train_idx = train_idx[keep[train_idx]]
    val_idx = val_idx[keep[val_idx]]
    test_idx = test_idx[keep[test_idx]]

    others = [m for m in MODALITIES if m != modality]
    frozen_heads = {
        m: load_projection(projections / f"{m}.pt", device=device) for m in others
    }
    frozen_projected = {
        m: _project(frozen_heads[m], raw[m], device) for m in others
    }
    anchor_all = np.mean([frozen_projected[m] for m in others], axis=0)
    anchor_all = anchor_all / (np.linalg.norm(anchor_all, axis=1, keepdims=True) + 1e-9)

    old_head_path = projections / f"{modality}.pt"
    if init_from_old and old_head_path.exists():
        new_head = load_projection(old_head_path, device=device)
    else:
        new_head = ProjectionHead(raw[modality].shape[1]).to(device)
    attr_head = nn.Linear(new_head.out_dim, attrs.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        list(new_head.parameters()) + list(attr_head.parameters()), lr=lr, weight_decay=1e-4
    )

    X = torch.as_tensor(raw[modality], dtype=torch.float32, device=device)
    A = torch.as_tensor(attrs, dtype=torch.float32, device=device)
    anchors = torch.as_tensor(anchor_all, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_all, dtype=torch.long, device=device)
    material_all = torch.as_tensor(_material_bins(attrs, train_idx), dtype=torch.long, device=device)

    frozen_paths = [projections / f"{m}.pt" for m in others]
    frozen_paths += [prototypes / "prototypes.npz", prototypes / "tree.json"]
    frozen_before = _hash_files([p for p in frozen_paths if p.exists()])

    best_val = -1.0
    best_state = {k: v.detach().clone() for k, v in new_head.state_dict().items()}
    history = []
    for epoch in range(1, epochs + 1):
        new_head.train()
        attr_head.train()
        perm = torch.randperm(len(train_idx))
        for start in range(0, len(train_idx), batch_size):
            idx = torch.as_tensor(
                train_idx[perm[start : start + batch_size]], dtype=torch.long, device=device
            )
            z = new_head(X[idx])
            loss = supcon_cross(z, anchors[idx], y[idx], temp)
            if modality in MATERIAL_MODALITIES:
                loss = loss + material_weight * material_within_category(
                    z, y[idx], material_all[idx], temp
                )
            loss = loss + 0.1 * F.mse_loss(attr_head(z), A[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        new_head.eval()
        with torch.no_grad():
            z_all = new_head(X).cpu().numpy()
        val_r1 = _retrieval(z_all, anchor_all, feats.categories, val_idx, "category")
        history.append({"epoch": epoch, "val_category_r1": val_r1})
        if val_r1 > best_val:
            best_val = val_r1
            best_state = {k: v.detach().clone() for k, v in new_head.state_dict().items()}

    new_head.load_state_dict(best_state)
    new_head.eval()
    with torch.no_grad():
        z_new = new_head(X).cpu().numpy()

    retrieval = {
        "val": {
            "category_r1": _retrieval(z_new, anchor_all, feats.categories, val_idx, "category"),
            "pair_r1": _retrieval(z_new, anchor_all, feats.categories, val_idx, "pair"),
        },
        "test": {
            "category_r1": _retrieval(z_new, anchor_all, feats.categories, test_idx, "category"),
            "pair_r1": _retrieval(z_new, anchor_all, feats.categories, test_idx, "pair"),
        },
        "test_category": {
            "category_r1": _retrieval(z_new, anchor_all, feats.categories, unseen_idx, "category"),
            "pair_r1": _retrieval(z_new, anchor_all, feats.categories, unseen_idx, "pair"),
        },
    }
    retention = _attribute_retention(
        raw[modality], z_new, attrs, train_idx, test_idx
    )
    alignment = float(np.mean(np.sum((z_new[test_idx] - anchor_all[test_idx]) ** 2, axis=1)))
    bank = PrototypeBank.load(prototypes / "prototypes.npz")
    drift = _prototype_drift(
        bank, modality, z_new, feats.categories, train_idx, seen_classes
    )

    frozen_after = _hash_files([p for p in frozen_paths if p.exists()])
    integrity = {
        "unchanged": frozen_before == frozen_after,
        "before": frozen_before,
        "after": frozen_after,
    }

    out.mkdir(parents=True, exist_ok=True)
    save_projection(out / f"{modality}.pt", new_head)
    metrics = {
        "modality": modality,
        "new_expert": str(new_expert_path) if new_expert_path else "same-as-current",
        "data": str(data_dir),
        "experts": str(experts),
        "projections": str(projections),
        "prototypes": str(prototypes),
        "device": device,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "temp": float(temp),
        "material_weight": float(material_weight) if modality in MATERIAL_MODALITIES else 0.0,
        "init_from_old": bool(init_from_old),
        "best_val_category_r1": float(best_val),
        "retrieval": retrieval,
        "attribute_retention": retention,
        "alignment_mse": alignment,
        "prototype_drift": drift,
        "frozen_integrity": integrity,
        "history": history,
    }
    (out / "anchor_report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "ANCHOR_REPORT.md").write_text(_render_report(metrics), encoding="utf-8")
    return metrics


def _fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _render_report(metrics: dict) -> str:
    lines = [
        "# Anchor retraining report",
        "",
        f"- modality replaced: **{metrics['modality']}**",
        f"- new expert: `{metrics['new_expert']}`",
        f"- projections: `{metrics['projections']}`",
        f"- prototypes (frozen): `{metrics['prototypes']}`",
        f"- epochs: {metrics['epochs']}; batch: {metrics['batch_size']}; temp: {metrics['temp']}",
        f"- material weight: {metrics['material_weight']}",
        "",
        "## Cross-modal alignment to frozen anchors",
        "",
        "| split | category R@1 | pair R@1 |",
        "|---|---:|---:|",
    ]
    for split, vals in metrics["retrieval"].items():
        lines.append(f"| {split} | {_fmt(vals['category_r1'])} | {_fmt(vals['pair_r1'])} |")
    lines += [
        "",
        "## Attribute retention",
        "",
        f"- raw mean R²: {_fmt(metrics['attribute_retention']['mean_raw_r2'])}",
        f"- projected mean R²: {_fmt(metrics['attribute_retention']['mean_projected_r2'])}",
        f"- retention: {metrics['attribute_retention']['mean_retention']:.1%}",
        "",
        "## Alignment and prototype drift",
        "",
        f"- alignment MSE (new vs frozen anchor): {_fmt(metrics['alignment_mse'])}",
        f"- prototype drift mean cosine: {_fmt(metrics['prototype_drift']['mean_cosine'])}",
        f"- prototype drift min cosine: {_fmt(metrics['prototype_drift']['min_cosine'])}",
        "",
        "## Frozen artifact integrity",
        "",
        f"- unchanged: **{metrics['frozen_integrity']['unchanged']}**",
        "",
        "## Verdict",
        "",
        f"- category R@1 (test): {_fmt(metrics['retrieval']['test']['category_r1'])}",
        f"- attribute retention: {metrics['attribute_retention']['mean_retention']:.1%}",
        f"- only the new `{metrics['modality']}` head is saved to `{metrics['modality']}.pt`",
        "- other projections and the prototype bank are untouched; pair R@1 remains RD2-blocked",
        "",
    ]
    if not metrics["frozen_integrity"]["unchanged"]:
        lines.append("**FAIL: a frozen artifact changed. Do not deploy this run.**")
    return "\n".join(lines)
