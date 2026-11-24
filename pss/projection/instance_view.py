"""Instance-level view holdout: a simulation proxy for RD2.

The current MuJoCo category dataset has four synchronized views per instance.
This experiment holds out views 2-3, trains the projection heads on views 0-1
with instance labels, and measures same-instance / different-view retrieval.
It validates the *algorithmic* instance-level alignment path without collecting
real synchronized visual-tactile data.

What it does NOT validate:
- new-instance generalization (the same instances appear in train views);
- real sensors (tactile is still the synthetic proxy);
- sim2real (this remains the RD2/R-Gate deliverable).

The report compares the canonical category projection against an instance
fine-tuned projection on the held-out views.
"""

from __future__ import annotations

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

MODALITIES = ("vision", "tactile", "geometry")
MATERIAL_MODALITIES = ("vision", "tactile")
ATTR_NAMES = ("hardness", "roughness", "thermal", "size_x", "size_y")


def _project(head, X: np.ndarray, device: str) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        xb = torch.as_tensor(X, dtype=torch.float32, device=device)
        return head(xb).cpu().numpy()


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _instance_retrieval(z1, z2, instances, idx, k: int = 5) -> dict:
    if len(idx) < 2:
        return {"r1": float("nan"), "r5": float("nan"), "positives": 0, "n": len(idx)}
    A = _l2(z1[idx])
    B = _l2(z2[idx])
    sim = A @ B.T
    np.fill_diagonal(sim, -np.inf)
    ids = np.asarray(instances)[idx]
    pos = ids[:, None] == ids[None, :]
    np.fill_diagonal(pos, False)
    if not pos.any():
        return {"r1": float("nan"), "r5": float("nan"), "positives": 0, "n": len(idx)}
    order = np.argsort(-sim, axis=1)
    hits1 = pos[np.arange(len(idx)), order[:, 0]]
    topk = min(k, len(idx))
    hits5 = np.array([pos[i, order[i, :topk]].any() for i in range(len(idx))])
    return {
        "r1": float(hits1.mean()),
        "r5": float(hits5.mean()),
        "positives": int(pos.sum()),
        "n": len(idx),
    }


def _instance_retrieval_qg(
    z_query, z_gallery, instances, query_idx, gallery_idx, k: int = 5
) -> dict:
    """Query test views against train views; positives share the instance id."""
    if len(query_idx) < 1 or len(gallery_idx) < 1:
        return {"r1": float("nan"), "r5": float("nan"), "positives": 0}
    A = _l2(z_query[query_idx])
    B = _l2(z_gallery[gallery_idx])
    sim = A @ B.T
    ids_query = np.asarray(instances)[query_idx]
    ids_gallery = np.asarray(instances)[gallery_idx]
    pos = ids_query[:, None] == ids_gallery[None, :]
    if not pos.any():
        return {"r1": float("nan"), "r5": float("nan"), "positives": 0}
    order = np.argsort(-sim, axis=1)
    hits1 = pos[np.arange(len(query_idx)), order[:, 0]]
    topk = min(k, len(gallery_idx))
    hits5 = np.array(
        [pos[i, order[i, :topk]].any() for i in range(len(query_idx))]
    )
    return {
        "r1": float(hits1.mean()),
        "r5": float(hits5.mean()),
        "positives": int(pos.sum()),
    }


def _category_retrieval(z1, z2, categories, idx) -> float:
    if len(idx) < 2:
        return float("nan")
    A = _l2(z1[idx])
    B = _l2(z2[idx])
    sim = A @ B.T
    np.fill_diagonal(sim, -np.inf)
    top = sim.argmax(1)
    labels = np.asarray(categories)[idx]
    return float(np.mean(labels[top] == labels))


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


def _evaluate(projected, categories, instances, raw, attrs, train_idx, test_idx) -> dict:
    pairs = {}
    query_gallery = {}
    for i, first in enumerate(MODALITIES):
        pairs[f"{first}-{first}"] = _instance_retrieval(
            projected[first], projected[first], instances, test_idx
        )
        query_gallery[f"{first}-{first}"] = _instance_retrieval_qg(
            projected[first], projected[first], instances, test_idx, train_idx
        )
        for second in MODALITIES[i + 1 :]:
            pairs[f"{first}-{second}"] = _instance_retrieval(
                projected[first], projected[second], instances, test_idx
            )
            query_gallery[f"{first}-{second}"] = _instance_retrieval_qg(
                projected[first], projected[second], instances, test_idx, train_idx
            )
    categories_r1 = {
        f"{first}-{second}": _category_retrieval(
            projected[first], projected[second], categories, test_idx
        )
        for i, first in enumerate(MODALITIES)
        for second in MODALITIES[i + 1 :]
    }
    retention = {
        m: _attribute_retention(
            raw[m], projected[m], attrs, train_idx, test_idx
        )
        for m in MODALITIES
    }
    return {
        "instance_retrieval": pairs,
        "instance_retrieval_qg": query_gallery,
        "category_r1": categories_r1,
        "attribute_retention": retention,
    }


def _material_bins(attrs: np.ndarray, train_idx: np.ndarray, n_bins: int = 3) -> np.ndarray:
    bins = []
    for column in (0, 1):
        edges = np.quantile(
            attrs[train_idx, column], np.linspace(0, 1, n_bins + 1)[1:-1]
        )
        bins.append(np.digitize(attrs[:, column], edges))
    return (bins[0] * n_bins + bins[1]).astype(np.int64)


def run_instance_view_experiment(
    data_dir: str | Path,
    experts: str | Path,
    projections: str | Path,
    out_dir: str | Path,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 5e-4,
    temp: float = 0.07,
    instance_weight: float = 1.0,
    category_weight: float = 0.5,
    material_weight: float = 0.5,
    seed: int = 0,
    train_views: tuple[int, ...] = (0, 1),
    test_views: tuple[int, ...] = (2, 3),
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(data_dir)
    experts = Path(experts)
    projections = Path(projections)
    vision = load_vision_expert(experts / "vision.pt", device=device)
    tactile = load_tactile_expert(experts / "tactile.pt", device=device)
    feats: PairedFeatures = load_paired_features(data_dir, vision, tactile)
    views = np.asarray(feats.views)
    train_idx = np.where(np.isin(views, train_views))[0]
    test_idx = np.where(np.isin(views, test_views))[0]
    categories = np.asarray(feats.categories)
    instances = np.asarray(feats.instances)
    attrs = np.asarray(feats.attrs, dtype=np.float32)
    raw = {"vision": feats.vision, "tactile": feats.tactile, "geometry": feats.geometry}

    train_instances = sorted(set(instances[train_idx]))
    instance_map = {v: i for i, v in enumerate(train_instances)}
    y_instance = np.array([instance_map.get(v, -1) for v in instances], dtype=np.int64)
    train_categories = sorted(set(categories[train_idx]))
    category_map = {v: i for i, v in enumerate(train_categories)}
    y_category = np.array([category_map.get(v, -1) for v in categories], dtype=np.int64)
    material = _material_bins(attrs, train_idx)
    by_instance: dict[str, list[int]] = {}
    for i in train_idx:
        by_instance.setdefault(str(instances[i]), []).append(int(i))
    instance_ids = list(by_instance)

    base_heads = {
        m: load_projection(projections / f"{m}.pt", device=device) for m in MODALITIES
    }
    base_projected = {
        m: _project(base_heads[m], raw[m], device) for m in MODALITIES
    }
    baseline = _evaluate(
        base_projected, categories, instances, raw, attrs, train_idx, test_idx
    )

    heads = {
        m: load_projection(projections / f"{m}.pt", device=device) for m in MODALITIES
    }
    for head in heads.values():
        for parameter in head.parameters():
            parameter.requires_grad = True
    attr_heads = {
        m: nn.Linear(heads[m].out_dim, attrs.shape[1]).to(device) for m in MODALITIES
    }
    optimizer = torch.optim.Adam(
        [p for head in heads.values() for p in head.parameters()]
        + [p for head in attr_heads.values() for p in head.parameters()],
        lr=lr,
        weight_decay=1e-4,
    )
    X = {
        m: torch.as_tensor(raw[m], dtype=torch.float32, device=device)
        for m in MODALITIES
    }
    A = torch.as_tensor(attrs, dtype=torch.float32, device=device)
    y_i = torch.as_tensor(y_instance, dtype=torch.long, device=device)
    y_c = torch.as_tensor(y_category, dtype=torch.long, device=device)
    mat = torch.as_tensor(material, dtype=torch.long, device=device)

    for _epoch in range(epochs):
        for head in heads.values():
            head.train()
        for head in attr_heads.values():
            head.train()
        rng = np.random.default_rng(seed + _epoch)
        order = rng.permutation(len(instance_ids))
        per_batch = max(1, batch_size // 2)
        for start in range(0, len(order), per_batch):
            batch_instances = [instance_ids[i] for i in order[start : start + per_batch]]
            batch_idx = [
                j for instance in batch_instances for j in by_instance[instance]
            ]
            idx = torch.as_tensor(batch_idx, dtype=torch.long, device=device)
            z = {m: heads[m](X[m][idx]) for m in MODALITIES}
            instance_loss = (
                supcon_cross(z["vision"], z["tactile"], y_i[idx], temp)
                + supcon_cross(z["tactile"], z["vision"], y_i[idx], temp)
                + supcon_cross(z["vision"], z["geometry"], y_i[idx], temp)
                + supcon_cross(z["geometry"], z["vision"], y_i[idx], temp)
                + supcon_cross(z["tactile"], z["geometry"], y_i[idx], temp)
                + supcon_cross(z["geometry"], z["tactile"], y_i[idx], temp)
            ) / 6.0
            category_loss = (
                supcon_cross(z["vision"], z["tactile"], y_c[idx], temp)
                + supcon_cross(z["tactile"], z["vision"], y_c[idx], temp)
                + supcon_cross(z["vision"], z["geometry"], y_c[idx], temp)
                + supcon_cross(z["geometry"], z["vision"], y_c[idx], temp)
                + supcon_cross(z["tactile"], z["geometry"], y_c[idx], temp)
                + supcon_cross(z["geometry"], z["tactile"], y_c[idx], temp)
            ) / 6.0
            material_loss = sum(
                material_within_category(z[m], y_c[idx], mat[idx], temp)
                for m in MATERIAL_MODALITIES
            ) / len(MATERIAL_MODALITIES)
            attr_loss = sum(
                F.mse_loss(attr_heads[m](z[m]), A[idx]) for m in MODALITIES
            ) / len(MODALITIES)
            loss = (
                instance_weight * instance_loss
                + category_weight * category_loss
                + material_weight * material_loss
                + 0.1 * attr_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    fine_heads = {m: heads[m].eval() for m in MODALITIES}
    fine_projected = {
        m: _project(fine_heads[m], raw[m], device) for m in MODALITIES
    }
    fine = _evaluate(
        fine_projected, categories, instances, raw, attrs, train_idx, test_idx
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for m, head in fine_heads.items():
        save_projection(out / f"{m}.pt", head)

    metrics = {
        "data": str(data_dir),
        "experts": str(experts),
        "projections_init": str(projections),
        "device": device,
        "split": {
            "train_views": [int(v) for v in train_views],
            "test_views": [int(v) for v in test_views],
        },
        "counts": {"train": len(train_idx), "test": len(test_idx)},
        "instances": len(train_instances),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "instance_weight": float(instance_weight),
        "category_weight": float(category_weight),
        "material_weight": float(material_weight),
        "baseline": baseline,
        "fine_tuned": fine,
        "note": (
            "Simulation proxy: same instances appear in train views; this measures "
            "view invariance, not new-instance generalization or sim2real."
        ),
    }
    (out / "instance_view_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "INSTANCE_VIEW_REPORT.md").write_text(
        _render_report(metrics), encoding="utf-8"
    )
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
        "# Instance-level view holdout",
        "",
        f"- data: `{metrics['data']}`",
        f"- train views: {metrics['split']['train_views']}; test views: {metrics['split']['test_views']}",
        f"- instances: {metrics['instances']}; train views: {metrics['counts']['train']}; test views: {metrics['counts']['test']}",
        f"- epochs: {metrics['epochs']}; batch: {metrics['batch_size']}",
        f"- instance weight: {metrics['instance_weight']}; category weight: {metrics['category_weight']}; material weight: {metrics['material_weight']}",
        "",
        "## Same-instance / different-view retrieval (test views)",
        "",
        "| pair | baseline R@1 | fine-tuned R@1 | baseline R@5 | fine-tuned R@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    base = metrics["baseline"]["instance_retrieval"]
    fine = metrics["fine_tuned"]["instance_retrieval"]
    for pair in base:
        lines.append(
            f"| {pair} | {_fmt(base[pair]['r1'])} | {_fmt(fine[pair]['r1'])} "
            f"| {_fmt(base[pair]['r5'])} | {_fmt(fine[pair]['r5'])} |"
        )
    lines += [
        "",
        "## Query(test views) -> gallery(train views), same instance",
        "",
        "| pair | baseline R@1 | fine-tuned R@1 | baseline R@5 | fine-tuned R@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    base_qg = metrics["baseline"]["instance_retrieval_qg"]
    fine_qg = metrics["fine_tuned"]["instance_retrieval_qg"]
    for pair in base_qg:
        lines.append(
            f"| {pair} | {_fmt(base_qg[pair]['r1'])} | {_fmt(fine_qg[pair]['r1'])} "
            f"| {_fmt(base_qg[pair]['r5'])} | {_fmt(fine_qg[pair]['r5'])} |"
        )
    lines += [
        "",
        "## Category R@1 (test views)",
        "",
        "| pair | baseline | fine-tuned |",
        "|---|---:|---:|",
    ]
    for pair, value in metrics["baseline"]["category_r1"].items():
        lines.append(
            f"| {pair} | {_fmt(value)} | {_fmt(metrics['fine_tuned']['category_r1'][pair])} |"
        )
    lines += [
        "",
        "## Attribute retention",
        "",
        "| modality | baseline | fine-tuned |",
        "|---|---:|---:|",
    ]
    for modality in MODALITIES:
        lines.append(
            f"| {modality} | {metrics['baseline']['attribute_retention'][modality]['mean_retention']:.1%} "
            f"| {metrics['fine_tuned']['attribute_retention'][modality]['mean_retention']:.1%} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        "- Instance fine-tuning should raise same-instance/different-view R@1 without collapsing category R@1.",
        "- This is a simulation proxy. The same instances appear in train views; it measures view invariance,",
        "  not new-instance generalization or real-sensor sim2real.",
        "- Real RD2/R-Gate remains deferred until synchronized real visual-tactile data exists.",
        "",
    ]
    return "\n".join(lines)
