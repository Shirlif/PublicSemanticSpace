"""T1 data diagnostics: is there instance-level signal before training?

Metrics per modality and fused:
- instance nearest-neighbor accuracy (leave-one-out) vs chance;
- same-instance vs different-instance cosine gap;
- category NN accuracy (control);
- cross-session consistency when session_id is available.

Use before spending time on instance-level SupCon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


@dataclass
class T1Data:
    features: dict[str, np.ndarray]
    object_ids: np.ndarray
    categories: np.ndarray | None = None
    session_ids: np.ndarray | None = None
    source: str = "unknown"


def load_npz(path: str | Path) -> T1Data:
    with np.load(Path(path), allow_pickle=False) as f:
        features = {
            key: f[key].astype(np.float32)
            for key in ("vision", "tactile", "geometry")
            if key in f.files
        }
        object_ids = np.asarray([str(x) for x in f["object_id"]])
        categories = (
            np.asarray([str(x) for x in f["category"]]) if "category" in f.files else None
        )
        session_ids = (
            np.asarray([str(x) for x in f["session_id"]])
            if "session_id" in f.files
            else None
        )
    if not features:
        raise ValueError("npz must contain at least one of vision/tactile/geometry")
    return T1Data(features, object_ids, categories, session_ids, source=str(path))


def load_expert_v1(
    data_dir: str | Path = "data/expert_v1",
    vision_ckpt: str | Path = "checkpoints/expert_v1/vision.pt",
    tactile_ckpt: str | Path = "checkpoints/tactile_wtmm/tactile_wtmm.pt",
) -> T1Data:
    import torch

    from pss.encoders.tactile.wtmm import WTMMTactileExpert, load_wtmm_tactile
    from pss.encoders.vision import load_vision_expert
    from pss.projection.features import load_paired_features

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision = load_vision_expert(vision_ckpt, device=device)
    tactile = WTMMTactileExpert(
        encoder=load_wtmm_tactile(tactile_ckpt, device=device),
        device=device,
        input_dim=64,
    )
    feats = load_paired_features(data_dir, vision, tactile)
    return T1Data(
        features={
            "vision": feats.vision,
            "tactile": feats.tactile,
            "geometry": feats.geometry,
        },
        object_ids=np.asarray(feats.instances),
        categories=np.asarray(feats.categories),
        session_ids=None,
        source=str(data_dir),
    )


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = _normalize(x)
    return x @ x.T


def _nn_stats(sim: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    n = len(labels)
    if n < 2:
        return 0.0, 0.0
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)
    top = sim.argmax(axis=1)
    correct = sum(int(labels[top[i]] == labels[i]) for i in range(n))
    counts = {lab: int((labels == lab).sum()) for lab in set(labels.tolist())}
    chance = float(np.mean([(counts[labels[i]] - 1) / (n - 1) for i in range(n)]))
    return correct / n, chance


def _similarity_gap(sim: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    same = labels[:, None] == labels[None, :]
    off = ~np.eye(len(labels), dtype=bool)
    same_mask = same & off
    diff_mask = ~same
    if not same_mask.any() or not diff_mask.any():
        return 0.0, 0.0, 0.0
    same_sim = float(sim[same_mask].mean())
    diff_sim = float(sim[diff_mask].mean())
    return same_sim, diff_sim, same_sim - diff_sim


def _cross_session(sim: np.ndarray, object_ids: np.ndarray, session_ids: np.ndarray) -> dict:
    same_obj = object_ids[:, None] == object_ids[None, :]
    diff_session = session_ids[:, None] != session_ids[None, :]
    mask = same_obj & diff_session
    if not mask.any():
        return {"available": False, "pairs": 0}
    diff_obj = ~same_obj
    return {
        "available": True,
        "pairs": int(mask.sum()),
        "cross_session_same_obj": float(sim[mask].mean()),
        "different_obj": float(sim[diff_obj].mean()),
        "gap": float(sim[mask].mean() - sim[diff_obj].mean()),
    }


def diagnose(data: T1Data, max_samples: int = 5000, seed: int = 0) -> dict:
    n = len(data.object_ids)
    if n > max_samples:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=max_samples, replace=False)
        data = T1Data(
            {m: v[keep] for m, v in data.features.items()},
            data.object_ids[keep],
            data.categories[keep] if data.categories is not None else None,
            data.session_ids[keep] if data.session_ids is not None else None,
            data.source,
        )
        n = max_samples

    result = {
        "n": int(n),
        "objects": int(len(set(data.object_ids.tolist()))),
        "categories": int(len(set(data.categories.tolist()))) if data.categories is not None else None,
        "sessions": int(len(set(data.session_ids.tolist()))) if data.session_ids is not None else None,
        "source": data.source,
        "per_modality": {},
    }

    fused_blocks = []
    for modality, features in data.features.items():
        if len(features) != n:
            continue
        sim = _cosine_matrix(features)
        inst_acc, inst_chance = _nn_stats(sim, data.object_ids)
        same_sim, diff_sim, gap = _similarity_gap(sim, data.object_ids)
        row = {
            "instance_nn": inst_acc,
            "instance_chance": inst_chance,
            "instance_gap": gap,
            "same_sim": same_sim,
            "diff_sim": diff_sim,
        }
        if data.categories is not None:
            cat_acc, cat_chance = _nn_stats(sim, data.categories)
            row["category_nn"] = cat_acc
            row["category_chance"] = cat_chance
        if data.session_ids is not None:
            row["cross_session"] = _cross_session(sim, data.object_ids, data.session_ids)
        result["per_modality"][modality] = row
        fused_blocks.append(_normalize(features))

    if fused_blocks:
        fused = np.concatenate(fused_blocks, axis=1)
        sim = _cosine_matrix(fused)
        inst_acc, inst_chance = _nn_stats(sim, data.object_ids)
        same_sim, diff_sim, gap = _similarity_gap(sim, data.object_ids)
        fused_row = {
            "instance_nn": inst_acc,
            "instance_chance": inst_chance,
            "instance_gap": gap,
            "same_sim": same_sim,
            "diff_sim": diff_sim,
        }
        if data.categories is not None:
            cat_acc, cat_chance = _nn_stats(sim, data.categories)
            fused_row["category_nn"] = cat_acc
            fused_row["category_chance"] = cat_chance
        if data.session_ids is not None:
            fused_row["cross_session"] = _cross_session(sim, data.object_ids, data.session_ids)
        result["fused"] = fused_row

    result["verdict"] = _verdict(result)
    return result


def _verdict(result: dict) -> str:
    fused = result.get("fused")
    if fused is None:
        return "no features"
    if fused["instance_nn"] < fused["instance_chance"] + 0.05 or fused["instance_gap"] < 0.02:
        return "weak: 原始特征几乎没有实例信号；先换传感器/专家，不要训实例级投影"

    weak_modalities = [
        name
        for name, row in result.get("per_modality", {}).items()
        if row["instance_nn"] < row["instance_chance"] + 0.05 or row["instance_gap"] < 0.02
    ]
    if weak_modalities:
        return (
            "partial: 融合特征能区分实例，但 "
            + ", ".join(weak_modalities)
            + " 的实例信号弱；跨模态实例检索会受限。先把弱模态的实例差异做出来，再训实例级投影"
        )
    if fused["instance_nn"] >= 0.5 and fused["instance_gap"] >= 0.1:
        return "strong: 原始特征已能区分实例，具备 T1 实例级训练价值"
    return "marginal: 有信号但偏弱；建议增加实例差异/视角/会话，再训实例级投影"


def format_report(result: dict) -> str:
    lines = [
        "=== T1 数据诊断 ===",
        f"source: {result['source']}",
        f"N={result['n']} objects={result['objects']} "
        f"categories={result['categories']} sessions={result['sessions']}",
        "",
        f"{'modality':<10} {'inst NN':>8} {'chance':>8} {'gap':>8} "
        f"{'cat NN':>8} {'cat chance':>10}",
        "-" * 58,
    ]
    rows = dict(result["per_modality"])
    if "fused" in result:
        rows["fused"] = result["fused"]
    for name, row in rows.items():
        cat_nn = row.get("category_nn")
        cat_ch = row.get("category_chance")
        lines.append(
            f"{name:<10} {row['instance_nn']:>8.3f} {row['instance_chance']:>8.3f} "
            f"{row['instance_gap']:>8.3f} "
            f"{(f'{cat_nn:.3f}' if cat_nn is not None else '-'):>8} "
            f"{(f'{cat_ch:.3f}' if cat_ch is not None else '-'):>10}"
        )
    if result.get("fused", {}).get("cross_session", {}).get("available"):
        cs = result["fused"]["cross_session"]
        lines += [
            "",
            f"跨会话：同物体跨会话={cs['cross_session_same_obj']:.3f} "
            f"不同物体={cs['different_obj']:.3f} gap={cs['gap']:.3f}",
        ]
    lines += ["", f"决策: {result['verdict']}"]
    return "\n".join(lines)
