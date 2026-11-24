"""Fifth priority: end-to-end / bilevel training with frozen encoders.

Trains projections + binder + typed relations + matcher jointly (and with
ablations), then evaluates recognition, re-ID, relation F1 and consistency.
The encoders are frozen; raw expert features are cached once.

Example:
    python -m pss.eval.scripts.run_bilevel --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from pss._console import ensure_utf8
from pss.encoders.data import crop_object, load_index, normalize_image
from pss.encoders.geometry import geometry_features
from pss.learning.bilevel import (
    EndToEndPSS,
    RawBatch,
    RawPairs,
    TrainConfig,
    consistency_metrics,
    evaluate_recognition,
    evaluate_reid,
    evaluate_relations,
    save_joint,
    train_bilevel,
)
from pss.memory.learned_metric import CATEGORIES, CATEGORY_TO_INDEX
from pss.memory.perception import PoseNoise, extract_frame, load_perception_models
from pss.memory.typed_relations import frame_targets
from pss.projection.features import load_paired_features
from pss.sim.dataset import load_sample

DATA = "data/category_pretrain"
TRACK = "data/tracking_bench_v2"
EXPERTS = "checkpoints/category_pretrain_experts"
PROJECTIONS = "checkpoints/phase45_projection"
OUT = "eval/results/bilevel"
CKPT = "checkpoints/bilevel"
N_SLOTS = 3


def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "n/a" if math.isnan(v) else f"{v:.{digits}f}"


def _safe(obj):
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _safe(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return _safe(obj.detach().cpu().numpy())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj


def _onehot(category: str | None) -> np.ndarray:
    out = np.zeros(len(CATEGORIES), dtype=np.float32)
    if category in CATEGORY_TO_INDEX:
        out[CATEGORY_TO_INDEX[category]] = 1.0
    return out


# ── feature caches ──────────────────────────────────────────────────────────


def category_data(device: str) -> tuple[dict, dict]:
    from pss.encoders.tactile import load_tactile_expert
    from pss.encoders.vision import load_vision_expert

    vision = load_vision_expert(f"{EXPERTS}/vision.pt", device=device)
    tactile = load_tactile_expert(f"{EXPERTS}/tactile.pt", device=device)
    feats = load_paired_features(DATA, vision, tactile)
    splits = np.array(feats.splits)
    train_cats = sorted({feats.categories[i] for i in np.where(splits == "train")[0]})
    cat_map = {c: i for i, c in enumerate(train_cats)}
    train_instances = sorted({feats.instances[i] for i in np.where(splits == "train")[0]})
    inst_map = {inst: i for i, inst in enumerate(train_instances)}

    def rows(split: str):
        idx = [
            i
            for i in np.where(splits == split)[0]
            if feats.categories[i] in cat_map
        ]
        if not idx:
            return None
        vision_x = torch.as_tensor(feats.vision[idx], dtype=torch.float32)
        tactile_x = torch.as_tensor(feats.tactile[idx], dtype=torch.float32)
        geometry_x = torch.as_tensor(feats.geometry[idx], dtype=torch.float32)
        labels = torch.as_tensor([cat_map[feats.categories[i]] for i in idx], dtype=torch.long)
        instances = torch.as_tensor([inst_map.get(feats.instances[i], -1) for i in idx], dtype=torch.long)
        pose = torch.zeros((len(idx), 3), dtype=torch.float32)
        category = torch.as_tensor(
            np.stack([_onehot(feats.categories[i]) for i in idx]), dtype=torch.float32
        )
        return {
            "vision": vision_x,
            "tactile": tactile_x,
            "geometry": geometry_x,
            "pose": pose,
            "category": category,
            "labels": labels,
            "instances": instances,
        }

    train = rows("train")
    test = rows("test")
    packed = {
        "vision": train["vision"],
        "tactile": train["tactile"],
        "geometry": train["geometry"],
        "pose": train["pose"],
        "category": train["category"],
        "labels": train["labels"],
        "instances": train["instances"],
        "train_vision": train["vision"],
        "train_tactile": train["tactile"],
        "train_geometry": train["geometry"],
        "train_pose": train["pose"],
        "train_category": train["category"],
        "train_labels": train["labels"],
        "test_vision": test["vision"],
        "test_tactile": test["tactile"],
        "test_geometry": test["geometry"],
        "test_pose": test["pose"],
        "test_category": test["category"],
        "test_labels": test["labels"],
        "n_classes": len(train_cats),
        "classes": train_cats,
    }
    return packed, {"n_train": len(train["labels"]), "n_test": len(test["labels"])}


def _frame_arrays(sample, frame, obs, models):
    vision, tactile, geometry, pose, category = [], [], [], [], []
    ids = []
    for o in obs[:N_SLOTS]:
        gid = str(o.gt_id)
        idx = [str(x) for x in sample["instance_ids"]].index(gid)
        geoms = [
            int(k) for k, v in sample["geom_to_instance"].items() if str(v) == gid
        ]
        crop = crop_object(
            sample["rgb"][frame],
            sample["segmentation"][frame],
            geoms[0] if geoms else int(sample["geom_to_instance"].get(0, 0)),
            size=models.image_size,
            preserve_aspect=True,
        )
        vision.append(normalize_image(crop))
        tactile.append(np.asarray(sample["tactile"][frame, idx]).reshape(-1).astype(np.float32))
        geometry.append(
            geometry_features(
                sample["depth"][frame],
                sample["segmentation"][frame],
                geoms[0] if geoms else 0,
            )
        )
        pose.append(np.asarray(o.pos, dtype=np.float32))
        category.append(_onehot(o.category or o.gt_category))
        ids.append(gid)
    if not vision:
        return None
    v = models.vision.encode(torch.as_tensor(np.stack(vision), dtype=torch.float32))
    t = models.tactile.encode(torch.as_tensor(np.stack(tactile), dtype=torch.float32))
    n = len(vision)
    out = {
        "vision": np.zeros((N_SLOTS, v.shape[1]), dtype=np.float32),
        "tactile": np.zeros((N_SLOTS, t.shape[1]), dtype=np.float32),
        "geometry": np.zeros((N_SLOTS, geometry[0].shape[0]), dtype=np.float32),
        "pose": np.zeros((N_SLOTS, 3), dtype=np.float32),
        "category": np.zeros((N_SLOTS, len(CATEGORIES)), dtype=np.float32),
        "mask": np.zeros(N_SLOTS, dtype=np.float32),
    }
    out["vision"][:n] = v.detach().cpu().numpy()
    out["tactile"][:n] = t.detach().cpu().numpy()
    out["geometry"][:n] = np.stack(geometry)
    out["pose"][:n] = np.stack(pose)
    out["category"][:n] = np.stack(category)
    out["mask"][:n] = 1.0
    return out, ids


def _pack_targets(target_lists):
    out = {}
    for name in ("spatial", "semantic", "tactile"):
        out[name] = torch.as_tensor(np.stack([t.as_dict()[name] for t in target_lists]), dtype=torch.float32)
    valid = torch.as_tensor(np.stack([t.valid for t in target_lists]), dtype=torch.float32)
    return out, valid


def build_raw_pairs(device: str) -> dict[str, RawPairs]:
    cache = Path(CKPT) / "raw_pairs_v2.pt"
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        return {
            split: RawPairs(
                pre=RawBatch(**payload[split]["pre"]),
                post=RawBatch(**payload[split]["post"]),
                labels=payload[split]["labels"],
                pre_targets=payload[split]["pre_targets"],
                post_targets=payload[split]["post_targets"],
                pre_valid=payload[split]["pre_valid"],
                post_valid=payload[split]["post_valid"],
                subsets=payload[split]["subsets"],
            )
            for split in payload
        }
    index = load_index(TRACK)
    models = load_perception_models(EXPERTS, PROJECTIONS, device=device)
    payload = {}
    for split in ("train", "val", "test"):
        pre_rows, post_rows, labels, pre_t, post_t, pre_v, post_v, subsets = (
            [], [], [], [], [], [], [], []
        )
        for entry in [
            e for e in index["entries"] if e.get("kind") == "occlusion" and e.get("split") == split
        ]:
            sample = load_sample(Path(TRACK) / entry["path"])
            f0 = int(sample["occlusion_start"]) - 1
            f1 = int(sample["occlusion_end"])
            pre_obs = extract_frame(sample, f0, models, PoseNoise(seed=0, enabled=False))
            post_obs = extract_frame(sample, f1, models, PoseNoise(seed=0, enabled=False))
            pre = _frame_arrays(sample, f0, pre_obs, models)
            post = _frame_arrays(sample, f1, post_obs, models)
            if pre is None or post is None:
                continue
            pre_arr, pre_ids = pre
            post_arr, post_ids = post
            lab = np.zeros((N_SLOTS, N_SLOTS), dtype=np.float32)
            for i, a in enumerate(pre_ids):
                for j, b in enumerate(post_ids):
                    lab[i, j] = float(a == b)
            pre_rows.append(pre_arr)
            post_rows.append(post_arr)
            labels.append(lab)
            pre_t.append(frame_targets(pre_obs))
            post_t.append(frame_targets(post_obs))
            subsets.append(str(entry.get("subset", "unknown")))
        pre_batch = {
            k: torch.as_tensor(np.stack([r[k] for r in pre_rows]), dtype=torch.float32)
            for k in ("vision", "tactile", "geometry", "pose", "category", "mask")
        }
        post_batch = {
            k: torch.as_tensor(np.stack([r[k] for r in post_rows]), dtype=torch.float32)
            for k in ("vision", "tactile", "geometry", "pose", "category", "mask")
        }
        pre_targets, pre_valid = _pack_targets(pre_t)
        post_targets, post_valid = _pack_targets(post_t)
        payload[split] = {
            "pre": pre_batch,
            "post": post_batch,
            "labels": torch.as_tensor(np.stack(labels), dtype=torch.float32),
            "pre_targets": pre_targets,
            "post_targets": post_targets,
            "pre_valid": pre_valid,
            "post_valid": post_valid,
            "subsets": subsets,
        }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    return {
        split: RawPairs(
            pre=RawBatch(**payload[split]["pre"]),
            post=RawBatch(**payload[split]["post"]),
            labels=payload[split]["labels"],
            pre_targets=payload[split]["pre_targets"],
            post_targets=payload[split]["post_targets"],
            pre_valid=payload[split]["pre_valid"],
            post_valid=payload[split]["post_valid"],
            subsets=payload[split]["subsets"],
        )
        for split in payload
    }


# ── run ─────────────────────────────────────────────────────────────────────


def run(device: str = "cuda", epochs: int = 60, seeds: int = 1) -> dict:
    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    cat, counts = category_data(device)
    pairs = build_raw_pairs(device)
    modes = (
        "joint",
        "joint_instance",
        "reid_heavy",
        "sequential",
        "no_consistency",
        "no_relation",
        "linear_head",
        "concat",
    )
    results = {}
    for mode in modes:
        runs = []
        for seed in range(seeds):
            config = TrainConfig(
                mode="joint" if mode in ("joint_instance", "reid_heavy") else mode,
                epochs=epochs,
                seed=seed,
                n_categories=int(cat["n_classes"]),
                task="instance" if mode == "joint_instance" else "category",
                lambda_task=0.1 if mode == "reid_heavy" else 1.0,
                lambda_reid=2.0 if mode == "reid_heavy" else 0.5,
            )
            model, info = train_bilevel(cat, pairs["train"], config, device=device)
            rec = evaluate_recognition(model, cat, device)
            reid = evaluate_reid(model, pairs["test"], device)
            rel = evaluate_relations(model, pairs["test"], device)
            cons = consistency_metrics(model, pairs["test"], device)
            runs.append({"recognition": rec, "reid": reid, "relations": rel, "consistency": cons})
            if seed == 0:
                save_joint(model, Path(CKPT) / f"joint_{mode}.pt")
        results[mode] = {
            "recognition": float(np.mean([r["recognition"] for r in runs])),
            "reid_overall": float(np.mean([r["reid"]["overall"] for r in runs])),
            "reid_near": float(np.mean([r["reid"]["per_subset"]["near_identical"] for r in runs])),
            "relation_f1": {
                k: float(np.mean([r["relations"][k]["f1"] for r in runs]))
                for k in ("spatial", "semantic", "tactile")
            },
            "consistency": {
                k: float(np.mean([r["consistency"][k] for r in runs]))
                for k in runs[0]["consistency"]
            },
            "runs": runs,
        }
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "device": device,
        "epochs": epochs,
        "seeds": seeds,
        "counts": counts,
        "n_pairs": {k: len(v.labels) for k, v in pairs.items()},
        "results": results,
    }
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bilevel.json").write_text(
        json.dumps(_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "bilevel.md").write_text(render(payload), encoding="utf-8")
    return payload


def render(payload: dict) -> str:
    lines = [
        "# End-to-end / bilevel training (frozen encoders)",
        "",
        f"> generated: {payload['generated']}",
        f"> device: `{payload['device']}`; epochs {payload['epochs']}; seeds {payload['seeds']}",
        f"> category train/test: {payload['counts']['n_train']} / {payload['counts']['n_test']}",
        f"> re-ID pairs: {payload['n_pairs']}",
        "",
        "| mode | recognition | re-ID | near-identical | spatial F1 | semantic F1 | tactile F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, r in payload["results"].items():
        lines.append(
            f"| {mode} | {_fmt(r['recognition'])} | {_fmt(r['reid_overall'])} | "
            f"{_fmt(r['reid_near'])} | {_fmt(r['relation_f1']['spatial'])} | "
            f"{_fmt(r['relation_f1']['semantic'])} | {_fmt(r['relation_f1']['tactile'])} |"
        )
    lines += [
        "",
        "## Consistency / audit constraints",
        "",
        "| mode | antisymmetry error | exclusivity violation | permutation MSE | stability MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, r in payload["results"].items():
        c = r["consistency"]
        lines.append(
            f"| {mode} | {_fmt(c['antisymmetry_error'])} | {_fmt(c['exclusivity_violation'])} | "
            f"{_fmt(c['permutation_mse'])} | {_fmt(c['stability_mse'])} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Encoders are frozen; only projections, binder, relation heads and matcher train.",
        "- `joint` trains all losses together; `sequential` trains recognition first,",
        "  then freezes projections and trains relations/matching.",
        "- `no_consistency`, `no_relation`, `linear_head`, `concat` are ablations.",
        "- The prototype support mean is the differentiable inner memory; the query loss is",
        "  the outer task (bilevel-style episodic training).",
        "- Consistency metrics are lower-is-better.",
        "",
    ]
    return "\n".join(lines)


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    payload = run(args.device, args.epochs, args.seeds)
    for mode, r in payload["results"].items():
        print(
            f"{mode}: rec {r['recognition']:.3f}, reid {r['reid_overall']:.3f}, "
            f"near {r['reid_near']:.3f}"
        )
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
