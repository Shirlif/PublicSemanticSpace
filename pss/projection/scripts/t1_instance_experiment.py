"""T1 instance-level contrastive experiment on synthetic data.

Split by view: each instance's views 0-5 train, views 6-7 test. This mimics
asynchronous, object-ID-keyed collection: the model has seen the same object
from other views, but not the test views.

Compare:
- category-level SupCon (positives = same category)
- instance-level SupCon (positives = same instance_id)

Metrics: instance R@1, pair R@1, category R@1 (vision -> tactile).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.encoders.tactile.wtmm import WTMMTactileExpert, load_wtmm_tactile
from pss.encoders.vision import load_vision_expert
from pss.projection.features import load_paired_features
from pss.projection.heads import ProjectionHead
from pss.projection.losses import supcon_cross

MODALITIES = ("vision", "tactile", "geometry")


def _retrieval(z1, z2, labels, idx, mode):
    z1 = F.normalize(torch.as_tensor(z1[idx]), dim=-1)
    z2 = F.normalize(torch.as_tensor(z2[idx]), dim=-1)
    top = (z1 @ z2.T).argmax(dim=1).tolist()
    if mode == "pair":
        return float(np.mean([idx[t] == idx[k] for k, t in enumerate(top)]))
    return float(np.mean([labels[idx[t]] == labels[idx[k]] for k, t in enumerate(top)]))


def train_once(feats, level: str, seed: int = 0, epochs: int = 60, batch_size: int = 64):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V = torch.as_tensor(feats.vision)
    T = torch.as_tensor(feats.tactile)
    G = torch.as_tensor(feats.geometry)
    A = torch.as_tensor(feats.attrs)
    views = np.array(feats.views)
    labels_raw = feats.instances if level == "instance" else feats.categories
    label_map = {v: i for i, v in enumerate(sorted(set(labels_raw)))}
    y = torch.as_tensor([label_map[v] for v in labels_raw], dtype=torch.long)

    train_idx = np.where(views < 6)[0]
    test_idx = np.where(views >= 6)[0]
    X = {"vision": V, "tactile": T, "geometry": G}
    heads = {m: ProjectionHead(X[m].shape[1], 128).to(device) for m in MODALITIES}
    attr_heads = {m: nn.Linear(128, A.shape[1]).to(device) for m in MODALITIES}
    opt = torch.optim.Adam(
        [p for h in heads.values() for p in h.parameters()]
        + [p for h in attr_heads.values() for p in h.parameters()],
        lr=1e-3,
        weight_decay=1e-4,
    )

    for _epoch in range(epochs):
        for h in heads.values():
            h.train()
        for h in attr_heads.values():
            h.train()
        perm = torch.randperm(len(train_idx))
        for start in range(0, len(train_idx), batch_size):
            idx = torch.as_tensor(train_idx[perm[start : start + batch_size].numpy()])
            yb = y[idx].to(device)
            z = {m: heads[m](X[m][idx].to(device)) for m in MODALITIES}
            loss = (
                supcon_cross(z["vision"], z["tactile"], yb)
                + supcon_cross(z["tactile"], z["vision"], yb)
                + supcon_cross(z["vision"], z["geometry"], yb)
                + supcon_cross(z["geometry"], z["vision"], yb)
                + supcon_cross(z["tactile"], z["geometry"], yb)
                + supcon_cross(z["geometry"], z["tactile"], yb)
            ) / 6
            attr_loss = sum(
                F.mse_loss(attr_heads[m](z[m]), A[idx].to(device)) for m in MODALITIES
            ) / 3
            loss = loss + 0.1 * attr_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        Z = {
            m: heads[m](X[m].to(device)).cpu().numpy() for m in MODALITIES
        }
    metrics = {
        "level": level,
        "train_frames": int(len(train_idx)),
        "test_frames": int(len(test_idx)),
        "instances": int(len(set(feats.instances))),
        "categories": int(len(set(feats.categories))),
        "instance_r1": _retrieval(Z["vision"], Z["tactile"], feats.instances, test_idx, "instance"),
        "pair_r1": _retrieval(Z["vision"], Z["tactile"], feats.instances, test_idx, "pair"),
        "category_r1": _retrieval(Z["vision"], Z["tactile"], feats.categories, test_idx, "category"),
        "instance_r1_reverse": _retrieval(Z["tactile"], Z["vision"], feats.instances, test_idx, "instance"),
        "category_r1_reverse": _retrieval(Z["tactile"], Z["vision"], feats.categories, test_idx, "category"),
    }
    return metrics


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_v1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/results/t1_instance.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision = load_vision_expert("checkpoints/expert_v1/vision.pt", device=device)
    tactile = WTMMTactileExpert(
        encoder=load_wtmm_tactile("checkpoints/tactile_wtmm/tactile_wtmm.pt", device=device),
        device=device,
        input_dim=64,
    )
    feats = load_paired_features(args.data, vision, tactile)

    category = train_once(feats, "category", seed=args.seed, epochs=args.epochs)
    instance = train_once(feats, "instance", seed=args.seed, epochs=args.epochs)
    metrics = {
        "data": args.data,
        "epochs": args.epochs,
        "seed": args.seed,
        "category_level": category,
        "instance_level": instance,
        "note": "View split: train views 0-5, test views 6-7; T1 asynchronous proxy.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== T1 实例级对照（视角切分：0-5 train / 6-7 test）===")
    print(f"instances={category['instances']} categories={category['categories']} "
          f"train={category['train_frames']} test={category['test_frames']}")
    print()
    print(f"{'训练标签':<10} {'instance R@1':>14} {'pair R@1':>10} {'category R@1':>14}")
    print("-" * 52)
    for name, m in (("category", category), ("instance", instance)):
        print(f"{name:<10} {m['instance_r1']:>14.3f} {m['pair_r1']:>10.3f} "
              f"{m['category_r1']:>14.3f}")
    print()
    print(f"反向（tactile→vision）:")
    print(f"  category-level: instance={category['instance_r1_reverse']:.3f} "
          f"category={category['category_r1_reverse']:.3f}")
    print(f"  instance-level: instance={instance['instance_r1_reverse']:.3f} "
          f"category={instance['category_r1_reverse']:.3f}")


if __name__ == "__main__":
    main()
