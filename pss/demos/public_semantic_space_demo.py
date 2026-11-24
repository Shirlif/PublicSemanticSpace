"""Minimal end-to-end public semantic space demo.

Three steps:
1. frozen expert per modality (open-source / pretrained / handcrafted);
2. one MLP projection head per modality -> 128-d public space;
3. prototype bank + recognition.

Run: python -m pss.demos.public_semantic_space_demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pss._console import ensure_utf8
from pss.encoders.data import crop_object, load_index
from pss.encoders.tactile.wtmm import WTMMTactileExpert, load_wtmm_tactile
from pss.encoders.vision import load_vision_expert
from pss.inference import PublicSemanticSpace
from pss.projection import load_projection
from pss.projection.features import load_paired_features
from pss.prototypes import build_bank, recognize
from pss.sim.dataset import load_sample

MODALITIES = ("vision", "tactile", "geometry")
WEIGHTS = {"vision": 1.0, "tactile": 0.5, "geometry": 0.95}


def run() -> dict:
    ensure_utf8()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. frozen experts
    vision = load_vision_expert("checkpoints/expert_v1/vision.pt", device=device)
    tactile = WTMMTactileExpert(
        encoder=load_wtmm_tactile("checkpoints/tactile_wtmm/tactile_wtmm.pt", device=device),
        device=device,
        input_dim=64,
    )

    # 2. per-modality MLP projection heads -> 128-d public space
    heads = {
        m: load_projection(f"checkpoints/projection_wtmm/{m}.pt", device=device)
        for m in MODALITIES
    }
    feats = load_paired_features("data/expert_v1", vision, tactile)
    with torch.no_grad():
        Z = {
            m: heads[m](
                torch.as_tensor(getattr(feats, m), dtype=torch.float32, device=device)
            ).cpu().numpy()
            for m in MODALITIES
        }

    # 3. prototype bank + recognition
    splits = np.array(feats.splits)
    classes = sorted({feats.categories[i] for i in np.where(splits == "train")[0]})
    train_idx = np.array(
        [i for i in np.where(splits == "train")[0] if feats.categories[i] in classes],
        dtype=int,
    )
    test_idx = np.array(
        [i for i in np.where(splits == "test")[0] if feats.categories[i] in classes],
        dtype=int,
    )
    bank = build_bank(feats, Z, train_idx, classes, version="demo")

    correct = 0
    sample = None
    for i in test_idx:
        result = recognize(
            {m: Z[m][i] for m in MODALITIES},
            bank,
            weights=WEIGHTS,
            use_tree=False,
        )
        ok = bool(result.ranked) and result.ranked[0][0] == feats.categories[i]
        correct += int(ok)
        if sample is None:
            sample = {
                "true": feats.categories[i],
                "top3": [(c, round(s, 3)) for c, s in result.ranked[:3]],
                "per_modality_top": result.modality_top,
                "conflict": result.conflict,
            }
    accuracy = correct / max(1, len(test_idx))

    print("=== 最小公共语义空间 Demo ===")
    print(f"设备: {device}")
    print(f"视觉专家: SmallCNN (256-d) | 触觉专家: WT-MM CNN (64-d) | 几何: 手工 16-d")
    print(f"公共空间: 128-d, L2 归一化")
    print(f"类别: {classes}")
    print(f"train/test: {len(train_idx)}/{len(test_idx)}")
    print(f"测试集 top-1: {accuracy:.3f}")
    print()
    print("示例识别:")
    print(f"  真值: {sample['true']}")
    print(f"  融合 Top-3: {sample['top3']}")
    print(f"  各模态 top-1: {sample['per_modality_top']}")
    print(f"  冲突标记: {sample['conflict']}")
    return {"accuracy": accuracy, "sample": sample, "classes": classes}


def run_samples(n: int = 12, with_geometry: bool = True, top_k: int = 3) -> dict:
    space = PublicSemanticSpace()
    if space.bank is None:
        space.build_bank("data/expert_v1")
    data_dir = Path("data/expert_v1")
    index = load_index(data_dir)
    entries = [
        e
        for e in index["entries"]
        if e["kind"] == "category" and e["split"] == "test"
    ]
    by_category: dict[str, list[dict]] = {}
    for entry in entries:
        by_category.setdefault(entry["category"], []).append(entry)
    categories = sorted(by_category)
    selected = []
    for k in range(n):
        category = categories[k % len(categories)]
        pool = by_category[category]
        selected.append(pool[(k // len(categories)) % len(pool)])

    print(f"=== 测试样本逐条结果（{len(selected)} 条，图 + 触觉{' + 几何' if with_geometry else ''}）===")
    header = f"{'#':>2}  {'真值':<10} {'V+T':<10} {'V+T+G':<10} {'触觉':<10} {'冲突':<5} Top-3"
    print(header)
    print("-" * len(header))
    vt_correct = vtg_correct = vt_conflict = vtg_conflict = 0
    rows = []
    for k, entry in enumerate(selected):
        sample = load_sample(data_dir / entry["path"])
        view = k % sample["rgb"].shape[0]
        geom_id = int(sample["geom_id"])
        crop = crop_object(sample["rgb"][view], sample["segmentation"][view], geom_id, size=64)
        tactile = sample["tactile"][view]
        vt = space.recognize(image=crop, tactile=tactile, top_k=top_k)
        geometry = (
            (sample["depth"][view], sample["segmentation"][view], geom_id)
            if with_geometry
            else None
        )
        vtg = (
            space.recognize(image=crop, tactile=tactile, geometry=geometry, top_k=top_k)
            if with_geometry
            else None
        )
        true = sample["category"]
        vt_pred = vt["ranked"][0][0] if vt["ranked"] else "-"
        vtg_pred = vtg["ranked"][0][0] if vtg and vtg["ranked"] else "-"
        tactile_top = vt["per_modality_top"].get("tactile", "-")
        conflict = vt["conflict"]
        vt_correct += int(vt_pred == true)
        vt_conflict += int(conflict)
        if vtg is not None:
            vtg_correct += int(vtg_pred == true)
            vtg_conflict += int(vtg["conflict"])
        top3 = ", ".join(f"{c}({s:+.2f})" for c, s in vt["ranked"])
        print(
            f"{k+1:>2}  {true:<10} {vt_pred:<10} {vtg_pred:<10} {tactile_top:<10} "
            f"{str(conflict):<5} {top3}"
        )
        rows.append(
            {
                "true": true,
                "vt_pred": vt_pred,
                "vtg_pred": vtg_pred,
                "tactile_top": tactile_top,
                "conflict": conflict,
                "top3": vt["ranked"],
            }
        )

    summary = {
        "n": len(selected),
        "vision_tactile_acc": vt_correct / max(1, len(selected)),
        "vision_tactile_geometry_acc": vtg_correct / max(1, len(selected)),
        "vision_tactile_conflict_rate": vt_conflict / max(1, len(selected)),
        "vision_tactile_geometry_conflict_rate": vtg_conflict / max(1, len(selected)),
        "rows": rows,
    }
    print()
    print(f"V+T top-1: {summary['vision_tactile_acc']:.3f} "
          f"| V+T+G top-1: {summary['vision_tactile_geometry_acc']:.3f}")
    print(f"冲突率: V+T {summary['vision_tactile_conflict_rate']:.3f} "
          f"| V+T+G {summary['vision_tactile_geometry_conflict_rate']:.3f}")
    return summary


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description="最小公共语义空间推理 Demo")
    ap.add_argument("--image", default=None, help="物体图片路径")
    ap.add_argument("--tactile", default=None, help="触觉数组 .npy（8x8 / 12x32 / 64 / 384）")
    ap.add_argument("--depth", default=None, help="可选：深度 .npy")
    ap.add_argument("--seg", default=None, help="可选：分割 .npy（通道 0 为 geom id）")
    ap.add_argument("--geom-id", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--build-bank", action="store_true", help="重建原型库后退出")
    ap.add_argument("--samples", type=int, default=0, help="逐条跑 N 个测试样本")
    ap.add_argument("--no-geometry", action="store_true", help="逐条模式不融合几何")
    args = ap.parse_args()

    if args.samples:
        run_samples(args.samples, with_geometry=not args.no_geometry, top_k=args.top_k)
        return

    if args.build_bank:
        space = PublicSemanticSpace()
        bank = space.build_bank("data/expert_v1")
        print(f"原型库已保存: {space.bank_path}")
        print(f"类别: {bank.categories}")
        return

    if args.image or args.tactile:
        space = PublicSemanticSpace()
        geometry = None
        if args.depth and args.seg:
            geometry = (np.load(args.depth), np.load(args.seg), args.geom_id)
        result = space.recognize(
            image=args.image,
            tactile=args.tactile,
            geometry=geometry,
            top_k=args.top_k,
        )
        print("=== 公共语义空间推理 ===")
        print(f"输入模态: {result['modalities']}")
        print("Top-k:")
        for i, (category, score) in enumerate(result["ranked"], 1):
            print(f"  {i}. {category:<12} {score:+.3f}")
        print(f"各模态 top-1: {result['per_modality_top']}")
        print(f"冲突标记: {result['conflict']}")
        return

    run()


if __name__ == "__main__":
    main()
