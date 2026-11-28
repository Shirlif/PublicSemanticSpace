"""U1: rerun the A-line readout on a real vision backbone (DINOv2).

Executable form of ``docs/plans/PSS-架构上界复核与下一阶段.md`` §6 (U1).

The synthetic A line uses a small task-trained CNN.  U1 swaps only the vision
modality for frozen DINOv2 (384-d), retrains the same 128-d projection with the
same recipe, and reruns the pre-registered readout arms.  Tactile and geometry
are unchanged, and the split is byte-identical, so the comparison isolates the
backbone x readout interaction.

The cached ``paired_features.npz`` is reused for tactile / geometry / labels;
only ``vision`` is replaced.  Row order is the same iteration over
``index["entries"]`` x views, and the script asserts the sample ids and views
match before writing.

Example
-------
python -u -m pss.eval.scripts.run_upper_bound_u1 \
  --out eval/results/upper_bound/u1 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pss._console import ensure_utf8
from pss.eval.readout_reasoning_preregistration import (
    check_readout_reasoning_preregistration,
)
from pss.eval.scripts.run_attribution_grid import (
    MODALITIES,
    _write_json,
    load_features,
    task_split,
    train_projection,
)
from pss.eval.scripts.run_fusion_ablation import run as run_fusion
from pss.eval.scripts.run_rerank_eval import run as run_rerank
from pss.projection.features import load_paired_features
from pss.projection.heads import save_projection

DINOV2_IMAGE_SIZE = 224


def _dinov2_vision_features(data_dir: str, device: str) -> tuple[np.ndarray, list[str], list[int]]:
    """Encode every dataset crop with frozen DINOv2, entry by entry.

    Encoding per entry keeps peak memory at 8 images instead of 3,840.
    """

    from pss.encoders.data import crop_object, load_index, normalize_image
    from pss.encoders.vision.dinov2 import load_dinov2_encoder
    from pss.sim.dataset import load_sample

    encoder = load_dinov2_encoder(device=device)
    index = load_index(Path(data_dir))
    preserve_aspect = bool(index.get("config", {}).get("crop_preserve_aspect", False))

    feats: list[np.ndarray] = []
    sample_ids: list[str] = []
    views: list[int] = []
    for entry in index["entries"]:
        if entry["kind"] != "category":
            continue
        sample = load_sample(Path(data_dir) / entry["path"])
        geom_id = int(sample["geom_id"])
        crops = [
            normalize_image(
                crop_object(
                    sample["rgb"][v],
                    sample["segmentation"][v],
                    geom_id,
                    size=DINOV2_IMAGE_SIZE,
                    preserve_aspect=preserve_aspect,
                )
            )
            for v in range(sample["rgb"].shape[0])
        ]
        batch = torch.as_tensor(np.stack(crops), dtype=torch.float32)
        with torch.inference_mode():
            z = encoder.encode(batch, batch_size=8)
        feats.append(z.detach().cpu().numpy().astype(np.float32))
        sample_ids.extend([str(entry["id"])] * len(crops))
        views.extend(list(range(len(crops))))
        if len(feats) % 50 == 0:
            print(f"[dinov2] {len(feats)} entries encoded", flush=True)

    return np.concatenate(feats, axis=0), sample_ids, views


def _build_cache(
    data_dir: str,
    out: Path,
    device: str,
    *,
    refresh: bool = False,
) -> Path:
    """Write ``paired_features.npz`` with DINOv2 vision and reused rest."""

    cache = out / "paired_features.npz"
    if cache.exists() and not refresh:
        print(f"[cache] reusing {cache}", flush=True)
        return cache

    # Reuse the existing synthetic cache for tactile/geometry/attrs/labels.
    base = load_features(data_dir, "checkpoints/category_pretrain_instance8_experts", out)
    vision, sample_ids, views = _dinov2_vision_features(data_dir, device)

    if list(map(str, base["sample_ids"])) != sample_ids:
        raise RuntimeError("sample id order changed; refusing to mix caches")
    if list(map(int, base["views"])) != views:
        raise RuntimeError("view order changed; refusing to mix caches")

    payload = {key: base[key] for key in base}
    payload["vision"] = vision
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    print(f"[cache] wrote {cache} vision={vision.shape}", flush=True)
    return cache


def _train_and_save_projection(
    feats: dict,
    out: Path,
    device: str,
    *,
    dim: int,
    epochs: int,
    seed: int,
) -> Path:
    task = task_split(feats, "view-instance")
    raw = {m: np.asarray(feats[m], dtype=np.float32) for m in MODALITIES}
    heads = train_projection(
        raw,
        np.asarray(feats["attrs"], dtype=np.float32),
        task["labels"],
        task["train_idx"],
        transform="res",
        dim=dim,
        seed=seed,
        epochs=epochs,
        device=device,
        return_heads=True,
    )[1]
    proj = out / "projection"
    proj.mkdir(parents=True, exist_ok=True)
    for m, head in heads.items():
        save_projection(proj / f"{m}.pt", head)
    print(f"[projection] wrote {proj} dim={dim} seed={seed}", flush=True)
    return proj


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _observe(fusion: dict, rerank: dict | None) -> dict:
    proto_acc = float(fusion["arms"]["proto_cosine"]["test_acc"])
    lda = fusion["arms"]["lda_concat_whiten"]
    observed = {
        "u_real_proto": {"test_acc": proto_acc},
        "u_real_mlp": {"test_acc": float(fusion["mlp_reference"]["acc"])},
        "u_real_lda_whiten": {
            "accuracy_delta_vs_proto": float(lda["test_acc"]) - proto_acc,
            "auroc": float(lda["auroc"]),
            "unseen_reject_at_95": float(lda["unseen_reject_at_95"]),
        },
    }
    if rerank is not None:
        prereg_k = int(rerank["rerank"]["prereg_k"])
        by_k = rerank["rerank"]["by_k"]
        row = by_k.get(prereg_k, by_k.get(str(prereg_k)))
        if row is None:
            raise KeyError(f"rerank payload has no k={prereg_k}: {list(by_k)}")
        gate_acc = float(rerank["gate"]["acc"])
        observed["u_real_rerank"] = {
            "accuracy_delta_vs_gate": float(row["test_acc"]) - gate_acc,
            "accuracy_delta_vs_proto": float(row["test_acc"]) - proto_acc,
        }
    return observed


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reuse = bool(getattr(args, "reuse", False))
    fusion_json = out / "fusion_ablation.json"
    rerank_json = out / "rerank_eval.json"
    if reuse and fusion_json.exists():
        fusion = json.loads(fusion_json.read_text(encoding="utf-8"))
        rerank = (
            json.loads(rerank_json.read_text(encoding="utf-8"))
            if rerank_json.exists() and not args.skip_rerank
            else None
        )
        proj = Path(fusion.get("projections", out / "projection"))
        print(f"[reuse] fusion={fusion_json} rerank={rerank_json if rerank else 'none'}", flush=True)
    else:
        _build_cache(args.data, out, args.device, refresh=args.refresh)
        feats = load_features(args.data, args.experts, out)
        proj = _train_and_save_projection(
            feats,
            out,
            args.device,
            dim=args.dim,
            epochs=args.projection_epochs,
            seed=args.projection_seed,
        )

        fusion_args = SimpleNamespace(
            data=args.data,
            experts=args.experts,
            projections=str(proj),
            regime="view-instance",
            out=str(out),
            seeds=list(args.seeds),
            device=args.device,
        )
        fusion = run_fusion(fusion_args)

        rerank = None
        if not args.skip_rerank:
            rerank_args = SimpleNamespace(
                data=args.data,
                experts=args.experts,
                projections=str(proj),
                regime="view-instance",
                out=str(out),
                seeds=list(args.seeds),
                k_grid=[3, 5, 10],
                prereg_k=5,
                epochs=args.rerank_epochs,
                device=args.device,
            )
            rerank = run_rerank(rerank_args)

    observed = _observe(fusion, rerank)
    check = check_readout_reasoning_preregistration(observed)

    control = _read_json(Path(args.control_json)) if args.control_json else None
    payload = {
        "stage": "u1",
        "backbone": "dinov2_small",
        "data": args.data,
        "projections": str(proj),
        "split": "view-instance (views 0-5 train / 6 val / 7 test)",
        "observed": observed,
        "fusion": fusion,
        "rerank": rerank,
        "control_fusion": control,
        "preregistration": check,
        "note": (
            "vision is frozen DINOv2 384-d; tactile/geometry and labels are "
            "the synthetic cache; projection is retrained with the same "
            "recipe (res, 128-d, 120 epochs)."
        ),
    }
    _write_json(out / "u1.json", payload)
    (out / "U1_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _render(p: dict) -> str:
    proto = p["observed"]["u_real_proto"]["test_acc"]
    lda = p["observed"]["u_real_lda_whiten"]
    lines = [
        "# U1：真实视觉骨干（DINOv2）上的 A 线读出",
        "",
        f"- data: `{p['data']}` · split: {p['split']}",
        f"- backbone: `{p['backbone']}` · projections: `{p['projections']}`",
        "",
        "## 1. 主表",
        "",
        "| arm | closed-set acc | Δ vs proto | AUROC | unseen reject @95 |",
        "|---|---:|---:|---:|---:|",
        f"| `u_real_proto` | {proto:.3f} | — | — | — |",
        f"| `u_real_lda_whiten` | {proto + lda['accuracy_delta_vs_proto']:.3f} | "
        f"{lda['accuracy_delta_vs_proto']:+.3f} | {lda['auroc']:.3f} | "
        f"{lda['unseen_reject_at_95']:.3f} |",
    ]
    if "u_real_rerank" in p["observed"]:
        rr = p["observed"]["u_real_rerank"]
        lines.append(
            f"| `u_real_rerank` | — | {rr['accuracy_delta_vs_gate']:+.3f} vs gate | — | — |"
        )
    lines += [
        f"| `u_real_mlp` | {p['observed']['u_real_mlp']['test_acc']:.3f} | — | — | — |",
        "",
    ]

    control = p.get("control_fusion")
    if control:
        c_proto = float(control["arms"]["proto_cosine"]["test_acc"])
        c_lda = control["arms"]["lda_concat_whiten"]
        lines += [
            "## 2. 与合成 A 线对照（同一读出，仅换骨干）",
            "",
            "| backbone | proto | lda_whiten | Δ vs proto |",
            "|---|---:|---:|---:|",
            f"| synthetic CNN | {c_proto:.3f} | {float(c_lda['test_acc']):.3f} | "
            f"{float(c_lda['test_acc']) - c_proto:+.3f} |",
            f"| DINOv2 | {proto:.3f} | {proto + lda['accuracy_delta_vs_proto']:.3f} | "
            f"{lda['accuracy_delta_vs_proto']:+.3f} |",
            "",
        ]

    check = p["preregistration"]
    lines += [
        "## 3. 预注册检查",
        "",
        f"- 检查 arm 数: {check['checked']} · 通过: **{check['ok']}**",
    ]
    if check["mismatches"]:
        lines.append("")
        for mismatch in check["mismatches"]:
            lines.append(f"- `{mismatch}`")
    lines += [
        "",
        "## 4. 判读",
        "",
        "- `u_real_lda_whiten` 的 bar 是 **Δ vs proto ≥ +0.08**，不是绝对精度；"
        "DINOv2 的尺度可以与合成不同。",
        "- 若 Δ < +0.03，0.948 判为合成代理特异，所有“读出上界”措辞降级。",
        "- `u_real_rerank` 是负对照：它的 bar 是 **Δ vs 同一闸门 ≤ +0.005**；"
        "若重排超过闸门，A2 的“线性读已是上限”被证伪，必须重开读出线。",
        "- CLIP 权重未缓存；本报告只覆盖 DINOv2，不能声称 VLM。",
        "",
        "## 5. 边界",
        "",
        "- 只换 vision；tactile/geometry 仍是合成专家，因此这是“backbone × readout”"
        "交互测试，不是端到端真实感知。",
        "- 投影头按同一配方（res, 128-d, 120 epochs）重训，仍有训练随机性；"
        "本表是单投影 seed 的诊断。",
        "- DINOv2 权重是 `facebook/dinov2-small`（384-d），不是 VLM。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/category_pretrain_instance8")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_instance8_experts")
    ap.add_argument("--out", default="eval/results/upper_bound/u1")
    ap.add_argument("--control-json", default="eval/results/readout_opt/fusion_ablation.json")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--projection-epochs", type=int, default=120)
    ap.add_argument("--projection-seed", type=int, default=0)
    ap.add_argument("--rerank-epochs", type=int, default=200)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-rerank", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    payload = run(args)
    print(json.dumps(payload["observed"], indent=2))
    check = payload["preregistration"]
    print(f"preregistration ok={check['ok']} checked={check['checked']}")
    for mismatch in check["mismatches"]:
        print(f"  - {mismatch}")


if __name__ == "__main__":
    main()
