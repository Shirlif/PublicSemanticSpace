"""U1b: CLIP/DINOv2 + three projection seeds + open-set recalibration.

Executable form of ``docs/plans/PSS-架构上界复核与下一阶段.md`` §10 (U1b).

U1 swapped the vision backbone for DINOv2 and found a closed-set gain
(+12.5 pt) but an open-set regression (AUROC 0.9866, reject@95 0.918).  U1b
asks whether that regression is fixable without giving up the gain:

  proto        deployed prototype cosine (reference)
  lda          per-modality PCA-whitening -> concat LDA, shrinkage chosen on
               val seen + held-out calibration half
  dual_head    closed-set from LDA, open-set from the prototype head

Three projection seeds measure training variance.  The held-out category split
is divided into calibration and test halves so the shrinkage choice never sees
the test half.

Example
-------
python -u -m pss.eval.scripts.run_upper_bound_u1b \
  --backbone clip --out eval/results/upper_bound/u1b_clip --device cuda
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
    _acc,
    _auroc,
    _write_json,
    build_bank,
    concat,
    load_features,
    task_split,
    train_projection,
)
from pss.eval.scripts.run_fusion_ablation import _proto_scores, _whiten
from pss.eval.scripts.run_readout_scorers import _reject_at
from pss.eval.scripts.run_upper_bound_u1 import _dinov2_vision_features
from pss.projection.heads import save_projection
from pss.prototypes.scorers import MahalanobisScorer, predict_classes

BACKBONES = ("dinov2", "clip")
CLIP_IMAGE_SIZE = 224


def _clip_vision_features(
    data_dir: str, device: str, model_id: str
) -> tuple[np.ndarray, list[str], list[int]]:
    from pss.encoders.data import crop_object, load_index, normalize_image
    from pss.encoders.vision.clip import load_clip_vision_encoder
    from pss.sim.dataset import load_sample

    encoder = load_clip_vision_encoder(device=device, model_id=model_id)
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
                    size=CLIP_IMAGE_SIZE,
                    preserve_aspect=preserve_aspect,
                )
            )
            for v in range(sample["rgb"].shape[0])
        ]
        batch = torch.as_tensor(np.stack(crops), dtype=torch.float32)
        with torch.inference_mode():
            z = encoder.encode(batch, batch_size=16)
        feats.append(z.detach().cpu().numpy().astype(np.float32))
        sample_ids.extend([str(entry["id"])] * len(crops))
        views.extend(list(range(len(crops))))
        if len(feats) % 50 == 0:
            print(f"[{model_id}] {len(feats)} entries encoded", flush=True)
    return np.concatenate(feats, axis=0), sample_ids, views


def _build_cache(
    data_dir: str, out: Path, device: str, backbone: str, clip_model: str, refresh: bool
) -> dict:
    cache = out / "paired_features.npz"
    if cache.exists() and not refresh:
        print(f"[cache] reusing {cache}", flush=True)
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}

    base = load_features(
        data_dir, "checkpoints/category_pretrain_instance8_experts", out
    )
    if backbone == "dinov2":
        vision, sample_ids, views = _dinov2_vision_features(data_dir, device)
    elif backbone == "clip":
        vision, sample_ids, views = _clip_vision_features(data_dir, device, clip_model)
    else:
        raise ValueError(f"unknown backbone {backbone!r}")

    if list(map(str, base["sample_ids"])) != sample_ids:
        raise RuntimeError("sample id order changed; refusing to mix caches")
    if list(map(int, base["views"])) != views:
        raise RuntimeError("view order changed; refusing to mix caches")

    payload = {key: base[key] for key in base}
    payload["vision"] = vision
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    print(f"[cache] wrote {cache} vision={vision.shape}", flush=True)
    z = np.load(cache, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _project(
    raw: dict[str, np.ndarray],
    attrs: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    out: Path,
    device: str,
    *,
    seed: int,
    dim: int,
    epochs: int,
) -> dict[str, np.ndarray]:
    heads = train_projection(
        raw, attrs, labels, train_idx,
        transform="res", dim=dim, seed=seed, epochs=epochs,
        device=device, return_heads=True,
    )[1]
    proj_dir = out / f"projection_seed{seed}"
    proj_dir.mkdir(parents=True, exist_ok=True)
    for m, head in heads.items():
        save_projection(proj_dir / f"{m}.pt", head)
    rep = {}
    with torch.no_grad():
        for m in MODALITIES:
            heads[m].eval()
            rep[m] = heads[m](torch.as_tensor(raw[m]).to(device)).cpu().numpy().astype(np.float64)
    return rep


def _metrics(acc: float, seen: np.ndarray, unseen: np.ndarray) -> dict:
    return {
        "acc": float(acc),
        "auroc": float(_auroc(seen, unseen)),
        "reject": float(_reject_at(seen, unseen)),
    }


def _eval_seed(
    feats: dict,
    task: dict,
    out: Path,
    device: str,
    *,
    seed: int,
    dim: int,
    epochs: int,
    shrinkages: tuple[float, ...],
) -> dict:
    raw = {m: np.asarray(feats[m], dtype=np.float32) for m in MODALITIES}
    attrs = np.asarray(feats["attrs"], dtype=np.float32)
    rep = _project(raw, attrs, task["labels"], task["train_idx"], out, device,
                   seed=seed, dim=dim, epochs=epochs)
    whitened = _whiten(rep, task["train_idx"])

    Xtr = concat(whitened, task["train_idx"])
    Xva = concat(whitened, task["val_idx"])
    Xte = concat(whitened, task["test_idx"])
    held = np.asarray(task["held_idx"])
    held_cal = held[::2]
    held_test = held[1::2]
    Xcal = concat(whitened, held_cal)
    Xunseen = concat(whitened, held_test)

    bank = build_bank(rep, task["train_idx"], task["labels"], task["classes"])
    proto_te = _proto_scores(rep, bank, task["test_idx"], MODALITIES)
    proto_cal = _proto_scores(rep, bank, held_cal, MODALITIES)
    proto_unseen = _proto_scores(rep, bank, held_test, MODALITIES)
    proto = _metrics(
        _acc(proto_te.argmax(1), task["yte"]),
        proto_te.max(1),
        proto_unseen.max(1),
    )

    best = None
    for s in shrinkages:
        scorer = MahalanobisScorer(shrinkage=s).fit(Xtr, task["ytr"])
        cal_seen = scorer.scores(Xva).max(1)
        cal_unseen = scorer.scores(Xcal).max(1)
        cal_auroc = float(_auroc(cal_seen, cal_unseen))
        if best is None or cal_auroc > best["cal_auroc"]:
            best = {
                "shrinkage": float(s),
                "cal_auroc": cal_auroc,
                "cal_reject": float(_reject_at(cal_seen, cal_unseen)),
                "scorer": scorer,
            }
    scorer = best["scorer"]
    lda_seen = scorer.scores(Xte).max(1)
    lda_unseen = scorer.scores(Xunseen).max(1)
    lda = _metrics(
        _acc(predict_classes(scorer, Xte), task["yte"]),
        lda_seen,
        lda_unseen,
    )

    # Dual head: closed-set from LDA, open-set from prototype cosine.
    dual = {
        "acc": lda["acc"],
        "auroc": proto["auroc"],
        "reject": proto["reject"],
    }
    return {
        "seed": int(seed),
        "shrinkage": best["shrinkage"],
        "cal_auroc": best["cal_auroc"],
        "cal_reject": best["cal_reject"],
        "proto": proto,
        "lda": lda,
        "dual": dual,
    }


def _agg(rows: list[dict], group: str) -> dict:
    return {
        key: {
            "mean": float(np.mean([r[group][key] for r in rows])),
            "std": float(np.std([r[group][key] for r in rows])),
        }
        for key in ("acc", "auroc", "reject")
    }


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feats = _build_cache(
        args.data, out, args.device, args.backbone, args.clip_model, args.refresh
    )
    task = task_split(feats, "view-instance")
    shrinkages = tuple(float(s) for s in args.shrinkage_grid)

    rows = []
    for seed in args.projection_seeds:
        row = _eval_seed(
            feats, task, out, args.device,
            seed=seed, dim=args.dim, epochs=args.projection_epochs,
            shrinkages=shrinkages,
        )
        rows.append(row)
        print(
            f"[seed {seed}] proto {row['proto']['acc']:.3f}/{row['proto']['auroc']:.3f}/"
            f"{row['proto']['reject']:.3f} lda {row['lda']['acc']:.3f}/"
            f"{row['lda']['auroc']:.3f}/{row['lda']['reject']:.3f} "
            f"dual {row['dual']['acc']:.3f}/{row['dual']['auroc']:.3f}/"
            f"{row['dual']['reject']:.3f} s={row['shrinkage']}",
            flush=True,
        )

    proto = _agg(rows, "proto")
    lda = _agg(rows, "lda")
    dual = _agg(rows, "dual")
    observed = {
        "u_real_proto": {"test_acc": proto["acc"]["mean"]},
        "u_real_lda_whiten": {
            "accuracy_delta_vs_proto": lda["acc"]["mean"] - proto["acc"]["mean"],
            "auroc": lda["auroc"]["mean"],
            "unseen_reject_at_95": lda["reject"]["mean"],
        },
        "u_real_dual_head": {
            "accuracy_delta_vs_proto": dual["acc"]["mean"] - proto["acc"]["mean"],
            "auroc": dual["auroc"]["mean"],
            "unseen_reject_at_95": dual["reject"]["mean"],
        },
    }
    check = check_readout_reasoning_preregistration(observed)

    payload = {
        "stage": "u1b",
        "backbone": args.backbone,
        "clip_model": args.clip_model if args.backbone == "clip" else None,
        "data": args.data,
        "split": "view-instance (views 0-5 train / 6 val / 7 test; held split cal/test)",
        "projection_seeds": list(args.projection_seeds),
        "shrinkage_grid": list(shrinkages),
        "rows": rows,
        "proto": proto,
        "lda": lda,
        "dual": dual,
        "observed": observed,
        "preregistration": check,
        "note": (
            "LDA shrinkage is chosen on val seen + the first half of the held "
            "categories; the test half is never used for selection.  dual_head "
            "uses LDA for closed-set and prototype cosine for open-set."
        ),
    }
    _write_json(out / "u1b.json", payload)
    (out / "U1B_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _row(name: str, group: dict) -> str:
    return (
        f"| `{name}` | {group['acc']['mean']:.3f} ± {group['acc']['std']:.3f} | "
        f"{group['auroc']['mean']:.3f} ± {group['auroc']['std']:.3f} | "
        f"{group['reject']['mean']:.3f} ± {group['reject']['std']:.3f} |"
    )


def _render(p: dict) -> str:
    lines = [
        "# U1b：真实骨干 + 3 投影种子 + 开集重校准",
        "",
        f"- backbone: `{p['backbone']}`"
        + (f" (`{p['clip_model']}`)" if p.get("clip_model") else ""),
        f"- data: `{p['data']}` · split: {p['split']}",
        f"- projection seeds: {p['projection_seeds']} · shrinkage grid: {p['shrinkage_grid']}",
        "",
        "## 1. 主表（3 投影种子，mean ± std）",
        "",
        "| arm | closed-set acc | AUROC | unseen reject @95 |",
        "|---|---:|---:|---:|",
        _row("proto", p["proto"]),
        _row("lda (shrinkage on cal)", p["lda"]),
        _row("dual_head (LDA + proto open-set)", p["dual"]),
        "",
        "## 2. 每 seed 明细",
        "",
        "| seed | shrinkage | proto acc | lda acc | lda AUROC | lda reject | dual AUROC | dual reject |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p["rows"]:
        lines.append(
            f"| {r['seed']} | {r['shrinkage']:.3f} | {r['proto']['acc']:.3f} | "
            f"{r['lda']['acc']:.3f} | {r['lda']['auroc']:.3f} | "
            f"{r['lda']['reject']:.3f} | {r['dual']['auroc']:.3f} | "
            f"{r['dual']['reject']:.3f} |"
        )
    check = p["preregistration"]
    lines += [
        "",
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
        "- `dual_head` 是 U1 开集失败的修复候选：闭集用 LDA，开集用原型 cosine。",
        "- 若 LDA 的开集在 shrinkage 重校准后仍不达标，而 dual_head 达标，"
        "结论是“读出要分维度”：闭集与开集不能用同一个头。",
        "- 若 dual_head 的闭集也掉，说明双头切换有代价，需要按类数/覆盖选择。",
        "",
        "## 5. 边界",
        "",
        "- 只换 vision；tactile/geometry 仍是合成专家。",
        "- shrinkage 在 val seen + held 前半选择，held 后半只报一次；"
        "不是全 held 调参。",
        "- 3 个投影种子只覆盖投影训练方差，不覆盖数据划分方差。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/category_pretrain_instance8")
    ap.add_argument("--backbone", choices=BACKBONES, default="clip")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out", default="eval/results/upper_bound/u1b")
    ap.add_argument("--projection-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--projection-epochs", type=int, default=120)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument(
        "--shrinkage-grid", nargs="+", type=float,
        default=[0.001, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    payload = run(args)
    print(json.dumps(payload["observed"], indent=2))
    check = payload["preregistration"]
    print(f"preregistration ok={check['ok']} checked={check['checked']}")
    for mismatch in check["mismatches"]:
        print(f"  - {mismatch}")


if __name__ == "__main__":
    main()
