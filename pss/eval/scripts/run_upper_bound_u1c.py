"""U1c: CLIP two-tower VLM proxy on the 16-class category task.

No real VLM weights are cached in the workspace (no Qwen-VL / LLaVA /
InternVL), so U1c is a **two-tower proxy**, not a VLM run:

* vision tower: frozen ``openai/clip-vit-base-patch16`` image encoder;
* text tower: the matching CLIP text encoder over the 16 seen category names.

The task is the semantic 16-class split of
``data/category_pretrain_instance8`` (``task_split(regime="category")``), not
the 384-class synthetic view-instance split used by U1/U1b.  Text prototypes
need real class names, and the held-out categories (cloth/tool/sponge/sheet)
provide the open set.

Arms
----
``proto``
    U1b recipe: 128-d learned projections, prototype cosine on the fused
    modalities (tactile/geometry are the synthetic experts, unchanged from
    U1/U1b).
``lda``
    U1b recipe: learned projection -> per-modality PCA whitening -> concat ->
    pooled-covariance LDA.  Shrinkage is chosen on validation seen plus the
    first half of the held-out categories.
``text_zero_shot``
    Cosine between CLIP's joint image embedding and the CLIP text prototypes
    of the 16 seen class names.  No learned projection.
``dual_head``
    Closed-set from ``lda``; open-set from the better of the text and
    prototype heads, selected on the calibration half of the held split
    (never on the test half).

Failure handling
----------------
If the CLIP text tower cannot be loaded, the script records the error and
falls back to Qwen2.5-1.5B hidden states ridge-aligned to the CLIP visual
class centroids.  That fallback is still a proxy and is labelled as such.

Example
-------
python -u -m pss.eval.scripts.run_upper_bound_u1c \
  --out eval/results/upper_bound/u1c --device cuda
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

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
)
from pss.eval.scripts.run_fusion_ablation import _proto_scores, _whiten
from pss.eval.scripts.run_readout_scorers import _reject_at
from pss.eval.scripts.run_upper_bound_u1b import _clip_vision_features, _project
from pss.prototypes.scorers import MahalanobisScorer, predict_classes

TEMPLATES = (
    "a photo of a {}.",
    "a close-up photo of a {}.",
    "a {} on a table.",
    "a small {}.",
)
QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def _l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


@dataclass
class TextTower:
    """Text encoder facade: ``embed(prompts) -> (n, d)`` L2-normalized."""

    source: str
    dim: int
    _embed: object
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def embed(self, prompts: list[str]) -> np.ndarray:
        return _l2_rows(self._embed(prompts))


def _clip_text_tower(model, tokenizer, device: str) -> TextTower:
    def embed(prompts: list[str]) -> np.ndarray:
        toks = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(device)
        with torch.inference_mode():
            return model.get_text_features(**toks).float().cpu().numpy()

    return TextTower(
        source="clip_text",
        dim=int(model.config.projection_dim),
        _embed=embed,
        extra={"model_id": getattr(model.config, "_name_or_path", "clip")},
    )


def _qwen_text_tower(device: str, model_id: str = QWEN_MODEL_ID) -> TextTower:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True).to(device).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def embed(prompts: list[str]) -> np.ndarray:
        toks = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.inference_mode():
            out = model(**toks, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        # Last non-padding token per prompt.
        lengths = toks["attention_mask"].sum(1) - 1
        rows = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        return rows.float().cpu().numpy()

    return TextTower(source="qwen2.5_1.5b_hidden", dim=int(model.config.hidden_size), _embed=embed)


def _load_text_tower(clip_model, tokenizer, device: str) -> TextTower:
    if clip_model is not None and tokenizer is not None:
        return _clip_text_tower(clip_model, tokenizer, device)
    try:
        return _qwen_text_tower(device)
    except Exception as exc:  # noqa: BLE001 - recorded, never silently swallowed
        raise RuntimeError(
            "CLIP text tower unavailable and Qwen2.5 fallback also failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _load_clip(stack: dict, model_id: str, device: str) -> tuple[object, object, str | None]:
    """Load the full CLIP model (vision projection + text tower), or record why not."""

    try:
        from transformers import CLIPModel, CLIPTokenizer

        model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
        tokenizer = CLIPTokenizer.from_pretrained(model_id, local_files_only=True)
        stack["clip_model"] = model
        stack["clip_tokenizer"] = tokenizer
        return model, tokenizer, None
    except Exception as exc:  # noqa: BLE001 - recorded in the payload
        error = f"{type(exc).__name__}: {exc}"
        stack["clip_error"] = error
        print(f"[text] CLIP model unavailable: {error}", flush=True)
        return None, None, error


def _build_cache(args, clip_model) -> dict:
    """Cache the paired features plus the CLIP 768-d and 512-d joint vision."""

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "paired_features.npz"
    if cache.exists() and not args.refresh:
        z = np.load(cache, allow_pickle=True)
        loaded = {k: z[k] for k in z.files}
        if "vision" in loaded and "vision_joint" in loaded:
            print(f"[cache] reusing {cache}", flush=True)
            return loaded

    base = load_features(args.data, args.experts, out)
    vision = None

    u1b_cache = Path(args.u1b_cache) if args.u1b_cache else None
    if u1b_cache is not None and u1b_cache.exists():
        z = np.load(u1b_cache, allow_pickle=True)
        if (
            "vision" in z.files
            and list(map(str, z["sample_ids"])) == list(map(str, base["sample_ids"]))
            and list(map(int, z["views"])) == list(map(int, base["views"]))
        ):
            vision = np.asarray(z["vision"], dtype=np.float32)
            print(f"[cache] reusing CLIP vision from {u1b_cache} {vision.shape}", flush=True)
        else:
            print(f"[cache] {u1b_cache} does not match the dataset; re-encoding", flush=True)

    if vision is None:
        vision, sample_ids, views = _clip_vision_features(
            args.data, args.device, args.clip_model
        )
        if list(map(str, base["sample_ids"])) != sample_ids:
            raise RuntimeError("sample id order changed; refusing to mix caches")
        if list(map(int, base["views"])) != views:
            raise RuntimeError("view order changed; refusing to mix caches")
        print(f"[cache] encoded CLIP vision {vision.shape}", flush=True)

    if clip_model is not None:
        projection = clip_model.visual_projection.weight.detach().cpu().numpy().astype(np.float64)
        joint = vision.astype(np.float64) @ projection.T
    else:
        # Fallback: no full CLIP model, so the 768-d hidden space is the joint space.
        joint = vision.astype(np.float64)
    joint = _l2_rows(joint).astype(np.float32)

    payload = {key: base[key] for key in base}
    payload["vision"] = vision
    payload["vision_joint"] = joint
    np.savez_compressed(cache, **payload)
    print(f"[cache] wrote {cache} vision={vision.shape} joint={joint.shape}", flush=True)
    z = np.load(cache, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _fit_ridge(text: np.ndarray, target: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Ridge map ``text (n,d) -> target (n,D)``; used only by the Qwen fallback."""

    text = np.asarray(text, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    gram = text.T @ text + lam * np.eye(text.shape[1])
    return np.linalg.solve(gram, text.T @ target)


def _text_prototypes(
    tower: TextTower,
    class_names: list[str],
    joint: np.ndarray,
    task: dict,
    *,
    align: bool,
) -> np.ndarray:
    """Build one text prototype per seen class.

    With ``align=False`` (CLIP text tower) the prototypes live in CLIP's joint
    embedding space.  With ``align=True`` (Qwen fallback) the text hidden
    states are ridge-aligned to the CLIP visual class centroids, which is a
    supervised proxy and is flagged in the report.
    """

    raw = []
    for name in class_names:
        prompts = [t.format(name) for t in TEMPLATES]
        emb = tower.embed(prompts).mean(0)
        raw.append(emb / max(np.linalg.norm(emb), 1e-12))
    protos = np.stack(raw).astype(np.float64)

    if align:
        centroids = []
        for name in class_names:
            rows = [i for i in task["train_idx"] if task["labels"][i] == name]
            centroids.append(joint[rows].mean(0))
        centroids = _l2_rows(np.stack(centroids))
        weight = _fit_ridge(protos, centroids)
        protos = protos @ weight

    return _l2_rows(protos)


def _metrics_from_scores(pred, y, seen, unseen) -> dict:
    return {
        "acc": _acc(pred, y),
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
    text_scores: np.ndarray,
) -> dict:
    raw = {m: np.asarray(feats[m], dtype=np.float32) for m in MODALITIES}
    attrs = np.asarray(feats["attrs"], dtype=np.float32)
    rep = _project(
        raw, attrs, task["labels"], task["train_idx"], out, device,
        seed=seed, dim=dim, epochs=epochs,
    )

    # U1b readout: all modalities, exactly as ``run_upper_bound_u1b`` does.
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
    proto_te_scores = _proto_scores(rep, bank, task["test_idx"], MODALITIES)
    proto_val_scores = _proto_scores(rep, bank, task["val_idx"], MODALITIES)
    proto_cal_scores = _proto_scores(rep, bank, held_cal, MODALITIES)
    proto_unseen_scores = _proto_scores(rep, bank, held_test, MODALITIES)
    proto = _metrics_from_scores(
        proto_te_scores.argmax(1),
        task["yte"],
        proto_te_scores.max(1),
        proto_unseen_scores.max(1),
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
    lda_te_scores = scorer.scores(Xte)
    lda_unseen_scores = scorer.scores(Xunseen)
    lda = _metrics_from_scores(
        predict_classes(scorer, Xte),
        task["yte"],
        lda_te_scores.max(1),
        lda_unseen_scores.max(1),
    )

    # Text zero-shot is seed-independent; recompute per row for a uniform table.
    text = _metrics_from_scores(
        text_scores[task["test_idx"]].argmax(1),
        task["yte"],
        text_scores[task["test_idx"]].max(1),
        text_scores[held_test].max(1),
    )

    # Dual head: LDA closed-set; open-set head chosen on the calibration half.
    text_cal_auroc = float(
        _auroc(text_scores[task["val_idx"]].max(1), text_scores[held_cal].max(1))
    )
    proto_cal_auroc = float(
        _auroc(proto_val_scores.max(1), proto_cal_scores.max(1))
    )
    open_head = "text" if text_cal_auroc > proto_cal_auroc else "proto"
    if open_head == "text":
        dual_auroc = text["auroc"]
        dual_reject = text["reject"]
    else:
        dual_auroc = proto["auroc"]
        dual_reject = proto["reject"]
    dual = {"acc": lda["acc"], "auroc": dual_auroc, "reject": dual_reject}

    return {
        "seed": int(seed),
        "shrinkage": best["shrinkage"],
        "cal_auroc": best["cal_auroc"],
        "cal_reject": best["cal_reject"],
        "open_head": open_head,
        "text_cal_auroc": text_cal_auroc,
        "proto_cal_auroc": proto_cal_auroc,
        "proto": proto,
        "lda": lda,
        "text_zero_shot": text,
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
    stack: dict = {}
    clip_model, clip_tokenizer, clip_error = _load_clip(stack, args.clip_model, args.device)
    feats = _build_cache(args, clip_model)
    task = task_split(feats, "category")

    tower = _load_text_tower(clip_model, clip_tokenizer, args.device)
    align = tower.source != "clip_text"
    if clip_error and align:
        tower.error = clip_error
        print(f"[text] falling back to {tower.source}: {clip_error}", flush=True)

    joint = np.asarray(feats["vision_joint"], dtype=np.float64)
    text_protos = _text_prototypes(
        tower, list(task["classes"]), joint, task, align=align
    )
    text_scores = joint @ text_protos.T

    shrinkages = tuple(float(s) for s in args.shrinkage_grid)
    rows = []
    for seed in args.projection_seeds:
        row = _eval_seed(
            feats, task, out, args.device,
            seed=seed, dim=args.dim, epochs=args.projection_epochs,
            shrinkages=shrinkages, text_scores=text_scores,
        )
        rows.append(row)
        print(
            f"[seed {seed}] proto {row['proto']['acc']:.3f}/{row['proto']['auroc']:.3f}/"
            f"{row['proto']['reject']:.3f} lda {row['lda']['acc']:.3f}/"
            f"{row['lda']['auroc']:.3f}/{row['lda']['reject']:.3f} "
            f"text {row['text_zero_shot']['acc']:.3f}/"
            f"{row['text_zero_shot']['auroc']:.3f}/"
            f"{row['text_zero_shot']['reject']:.3f} "
            f"dual[{row['open_head']}] {row['dual']['acc']:.3f}/"
            f"{row['dual']['auroc']:.3f}/{row['dual']['reject']:.3f} "
            f"s={row['shrinkage']}",
            flush=True,
        )

    proto = _agg(rows, "proto")
    lda = _agg(rows, "lda")
    text = _agg(rows, "text_zero_shot")
    dual = _agg(rows, "dual")
    observed = {
        "u_real_proto": {"test_acc": proto["acc"]["mean"]},
        "u_real_vlm_proxy": {
            "accuracy_delta_vs_proto": dual["acc"]["mean"] - proto["acc"]["mean"],
            "auroc": dual["auroc"]["mean"],
            "unseen_reject_at_95": dual["reject"]["mean"],
        },
    }
    check = check_readout_reasoning_preregistration(observed)

    payload = {
        "stage": "u1c",
        "proxy": "clip_two_tower",
        "backbone": "clip_vit_base_patch16 (frozen)",
        "clip_model": args.clip_model,
        "text_source": tower.source,
        "text_dim": int(tower.dim),
        "text_templates": list(TEMPLATES),
        "clip_error": stack.get("clip_error"),
        "data": args.data,
        "regime": "category (16 seen classes, 4 held-out categories)",
        "split": (
            "category train/val/test by variation index; held-out categories "
            "(cloth/tool/sponge/sheet) split into calibration/test halves"
        ),
        "classes": list(task["classes"]),
        "projection_seeds": list(args.projection_seeds),
        "shrinkage_grid": list(shrinkages),
        "rows": rows,
        "proto": proto,
        "lda": lda,
        "text_zero_shot": text,
        "dual": dual,
        "observed": observed,
        "preregistration": check,
        "note": (
            "Two-tower CLIP proxy, not a real VLM: no VLM weights are cached. "
            "proto/lda use the U1b recipe verbatim (learned 128-d projections, "
            "PCA whitening, concat, pooled-covariance LDA); text_zero_shot uses "
            "CLIP's joint image/text embedding. dual_head uses LDA for the "
            "closed set and the text-or-prototype head selected on the held "
            "calibration half for the open set; the test half is never used for "
            "selection."
        ),
    }
    _write_json(out / "u1c.json", payload)
    (out / "U1C_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _row(name: str, group: dict) -> str:
    return (
        f"| `{name}` | {group['acc']['mean']:.3f} ± {group['acc']['std']:.3f} | "
        f"{group['auroc']['mean']:.3f} ± {group['auroc']['std']:.3f} | "
        f"{group['reject']['mean']:.3f} ± {group['reject']['std']:.3f} |"
    )


def _render(p: dict) -> str:
    heads = sorted({r["open_head"] for r in p["rows"]})
    head_label = "/".join(heads)
    lines = [
        "# U1c：CLIP 双塔 VLM 代理（16 类 category）",
        "",
        f"- vision: `{p['clip_model']}`（冻结）· text: `{p['text_source']}`",
        f"- data: `{p['data']}` · regime: {p['regime']}",
        f"- classes: {', '.join(p['classes'])}",
        f"- projection seeds: {p['projection_seeds']} · shrinkage grid: {p['shrinkage_grid']}",
        "",
        "## 1. 主表（3 投影种子，mean ± std）",
        "",
        "| arm | closed-set acc | AUROC | unseen reject @95 |",
        "|---|---:|---:|---:|",
        _row("proto", p["proto"]),
        _row("lda (shrinkage on cal)", p["lda"]),
        _row("text_zero_shot", p["text_zero_shot"]),
        _row(f"dual_head (LDA + {head_label} open-set)", p["dual"]),
        "",
        "## 2. 每 seed 明细",
        "",
        "| seed | shrinkage | open head | text cal AUROC | proto cal AUROC | "
        "proto acc | lda acc | text acc | dual AUROC | dual reject |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p["rows"]:
        lines.append(
            f"| {r['seed']} | {r['shrinkage']:.3f} | {r['open_head']} | "
            f"{r['text_cal_auroc']:.3f} | {r['proto_cal_auroc']:.3f} | "
            f"{r['proto']['acc']:.3f} | {r['lda']['acc']:.3f} | "
            f"{r['text_zero_shot']['acc']:.3f} | {r['dual']['auroc']:.3f} | "
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
        f"- 文本塔来源：`{p['text_source']}`"
        + ("（CLIP 文本塔加载失败后的代理）" if p["text_source"] != "clip_text" else ""),
        "- `u_real_vlm_proxy` 映射到 `dual_head`：闭集用 LDA，开集用 cal 上选出的"
        "文本/原型头。开集头的选择只用 held 的 calibration 半，test 半只报一次。",
        "- 若 `text_zero_shot` 接近随机，说明合成 primitive 渲染不在 CLIP 文本的"
        "语义分布内；这不是“VLM 无效”，而是该代理在此数据上不可用。",
        "",
        "## 5. 边界",
        "",
        "- **这是 two-tower 代理，不是真实 VLM**：工作区没有 Qwen-VL/LLaVA/InternVL 权重，"
        "文本塔只覆盖 16 个类名。",
        "- 任务是 16 类 category，不是 U1b 的 384 类 view-instance；两者数字不可直接相减。",
        "- 文本原型只覆盖 seen 的 16 个类名；held-out 的 4 个类别只用于开集。",
        "- `proto`/`lda` 复用 U1b 的多模态读出（CLIP vision + 合成 tactile/geometry），"
        "因此 LDA 不是纯视觉；文本塔仍只作用于 CLIP 视觉/文本联合空间。",
        "- 3 个投影种子只覆盖投影训练方差，不覆盖数据划分方差。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/category_pretrain_instance8")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_instance8_experts")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--u1b-cache", default="eval/results/upper_bound/u1b_clip/paired_features.npz")
    ap.add_argument("--out", default="eval/results/upper_bound/u1c")
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
