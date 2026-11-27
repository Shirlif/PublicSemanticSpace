"""A4: mix across modalities before discriminating.

P1 isolated the loss: the same pooled-covariance Mahalanobis reaches 0.930 on a
384-d concatenated public representation but only 0.763 when fitted per
modality and fused at the score level.  This script tests the "mix first"
family directly:

  lda_concat             concat(3 modalities) -> pooled-covariance Mahalanobis
  lda_concat_whiten      per-modality PCA-whitening -> concat -> Mahalanobis
  lda_concat_weighted    bounded per-modality weights (chosen on val) -> concat
  lda_subset_dropout     one model per non-empty modality subset (missing inputs)
  lda_per_modality_fused negative control: score-level fusion

LDA here means exactly "class means + one pooled covariance", which is what
:class:`pss.prototypes.scorers.MahalanobisScorer` fits.  Three extra checks go
beyond P1:

- **Hot update**: median time to absorb one class.
- **Incremental forgetting**: fit on half the classes, add the other half one
  by one with a frozen precision, and re-measure the original classes.
- **Missing modality**: per-subset models, so a dropped input has an honest
  number instead of a zero-imputed one.

Example
-------
python -m pss.eval.scripts.run_fusion_ablation --out eval/results/readout_opt
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from pss._console import ensure_utf8
from pss.eval.readout_reasoning_preregistration import (
    check_readout_reasoning_preregistration,
)
from pss.eval.scripts.run_attribution_grid import (
    DEFAULT_WEIGHTS,
    MODALITIES,
    _acc,
    _auroc,
    _write_json,
    build_bank,
    concat,
    fit_mlp,
    load_features,
    task_split,
)
from pss.eval.scripts.run_readout_scorers import (
    _fused_modality_arm,
    _median_update_ms,
    _reject_at,
)
from pss.projection.heads import load_projection
from pss.prototypes.scorers import MahalanobisScorer, predict_classes

SUBSETS = (
    ("vision",),
    ("tactile",),
    ("geometry",),
    ("vision", "tactile"),
    ("vision", "geometry"),
    ("tactile", "geometry"),
    MODALITIES,
)

WEIGHT_GRID = (0.2, 0.5, 1.0, 1.5, 2.0)


# --------------------------------------------------------------------------- #
# representation helpers
# --------------------------------------------------------------------------- #
def _subset(rep, rows, modalities) -> np.ndarray:
    return np.concatenate([rep[m][rows] for m in modalities], axis=1)


def _weighted(rep, rows, weights) -> np.ndarray:
    return np.concatenate(
        [weights[m] * rep[m][rows] for m in MODALITIES], axis=1
    )


def _whiten(rep, train_idx) -> dict[str, np.ndarray]:
    out = {}
    for m in MODALITIES:
        pca = PCA(n_components=rep[m].shape[1], whiten=True, random_state=0)
        pca.fit(rep[m][train_idx])
        out[m] = pca.transform(rep[m]).astype(np.float64)
    return out


def _proto_scores(rep, bank, rows, modalities, weights=DEFAULT_WEIGHTS) -> np.ndarray:
    total = sum(weights[m] for m in modalities)
    scores = None
    for m in modalities:
        z = np.asarray([v / max(np.linalg.norm(v), 1e-12) for v in rep[m][rows]])
        term = weights[m] * (z @ bank.protos[m].T.astype(np.float64))
        scores = term if scores is None else scores + term
    return scores / max(total, 1e-12)


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def _lda_arm(rep, task, modalities=MODALITIES) -> dict:
    Xtr = _subset(rep, task["train_idx"], modalities)
    Xte = _subset(rep, task["test_idx"], modalities)
    Xhe = _subset(rep, task["held_idx"], modalities)
    scorer = MahalanobisScorer().fit(Xtr, task["ytr"])
    seen = scorer.scores(Xte).max(axis=1)
    unseen = scorer.scores(Xhe).max(axis=1)
    return {
        "scorer": scorer,
        "test_acc": float((predict_classes(scorer, Xte) == task["yte"]).mean()),
        "seen": seen,
        "unseen": unseen,
        "hot_ms": _median_update_ms(scorer, Xtr, task["ytr"]),
    }


def _choose_weights(rep, task) -> tuple[dict[str, float], float]:
    """Coordinate grid on val only; bounded to [0.2, 2.0] (the A4 constraint)."""

    best_weights = {m: 1.0 for m in MODALITIES}
    best_acc = -1.0
    for i, combo in enumerate(itertools.product(WEIGHT_GRID, repeat=len(MODALITIES))):
        weights = dict(zip(MODALITIES, combo))
        Xtr = _weighted(rep, task["train_idx"], weights)
        Xva = _weighted(rep, task["val_idx"], weights)
        scorer = MahalanobisScorer().fit(Xtr, task["ytr"])
        acc = float((predict_classes(scorer, Xva) == task["yva"]).mean())
        if acc > best_acc:
            best_acc, best_weights = acc, weights
        if (i + 1) % 25 == 0:
            print(f"[weights] {i + 1} combos, best val {best_acc:.3f}", flush=True)
    return best_weights, best_acc


def _weighted_arm(rep, task) -> tuple[dict, dict[str, float], float]:
    weights, val_acc = _choose_weights(rep, task)
    Xtr = _weighted(rep, task["train_idx"], weights)
    Xte = _weighted(rep, task["test_idx"], weights)
    Xhe = _weighted(rep, task["held_idx"], weights)
    scorer = MahalanobisScorer().fit(Xtr, task["ytr"])
    return (
        {
            "scorer": scorer,
            "test_acc": float((predict_classes(scorer, Xte) == task["yte"]).mean()),
            "seen": scorer.scores(Xte).max(axis=1),
            "unseen": scorer.scores(Xhe).max(axis=1),
            "hot_ms": _median_update_ms(scorer, Xtr, task["ytr"]),
        },
        weights,
        val_acc,
    )


def _incremental_forgetting(rep, task, modalities=MODALITIES, n_add: int = 50,
                            seed: int = 0) -> dict:
    Xtr = _subset(rep, task["train_idx"], modalities)
    ytr = task["ytr"]
    n_classes = len(task["classes"])
    order = np.random.default_rng(seed).permutation(n_classes)
    split = n_classes // 2
    base = order[:split]
    added = order[split:split + n_add]

    scorer = MahalanobisScorer().fit(Xtr[np.isin(ytr, base)], ytr[np.isin(ytr, base)])
    test_mask = np.isin(task["yte"], base)
    Xte = _subset(rep, task["test_idx"], modalities)[test_mask]
    yte = task["yte"][test_mask]
    before = float((predict_classes(scorer, Xte) == yte).mean())
    for label in added:
        scorer.add_class(int(label), Xtr[ytr == label])
    after = float((predict_classes(scorer, Xte) == yte).mean())
    return {
        "before_add": before,
        "after_add": after,
        "delta": after - before,
        "n_base_classes": int(split),
        "n_added": int(len(added)),
    }


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    out = Path(args.out)
    feats = load_features(args.data, args.experts, out)
    task = task_split(feats, args.regime)
    raw = {m: np.asarray(feats[m], dtype=np.float32) for m in MODALITIES}
    device = args.device

    heads = {
        m: load_projection(Path(args.projections) / f"{m}.pt", device=device)
        for m in MODALITIES
    }
    rep: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for m in MODALITIES:
            heads[m].eval()
            rep[m] = heads[m](torch.as_tensor(raw[m]).to(device)).cpu().numpy().astype(np.float64)
    whitened = _whiten(rep, task["train_idx"])
    print("[rep] projected and whitened", flush=True)

    # matched MLP reference
    Xtr, Xva, Xte = (concat(rep, task[k]) for k in ("train_idx", "val_idx", "test_idx"))
    probs = [
        fit_mlp(Xtr, task["ytr"], Xva, task["yva"], Xte, len(task["classes"]),
                seed=seed, device=device)[0]
        for seed in args.seeds
    ]
    mlp_probs = np.mean(probs, axis=0)
    mlp_acc = _acc(mlp_probs.argmax(1), task["yte"])
    print(f"[mlp] reference acc {mlp_acc:.3f}", flush=True)

    arms: dict[str, dict] = {}

    # reference: deployed prototype cosine
    bank = build_bank(rep, task["train_idx"], task["labels"], task["classes"])
    proto_scores = _proto_scores(rep, bank, task["test_idx"], MODALITIES)
    proto_seen = proto_scores.max(axis=1)
    proto_unseen = _proto_scores(rep, bank, task["held_idx"], MODALITIES).max(axis=1)
    arms["proto_cosine"] = {
        "test_acc": float((proto_scores.argmax(axis=1) == task["yte"]).mean()),
        "seen": proto_seen,
        "unseen": proto_unseen,
        "hot_ms": 0.0,
    }

    # mix-first family
    lda = _lda_arm(rep, task)
    arms["lda_concat"] = lda
    print(f"[lda] concat acc {lda['test_acc']:.3f}", flush=True)
    arms["lda_concat_whiten"] = _lda_arm(whitened, task)
    print(f"[lda] whitened acc {arms['lda_concat_whiten']['test_acc']:.3f}", flush=True)
    weighted, weights, weight_val_acc = _weighted_arm(rep, task)
    arms["lda_concat_weighted"] = weighted
    print(f"[lda] weighted acc {weighted['test_acc']:.3f} weights {weights}", flush=True)

    # one model per modality subset, for missing inputs
    subset_table = {}
    for subset in SUBSETS:
        key = "+".join(subset)
        model = _lda_arm(rep, task, modalities=subset)
        proto_subset = _proto_scores(rep, bank, task["test_idx"], subset)
        subset_table[key] = {
            "lda_acc": model["test_acc"],
            "proto_acc": float((proto_subset.argmax(axis=1) == task["yte"]).mean()),
            "hot_ms": model["hot_ms"],
        }
        print(f"[subset] {key}: lda {model['test_acc']:.3f}", flush=True)
    arms["lda_subset_dropout"] = dict(arms["lda_concat"])  # full-modality row

    # negative control: score-level fusion (P1's proto_mahalanobis)
    control_seen, control_unseen, control_acc, control_hot = _fused_modality_arm(
        MahalanobisScorer, rep, task
    )
    arms["lda_per_modality_fused"] = {
        "test_acc": control_acc,
        "seen": control_seen,
        "unseen": control_unseen,
        "hot_ms": control_hot,
    }

    # finalise metrics
    for name, arm in arms.items():
        arm["accuracy_delta_vs_mlp"] = float(arm["test_acc"] - mlp_acc)
        arm["auroc"] = _auroc(arm["seen"], arm["unseen"])
        arm["unseen_reject_at_95"] = _reject_at(arm["seen"], arm["unseen"])
        arm["seed_std"] = 0.0
        arm.pop("seen", None)
        arm.pop("unseen", None)
        arm.pop("scorer", None)

    forgetting = _incremental_forgetting(rep, task)
    print(f"[forgetting] {forgetting}", flush=True)
    check = check_readout_reasoning_preregistration({
        name: {
            "accuracy_delta_vs_mlp": arms[name]["accuracy_delta_vs_mlp"],
            "auroc": arms[name]["auroc"],
            "unseen_reject_at_95": arms[name]["unseen_reject_at_95"],
            "hot_update_ms": arms[name]["hot_ms"],
            "seed_std": arms[name]["seed_std"],
        }
        for name in (
            "lda_concat", "lda_concat_whiten", "lda_concat_weighted",
            "lda_subset_dropout", "lda_per_modality_fused",
        )
    })

    payload = {
        "data": args.data,
        "projections": args.projections,
        "regime": args.regime,
        "classes": len(task["classes"]),
        "counts": {k: int(len(task[k])) for k in ("train_idx", "val_idx", "test_idx", "held_idx")},
        "mlp_reference": {"acc": float(mlp_acc), "seeds": list(args.seeds)},
        "weights": {"chosen": weights, "val_acc": float(weight_val_acc),
                    "grid": list(WEIGHT_GRID)},
        "arms": arms,
        "subset_dropout": subset_table,
        "incremental_forgetting": forgetting,
        "preregistration": check,
        "note": (
            "LDA = class means + one pooled covariance (MahalanobisScorer). "
            "Incremental addition moves means only; the precision is frozen, "
            "which is the hot-update story being tested."
        ),
    }
    _write_json(out / "fusion_ablation.json", payload)
    (out / "FUSION_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _render(p: dict) -> str:
    lines = [
        "# A4：先混合、再判别（融合实验）",
        "",
        f"- data: `{p['data']}` · projections: `{p['projections']}`",
        f"- regime: {p['regime']} · classes: {p['classes']} · counts: {p['counts']}",
        f"- MLP 参照: **{p['mlp_reference']['acc']:.3f}**",
        f"- 受约束权重（val 选取，grid {p['weights']['grid']}）: "
        f"{p['weights']['chosen']} · val acc {p['weights']['val_acc']:.3f}",
        "",
        "## 1. 主表",
        "",
        "| arm | closed-set acc | Δ vs MLP | AUROC | unseen reject @95 | hot update ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, arm in p["arms"].items():
        lines.append(
            f"| `{name}` | {arm['test_acc']:.3f} | {arm['accuracy_delta_vs_mlp']:+.3f} | "
            f"{arm['auroc']:.3f} | {arm['unseen_reject_at_95']:.3f} | {arm['hot_ms']:.3f} |"
        )

    lines += [
        "",
        "## 2. 缺失模态（每个模态子集单独拟合）",
        "",
        "| 可用模态 | LDA acc | 原型 cosine acc | hot ms |",
        "|---|---:|---:|---:|",
    ]
    for key, row in p["subset_dropout"].items():
        lines.append(
            f"| {key} | {row['lda_acc']:.3f} | {row['proto_acc']:.3f} | {row['hot_ms']:.3f} |"
        )

    f = p["incremental_forgetting"]
    check = p["preregistration"]
    lines += [
        "",
        "## 3. 增量遗忘（冻结精度，逐个加类）",
        "",
        f"- 基类 {f['n_base_classes']} 个；逐个加入 {f['n_added']} 个新类",
        f"- 基类 test acc：加入前 {f['before_add']:.3f} → 加入后 {f['after_add']:.3f} "
        f"(Δ {f['delta']:+.3f})",
        "",
        "## 4. 预注册检查",
        "",
        f"- 检查 arm 数: {check['checked']} · 通过: **{check['ok']}**",
    ]
    if check["mismatches"]:
        lines.append("")
        for mismatch in check["mismatches"]:
            lines.append(f"- `{mismatch}`")
    lines += [
        "",
        "## 5. 判读",
        "",
        "1. `lda_concat` 是主候选：它把跨模态组合放进同一个协方差里，而不是在每个模态内部先做一次判别。",
        "2. `lda_per_modality_fused` 是负对照。若它意外追平主候选，说明 P1 的\"融合吃掉收益\"解释不成立，",
        "   应检查融合的标准化方式而不是继续加模型。",
        "3. 增量遗忘接近 0 才支持\"冻结精度 + 更新均值\"可以当作热更新；否则必须退回按窗口重估。",
        "4. 缺失模态表比零填充更诚实：每个子集有自己的模型，代价是 7 份均值。",
        "",
        "## 6. 边界",
        "",
        "- LDA 是确定性拟合，`seed_std` 为 0 是结构性的，不是稳定性证据。",
        "- 池化协方差在类数增加时冻结，长期漂移未测；增量表只覆盖 50 个新类。",
        "- 合成代理数据、小编码器。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/category_pretrain_instance8")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_instance8_experts")
    ap.add_argument("--projections", default="checkpoints/category_pretrain_instance8_projection")
    ap.add_argument("--regime", choices=["category", "view-instance"], default="view-instance")
    ap.add_argument("--out", default="eval/results/readout_opt")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    payload = run(args)
    print(f"MLP reference: {payload['mlp_reference']['acc']:.3f}")
    for name, arm in payload["arms"].items():
        print(
            f"  {name:<24} acc={arm['test_acc']:.3f} "
            f"delta={arm['accuracy_delta_vs_mlp']:+.3f} "
            f"auroc={arm['auroc']:.3f} reject@95={arm['unseen_reject_at_95']:.3f} "
            f"hot={arm['hot_ms']:.3f}ms"
        )
    check = payload["preregistration"]
    print(f"preregistration ok={check['ok']} checked={check['checked']}")
    for mismatch in check["mismatches"]:
        print(f"  - {mismatch}")


if __name__ == "__main__":
    main()
