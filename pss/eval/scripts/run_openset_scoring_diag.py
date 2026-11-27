"""CLI: is a discriminative head's weak open-set behaviour *overfitting*, or a
bad OOD score? Hold the representation fixed, vary the scorer.

Trains one MLP head on the PSS projected space (category task), then compares
four novelty scores on the SAME penultimate features:

  max-softmax prob  (the usual, known to be overconfident OOD)
  energy            (logsumexp of logits)
  Mahalanobis       (tied per-class means + pooled covariance)
  kNN distance      (mean distance to k nearest train features)

Plus reference scorers on the projected space itself (prototype cosine, kNN).

Reports closed-set accuracy, train-vs-test gap, open-set AUROC, and the
operating point (unseen reject at 95% seen accept) — the metric that cannot be
gamed by rejecting everything.

Example:
    python -m pss.eval.scripts.run_openset_scoring_diag \
        --data data/category_pretrain \
        --experts checkpoints/category_pretrain_experts \
        --projections checkpoints/phase45_projection \
        --out eval/results/openset_scoring_diag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from pss._console import ensure_utf8
from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision import load_vision_expert
from pss.eval.scripts.run_attribution_ablation import (
    DEFAULT_WEIGHTS, MODALITIES, Bottleneck, _acc, _features, _fit_torch, _probs,
    build_bank_labels, fused_scores, set_seed,
)
from pss.projection.features import load_paired_features
from pss.projection.heads import load_projection


def _concat(rep, rows):
    return np.concatenate([rep[m][rows] for m in MODALITIES], axis=1)


def _logits(model, X):
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).numpy()


def _auroc_op(score_seen, score_unseen):
    """AUROC + operating point. ``score`` is in-distribution (higher = seen)."""
    anomaly_seen = -np.asarray(score_seen, dtype=np.float64)
    anomaly_unseen = -np.asarray(score_unseen, dtype=np.float64)
    y = np.concatenate([np.zeros(len(anomaly_seen)), np.ones(len(anomaly_unseen))])
    a = np.concatenate([anomaly_seen, anomaly_unseen])
    auc = float(roc_auc_score(y, a))
    t = float(np.quantile(anomaly_seen, 0.95))  # 95% seen accepted
    reject = float((anomaly_unseen > t).mean())
    return auc, reject


def _mahalanobis_scorer(Ztr, ytr, n_classes):
    mus = np.stack([Ztr[ytr == c].mean(0) for c in range(n_classes)])
    resid = Ztr - mus[ytr]
    cov = (resid.T @ resid) / max(len(Ztr), 1)
    cov += 1e-3 * np.eye(cov.shape[0])
    inv = np.linalg.pinv(cov)

    def score(Z):
        best = np.full(len(Z), np.inf)
        for c in range(n_classes):
            d = Z - mus[c]
            m = np.einsum("ij,jk,ik->i", d, inv, d)
            best = np.minimum(best, m)
        return -best  # in-distribution score = negative min distance

    return score


def _knn_scorer(Ztr, k=5):
    nn = NearestNeighbors(n_neighbors=min(k, len(Ztr))).fit(Ztr)

    def score(Z):
        dist, _ = nn.kneighbors(Z)
        return -dist.mean(1)

    return score


def run(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision = load_vision_expert(Path(args.experts) / "vision.pt", device=device)
    tactile = load_tactile_expert(Path(args.experts) / "tactile.pt", device=device)
    feats = load_paired_features(Path(args.data), vision, tactile)
    raw = {"vision": feats.vision, "tactile": feats.tactile, "geometry": feats.geometry}
    heads = {m: load_projection(Path(args.projections) / f"{m}.pt", device=device) for m in MODALITIES}
    projected = {}
    for m in MODALITIES:
        heads[m].eval()
        with torch.no_grad():
            projected[m] = heads[m](
                torch.as_tensor(raw[m], dtype=torch.float32, device=device)
            ).cpu().numpy().astype(np.float64)

    splits = np.array(feats.splits)
    labels = np.array([str(x) for x in feats.categories])
    train_idx = np.where(splits == "train")[0]
    val_idx = np.where(splits == "val")[0]
    test_idx = np.where(splits == "test")[0]
    held_idx = np.where(splits == "test_category")[0]
    classes = sorted({str(labels[i]) for i in train_idx})
    cls_to_i = {c: i for i, c in enumerate(classes)}
    ytr = np.array([cls_to_i[str(labels[i])] for i in train_idx])
    yva = np.array([cls_to_i[str(labels[i])] for i in val_idx])
    yte = np.array([cls_to_i[str(labels[i])] for i in test_idx])

    Xtr, Xva, Xte, Xheld = (_concat(projected, r) for r in (train_idx, val_idx, test_idx, held_idx))

    # ---- one discriminative head, fixed representation ----------------- #
    set_seed(0)
    model = _fit_torch(Bottleneck(Xtr.shape[1], len(classes), 256), Xtr, ytr, Xva, yva)
    train_acc = _acc(_probs(model, Xtr).argmax(1), ytr)
    val_acc = _acc(_probs(model, Xva).argmax(1), yva)
    test_acc = _acc(_probs(model, Xte).argmax(1), yte)

    Ztr = _features(model, Xtr)
    Zte = _features(model, Xte)
    Zhe = _features(model, Xheld)
    probs_te, probs_he = _probs(model, Xte), _probs(model, Xheld)
    logit_te, logit_he = _logits(model, Xte), _logits(model, Xheld)

    mahal = _mahalanobis_scorer(Ztr, ytr, len(classes))
    knn = _knn_scorer(Ztr, k=5)

    scorers = {
        "mlp_max_softmax": (probs_te.max(1), probs_he.max(1)),
        "mlp_energy": (_logsumexp(logit_te), _logsumexp(logit_he)),
        "mlp_mahalanobis": (mahal(Zte), mahal(Zhe)),
        "mlp_knn": (knn(Zte), knn(Zhe)),
    }

    # ---- reference: prototype + kNN directly on the projected space ---- #
    bank = build_bank_labels(projected, train_idx, labels, classes)
    proto_seen = fused_scores(projected, bank, test_idx, DEFAULT_WEIGHTS).max(1)
    proto_he = fused_scores(projected, bank, held_idx, DEFAULT_WEIGHTS).max(1)
    scorers["prototype_cosine"] = (proto_seen, proto_he)

    Ptr, Pte, Phe = (_concat(projected, r) for r in (train_idx, test_idx, held_idx))
    proj_knn = _knn_scorer(Ptr, k=5)
    scorers["projected_knn"] = (proj_knn(Pte), proj_knn(Phe))

    table = {}
    for name, (s_seen, s_unseen) in scorers.items():
        auc, reject = _auroc_op(s_seen, s_unseen)
        table[name] = {"auroc": auc, "unseen_reject_at_95_seen_accept": reject}

    payload = {
        "data": str(args.data),
        "regime": "category",
        "classes": len(classes),
        "counts": {"train": len(train_idx), "val": len(val_idx),
                   "test": len(test_idx), "test_category": len(held_idx)},
        "head": {"train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
                 "train_test_gap": train_acc - test_acc},
        "scorers": table,
        "note": "Same MLP representation for the four mlp_* scorers; only the OOD "
                "score changes. If a better scorer recovers open-set without losing "
                "closed-set accuracy, the weakness is scoring, not overfitting.",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "openset_scoring.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "OPENSET_SCORING_REPORT.md").write_text(render_report(payload), encoding="utf-8")
    return payload


def _logsumexp(logits: np.ndarray) -> np.ndarray:
    m = logits.max(1, keepdims=True)
    return (m + np.log(np.exp(logits - m).sum(1, keepdims=True))).squeeze(1)


def render_report(p: dict) -> str:
    h = p["head"]
    lines = [
        "# 开集打分诊断：过拟合还是打分方式？",
        "",
        f"- data: `{p['data']}`  ·  类别: {p['classes']}  ·  counts: {p['counts']}",
        "",
        "## 0. 同一个 MLP 头是否过拟合",
        "",
        "| train acc | val acc | test acc | train−test gap |",
        "|---:|---:|---:|---:|",
        f"| {h['train_acc']:.3f} | {h['val_acc']:.3f} | {h['test_acc']:.3f} | {h['train_test_gap']:.3f} |",
        "",
        "## 1. 同一表示、不同开集打分器",
        "",
        "| scorer | AUROC | unseen reject @95% seen accept |",
        "|---|---:|---:|",
    ]
    for name, v in p["scorers"].items():
        lines.append(f"| `{name}` | {v['auroc']:.3f} | {v['unseen_reject_at_95_seen_accept']:.3f} |")
    lines += [
        "",
        "## 2. 判读",
        "",
        "- 若 `mlp_mahalanobis` / `mlp_knn` 的 AUROC 明显高于 `mlp_max_softmax`，说明低开集分",
        "  **主要是 OOD 打分器的问题，不是表示过拟合**：softmax 头对未见类天然过度自信。",
        "- 若换打分器后仍显著低于 `prototype_cosine`，才是表示的开放集能力真的更弱。",
        "- `train−test gap` 大才支持'过拟合'的说法；gap 小则不支持。",
        "- 看 `unseen reject @95% seen accept`，不要只看 AUROC：拒绝一切的模型 AUROC 可以很高但没用。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/category_pretrain")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    ap.add_argument("--projections", default="checkpoints/phase45_projection")
    ap.add_argument("--out", default="eval/results/openset_scoring_diag")
    args = ap.parse_args()
    payload = run(args)
    print("Head:", json.dumps(payload["head"], ensure_ascii=False))
    print("OOD scorers:")
    for name, v in payload["scorers"].items():
        print(f"  {name:<20} AUROC={v['auroc']:.3f}  reject@95={v['unseen_reject_at_95_seen_accept']:.3f}")
    print(f"  report: {Path(args.out) / 'OPENSET_SCORING_REPORT.md'}")


if __name__ == "__main__":
    main()
