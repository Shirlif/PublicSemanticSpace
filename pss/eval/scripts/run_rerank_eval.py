"""A2: does a learned rerank over the A4 gate beat the gate itself?

A4's winner is ``lda_concat_whiten`` (0.948).  A reranker can only reorder the
classes the gate proposes, so its ceiling is the gate's top-k recall.  This
script measures that ceiling first, then trains a learned bilinear metric over
the candidates.

Candidate leakage is the trap
-----------------------------
If the candidates for a training row are produced by a gate fitted on that same
row, every true class is a candidate and the reranker learns an easier problem
than deployment.  The reranker-train candidates therefore come from a gate
fitted on views 0-3 only and are generated for views 4-5.  The full-train gate
is used for val and test.

The learned metric
------------------
``score(x, c) = x^T W mu_c - 0.5 mu_c^T W_sym mu_c``, initialised at the LDA
precision.  At initialisation it ranks exactly like the gate; it can only move
away by learning.  Class means stay the hot-updatable parameters.

Example
-------
python -u -m pss.eval.scripts.run_rerank_eval --out eval/results/readout_opt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.eval.readout_reasoning_preregistration import (
    check_readout_reasoning_preregistration,
)
from pss.eval.scripts.run_attribution_grid import (
    MODALITIES,
    _acc,
    _auroc,
    _write_json,
    concat,
    fit_mlp,
    load_features,
    recall_profile,
    task_split,
)
from pss.eval.scripts.run_fusion_ablation import _whiten
from pss.eval.scripts.run_readout_scorers import _median_update_ms, _reject_at
from pss.projection.heads import load_projection
from pss.prototypes.scorers import MahalanobisScorer, predict_classes


# --------------------------------------------------------------------------- #
# candidates
# --------------------------------------------------------------------------- #
def topk_candidates(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.asarray(scores), axis=1)[:, :k]


def candidate_mask(candidates: np.ndarray, n_classes: int) -> np.ndarray:
    mask = np.zeros((len(candidates), n_classes), dtype=bool)
    rows = np.repeat(np.arange(len(candidates)), candidates.shape[1])
    mask[rows, candidates.ravel()] = True
    return mask


# --------------------------------------------------------------------------- #
# learned bilinear metric
# --------------------------------------------------------------------------- #
class BilinearMetric(nn.Module):
    def __init__(self, dim: int, init_precision: np.ndarray):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.W.weight.data.copy_(
            torch.as_tensor(init_precision, dtype=torch.float32)
        )
        self.register_buffer("means", torch.zeros(0, dim))

    def set_means(self, means: np.ndarray) -> None:
        self.means = torch.as_tensor(
            means, dtype=torch.float32, device=self.W.weight.device
        )

    def update_class(self, label: int, Z_c: np.ndarray) -> None:
        self.means[label] = torch.as_tensor(
            np.asarray(Z_c, dtype=np.float32).mean(0),
            device=self.means.device,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        W = self.W.weight
        W_sym = 0.5 * (W + W.T)
        mu = self.means
        quad = -0.5 * torch.einsum("ij,jk,ik->i", mu, W_sym, mu)
        return (X @ W.T) @ mu.T + quad


def train_bilinear(Xtr, ytr, cand_tr, means, precision, Xva, yva, cand_va,
                   device, *, epochs=200, patience=20, lr=1e-3, wd=1e-2,
                   seed=0) -> tuple[BilinearMetric, float, int]:
    n_classes = len(means)
    torch.manual_seed(seed)
    model = BilinearMetric(precision.shape[0], precision).to(device)
    model.set_means(means)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    mask_tr = torch.as_tensor(candidate_mask(cand_tr, n_classes), device=device)
    keep = mask_tr[torch.arange(len(ytr_t), device=device), ytr_t]
    Xtr_t, ytr_t, mask_tr = Xtr_t[keep], ytr_t[keep], mask_tr[keep]

    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.as_tensor(yva, dtype=torch.long, device=device)
    mask_va = torch.as_tensor(candidate_mask(cand_va, n_classes), device=device)

    best_acc, best_state, stale = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        logits = model(Xtr_t).masked_fill(~mask_tr, -1e4)
        loss = F.cross_entropy(logits, ytr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva_t).masked_fill(~mask_va, -1e4)
            acc = float((val_logits.argmax(1) == yva_t).float().mean())
        if acc > best_acc:
            best_acc, stale = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc, int(keep.sum())


@torch.no_grad()
def eval_bilinear(model, X, y, candidates, n_classes, device) -> float:
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    mask = torch.as_tensor(candidate_mask(candidates, n_classes), device=device)
    logits = model(X_t).masked_fill(~mask, -1e4)
    return float((logits.argmax(1) == y_t).float().mean())


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    out = Path(args.out)
    feats = load_features(args.data, args.experts, out)
    task = task_split(feats, args.regime)
    raw = {m: np.asarray(feats[m], dtype=np.float32) for m in MODALITIES}
    device = args.device
    n_classes = len(task["classes"])

    heads = {
        m: load_projection(Path(args.projections) / f"{m}.pt", device=device)
        for m in MODALITIES
    }
    rep = {}
    with torch.no_grad():
        for m in MODALITIES:
            heads[m].eval()
            rep[m] = heads[m](torch.as_tensor(raw[m]).to(device)).cpu().numpy().astype(np.float64)
    whitened = _whiten(rep, task["train_idx"])
    print("[rep] projected and whitened", flush=True)

    Xtr = concat(whitened, task["train_idx"])
    Xva = concat(whitened, task["val_idx"])
    Xte = concat(whitened, task["test_idx"])
    Xhe = concat(whitened, task["held_idx"])

    # full gate
    gate = MahalanobisScorer().fit(Xtr, task["ytr"])
    gate_scores = gate.scores(Xte)
    gate_acc = float((predict_classes(gate, Xte) == task["yte"]).mean())
    gate_seen = gate_scores.max(axis=1)
    gate_unseen = gate.scores(Xhe).max(axis=1)
    gate_recall = recall_profile(gate_scores, task["yte"], ks=(1, 3, 5, 10))
    gate_hot = _median_update_ms(gate, Xtr, task["ytr"])
    print(f"[gate] acc {gate_acc:.3f} r@5 {gate_recall['recall@5']:.3f}", flush=True)

    # reference: full learned head on the same representation
    probs = [
        fit_mlp(Xtr, task["ytr"], Xva, task["yva"], Xte, n_classes,
                seed=seed, device=device)[0]
        for seed in args.seeds
    ]
    mlp_acc = _acc(np.mean(probs, axis=0).argmax(1), task["yte"])
    print(f"[mlp] whitened concat {mlp_acc:.3f}", flush=True)

    # candidate provenance: gate fitted on views 0-3, candidates for views 4-5
    views = np.asarray(feats["views"])
    train_rows = np.asarray(task["train_idx"])
    y_of_row = {int(i): int(y) for i, y in zip(task["train_idx"], task["ytr"])}
    gate_rows = np.asarray([i for i in train_rows if views[i] <= 3])
    rerank_rows = np.asarray([i for i in train_rows if views[i] >= 4])
    y_gate = np.asarray([y_of_row[int(i)] for i in gate_rows])
    y_rerank = np.asarray([y_of_row[int(i)] for i in rerank_rows])
    gate_fit = MahalanobisScorer().fit(concat(whitened, gate_rows), y_gate)
    X_rerank = concat(whitened, rerank_rows)
    gate_fit_scores = gate_fit.scores(X_rerank)
    print(f"[provenance] gate rows {len(gate_rows)} rerank rows {len(rerank_rows)}", flush=True)

    by_k = {}
    for k in args.k_grid:
        cand_rr = topk_candidates(gate_fit_scores, k)
        cand_va = topk_candidates(gate.scores(Xva), k)
        cand_te = topk_candidates(gate_scores, k)
        model, val_acc, kept = train_bilinear(
            X_rerank, y_rerank, cand_rr,
            gate.means_, gate.precision_,
            Xva, task["yva"], cand_va,
            device, epochs=args.epochs, seed=0,
        )
        test_acc = eval_bilinear(model, Xte, task["yte"], cand_te, n_classes, device)
        gate_miss = 1.0 - float(
            (cand_te == task["yte"][:, None]).any(axis=1).mean()
        )
        by_k[int(k)] = {
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
            "train_rows_kept": int(kept),
            "gate_miss_test": float(gate_miss),
        }
        print(
            f"[rerank k={k}] val {val_acc:.3f} test {test_acc:.3f} "
            f"gate_miss {gate_miss:.3f}",
            flush=True,
        )

    chosen_k = int(args.prereg_k)
    rerank_acc = by_k[chosen_k]["test_acc"]

    check = check_readout_reasoning_preregistration({
        "rerank_bilinear": {
            "test_acc": rerank_acc,
            "auroc": _auroc(gate_seen, gate_unseen),
            "hot_update_ms": gate_hot,
        },
        "mlp_whiten_concat": {"test_acc": float(mlp_acc)},
    })

    payload = {
        "data": args.data,
        "projections": args.projections,
        "regime": args.regime,
        "classes": n_classes,
        "counts": {k: int(len(task[k])) for k in ("train_idx", "val_idx", "test_idx", "held_idx")},
        "gate": {
            "acc": gate_acc,
            "auroc": _auroc(gate_seen, gate_unseen),
            "unseen_reject_at_95": _reject_at(gate_seen, gate_unseen),
            "hot_ms": float(gate_hot),
            "recall": gate_recall,
        },
        "mlp_whiten_concat": {"acc": float(mlp_acc), "seeds": list(args.seeds)},
        "rerank": {
            "prereg_k": chosen_k,
            "k_grid": [int(k) for k in args.k_grid],
            "by_k": by_k,
        },
        "preregistration": check,
        "note": (
            "Rerank candidates for training rows come from a gate fitted on "
            "views 0-3; val/test use the full-train gate. Gate miss rate is "
            "the hard ceiling."
        ),
    }
    _write_json(out / "rerank_eval.json", payload)
    (out / "RERANK_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _render(p: dict) -> str:
    g = p["gate"]
    lines = [
        "# A2：A4 闸门上的学习式重排",
        "",
        f"- data: `{p['data']}` · projections: `{p['projections']}`",
        f"- regime: {p['regime']} · classes: {p['classes']} · counts: {p['counts']}",
        "",
        "## 1. 闸门与参考",
        "",
        "| 项 | 值 |",
        "|---|---:|",
        f"| 闸门 closed-set acc | {g['acc']:.3f} |",
        f"| 闸门 AUROC | {g['auroc']:.3f} |",
        f"| 闸门 unseen reject @95 | {g['unseen_reject_at_95']:.3f} |",
        f"| 闸门 hot update | {g['hot_ms']:.3f} ms |",
        f"| 闸门 recall@1 | {g['recall']['recall@1']:.3f} |",
        f"| 闸门 recall@3 | {g['recall']['recall@3']:.3f} |",
        f"| 闸门 recall@5 | {g['recall']['recall@5']:.3f} |",
        f"| 闸门 recall@10 | {g['recall']['recall@10']:.3f} |",
        f"| 闸门 k@95 | {g['recall']['k_at_95']} |",
        f"| MLP（同表示，{len(p['mlp_whiten_concat']['seeds'])} seeds） | "
        f"{p['mlp_whiten_concat']['acc']:.3f} |",
        "",
        "## 2. 学习式重排（bilinear metric，初始化于 LDA 精度）",
        "",
        "| k | val acc | test acc | 训练行保留 | 闸门漏检率 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k, row in p["rerank"]["by_k"].items():
        lines.append(
            f"| {k} | {row['val_acc']:.3f} | {row['test_acc']:.3f} | "
            f"{row['train_rows_kept']} | {row['gate_miss_test']:.3f} |"
        )
    check = p["preregistration"]
    lines += [
        "",
        f"预注册 arm：`rerank_bilinear`（k={p['rerank']['prereg_k']}）· "
        f"通过 **{check['ok']}**",
    ]
    if check["mismatches"]:
        lines.append("")
        for mismatch in check["mismatches"]:
            lines.append(f"- `{mismatch}`")
    lines += [
        "",
        "## 3. 判读",
        "",
        "1. 重排的上限是 `1 - 闸门漏检率`；test acc 超过这个数的部分不可能来自重排。",
        "2. 与闸门自身 acc 比较才是 A2 的增量；与 MLP 比较只是说明\"学习式头\"的水平。",
        "3. 若重排不高于闸门，结论是**线性度量已到读出上限**，A2 应停止，",
        "   把预算转到 A5（长期增量/协方差刷新）或 B 线。",
        "4. 候选集来源与部署闸门不完全一致（训练用 views 0-3 的闸门），这是防泄漏的代价，",
        "   解读时需保留。",
        "",
        "## 4. 边界",
        "",
        "- bilinear 只学度量，不改变类均值；均值仍是热更新参数。",
        "- 训练行 768 条、384 类，正则较强；更大数据可能改变结论。",
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
    ap.add_argument("--k-grid", nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument("--prereg-k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    payload = run(args)
    g = payload["gate"]
    print(f"gate acc {g['acc']:.3f} r@5 {g['recall']['recall@5']:.3f} "
          f"auroc {g['auroc']:.3f}")
    print(f"mlp {payload['mlp_whiten_concat']['acc']:.3f}")
    for k, row in payload["rerank"]["by_k"].items():
        print(f"  rerank k={k}: test {row['test_acc']:.3f} "
              f"gate_miss {row['gate_miss_test']:.3f}")
    check = payload["preregistration"]
    print(f"preregistration ok={check['ok']}")
    for mismatch in check["mismatches"]:
        print(f"  - {mismatch}")


if __name__ == "__main__":
    main()
