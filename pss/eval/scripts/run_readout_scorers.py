"""A1: distance-based scorers on the deployed public space.

Holds the representation fixed (the production projection) and varies only the
class scorer, which is the §5.5 protocol applied to the PSS readout:

  proto                weighted cosine over per-modality prototypes (deployed)
  proto_mahalanobis    per-modality pooled-covariance Mahalanobis, fused
  proto_knn            per-modality class-conditional kNN, fused
  concat_mahalanobis   one 384-d representation, no fusion
  concat_knn           one 384-d representation, no fusion
  concat_cosine        one 384-d representation, no fusion

Reports closed-set accuracy, open-set AUROC, the operating point (unseen
reject at 95% seen accept), and a hot-update proxy.  The measured arms are
checked against ``pss/eval/readout_reasoning_preregistration.py``.

Example
-------
python -m pss.eval.scripts.run_readout_scorers \
  --projections checkpoints/category_pretrain_instance8_projection \
  --out eval/results/readout_opt
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

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
    fused_scores,
    load_features,
    task_split,
)
from pss.projection.heads import load_projection
from pss.prototypes.scorers import (
    CosineProtoScorer,
    KNNScorer,
    MahalanobisScorer,
    fit_novelty_map,
    fuse_scores,
)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _closed_set_acc(scores: np.ndarray, y: np.ndarray) -> float:
    return float((np.asarray(scores).argmax(axis=1) == np.asarray(y)).mean())


def _reject_at(seen: np.ndarray, unseen: np.ndarray, acceptance: float = 0.95) -> float:
    threshold = float(np.quantile(np.asarray(seen), 1.0 - acceptance))
    return float((np.asarray(unseen) < threshold).mean())


def _median_update_ms(scorer, Z: np.ndarray, y: np.ndarray, n: int = 50) -> float:
    """Median time to absorb one class's samples, on a copy of the scorer."""

    scorer = copy.deepcopy(scorer)
    rng = np.random.default_rng(0)
    rows = {int(c): Z[y == c] for c in scorer.classes_}
    times = []
    for _ in range(n):
        c = int(rng.choice(scorer.classes_))
        start = time.perf_counter()
        scorer.update_class(c, rows[c])
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(times))


def _arm_metrics(seen: np.ndarray, unseen: np.ndarray, acc: float,
                 hot_update_ms: float, seed_std: float = 0.0) -> dict:
    return {
        "test_acc": float(acc),
        "seed_std": float(seed_std),
        "auroc": _auroc(seen, unseen),
        "unseen_reject_at_95": _reject_at(seen, unseen),
        "hot_update_ms": float(hot_update_ms),
    }


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def _fused_modality_arm(scorer_cls, rep, task, *, hot_update: bool = True, **kwargs):
    scorers = {
        m: scorer_cls(**kwargs).fit(rep[m][task["train_idx"]], task["ytr"])
        for m in MODALITIES
    }
    rank = fuse_scores(
        {m: scorers[m].scores(rep[m][task["test_idx"]]) for m in MODALITIES},
        DEFAULT_WEIGHTS,
    )
    acc = _closed_set_acc(rank, task["yte"])

    maps = {m: fit_novelty_map(scorers[m], rep[m][task["train_idx"]]) for m in MODALITIES}
    weights = DEFAULT_WEIGHTS
    total = sum(weights[m] for m in MODALITIES)

    def novelty(rows):
        return sum(
            weights[m] * maps[m](scorers[m].scores(rep[m][rows]).max(axis=1))
            for m in MODALITIES
        ) / total

    hot = 0.0
    if hot_update:
        hot = float(
            np.mean([
                _median_update_ms(scorers[m], rep[m][task["train_idx"]], task["ytr"])
                for m in MODALITIES
            ])
        )
    return novelty(task["test_idx"]), novelty(task["held_idx"]), acc, hot


def _concat_arm(scorer, rep, task, *, hot_update: bool = True):
    Xtr = concat(rep, task["train_idx"])
    scorer.fit(Xtr, task["ytr"])
    seen = scorer.scores(concat(rep, task["test_idx"])).max(axis=1)
    unseen = scorer.scores(concat(rep, task["held_idx"])).max(axis=1)
    acc = _closed_set_acc(scorer.scores(concat(rep, task["test_idx"])), task["yte"])
    hot = _median_update_ms(scorer, Xtr, task["ytr"]) if hot_update else 0.0
    return seen, unseen, acc, hot


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

    # matched MLP reference, fixed representation
    Xtr, Xva, Xte = (concat(rep, task[k]) for k in ("train_idx", "val_idx", "test_idx"))
    probs = [
        fit_mlp(
            Xtr, task["ytr"], Xva, task["yva"], Xte, len(task["classes"]),
            seed=seed, device=device,
        )[0]
        for seed in args.seeds
    ]
    mlp_probs = np.mean(probs, axis=0)
    mlp_acc = _acc(mlp_probs.argmax(1), task["yte"])
    mlp_per_seed = [_acc(p.argmax(1), task["yte"]) for p in probs]

    arms: dict[str, dict] = {}

    # reference: deployed fused cosine prototypes
    bank = build_bank(rep, task["train_idx"], task["labels"], task["classes"])
    seen = fused_scores(rep, bank, task["test_idx"]).max(axis=1)
    unseen = fused_scores(rep, bank, task["held_idx"]).max(axis=1)
    cosine_proto = {m: CosineProtoScorer().fit(rep[m][task["train_idx"]], task["ytr"])
                    for m in MODALITIES}
    hot = float(np.mean([
        _median_update_ms(cosine_proto[m], rep[m][task["train_idx"]], task["ytr"])
        for m in MODALITIES
    ]))
    arms["proto"] = _arm_metrics(
        seen, unseen,
        _closed_set_acc(fused_scores(rep, bank, task["test_idx"]), task["yte"]),
        hot,
    )

    # per-modality Mahalanobis / kNN, fused
    for name, cls, kwargs in (
        ("proto_mahalanobis", MahalanobisScorer, {}),
        ("proto_knn", KNNScorer, {"k": args.knn_k}),
    ):
        seen, unseen, acc, hot = _fused_modality_arm(cls, rep, task, **kwargs)
        arms[name] = _arm_metrics(seen, unseen, acc, hot)

    # single concatenated representation: the §5.5 protocol
    for name, scorer in (
        ("concat_cosine", CosineProtoScorer()),
        ("concat_mahalanobis", MahalanobisScorer()),
        ("concat_knn", KNNScorer(k=args.knn_k)),
    ):
        seen, unseen, acc, hot = _concat_arm(scorer, rep, task)
        arms[name] = _arm_metrics(seen, unseen, acc, hot)

    for arm in arms.values():
        arm["accuracy_delta_vs_mlp"] = float(arm["test_acc"] - mlp_acc)

    observed = {
        name: {
            "accuracy_delta_vs_mlp": arms[name]["accuracy_delta_vs_mlp"],
            "auroc": arms[name]["auroc"],
            "unseen_reject_at_95": arms[name]["unseen_reject_at_95"],
            "hot_update_ms": arms[name]["hot_update_ms"],
            "seed_std": arms[name]["seed_std"],
        }
        for name in ("proto", "proto_mahalanobis", "proto_knn")
    }
    check = check_readout_reasoning_preregistration(observed)

    payload = {
        "data": args.data,
        "experts": args.experts,
        "projections": args.projections,
        "regime": args.regime,
        "classes": len(task["classes"]),
        "counts": {k: int(len(task[k])) for k in ("train_idx", "val_idx", "test_idx", "held_idx")},
        "mlp_reference": {
            "acc": float(mlp_acc),
            "per_seed": [float(v) for v in mlp_per_seed],
            "seeds": list(args.seeds),
        },
        "arms": arms,
        "preregistration": check,
        "note": (
            "Mahalanobis update moves the class mean and keeps the pooled "
            "precision frozen; hot_update_ms measures only that update."
        ),
    }
    _write_json(out / "scorers.json", payload)
    (out / "SCORERS_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _render(p: dict) -> str:
    lines = [
        "# A1：距离开集打分器（部署公共空间）",
        "",
        f"- data: `{p['data']}`",
        f"- projections: `{p['projections']}`",
        f"- regime: {p['regime']} · classes: {p['classes']} · counts: {p['counts']}",
        f"- MLP 参照（同表示，{len(p['mlp_reference']['seeds'])} seeds）: "
        f"acc **{p['mlp_reference']['acc']:.3f}** "
        f"(per seed {', '.join(f'{v:.3f}' for v in p['mlp_reference']['per_seed'])})",
        "",
        "## 1. 主表",
        "",
        "| arm | closed-set acc | Δ vs MLP | AUROC | unseen reject @95 | hot update ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, arm in p["arms"].items():
        lines.append(
            f"| `{name}` | {arm['test_acc']:.3f} | {arm['accuracy_delta_vs_mlp']:+.3f} | "
            f"{arm['auroc']:.3f} | {arm['unseen_reject_at_95']:.3f} | "
            f"{arm['hot_update_ms']:.3f} |"
        )
    check = p["preregistration"]
    lines += [
        "",
        "## 2. 预注册检查",
        "",
        f"- 检查 arm 数: {check['checked']} · 通过: **{check['ok']}**",
    ]
    if check["mismatches"]:
        lines.append("")
        for mismatch in check["mismatches"]:
            lines.append(f"- `{mismatch}`")
    lines += [
        "",
        "## 3. 判读",
        "",
        "1. 闭集看 `Δ vs MLP`，开集看 `AUROC` 与 `unseen reject @95`；三者必须同时成立，",
        "   否则只是把拒绝率挪了个位置。",
        "2. `proto_*` 是部署形态（分模态 + 融合）；`concat_*` 是 §5.5 的同表示对照。",
        "   若只有 concat 变好而 proto 不变，说明瓶颈在**融合**，属于 A4 而不是 A1。",
        "3. `hot_update_ms` 对 Mahalanobis 只计类均值更新，池化协方差是冻结的；",
        "   这不是精确重估，报告里必须保留这句话。",
        "",
        "## 4. 边界",
        "",
        "- 合成代理数据、小编码器；结论需在真实 VLM 骨架上重跑。",
        "- 384 类 6-shot，Mahalanobis 用池化协方差 + shrinkage；单类协方差在此设定下奇异。",
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
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    payload = run(args)
    print(f"MLP reference acc: {payload['mlp_reference']['acc']:.3f}")
    for name, arm in payload["arms"].items():
        print(
            f"  {name:<20} acc={arm['test_acc']:.3f} "
            f"delta={arm['accuracy_delta_vs_mlp']:+.3f} "
            f"auroc={arm['auroc']:.3f} reject@95={arm['unseen_reject_at_95']:.3f} "
            f"hot={arm['hot_update_ms']:.3f}ms"
        )
    check = payload["preregistration"]
    print(f"preregistration ok={check['ok']} checked={check['checked']}")
    for mismatch in check["mismatches"]:
        print(f"  - {mismatch}")


if __name__ == "__main__":
    main()
