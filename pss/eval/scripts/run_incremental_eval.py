"""A5: does refreshing the pooled covariance remove incremental forgetting?

A4 measured −1.6 pt on old classes after 50 additions with a frozen precision.
This script adds the controls that measurement lacked:

  incr_frozen      precision frozen (A4 behaviour), means updated
  incr_refit_10    refit every 10 added classes
  incr_refit_25    refit every 25 added classes
  incr_refit_all   refit after every added class (accuracy upper bound)
  cosine_whiten    per-modality whitening + cosine prototypes, no covariance
  batch_joint      fit on base+added jointly (no-forgetting reference)

Evaluation uses val+test (two unseen views per class) rather than test alone,
because the test split has one row per class and the forgetting effect is
smaller than the sampling noise there.

Example
-------
python -u -m pss.eval.scripts.run_incremental_eval --out eval/results/readout_opt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pss._console import ensure_utf8
from pss.eval.readout_reasoning_preregistration import (
    check_readout_reasoning_preregistration,
)
from pss.eval.scripts.run_attribution_grid import (
    MODALITIES,
    _write_json,
    concat,
    load_features,
    task_split,
)
from pss.eval.scripts.run_fusion_ablation import _whiten
from pss.projection.heads import load_projection
from pss.prototypes.incremental import IncrementalLDABank
from pss.prototypes.scorers import CosineProtoScorer, predict_classes

POLICIES = (
    ("incr_frozen", 0),
    ("incr_refit_10", 10),
    ("incr_refit_25", 25),
    ("incr_refit_all", 1),
)


def _acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float((np.asarray(pred) == np.asarray(y)).mean())


def _split_by_class(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.isin(y, classes)


def _run_lda_policy(X_base, y_base, added_rows, added_order,
                    X_eval, y_eval, base_eval, added_eval,
                    refit_every: int, traj_every: int) -> dict:
    bank = IncrementalLDABank(refit_every=refit_every).fit(X_base, y_base)
    base_before = _acc(bank.predict(X_eval[base_eval]), y_eval[base_eval])

    trajectory = [{"n_added": 0, "base_acc": base_before}]
    for i, label in enumerate(added_order, 1):
        bank.add_class(int(label), added_rows[int(label)])
        if traj_every and i % traj_every == 0:
            trajectory.append(
                {"n_added": i, "base_acc": _acc(bank.predict(X_eval[base_eval]), y_eval[base_eval])}
            )

    pred = bank.predict(X_eval)
    base_after = _acc(pred[base_eval], y_eval[base_eval])
    added_acc = _acc(pred[added_eval], y_eval[added_eval])
    overall = _acc(pred, y_eval)
    return {
        "base_before": float(base_before),
        "base_after": float(base_after),
        "forgetting": float(base_before - base_after),
        "added_acc": float(added_acc),
        "overall_acc": float(overall),
        "n_refits": int(bank.n_refits),
        "refit_seconds": float(bank.refit_seconds),
        "mean_update_ms": float(bank.mean_update_seconds * 1000.0 / max(len(added_order), 1)),
        "trajectory": trajectory,
    }


def _run_cosine_policy(X_base, y_base, added_rows, added_order,
                       X_eval, y_eval, base_eval, added_eval) -> dict:
    scorer = CosineProtoScorer().fit(X_base, y_base)
    for label in added_order:
        scorer.add_class(int(label), added_rows[int(label)])
    pred = predict_classes(scorer, X_eval)
    base_before = _acc(
        predict_classes(CosineProtoScorer().fit(X_base, y_base), X_eval[base_eval]),
        y_eval[base_eval],
    )
    base_after = _acc(pred[base_eval], y_eval[base_eval])
    return {
        "base_before": float(base_before),
        "base_after": float(base_after),
        "forgetting": float(base_before - base_after),
        "added_acc": _acc(pred[added_eval], y_eval[added_eval]),
        "overall_acc": _acc(pred, y_eval),
        "n_refits": 0,
        "refit_seconds": 0.0,
        "mean_update_ms": 0.0,
        "trajectory": [],
    }


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
    rep = {}
    with torch.no_grad():
        for m in MODALITIES:
            heads[m].eval()
            rep[m] = heads[m](torch.as_tensor(raw[m]).to(device)).cpu().numpy().astype(np.float64)
    whitened = _whiten(rep, task["train_idx"])
    print("[rep] projected and whitened", flush=True)

    Xtr = concat(whitened, task["train_idx"])
    ytr = np.asarray(task["ytr"])
    eval_idx = np.concatenate([task["val_idx"], task["test_idx"]])
    X_eval = concat(whitened, eval_idx)
    y_eval = np.concatenate([task["yva"], task["yte"]])

    n_classes = len(task["classes"])
    rng = np.random.default_rng(args.split_seed)
    order = rng.permutation(n_classes)
    base_classes = order[: n_classes // 2]
    added_classes = order[n_classes // 2 : n_classes // 2 + args.n_added]
    added_order = rng.permutation(added_classes)

    base_train = _split_by_class(ytr, base_classes)
    X_base, y_base = Xtr[base_train], ytr[base_train]
    added_rows = {int(c): Xtr[ytr == c] for c in added_classes}
    joint_classes = np.concatenate([base_classes, added_classes])
    joint_train = _split_by_class(ytr, joint_classes)

    base_eval = _split_by_class(y_eval, base_classes)
    added_eval = _split_by_class(y_eval, added_classes)

    print(
        f"[split] base {len(base_classes)} added {len(added_classes)} "
        f"base_eval {int(base_eval.sum())} added_eval {int(added_eval.sum())}",
        flush=True,
    )

    results = {}
    for name, every in POLICIES:
        results[name] = _run_lda_policy(
            X_base, y_base, added_rows, added_order,
            X_eval, y_eval, base_eval, added_eval,
            refit_every=every, traj_every=args.traj_every,
        )
        print(f"[{name}] forgetting {results[name]['forgetting']:+.3f} "
              f"added {results[name]['added_acc']:.3f}", flush=True)

    results["cosine_whiten"] = _run_cosine_policy(
        X_base, y_base, added_rows, added_order,
        X_eval, y_eval, base_eval, added_eval,
    )
    print(f"[cosine_whiten] forgetting {results['cosine_whiten']['forgetting']:+.3f} "
          f"added {results['cosine_whiten']['added_acc']:.3f}", flush=True)

    joint = IncrementalLDABank(refit_every=0).fit(Xtr[joint_train], ytr[joint_train])
    pred = joint.predict(X_eval)
    results["batch_joint"] = {
        "base_before": None,
        "base_after": _acc(pred[base_eval], y_eval[base_eval]),
        "forgetting": None,
        "added_acc": _acc(pred[added_eval], y_eval[added_eval]),
        "overall_acc": _acc(pred, y_eval),
        "n_refits": int(joint.n_refits),
        "refit_seconds": float(joint.refit_seconds),
        "mean_update_ms": None,
        "trajectory": [],
    }
    print(f"[batch_joint] base {results['batch_joint']['base_after']:.3f} "
          f"added {results['batch_joint']['added_acc']:.3f}", flush=True)

    check = check_readout_reasoning_preregistration({
        name: {
            "forgetting": results[name]["forgetting"],
            "added_acc": results[name]["added_acc"],
        }
        for name in ("incr_frozen", "incr_refit_25", "incr_refit_all")
    })

    payload = {
        "data": args.data,
        "projections": args.projections,
        "regime": args.regime,
        "classes": n_classes,
        "counts": {
            "base_classes": int(len(base_classes)),
            "added_classes": int(len(added_classes)),
            "base_eval": int(base_eval.sum()),
            "added_eval": int(added_eval.sum()),
        },
        "split_seed": int(args.split_seed),
        "policies": results,
        "preregistration": check,
        "note": (
            "forgetting = base_before - base_after on unseen views; "
            "batch_joint is the joint-training reference. Refits reuse a "
            "stored feature reservoir, which is what keeps them exact."
        ),
    }
    _write_json(out / "incremental_eval.json", payload)
    (out / "INCREMENTAL_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _render(p: dict) -> str:
    lines = [
        "# A5：增量加类与协方差刷新",
        "",
        f"- data: `{p['data']}` · projections: `{p['projections']}`",
        f"- classes: {p['classes']} · split seed: {p['split_seed']} · counts: {p['counts']}",
        "",
        "## 1. 主表（评估 = val+test，每类 2 个未见视角）",
        "",
        "| policy | base before | base after | forgetting | added acc | overall | refits | refit s | mean ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in p["policies"].items():
        before = "n/a" if row["base_before"] is None else f"{row['base_before']:.3f}"
        forgetting = "n/a" if row["forgetting"] is None else f"{row['forgetting']:+.3f}"
        mean_ms = "n/a" if row["mean_update_ms"] is None else f"{row['mean_update_ms']:.3f}"
        lines.append(
            f"| `{name}` | {before} | {row['base_after']:.3f} | {forgetting} | "
            f"{row['added_acc']:.3f} | {row['overall_acc']:.3f} | {row['n_refits']} | "
            f"{row['refit_seconds']:.3f} | {mean_ms} |"
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
        "## 3. 基类精度轨迹",
        "",
        "| policy | n_added → base acc |",
        "|---|---|",
    ]
    for name in ("incr_frozen", "incr_refit_25", "incr_refit_all"):
        traj = p["policies"][name]["trajectory"]
        if not traj:
            continue
        points = ", ".join(f"{t['n_added']}→{t['base_acc']:.3f}" for t in traj)
        lines.append(f"| `{name}` | {points} |")

    lines += [
        "",
        "## 4. 判读",
        "",
        "1. `batch_joint` 是同一类集合下的联合训练参照；增量策略的 base_after 应尽量接近它。",
        "2. `cosine_whiten` 不做全协方差，若它的 forgetting 更小，说明遗忘来自协方差估计而非均值更新。",
        "3. `incr_refit_all` 是精度上限；`incr_refit_25` 若与它持平，说明 25 类一次的刷新足够。",
        "4. 刷新成本要按 refit s 和 refit 次数一起看，不能只看精度。",
        "",
        "## 5. 边界",
        "",
        "- 评估行只有每类 2 个未见视角；小样本噪声约 ±1 点。",
        "- 刷新用存储的特征 reservoir，部署中需要保留样本或可重放的摘要。",
        "- 合成代理数据、小编码器；真实骨干需重跑。",
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
    ap.add_argument("--n-added", type=int, default=192)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--traj-every", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    payload = run(args)
    for name, row in payload["policies"].items():
        forgetting = "n/a" if row["forgetting"] is None else f"{row['forgetting']:+.3f}"
        print(f"  {name:<16} forgetting {forgetting} added {row['added_acc']:.3f} "
              f"overall {row['overall_acc']:.3f} refits {row['n_refits']}")
    check = payload["preregistration"]
    print(f"preregistration ok={check['ok']}")
    for mismatch in check["mismatches"]:
        print(f"  - {mismatch}")


if __name__ == "__main__":
    main()
