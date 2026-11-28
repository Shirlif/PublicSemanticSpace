"""U4: long-term incremental maintenance (delete / merge on top of A5).

Executable form of ``docs/plans/PSS-架构上界复核与下一阶段.md`` §6 (U4).

A5 measured the 1.3 pt class-dilution floor for *addition* only.  U4 asks
whether that floor survives the two operations a deployed bank actually sees:

  - delete 10% of classes (dead objects, retired labels)
  - merge 5% of classes (duplicate entities collapsed)

For each refresh policy (frozen / every 10 / every class) it records base
forgetting, added accuracy, post-delete / post-merge accuracy, rollback
exactness and covariance drift.  The pre-registered arm is the recommended
policy ``incr_refit_10``.

Example
-------
python -u -m pss.eval.scripts.run_upper_bound_u4 \
  --out eval/results/upper_bound/u4 --device cuda
"""

from __future__ import annotations

import argparse
import json
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

POLICIES = (
    ("incr_frozen", 0),
    ("incr_refit_10", 10),
    ("incr_refit_all", 1),
)


def _acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float((np.asarray(pred) == np.asarray(y)).mean())


def _mask(y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.isin(y, labels)


def _relative_change(before: np.ndarray, after: np.ndarray) -> float:
    denom = float(np.linalg.norm(before))
    return float(np.linalg.norm(after - before) / max(denom, 1e-12))


def _run_policy(
    *,
    X_base: np.ndarray,
    y_base: np.ndarray,
    added_rows: dict[int, np.ndarray],
    added_order: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    base_classes: np.ndarray,
    added_classes: np.ndarray,
    refit_every: int,
    delete_fraction: float,
    merge_fraction: float,
    seed: int,
) -> dict:
    bank = IncrementalLDABank(refit_every=refit_every).fit(X_base, y_base)
    base_mask = _mask(y_eval, base_classes)
    added_mask = _mask(y_eval, added_classes)
    base_before = _acc(bank.predict(X_eval[base_mask]), y_eval[base_mask])
    precision_before_add = bank.precision_.copy()

    for label in added_order:
        bank.add_class(int(label), added_rows[int(label)])

    pred_add = bank.predict(X_eval)
    base_after_add = _acc(pred_add[base_mask], y_eval[base_mask])
    added_after_add = _acc(pred_add[added_mask], y_eval[added_mask])
    precision_after_add = bank.precision_.copy()
    snapshot = bank.snapshot()
    pred_snapshot = pred_add.copy()

    all_classes = np.concatenate([base_classes, added_classes])
    rng = np.random.default_rng(seed)
    n_delete = max(1, int(round(delete_fraction * len(all_classes))))
    deleted = rng.choice(all_classes, size=n_delete, replace=False)
    for label in deleted:
        if int(label) in bank.classes_:
            bank.remove_class(int(label))

    remaining = np.array([c for c in all_classes if c not in set(map(int, deleted))])
    base_remaining = np.array([c for c in base_classes if c not in set(map(int, deleted))])
    added_remaining = np.array([c for c in added_classes if c not in set(map(int, deleted))])
    pred_delete = bank.predict(X_eval)
    base_after_delete = (
        _acc(pred_delete[_mask(y_eval, base_remaining)], y_eval[_mask(y_eval, base_remaining)])
        if len(base_remaining) else None
    )
    added_after_delete = (
        _acc(pred_delete[_mask(y_eval, added_remaining)], y_eval[_mask(y_eval, added_remaining)])
        if len(added_remaining) else None
    )

    n_merge = max(1, int(round(merge_fraction * len(remaining))))
    pairs = []
    pool = list(map(int, remaining))
    rng.shuffle(pool)
    while len(pairs) < n_merge and len(pool) >= 2:
        a, b = pool.pop(), pool.pop()
        pairs.append((a, b))
    merged_from = set()
    for src, dst in pairs:
        if src in bank.classes_ and dst in bank.classes_:
            bank.merge_classes(src, dst)
            merged_from.add(src)

    keep_base = [c for c in base_remaining if int(c) not in merged_from]
    keep_added = [c for c in added_remaining if int(c) not in merged_from]
    pred_merge = bank.predict(X_eval)
    base_after_merge = (
        _acc(pred_merge[_mask(y_eval, np.asarray(keep_base))], y_eval[_mask(y_eval, np.asarray(keep_base))])
        if keep_base else None
    )
    added_after_merge = (
        _acc(pred_merge[_mask(y_eval, np.asarray(keep_added))], y_eval[_mask(y_eval, np.asarray(keep_added))])
        if keep_added else None
    )
    precision_after_merge = bank.precision_.copy()

    bank.restore(snapshot)
    pred_restored = bank.predict(X_eval)
    rollback_exact = bool(np.array_equal(pred_restored, pred_snapshot))

    return {
        "base_before": float(base_before),
        "base_after_add": float(base_after_add),
        "forgetting": float(base_before - base_after_add),
        "added_acc": float(added_after_add),
        "base_after_delete": base_after_delete,
        "added_after_delete": added_after_delete,
        "base_after_merge": base_after_merge,
        "added_after_merge": added_after_merge,
        "n_deleted": int(n_delete),
        "n_merged": int(len(merged_from)),
        "rollback_exact": rollback_exact,
        "drift_add": _relative_change(precision_before_add, precision_after_add),
        "drift_delete": _relative_change(precision_after_add, precision_after_merge),
        "cond_before_add": float(np.linalg.cond(precision_before_add)),
        "cond_after_merge": float(np.linalg.cond(precision_after_merge)),
        "n_refits": int(bank.n_refits),
        "refit_seconds": float(bank.refit_seconds),
    }


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feats = load_features(args.data, args.experts, out)
    task = task_split(feats, args.regime)
    raw = {m: np.asarray(feats[m], dtype=np.float32) for m in MODALITIES}
    heads = {
        m: load_projection(Path(args.projections) / f"{m}.pt", device=args.device)
        for m in MODALITIES
    }
    rep = {}
    with torch.no_grad():
        for m in MODALITIES:
            heads[m].eval()
            rep[m] = heads[m](torch.as_tensor(raw[m]).to(args.device)).cpu().numpy().astype(np.float64)
    whitened = _whiten(rep, task["train_idx"])

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

    base_train = _mask(ytr, base_classes)
    X_base, y_base = Xtr[base_train], ytr[base_train]
    added_rows = {int(c): Xtr[ytr == c] for c in added_classes}

    print(
        f"[split] base {len(base_classes)} added {len(added_classes)} "
        f"eval {int(_mask(y_eval, base_classes).sum())}+{int(_mask(y_eval, added_classes).sum())}",
        flush=True,
    )

    results = {}
    for name, every in POLICIES:
        results[name] = _run_policy(
            X_base=X_base,
            y_base=y_base,
            added_rows=added_rows,
            added_order=added_order,
            X_eval=X_eval,
            y_eval=y_eval,
            base_classes=base_classes,
            added_classes=added_classes,
            refit_every=every,
            delete_fraction=args.delete_fraction,
            merge_fraction=args.merge_fraction,
            seed=args.split_seed,
        )
        row = results[name]
        print(
            f"[{name}] forgetting {row['forgetting']:+.3f} added {row['added_acc']:.3f} "
            f"delete base {row['base_after_delete']} merge base {row['base_after_merge']} "
            f"rollback {row['rollback_exact']}",
            flush=True,
        )

    observed = {
        "u_incr_delete_merge": {
            "forgetting": results["incr_refit_10"]["forgetting"],
            "added_acc": results["incr_refit_10"]["added_acc"],
        }
    }
    check = check_readout_reasoning_preregistration(observed)

    payload = {
        "stage": "u4",
        "data": args.data,
        "projections": args.projections,
        "regime": args.regime,
        "classes": n_classes,
        "split_seed": int(args.split_seed),
        "delete_fraction": float(args.delete_fraction),
        "merge_fraction": float(args.merge_fraction),
        "policies": results,
        "observed": observed,
        "preregistration": check,
        "note": (
            "forgetting = base_before - base_after_add (A5 definition); delete "
            "and merge are applied after all additions; rollback restores the "
            "post-add snapshot and checks prediction equality."
        ),
    }
    _write_json(out / "u4.json", payload)
    (out / "U4_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _render(p: dict) -> str:
    lines = [
        "# U4：长期增量维护（删除 / 合并）",
        "",
        f"- data: `{p['data']}` · projections: `{p['projections']}`",
        f"- classes {p['classes']} · delete {p['delete_fraction']:.0%} · "
        f"merge {p['merge_fraction']:.0%} · split seed {p['split_seed']}",
        "",
        "## 1. 主表",
        "",
        "| policy | base before | after add | forgetting | added acc | after delete | after merge | rollback | refits |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for name, r in p["policies"].items():
        lines.append(
            f"| `{name}` | {r['base_before']:.3f} | {r['base_after_add']:.3f} | "
            f"{r['forgetting']:+.3f} | {r['added_acc']:.3f} | "
            f"{_fmt(r['base_after_delete'])} | {_fmt(r['base_after_merge'])} | "
            f"{r['rollback_exact']} | {r['n_refits']} |"
        )
    lines += [
        "",
        "## 2. 协方差漂移",
        "",
        "| policy | drift add | drift delete+merge | cond before | cond after |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in p["policies"].items():
        lines.append(
            f"| `{name}` | {r['drift_add']:.3f} | {r['drift_delete']:.3f} | "
            f"{r['cond_before_add']:.1f} | {r['cond_after_merge']:.1f} |"
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
        "- 主 bar 是 `incr_refit_10` 的 `forgetting ≤ 0.03` 与 `added_acc ≥ 0.93`；"
        "删除/合并后的精度是诊断，不是主 bar。",
        "- 若 `rollback_exact=False`，说明删除/合并破坏了可回滚性，部署前必须修。",
        "- 若 delete/merge 后 base 精度掉得比 add 更多，说明 1.3 pt 地板只在“加类”成立，"
        "删除/合并需要单独的重估策略。",
        "",
        "## 5. 边界",
        "",
        "- 删除/合并的类集合按固定 seed 抽取，不是真实生命周期分布。",
        "- 合并用随机类对，不模拟“同一实体的重复 ID”这种更难的情形。",
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
    ap.add_argument("--out", default="eval/results/upper_bound/u4")
    ap.add_argument("--n-added", type=int, default=192)
    ap.add_argument("--delete-fraction", type=float, default=0.10)
    ap.add_argument("--merge-fraction", type=float, default=0.05)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    payload = run(args)
    for name, r in payload["policies"].items():
        print(f"  {name:<16} forgetting {r['forgetting']:+.3f} added {r['added_acc']:.3f}")
    check = payload["preregistration"]
    print(f"preregistration ok={check['ok']} checked={check['checked']}")
    for mismatch in check["mismatches"]:
        print(f"  - {mismatch}")


if __name__ == "__main__":
    main()
