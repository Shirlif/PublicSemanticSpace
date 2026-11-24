"""P3: incremental category update, PSS package add vs transformer fine-tune.

Base model: 16 seen categories. New category: cloth (held out). Both methods
see the same cloth train/test split. PSS adds one prototype; the transformer
fine-tunes its head (last-layer) or the whole model.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.eval.scripts.run_p2_openset import (
    MODALITIES,
    _tensor_tokens,
    _train_transformer,
)
from pss.learning import PackageManager, PrototypePackage
from pss.learning.review import bank_from_package
from pss.prototypes import PrototypeBank, recognize
from pss.prototypes.build import extract_projected_features

NEW_CATEGORY = "cloth"


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _old_test(feats, bank):
    return [
        i
        for i, (s, c) in enumerate(zip(feats.splits, feats.categories))
        if s == "test" and c in bank.categories
    ]


def _new_indices(feats):
    return [
        i
        for i, (s, c) in enumerate(zip(feats.splits, feats.categories))
        if s == "test_category" and c == NEW_CATEGORY
    ]


def _accuracy(bank, Z, indices, weights, feats):
    correct = 0
    for i in indices:
        query = {m: Z[m][i] for m in MODALITIES}
        result = recognize(query, bank, weights=weights, use_tree=False, threshold=None)
        if result.ranked and result.ranked[0][0] == feats.categories[i]:
            correct += 1
    return correct / max(1, len(indices))


def _pss_update(base_bank, Z, train_idx, weights):
    vectors = {
        m: _unit(Z[m][train_idx].mean(axis=0)).astype(np.float32) for m in MODALITIES
    }
    stats = {
        m: {
            "sample_count": len(train_idx),
            "variance": float(
                np.mean(1.0 - np.asarray(Z[m][train_idx], dtype=float) @ vectors[m])
            ),
        }
        for m in MODALITIES
    }
    base = PrototypePackage(
        version="v1.0",
        categories=base_bank.categories,
        protos={m: base_bank.protos[m] for m in base_bank.protos},
        stats=base_bank.stats,
    )
    start = time.perf_counter()
    new = base.append_category(
        NEW_CATEGORY, vectors, stats, version="v1.1", changelog=f"add {NEW_CATEGORY}"
    )
    with tempfile.TemporaryDirectory() as tmp:
        manager = PackageManager(Path(tmp))
        manager.install(base)
        manager.install(new)
        manager.activate("v1.0")
        manager.activate("v1.1")
        active = manager.active_package()
        rolled = manager.rollback()
    elapsed = time.perf_counter() - start
    return bank_from_package(active), elapsed, new, rolled


def _expand_head(model, n_new_total: int):
    old = model.head
    new_head = nn.Linear(old.in_features, n_new_total)
    with torch.no_grad():
        new_head.weight[: old.out_features] = old.weight
        new_head.bias[: old.out_features] = old.bias
    model.head = new_head
    return model


def _fisher(model, Z, idx, classes, feats, device):
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    class_to_idx = {c: i for i, c in enumerate(classes)}
    x = _tensor_tokens(Z, idx, device)
    y = torch.as_tensor(
        [class_to_idx[feats.categories[i]] for i in idx],
        dtype=torch.long,
        device=device,
    )
    model.eval()
    batches = 0
    for start in range(0, len(y), 16):
        logits = model([t[start : start + 16] for t in x])
        loss = F.cross_entropy(logits, y[start : start + 16])
        model.zero_grad()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
        batches += 1
    if batches:
        fisher = {k: v / batches for k, v in fisher.items()}
    model.zero_grad()
    return fisher


def _fine_tune(
    model,
    Z,
    train_idx,
    n_old: int,
    epochs: int,
    mode: str,
    device: str,
    old_train_idx=None,
    feats=None,
    classes=None,
    fisher=None,
    theta0=None,
    ewc_lambda: float = 100.0,
):
    model = _expand_head(model, n_old + 1)
    if mode == "last_layer":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True
    x = _tensor_tokens(Z, train_idx, device)
    y = torch.full((len(train_idx),), n_old, dtype=torch.long, device=device)
    lr = 1e-3 if mode == "last_layer" else 1e-4
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    x_old = y_old = None
    if mode == "replay" and old_train_idx:
        class_to_idx = {c: i for i, c in enumerate(classes or [])}
        x_old = _tensor_tokens(Z, old_train_idx, device)
        y_old = torch.as_tensor(
            [class_to_idx[feats.categories[i]] for i in old_train_idx],
            dtype=torch.long,
            device=device,
        )

    start = time.perf_counter()
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(y), device=device)
        for s in range(0, len(perm), 16):
            idx = perm[s : s + 16]
            xb = [t[idx] for t in x]
            yb = y[idx]
            if x_old is not None:
                replay_idx = torch.randint(0, len(y_old), (len(idx),), device=device)
                xb = [torch.cat([a, b[replay_idx]], dim=0) for a, b in zip(xb, x_old)]
                yb = torch.cat([yb, y_old[replay_idx]], dim=0)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if mode == "ewc" and fisher is not None and theta0 is not None:
                penalty = sum(
                    (fisher[name] * (p - theta0[name]) ** 2).sum()
                    for name, p in model.named_parameters()
                    if name in fisher and p.shape == theta0[name].shape
                )
                loss = loss + ewc_lambda * penalty
            opt.zero_grad()
            loss.backward()
            opt.step()
    elapsed = time.perf_counter() - start
    model.eval()
    return model, elapsed


def _model_accuracy(model, Z, indices, feats, classes, device):
    x = _tensor_tokens(Z, indices, device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).cpu().numpy()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    correct = sum(
        int(pred[k] == class_to_idx[feats.categories[i]]) for k, i in enumerate(indices)
    )
    return correct / max(1, len(indices))


def run_p3(
    data_dir,
    expert_dir,
    projection_dir,
    prototype_dir,
    out_dir,
    seeds: int = 5,
    base_epochs: int = 200,
    finetune_epochs: int = 50,
    device: str = "cpu",
) -> dict:
    feats, Z = extract_projected_features(data_dir, expert_dir, projection_dir, device)
    bank = PrototypeBank.load(Path(prototype_dir) / "prototypes.npz")
    weights = {"vision": 1.0, "tactile": 0.5, "geometry": 1.0}
    metrics_path = Path(prototype_dir) / "metrics.json"
    if metrics_path.exists():
        weights = json.loads(metrics_path.read_text(encoding="utf-8")).get("weights", weights)

    old_idx = _old_test(feats, bank)
    old_train_idx = [
        i
        for i, (s, c) in enumerate(zip(feats.splits, feats.categories))
        if s == "train" and c in bank.categories
    ]
    new_idx = _new_indices(feats)
    split = max(1, int(len(new_idx) * 0.6))
    new_train, new_test = new_idx[:split], new_idx[split:]

    pss_before = _accuracy(bank, Z, old_idx, weights, feats)
    pss_new_bank, pss_time, pss_pkg, rolled = _pss_update(
        bank, Z, new_train, weights
    )
    pss_old_after = _accuracy(pss_new_bank, Z, old_idx, weights, feats)
    pss_new_after = _accuracy(pss_new_bank, Z, new_test, weights, feats)

    models = []
    training = []
    for seed in range(seeds):
        model, info = _train_transformer(Z, feats, bank, seed, base_epochs, device)
        models.append(model)
        training.append({"seed": seed, "best_val_acc": info["best_val_acc"]})
    old_classes = sorted(bank.categories)
    new_classes = old_classes + [NEW_CATEGORY]

    variants = {"last_layer": [], "full": [], "replay": [], "ewc": []}
    for seed, base_model in enumerate(models):
        old_before = _model_accuracy(base_model, Z, old_idx, feats, old_classes, device)
        fisher = _fisher(base_model, Z, old_train_idx, old_classes, feats, device)
        theta0 = {
            name: p.detach().clone() for name, p in base_model.named_parameters()
        }
        for mode in variants:
            import copy

            ft_model = copy.deepcopy(base_model)
            ft_model, elapsed = _fine_tune(
                ft_model,
                Z,
                new_train,
                len(old_classes),
                finetune_epochs,
                mode,
                device,
                old_train_idx=old_train_idx,
                feats=feats,
                classes=old_classes,
                fisher=fisher if mode == "ewc" else None,
                theta0=theta0 if mode == "ewc" else None,
            )
            old_after = _model_accuracy(
                ft_model, Z, old_idx, feats, new_classes, device
            )
            new_after = _model_accuracy(
                ft_model, Z, new_test, feats, new_classes, device
            )
            variants[mode].append(
                {
                    "seed": seed,
                    "old_before": old_before,
                    "old_after": old_after,
                    "new_after": new_after,
                    "forgetting": old_before - old_after,
                    "update_time": elapsed,
                }
            )

    def agg(rows, key):
        return float(np.mean([r[key] for r in rows])), float(np.std([r[key] for r in rows]))

    tr = {}
    for mode, rows in variants.items():
        tr[mode] = {
            "old_before": agg(rows, "old_before")[0],
            "old_after_mean": agg(rows, "old_after")[0],
            "old_after_std": agg(rows, "old_after")[1],
            "new_after_mean": agg(rows, "new_after")[0],
            "new_after_std": agg(rows, "new_after")[1],
            "forgetting_mean": agg(rows, "forgetting")[0],
            "forgetting_std": agg(rows, "forgetting")[1],
            "update_time_mean": agg(rows, "update_time")[0],
            "rows": rows,
        }

    best_tr_forgetting = min(
        tr["last_layer"]["forgetting_mean"],
        tr["full"]["forgetting_mean"],
        tr["replay"]["forgetting_mean"],
        tr["ewc"]["forgetting_mean"],
    )
    h3 = {
        "pss_forgetting": pss_before - pss_old_after,
        "best_transformer_forgetting": best_tr_forgetting,
        "forgetting_gap": best_tr_forgetting - (pss_before - pss_old_after),
        "pss_update_time": pss_time,
        "transformer_update_time": min(
            tr["last_layer"]["update_time_mean"],
            tr["full"]["update_time_mean"],
            tr["replay"]["update_time_mean"],
            tr["ewc"]["update_time_mean"],
        ),
        "time_ratio": min(
            tr["last_layer"]["update_time_mean"],
            tr["full"]["update_time_mean"],
            tr["replay"]["update_time_mean"],
            tr["ewc"]["update_time_mean"],
        )
        / max(pss_time, 1e-9),
        "forgetting_pass": bool(
            best_tr_forgetting - (pss_before - pss_old_after) >= 0.05
        ),
        "time_pass": bool(
            min(tr["last_layer"]["update_time_mean"], tr["full"]["update_time_mean"])
            / max(pss_time, 1e-9)
            >= 10.0
        ),
    }
    metrics = {
        "data_dir": str(data_dir),
        "n_old_test": len(old_idx),
        "n_new_train": len(new_train),
        "n_new_test": len(new_test),
        "pss": {
            "old_before": pss_before,
            "old_after": pss_old_after,
            "new_after": pss_new_after,
            "forgetting": pss_before - pss_old_after,
            "update_time": pss_time,
            "rollback": rolled,
            "categories": pss_new_bank.categories,
        },
        "transformer": tr,
        "h3": h3,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p3_incremental.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "p3_incremental.md", metrics)
    return metrics


def _write_report(path: Path, metrics: dict) -> None:
    pss = metrics["pss"]
    tr = metrics["transformer"]
    h3 = metrics["h3"]
    lines = [
        "# P3: incremental category update",
        "",
        f"- old test: {metrics['n_old_test']}, new train/test: {metrics['n_new_train']} / {metrics['n_new_test']}",
        "",
        "| method | old before | old after | new after | forgetting | update time |",
        "|---|---:|---:|---:|---:|---:|",
        f"| PSS prototype add | {pss['old_before']:.3f} | {pss['old_after']:.3f} | "
        f"{pss['new_after']:.3f} | {pss['forgetting']:.3f} | {pss['update_time']*1000:.2f} ms |",
    ]
    for mode in ("last_layer", "full", "replay", "ewc"):
        r = tr[mode]
        lines.append(
            f"| Transformer {mode} | {r['old_before']:.3f} | "
            f"{r['old_after_mean']:.3f} ± {r['old_after_std']:.3f} | "
            f"{r['new_after_mean']:.3f} ± {r['new_after_std']:.3f} | "
            f"{r['forgetting_mean']:.3f} | {r['update_time_mean']*1000:.1f} ms |"
        )
    lines += [
        "",
        f"**H3 forgetting**: PSS {h3['pss_forgetting']:.3f} vs best transformer "
        f"{h3['best_transformer_forgetting']:.3f}, gap {h3['forgetting_gap']:+.3f} "
        f"-> {'PASS' if h3['forgetting_pass'] else 'FAIL'}",
        f"**H3 time**: ratio {h3['time_ratio']:.1f}x -> "
        f"{'PASS' if h3['time_pass'] else 'FAIL'}",
        "",
        f"PSS rollback: {h3 and pss['rollback']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/category_pretrain")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    ap.add_argument("--projections", default="checkpoints/phase45_projection")
    ap.add_argument("--prototypes", default="checkpoints/phase45_prototypes")
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base-epochs", type=int, default=200)
    ap.add_argument("--finetune-epochs", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    metrics = run_p3(
        args.data,
        args.experts,
        args.projections,
        args.prototypes,
        args.out,
        seeds=args.seeds,
        base_epochs=args.base_epochs,
        finetune_epochs=args.finetune_epochs,
        device=args.device,
    )
    print("P3 complete:")
    print(
        f"  PSS old {metrics['pss']['old_before']:.3f} -> {metrics['pss']['old_after']:.3f}, "
        f"new {metrics['pss']['new_after']:.3f}, time {metrics['pss']['update_time']*1000:.2f} ms"
    )
    for mode in ("last_layer", "full", "replay", "ewc"):
        r = metrics["transformer"][mode]
        print(
            f"  TR {mode}: old {r['old_before']:.3f} -> {r['old_after_mean']:.3f}, "
            f"new {r['new_after_mean']:.3f}, time {r['update_time_mean']*1000:.1f} ms"
        )
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
