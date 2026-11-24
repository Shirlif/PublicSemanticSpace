"""P1-8: prototype quality governance under poisoned incremental samples.

Builds the new cloth prototype from clean or corrupted new-class features and
compares mean, trimmed mean, medoid, and coordinate median. Old prototypes are
never touched, so old-class accuracy should stay flat; the question is how much
new-class accuracy and outlier detection survive poisoning.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.scripts.run_p3_incremental import NEW_CATEGORY, _accuracy, _unit
from pss.learning import PackageManager, PrototypePackage
from pss.learning.review import bank_from_package
from pss.prototypes import PrototypeBank
from pss.prototypes.build import extract_projected_features


def _new_indices(feats):
    return [
        i
        for i, (s, c) in enumerate(zip(feats.splits, feats.categories))
        if s == "test_category" and c == NEW_CATEGORY
    ]


def _old_test(feats, bank):
    return [
        i
        for i, (s, c) in enumerate(zip(feats.splits, feats.categories))
        if s == "test" and c in bank.categories
    ]


def _prototype(Z, idx, mode):
    rows = np.stack([np.asarray(Z[m][idx], dtype=float) for m in Z], axis=1)
    # rows: [n, n_mod, d]; compute per modality.
    out = {}
    for mi, m in enumerate(Z):
        x = np.asarray(Z[m][idx], dtype=float)
        mean = _unit(x.mean(axis=0))
        if mode == "mean":
            v = mean
        elif mode == "trimmed":
            sims = x @ mean
            keep = max(1, int(len(x) * 0.8))
            order = np.argsort(-sims)[:keep]
            v = _unit(x[order].mean(axis=0))
        elif mode == "medoid":
            sims = x @ mean
            v = _unit(x[int(np.argmax(sims))])
        elif mode == "median":
            v = _unit(np.median(x, axis=0))
        else:
            raise ValueError(mode)
        out[m] = v.astype(np.float32)
    return out


def _stats(Z, idx, vector):
    out = {}
    for m in Z:
        x = np.asarray(Z[m][idx], dtype=float)
        sims = x @ vector[m]
        out[m] = {
            "sample_count": len(idx),
            "variance": float(np.mean(1.0 - sims)),
            "p10_similarity": float(np.percentile(sims, 10)),
            "min_similarity": float(np.min(sims)),
        }
    return out


def run(
    data_dir="data/category_pretrain",
    expert_dir="checkpoints/category_pretrain_experts",
    projection_dir="checkpoints/phase45_projection",
    prototype_dir="checkpoints/phase45_prototypes",
    out_dir="eval/results/pss_vs_transformer",
    seeds=5,
    poison_fractions=(0.0, 0.1, 0.3),
):
    feats, Z = extract_projected_features(data_dir, expert_dir, projection_dir, "cpu")
    bank = PrototypeBank.load(Path(prototype_dir) / "prototypes.npz")
    weights = {"vision": 1.0, "tactile": 0.5, "geometry": 1.0}
    mp = Path(prototype_dir) / "metrics.json"
    if mp.exists():
        weights = json.loads(mp.read_text(encoding="utf-8")).get("weights", weights)

    old_idx = _old_test(feats, bank)
    new_idx = _new_indices(feats)
    split = max(1, int(len(new_idx) * 0.6))
    new_train, new_test = new_idx[:split], new_idx[split:]
    other_train = [
        i
        for i, (s, c) in enumerate(zip(feats.splits, feats.categories))
        if s == "train" and c in bank.categories
    ]

    results = {}
    for frac in poison_fractions:
        rows = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            clean = list(new_train)
            n_bad = int(round(len(clean) * frac))
            bad_idx = []
            if n_bad:
                bad_idx = list(rng.choice(other_train, size=n_bad, replace=False))
            train_idx = clean + bad_idx
            for mode in ("mean", "trimmed", "medoid", "median"):
                vector = _prototype(Z, train_idx, mode)
                stats = _stats(Z, train_idx, vector)
                clean_vector = _prototype(Z, clean, mode)
                clean_stats = _stats(Z, clean, clean_vector)
                protos = {m: np.concatenate([bank.protos[m], vector[m][None]], axis=0)
                          for m in bank.protos}
                categories = list(bank.categories) + [NEW_CATEGORY]
                stats_all = dict(bank.stats)
                stats_all[NEW_CATEGORY] = stats
                new_bank = PrototypeBank(categories, protos, stats_all, version="v1.1")
                old_acc = _accuracy(new_bank, Z, old_idx, weights, feats)
                new_acc = _accuracy(new_bank, Z, new_test, weights, feats)
                # Outlier flag: similarity below the 10th percentile of clean
                # in-class similarity is treated as suspicious.
                flagged = 0
                if bad_idx:
                    for m in Z:
                        sims = np.asarray(Z[m][bad_idx], dtype=float) @ vector[m]
                        flagged += int(
                            np.sum(sims < clean_stats[m]["p10_similarity"])
                        )
                    flagged /= max(1, len(bad_idx) * len(Z))
                rows.append(
                    {
                        "seed": seed,
                        "prototype": mode,
                        "old_accuracy": old_acc,
                        "new_accuracy": new_acc,
                        "poison_flagged": flagged,
                        "variance": float(np.mean([stats[m]["variance"] for m in stats])),
                    }
                )

        agg = {}
        for mode in ("mean", "trimmed", "medoid", "median"):
            vals = [r for r in rows if r["prototype"] == mode]
            agg[mode] = {
                "old_accuracy": float(np.mean([r["old_accuracy"] for r in vals])),
                "new_accuracy": float(np.mean([r["new_accuracy"] for r in vals])),
                "new_accuracy_std": float(np.std([r["new_accuracy"] for r in vals])),
                "poison_flagged": float(np.mean([r["poison_flagged"] for r in vals])),
                "variance": float(np.mean([r["variance"] for r in vals])),
            }
        results[f"poison_{int(frac*100)}"] = agg

    # Rollback timing on the clean trimmed package.
    base = PrototypePackage(
        version="v1.0",
        categories=bank.categories,
        protos={m: bank.protos[m] for m in bank.protos},
        stats=bank.stats,
    )
    vector = _prototype(Z, new_train, "trimmed")
    stats = _stats(Z, new_train, vector)
    start = time.perf_counter()
    new = base.append_category(NEW_CATEGORY, vector, stats, version="v1.1")
    with tempfile.TemporaryDirectory() as tmp:
        manager = PackageManager(Path(tmp))
        manager.install(base)
        manager.install(new)
        manager.activate("v1.0")
        manager.activate("v1.1")
        manager.rollback()
    rollback_time = time.perf_counter() - start

    metrics = {
        "data_dir": str(data_dir),
        "n_old_test": len(old_idx),
        "n_new_train": len(new_train),
        "n_new_test": len(new_test),
        "results": results,
        "rollback_time": rollback_time,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p18_prototype_governance.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p18_prototype_governance.md", metrics)
    return metrics


def _report(path, metrics):
    lines = [
        "# P1-8: prototype quality governance",
        "",
        f"- old test: {metrics['n_old_test']}, new train/test: "
        f"{metrics['n_new_train']} / {metrics['n_new_test']}",
        f"- rollback time: {metrics['rollback_time']*1000:.1f} ms",
        "",
    ]
    for poison, rows in metrics["results"].items():
        lines += [
            f"## {poison}",
            "",
            "| prototype | old acc | new acc | poison flagged | variance |",
            "|---|---:|---:|---:|---:|",
        ]
        for mode in ("mean", "trimmed", "medoid", "median"):
            r = rows[mode]
            lines.append(
                f"| {mode} | {r['old_accuracy']:.3f} | "
                f"{r['new_accuracy']:.3f} ± {r['new_accuracy_std']:.3f} | "
                f"{r['poison_flagged']:.3f} | {r['variance']:.4f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/category_pretrain")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    ap.add_argument("--projections", default="checkpoints/phase45_projection")
    ap.add_argument("--prototypes", default="checkpoints/phase45_prototypes")
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    m = run(args.data, args.experts, args.projections, args.prototypes, args.out, args.seeds)
    for poison, rows in m["results"].items():
        print(poison)
        for mode in ("mean", "trimmed", "medoid", "median"):
            r = rows[mode]
            print(f"  {mode}: old {r['old_accuracy']:.3f}, new {r['new_accuracy']:.3f}, flagged {r['poison_flagged']:.3f}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
