"""Modality conflict experiment: PSS fixed fusion vs transformer attention.

Both models are trained/calibrated on clean seen data. At test time one modality
is corrupted (shuffled, noised, or zeroed). We measure closed-set seen accuracy,
PSS conflict-abstention coverage, and conflict-aware re-fusion.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from pss._console import ensure_utf8
from pss.eval.scripts.run_p2_openset import (
    MODALITIES,
    _split_indices,
    _tensor_tokens,
    _train_transformer,
)
from pss.prototypes import PrototypeBank, recognize
from pss.prototypes.build import extract_projected_features
from pss.prototypes.conflict import conflict_aware_recognize


def _seen_test(feats, bank):
    return _split_indices(feats, bank, "test", True)


def _corrupt(Z, idx, modality: str | None, op: str, rng, scale: float = 1.0):
    Zc = {m: np.array(Z[m], copy=True) for m in Z}
    if modality is None or op == "clean":
        return Zc
    if op == "shuffle":
        perm = rng.permutation(idx)
        Zc[modality][idx] = Z[modality][perm]
    elif op == "noise":
        std = float(np.std(Z[modality]))
        Zc[modality][idx] += rng.normal(
            scale=scale * max(std, 1e-6), size=Zc[modality][idx].shape
        ).astype(np.float32)
    elif op == "zero":
        Zc[modality][idx] = 0.0
    else:
        raise ValueError(f"unknown corruption op: {op}")
    return Zc


def _query(Z, i):
    return {m: Z[m][i] for m in MODALITIES}


def _pss_predict(Z, bank, weights, idx, feats, mode: str):
    correct = 0
    accepted = 0
    conflicts = 0
    for i in idx:
        query = _query(Z, i)
        result = recognize(
            query,
            bank,
            weights=weights,
            use_tree=False,
            threshold=None,
            conflict_weight=0.75,
        )
        conflicts += int(result.conflict)
        gt = feats.categories[i]
        if mode == "abstain" and result.conflict:
            continue
        accepted += 1
        if mode == "drop" and result.conflict:
            per_top = {}
            for m in MODALITIES:
                sims = bank.modality_scores(query[m], m)
                per_top[m] = bank.categories[int(np.argmax(sims))]
            fused_top = result.ranked[0][0] if result.ranked else None
            keep = [m for m in MODALITIES if per_top[m] == fused_top]
            if keep:
                denom = sum(weights[m] for m in keep)
                scores = []
                for c_idx, c in enumerate(bank.categories):
                    s = sum(
                        weights[m] * float(bank.protos[m][c_idx] @ _unit(query[m]))
                        for m in keep
                    ) / denom
                    scores.append((c, s))
                pred = max(scores, key=lambda x: x[1])[0]
            else:
                pred = fused_top
        else:
            pred = result.ranked[0][0] if result.ranked else None
        correct += int(pred == gt)
    return {
        "accuracy": correct / max(1, len(idx)),
        "coverage": accepted / max(1, len(idx)),
        "accepted_accuracy": correct / max(1, accepted),
        "conflict_rate": conflicts / max(1, len(idx)),
    }


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _pss_reliable(Z, bank, weights, idx, feats):
    correct = 0
    accepted = 0
    conflicts = 0
    for i in idx:
        decision = conflict_aware_recognize(_query(Z, i), bank, weights)
        conflicts += int(decision.conflict)
        if decision.category is None:
            continue
        accepted += 1
        correct += int(decision.category == feats.categories[i])
    return {
        "accuracy": correct / max(1, len(idx)),
        "coverage": accepted / max(1, len(idx)),
        "accepted_accuracy": correct / max(1, accepted),
        "conflict_rate": conflicts / max(1, len(idx)),
    }


def _transformer_accuracy(model, Z, idx, feats, bank, device):
    x = _tensor_tokens(Z, idx, device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).cpu().numpy()
    class_to_idx = {c: i for i, c in enumerate(sorted(bank.categories))}
    correct = sum(
        int(pred[k] == class_to_idx[feats.categories[i]]) for k, i in enumerate(idx)
    )
    return correct / max(1, len(idx))


def run_conflict(
    data_dir,
    expert_dir,
    projection_dir,
    prototype_dir,
    out_dir,
    seeds: int = 5,
    epochs: int = 200,
    device: str = "cpu",
) -> dict:
    feats, Z = extract_projected_features(data_dir, expert_dir, projection_dir, device)
    bank = PrototypeBank.load(Path(prototype_dir) / "prototypes.npz")
    weights = {"vision": 1.0, "tactile": 0.5, "geometry": 1.0}
    metrics_path = Path(prototype_dir) / "metrics.json"
    if metrics_path.exists():
        weights = json.loads(metrics_path.read_text(encoding="utf-8")).get("weights", weights)
    test_idx = _seen_test(feats, bank)

    # Fix the placeholder in _pss_fixed by using a local version.
    def pss_fixed(Zc):
        correct = 0
        conflicts = 0
        for i in test_idx:
            r = recognize(
                _query(Zc, i),
                bank,
                weights=weights,
                use_tree=False,
                threshold=None,
                conflict_weight=0.75,
            )
            conflicts += int(r.conflict)
            if r.ranked and r.ranked[0][0] == feats.categories[i]:
                correct += 1
        return {
            "accuracy": correct / max(1, len(test_idx)),
            "coverage": 1.0,
            "accepted_accuracy": correct / max(1, len(test_idx)),
            "conflict_rate": conflicts / max(1, len(test_idx)),
        }

    conditions = [
        ("clean", None, "clean", 1.0),
        ("tactile_shuffle", "tactile", "shuffle", 1.0),
        ("tactile_noise_0.5", "tactile", "noise", 0.5),
        ("tactile_noise_1.0", "tactile", "noise", 1.0),
        ("tactile_zero", "tactile", "zero", 1.0),
        ("vision_shuffle", "vision", "shuffle", 1.0),
        ("geometry_shuffle", "geometry", "shuffle", 1.0),
        ("vision_zero", "vision", "zero", 1.0),
        ("geometry_zero", "geometry", "zero", 1.0),
    ]

    models = []
    for seed in range(seeds):
        model, _ = _train_transformer(Z, feats, bank, seed, epochs, device)
        models.append(model)

    results = {}
    rng = np.random.default_rng(0)
    for name, modality, op, scale in conditions:
        Zc = _corrupt(Z, test_idx, modality, op, rng, scale)
        fixed = pss_fixed(Zc)
        abstain = _pss_predict(Zc, bank, weights, test_idx, feats, "abstain")
        drop = _pss_predict(Zc, bank, weights, test_idx, feats, "drop")
        reliable = _pss_reliable(Zc, bank, weights, test_idx, feats)
        tr = [ _transformer_accuracy(m, Zc, test_idx, feats, bank, device) for m in models ]
        results[name] = {
            "pss_fixed": fixed,
            "pss_abstain": abstain,
            "pss_drop": drop,
            "pss_reliable": reliable,
            "transformer": {
                "accuracy_mean": float(np.mean(tr)),
                "accuracy_std": float(np.std(tr)),
            },
        }

    metrics = {
        "data_dir": str(data_dir),
        "n_test": len(test_idx),
        "weights": weights,
        "conditions": results,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "modality_conflict.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "modality_conflict.md", metrics)
    return metrics


def _write_report(path: Path, metrics: dict) -> None:
    lines = [
        "# Modality conflict: PSS fixed fusion vs transformer attention",
        "",
        f"- data: `{metrics['data_dir']}`",
        f"- seen test samples: {metrics['n_test']}",
        f"- PSS weights: {metrics['weights']}",
        "",
        "| condition | PSS fixed | PSS abstain (cov/acc) | PSS reliable (cov/acc) | PSS drop | Transformer |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in metrics["conditions"].items():
        a = r["pss_abstain"]
        rel = r["pss_reliable"]
        lines.append(
            f"| {name} | {r['pss_fixed']['accuracy']:.3f} | "
            f"{a['coverage']:.2f}/{a['accepted_accuracy']:.3f} | "
            f"{rel['coverage']:.2f}/{rel['accepted_accuracy']:.3f} | "
            f"{r['pss_drop']['accuracy']:.3f} | "
            f"{r['transformer']['accuracy_mean']:.3f} ± "
            f"{r['transformer']['accuracy_std']:.3f} |"
        )
    lines += [
        "",
        "Both models see clean training data; only the test modality is corrupted.",
        "Conflict-aware PSS uses per-modality top-1 disagreement; abstain drops conflicted samples, drop re-fuses without disagreeing modalities.",
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
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    metrics = run_conflict(
        args.data,
        args.experts,
        args.projections,
        args.prototypes,
        args.out,
        seeds=args.seeds,
        epochs=args.epochs,
        device=args.device,
    )
    for name, r in metrics["conditions"].items():
        print(
            f"  {name:20s} PSS fixed={r['pss_fixed']['accuracy']:.3f} "
            f"abstain={r['pss_abstain']['coverage']:.2f}/"
            f"{r['pss_abstain']['accepted_accuracy']:.3f} "
            f"drop={r['pss_drop']['accuracy']:.3f} "
            f"TR={r['transformer']['accuracy_mean']:.3f}"
        )
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
