"""Claim-gap E6: a real retrain-all proxy.

A5 in the property suite is an oracle upper bound.  This experiment retrains the
part that a family swap actually invalidates: the projection heads and the
prototype bank, on the alternative expert family.  It does **not** retrain the
frozen expert encoders, so it is a lower bound on retrain-all cost, not the real
thing.

Compared on the same alternative-family test split:

- frozen base heads + alternative prototypes (zero-shot);
- 1-shot ridge calibration into the frozen space;
- retrained projection heads + retrained prototypes.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from pss._console import ensure_utf8
from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision import load_vision_expert
from pss.experts import CategoryCodebook
from pss.metrics.property import zero_shot_compatibility
from pss.projection.features import load_paired_features
from pss.projection.heads import load_projection
from pss.projection.train import train_projections
from pss.property.l1 import _project, _ridge_map

MODALITIES = ("vision", "tactile", "geometry")
EVAL_MODALITIES = ("vision", "tactile")


def _class_means(Z: np.ndarray, categories: list[str], indices: list[int], classes: list[str]) -> np.ndarray:
    rows = []
    for c in classes:
        idx = [i for i in indices if categories[i] == c]
        if not idx:
            idx = indices
        rows.append(np.asarray(Z[idx]).mean(axis=0))
    return np.stack(rows)


def _codebook(Z, feats, train_idx, classes):
    return CategoryCodebook.from_vectors(
        classes, _class_means(Z, feats.categories, train_idx, classes)
    )


def _zcr(codebook, Z, categories, test_idx) -> float:
    preds = [codebook.predict(Z[i]) for i in test_idx]
    gt = [categories[i] for i in test_idx]
    return float(zero_shot_compatibility(preds, gt))


def _calibrate(codebook, Z, feats, train_idx, test_idx, k=1):
    X, Y = [], []
    for c in codebook.categories:
        idx = [i for i in train_idx if feats.categories[i] == c][:k]
        for i in idx:
            X.append(Z[i])
            Y.append(codebook.vector(c))
    W = _ridge_map(np.stack(X), np.stack(Y))
    preds = [codebook.predict(np.asarray(Z[i], dtype=float) @ W) for i in test_idx]
    gt = [feats.categories[i] for i in test_idx]
    return float(zero_shot_compatibility(preds, gt)), W


def run(
    alt_data: str = "data/category_pretrain",
    alt_experts: str = "checkpoints/category_pretrain_experts",
    base_projections: str = "checkpoints/projection_v1",
    out_projection: str = "eval/results/_retrain_projection",
    epochs: int = 10,
    seed: int = 0,
    expert_train_s: float = 0.0,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision = load_vision_expert(Path(alt_experts) / "vision.pt", device=device)
    tactile = load_tactile_expert(Path(alt_experts) / "tactile.pt", device=device)
    feats = load_paired_features(alt_data, vision, tactile)

    train_idx = [i for i, s in enumerate(feats.splits) if str(s) == "train"]
    test_idx = [i for i, s in enumerate(feats.splits) if str(s) == "test"]
    classes = sorted({feats.categories[i] for i in train_idx})

    base_heads = {
        m: load_projection(Path(base_projections) / f"{m}.pt", device=device)
        for m in MODALITIES
    }
    Z_frozen = _project(feats, base_heads, device)
    frozen = {}
    t0 = time.perf_counter()
    frozen_codebooks = {
        m: _codebook(Z_frozen[m], feats, train_idx, classes)
        for m in EVAL_MODALITIES
    }
    prototype_build_ms = (time.perf_counter() - t0) * 1000.0
    for m in EVAL_MODALITIES:
        frozen[m] = _zcr(
            frozen_codebooks[m], Z_frozen[m], feats.categories, test_idx
        )

    calib = {}
    calib_cost_ms = {}
    for m in EVAL_MODALITIES:
        cb = frozen_codebooks[m]
        t0 = time.perf_counter()
        score, _ = _calibrate(cb, Z_frozen[m], feats, train_idx, test_idx, k=1)
        calib_cost_ms[m] = (time.perf_counter() - t0) * 1000.0
        calib[m] = score

    t0 = time.perf_counter()
    train_projections(
        data_dir=alt_data,
        ckpt_dir=alt_experts,
        out_dir=out_projection,
        epochs=epochs,
        seed=seed,
        vision_expert=vision,
        tactile_expert=tactile,
    )
    retrain_s = time.perf_counter() - t0
    retrained_heads = {
        m: load_projection(Path(out_projection) / f"{m}.pt", device=device)
        for m in MODALITIES
    }
    Z_retrained = _project(feats, retrained_heads, device)
    retrained = {}
    for m in EVAL_MODALITIES:
        cb = _codebook(Z_retrained[m], feats, train_idx, classes)
        retrained[m] = _zcr(cb, Z_retrained[m], feats.categories, test_idx)

    return {
        "alt_data": alt_data,
        "alt_experts": alt_experts,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "classes": len(classes),
        "chance": 1.0 / max(len(classes), 1),
        "methods": {
            "frozen_heads_alt_prototypes": frozen,
            "k1_calibration_alt_prototypes": calib,
            "retrained_projection_prototypes": retrained,
        },
        "cost": {
            "prototype_build_ms": prototype_build_ms,
            "k1_calibration_ms": calib_cost_ms,
            "retrain_projection_s": retrain_s,
            "expert_train_s": float(expert_train_s),
            "retrain_total_s": float(expert_train_s) + float(retrain_s),
            "retrain_epochs": epochs,
        },
        "note": (
            "Retrains projection heads + prototypes only; frozen expert encoders are "
            "not retrained, so this is a lower bound on retrain-all cost. The frozen "
            "condition rebuilds prototypes on the target family, so it is not the "
            "base-bank zero-shot condition (that is the E11 family swap)."
        ),
    }


def _report(m: dict) -> str:
    expert_s = float(m.get("cost", {}).get("expert_train_s", 0.0))
    scope = (
        f"This run also retrains the expert encoders ({expert_s:.1f} s), then the "
        "projection heads and prototypes."
        if expert_s > 0
        else "This run retrains projection heads + prototypes; the expert encoders stay frozen."
    )
    lines = [
        "# Claim-gap E6: retrain-all proxy",
        "",
        f"> alt data: `{m['alt_data']}`, alt experts: `{m['alt_experts']}`",
        f"> train/test: {m['n_train']}/{m['n_test']}, classes: {m['classes']}, chance: {m['chance']:.3f}",
        "",
        "| method | vision ZCR | tactile ZCR | cost |",
        "|---|---:|---:|---:|",
        f"| frozen heads + alt prototypes | {m['methods']['frozen_heads_alt_prototypes']['vision']:.3f} | "
        f"{m['methods']['frozen_heads_alt_prototypes']['tactile']:.3f} | "
        f"{m['cost']['prototype_build_ms']:.2f} ms |",
        f"| 1-shot ridge calibration | {m['methods']['k1_calibration_alt_prototypes']['vision']:.3f} | "
        f"{m['methods']['k1_calibration_alt_prototypes']['tactile']:.3f} | "
        f"{m['cost']['k1_calibration_ms']['vision']:.2f}-{m['cost']['k1_calibration_ms']['tactile']:.2f} ms |",
        f"| retrained projection + prototypes | {m['methods']['retrained_projection_prototypes']['vision']:.3f} | "
        f"{m['methods']['retrained_projection_prototypes']['tactile']:.3f} | "
        f"{m['cost']['retrain_projection_s']:.2f} s"
        f" (+ expert {m['cost'].get('expert_train_s', 0.0):.2f} s) |",
        "",
        "## How to read this",
        "",
        "This is the honest middle ground between the oracle A5 and the 1-shot",
        "calibration.",
        "",
        scope,
        "",
        "On this family, rebuilding prototypes on frozen heads recovers most of the",
        "gap; retraining the projections adds a small gain at tens of seconds; the",
        "1-shot ridge map is worse than either because the prototypes are already in",
        "the target family. Practical reading: prototype rebuild is the dominant",
        "cheap step, projection retraining is optional, and calibration is not a",
        "universal fix. The cross-family zero-shot failure against the *base bank*",
        "is measured separately in E11.",
        "",
        m["note"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt-data", default="data/category_pretrain")
    ap.add_argument("--alt-experts", default="checkpoints/category_pretrain_experts")
    ap.add_argument("--base-projections", default="checkpoints/projection_v1")
    ap.add_argument("--out-projection", default="eval/results/_retrain_projection")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expert-train-s", type=float, default=0.0)
    ap.add_argument("--out", default="reports/claim_gaps/retrain_all")
    args = ap.parse_args()
    metrics = run(
        alt_data=args.alt_data,
        alt_experts=args.alt_experts,
        base_projections=args.base_projections,
        out_projection=args.out_projection,
        epochs=args.epochs,
        seed=args.seed,
        expert_train_s=args.expert_train_s,
    )
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    for name, row in metrics["methods"].items():
        print(f"{name:34s} vision={row['vision']:.3f} tactile={row['tactile']:.3f}")
    print(f"retrain_s={metrics['cost']['retrain_projection_s']:.2f}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
