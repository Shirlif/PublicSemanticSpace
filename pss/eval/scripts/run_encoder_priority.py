"""First-priority encoder swap: frozen backbones + calibrated projections.

Implements the "换编码器，不动 PSS 接口" item from
``PSS-认知层架构定位.md``:

- frozen CLIP / DINOv2 vision encoders (plus DINOv2 patch-token VLAD);
- a stronger conv tactile encoder;
- projection heads with PCA-whitening and a residual skip (alignment /
  calibration);
- the same three downstream tests: open-set recognition, set-level re-ID, and
  the P5b perception benchmark.

The PSS interfaces never change. A variant only changes which frozen encoder
feeds the existing 128-d projection head and which projection checkpoint the
evaluators load.

Example:
    python -m pss.eval.scripts.run_encoder_priority --stage all
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from pss._console import ensure_utf8
from pss.encoders.data import load_category_examples
from pss.encoders.registry import build_tactile_encoder, build_vision_encoder
from pss.encoders.train import train_all
from pss.encoders.vision.clip import clip_available
from pss.encoders.vision.dinov2 import load_dinov2_encoder
from pss.encoders.vision.local import fit_vlad_codebook, save_vlad
from pss.eval.baselines.entity_transformer import build_pairs
from pss.eval.relational_perception_bench import run as run_p5b
from pss.memory.graph_matcher import evaluate_graph, save_graph, train_graph
from pss.memory.learned_metric import TACTILE_TOKEN_DIM
from pss.projection.calibration import (
    calibrate_fusion,
    calibrated_confidence,
    calibrated_scores,
    reliability_weights,
)
from pss.projection.train import train_projections
from pss.prototypes.bank import build_bank
from pss.prototypes.build import extract_projected_features, run_phase5

DATA = "data/category_pretrain"
TRACK = "data/tracking_bench_v2"
BASE_EXPERTS = "checkpoints/category_pretrain_experts"
BASE_PROJECTION = "checkpoints/phase45_projection"
OUT_ROOT = "checkpoints/encoder_priority"
RESULTS = "eval/results/encoder_priority"


@dataclass
class Variant:
    name: str
    vision: str = "smallcnn"
    tactile: str = "mlp"
    whiten_dim: int | None = None
    whiten_modalities: tuple[str, ...] | None = None
    residual: bool = False
    vlad_k: int = 16
    notes: str = ""

    @property
    def dir(self) -> Path:
        return Path(OUT_ROOT) / self.name

    @property
    def projection_dir(self) -> Path:
        return self.dir / "projection"

    @property
    def tactical_dir(self) -> Path:
        # Directory the registry uses for the tactile checkpoint.
        return self.dir if self.tactile == "conv" else Path(BASE_EXPERTS)


VARIANTS = [
    Variant("base", "smallcnn", "mlp", notes="current baseline"),
    Variant(
        "calibrated",
        "smallcnn",
        "mlp",
        whiten_dim=128,
        whiten_modalities=("vision", "geometry"),
        residual=True,
        notes="baseline encoders + whitening (vision/geometry) + residual projections",
    ),
    Variant(
        "residual_only",
        "smallcnn",
        "mlp",
        residual=True,
        notes="ablation: residual skip only, no whitening",
    ),
    Variant(
        "whiten_only",
        "smallcnn",
        "mlp",
        whiten_dim=128,
        whiten_modalities=("vision", "geometry"),
        residual=False,
        notes="ablation: vision/geometry whitening only, no residual",
    ),
    Variant(
        "dinov2_plain",
        "dinov2",
        "mlp",
        notes="ablation: frozen DINOv2 with the uncalibrated plain MLP head",
    ),
    Variant(
        "dinov2",
        "dinov2",
        "mlp",
        whiten_dim=96,
        whiten_modalities=("vision", "geometry"),
        residual=True,
        notes="frozen DINOv2 CLS + calibrated projection",
    ),
    Variant(
        "dinov2_vlad",
        "dinov2_vlad",
        "mlp",
        whiten_dim=128,
        whiten_modalities=("vision", "geometry"),
        residual=True,
        notes="frozen DINOv2 patch tokens + VLAD + calibrated projection",
    ),
    Variant(
        "conv_tactile",
        "smallcnn",
        "conv",
        whiten_dim=128,
        whiten_modalities=("vision", "geometry"),
        residual=True,
        notes="conv tactile encoder + calibrated projection",
    ),
    Variant(
        "clip",
        "clip",
        "mlp",
        whiten_dim=128,
        whiten_modalities=("vision", "geometry"),
        residual=True,
        notes="frozen CLIP ViT-L/14; skipped when weights are offline",
    ),
]
BY_NAME = {v.name: v for v in VARIANTS}


def _mean_std(values):
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj


# ── encoder preparation ──────────────────────────────────────────────────────


def prepare_tactile(variant: Variant, device: str) -> None:
    if variant.tactile != "conv":
        return
    ckpt = variant.dir / "tactile_conv.pt"
    if ckpt.exists():
        return
    variant.dir.mkdir(parents=True, exist_ok=True)
    print(f"[tactile] training conv encoder -> {variant.dir}")
    train_all(
        DATA,
        variant.dir,
        epochs=60,
        image_size=64,
        seed=0,
        tactile_arch="conv",
    )


def prepare_vlad(variant: Variant, device: str) -> Path:
    path = variant.dir / "vlad_codebook.pt"
    if path.exists():
        return path
    variant.dir.mkdir(parents=True, exist_ok=True)
    examples = load_category_examples(DATA, size=224)
    train_images = np.stack([e.image for e in examples if e.split == "train"])
    backbone = load_dinov2_encoder(device=device)
    print(f"[vlad] fitting K={variant.vlad_k} on {len(train_images)} train crops")
    vlad = fit_vlad_codebook(
        backbone,
        torch.as_tensor(train_images),
        k=variant.vlad_k,
        seed=0,
        device=device,
    )
    save_vlad(path, vlad, meta={"source": "dinov2-small", "split": "train"})
    return path


def build_encoders(variant: Variant, device: str):
    vision = build_vision_encoder(
        variant.vision,
        expert_dir=BASE_EXPERTS,
        device=device,
        codebook=prepare_vlad(variant, device) if variant.vision == "dinov2_vlad" else None,
    )
    if variant.tactile == "conv":
        tactile = build_tactile_encoder("conv", expert_dir=variant.dir, device=device)
    else:
        tactile = build_tactile_encoder("mlp", expert_dir=BASE_EXPERTS, device=device)
    return vision, tactile


def prepare_projection(variant: Variant, device: str) -> dict | None:
    if variant.name == "base":
        return None
    if variant.projection_dir.joinpath("vision.pt").exists():
        return None
    vision, tactile = build_encoders(variant, device)
    print(f"[projection] training {variant.name} -> {variant.projection_dir}")
    return train_projections(
        DATA,
        BASE_EXPERTS,
        variant.projection_dir,
        epochs=120,
        batch_size=128,
        seed=0,
        vision_expert=vision,
        tactile_expert=tactile,
        whiten_dim=variant.whiten_dim,
        whiten_modalities=variant.whiten_modalities,
        residual=variant.residual,
    )


def ensure_prepared(variant: Variant, device: str) -> None:
    """Make the encoders/checkpoints a downstream stage needs, once."""
    prepare_tactile(variant, device)
    if variant.vision == "dinov2_vlad":
        prepare_vlad(variant, device)
    prepare_projection(variant, device)


# ── stage: open-set / closed-set ─────────────────────────────────────────────


def stage_recognition(variant: Variant, device: str) -> dict:
    vision, tactile = build_encoders(variant, device)
    projection = variant.projection_dir if variant.name != "base" else Path(BASE_PROJECTION)
    metrics = run_phase5(
        DATA,
        BASE_EXPERTS,
        projection,
        variant.dir / "prototypes",
        seed=0,
        vision_expert=vision,
        tactile_expert=tactile,
    )

    # Calibrated reliability + temperature, and a calibrated open-set score.
    feats, Z = extract_projected_features(
        DATA, BASE_EXPERTS, projection, vision_expert=vision, tactile_expert=tactile
    )
    splits = np.array(feats.splits)
    classes = sorted({feats.categories[i] for i in np.where(splits == "train")[0]})
    train_idx = np.array(
        [i for i in np.where(splits == "train")[0] if feats.categories[i] in classes]
    )
    val_idx = np.array(
        [i for i in np.where(splits == "val")[0] if feats.categories[i] in classes]
    )
    seen_test = np.array(
        [i for i in np.where(splits == "test")[0] if feats.categories[i] in classes]
    )
    unseen = np.where(splits == "test_category")[0]
    bank = build_bank(feats, Z, train_idx, classes)
    labels = np.array([classes.index(feats.categories[i]) for i in val_idx])
    calib = calibrate_fusion({m: Z[m][val_idx] for m in Z}, bank.protos, labels)
    weights = reliability_weights(calib)

    seen_scores = np.array(
        [
            calibrated_scores({m: Z[m][i] for m in Z}, bank.protos, calib).max()
            for i in seen_test
        ]
    )
    unseen_scores = np.array(
        [
            calibrated_scores({m: Z[m][i] for m in Z}, bank.protos, calib).max()
            for i in unseen
        ]
    )
    seen_conf = np.array(
        [
            calibrated_confidence({m: Z[m][i] for m in Z}, bank.protos, calib, classes)
            for i in seen_test
        ]
    )
    unseen_conf = np.array(
        [
            calibrated_confidence({m: Z[m][i] for m in Z}, bank.protos, calib, classes)
            for i in unseen
        ]
    )
    calibrated_auroc = float("nan")
    if len(seen_scores) and len(unseen_scores):
        calibrated_auroc = float(
            roc_auc_score(
                np.r_[np.ones(len(seen_scores)), np.zeros(len(unseen_scores))],
                np.r_[seen_scores, unseen_scores],
            )
        )
    return {
        "phase5": metrics,
        "calibration": {m: c.to_dict() for m, c in calib.items()},
        "calibrated_weights": weights,
        "calibrated_open_set_auroc": calibrated_auroc,
        "calibrated_confidence": {
            "seen_mean": float(np.mean(seen_conf)) if len(seen_conf) else None,
            "unseen_mean": float(np.mean(unseen_conf)) if len(unseen_conf) else None,
        },
    }


# ── stage: re-ID ─────────────────────────────────────────────────────────────


def _agg(runs):
    subsets = ("mixed", "samecat", "near_identical")
    return {
        "overall": _mean_std([r["reid_accuracy"] for r in runs]),
        "per_subset": {
            s: _mean_std([r["per_subset"][s]["reid_accuracy"] for r in runs]) for s in subsets
        },
    }


def stage_reid(variant: Variant, device: str, seeds: int, epochs: int) -> dict:
    vision, tactile = build_encoders(variant, device)
    projection = variant.projection_dir if variant.name != "base" else Path(BASE_PROJECTION)
    pairs = {}
    for split in ("train", "val", "test"):
        pairs[split] = build_pairs(
            TRACK,
            BASE_EXPERTS,
            str(projection),
            split,
            device=device,
            include_tactile=True,
            vision_expert=vision,
            tactile_expert=tactile,
        )
    runs, best_model, best_val = [], None, -1.0
    for seed in range(seeds):
        model, info = train_graph(
            pairs["train"],
            pairs["val"],
            epochs=epochs,
            seed=seed,
            device=device,
            token_dim=TACTILE_TOKEN_DIM,
        )
        runs.append(evaluate_graph(model, pairs["test"], device))
        if info["best_val_reid"] > best_val:
            best_val, best_model = info["best_val_reid"], model
    save_graph(best_model, variant.dir / "graph_matcher.pt")
    return {
        "n_train": len(pairs["train"]),
        "n_val": len(pairs["val"]),
        "n_test": len(pairs["test"]),
        "best_val_reid": best_val,
        "aggregate": _agg(runs),
        "runs": runs,
    }


# ── stage: P5b ───────────────────────────────────────────────────────────────


def stage_p5b(variant: Variant, device: str, seeds: int, epochs: int) -> dict:
    vision, tactile = build_encoders(variant, device)
    projection = variant.projection_dir if variant.name != "base" else Path(BASE_PROJECTION)
    return run_p5b(
        out_dir=str(variant.dir / "p5b"),
        seeds=seeds,
        n_train_per_pair=80,
        n_test_per_pair=10,
        epochs=epochs,
        noise_sigma=0.01,
        device=device,
        expert_dir=BASE_EXPERTS,
        projection_dir=str(projection),
        vision_expert=vision,
        tactile_expert=tactile,
    )


# ── orchestration ────────────────────────────────────────────────────────────


def available_variants(allow_download: bool) -> list[Variant]:
    out = []
    for variant in VARIANTS:
        if variant.vision == "clip" and not (
            allow_download or clip_available()
        ):
            continue
        out.append(variant)
    return out


def run(
    stage: str = "all",
    only: list[str] | None = None,
    device: str = "cuda",
    seeds: int = 3,
    epochs: int = 200,
    p5b_seeds: int = 2,
    p5b_epochs: int = 200,
    allow_download: bool = False,
) -> dict:
    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    requested = [v for v in VARIANTS if not only or v.name in only]
    usable = {v.name for v in available_variants(allow_download)}
    variants = [v for v in requested if v.name in usable]
    skipped = [
        {
            "name": v.name,
            "reason": "CLIP vision weights not in the local Hugging Face cache",
        }
        for v in requested
        if v.name not in usable
    ]
    out = Path(RESULTS)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "encoder_priority.json"
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"generated": None, "device": device, "variants": {}}
    )
    payload["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    payload["device"] = device
    if skipped:
        payload["skipped"] = skipped

    for variant in variants:
        entry = payload["variants"].setdefault(variant.name, {"config": variant.__dict__})
        entry["config"] = variant.__dict__
        entry.pop("error", None)
        print(f"\n=== {variant.name} ({variant.vision} vision / {variant.tactile} tactile) ===")
        try:
            ensure_prepared(variant, device)
            if stage in ("all", "recognition"):
                entry["recognition"] = stage_recognition(variant, device)
                print(
                    f"  closed-set test {entry['recognition']['phase5']['fusion']['test_acc']:.3f}; "
                    f"open-set AUROC {entry['recognition']['phase5']['open_set']['auroc']:.3f} "
                    f"(calibrated {entry['recognition']['calibrated_open_set_auroc']:.3f})"
                )
            if stage in ("all", "reid"):
                entry["reid"] = stage_reid(variant, device, seeds, epochs)
                agg = entry["reid"]["aggregate"]["overall"]
                print(
                    f"  re-ID {agg['mean']:.3f} ± {agg['std']:.3f}; "
                    f"near {entry['reid']['aggregate']['per_subset']['near_identical']['mean']:.3f}"
                )
            if stage in ("all", "p5b"):
                entry["p5b"] = stage_p5b(variant, device, p5b_seeds, p5b_epochs)
                print(
                    f"  P5b ID/OOD {entry['p5b']['pss']['id_success'][0]:.3f}/"
                    f"{entry['p5b']['pss']['ood_success'][0]:.3f}; "
                    f"TR {entry['p5b']['transformer']['id_success'][0]:.3f}/"
                    f"{entry['p5b']['transformer']['ood_success'][0]:.3f}"
                )
        except Exception as exc:  # keep other variants running
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  [error] {entry['error']}")
        path.write_text(
            json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (out / "encoder_priority.md").write_text(render(payload), encoding="utf-8")
    print(f"\noutput: {path}")
    return payload


# ── report ───────────────────────────────────────────────────────────────────


def _f(value, digits=3):
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def render(payload: dict) -> str:
    lines = [
        "# Encoder priority: frozen backbones + calibrated projections",
        "",
        f"> generated: {payload.get('generated')}",
        f"> device: `{payload.get('device')}`",
        "",
        "| variant | vision | tactile | projection | closed-set test | open-set AUROC | calibrated AUROC | re-ID overall | re-ID near | P5b ID / OOD |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in payload["variants"].items():
        cfg = entry.get("config", {})
        rec = entry.get("recognition", {})
        p5 = rec.get("phase5", {})
        rei = entry.get("reid", {}).get("aggregate", {})
        p5b = entry.get("p5b", {}).get("pss", {})
        if cfg.get("whiten_dim") is None:
            proj = "residual" if cfg.get("residual") else "none"
        else:
            mods = "+".join(cfg.get("whiten_modalities") or ("all",))
            proj = (
                f"whiten{cfg.get('whiten_dim')}({mods})"
                f"{'+res' if cfg.get('residual') else ''}"
            )
        if entry.get("error"):
            lines.append(f"| `{name}` | {cfg.get('vision')} | {cfg.get('tactile')} | {proj} | error | | | | | |")
            continue
        lines.append(
            f"| `{name}` | {cfg.get('vision')} | {cfg.get('tactile')} | {proj} | "
            f"{_f(p5.get('fusion', {}).get('test_acc'))} | "
            f"{_f(p5.get('open_set', {}).get('auroc'))} | "
            f"{_f(rec.get('calibrated_open_set_auroc'))} | "
            f"{_f(rei.get('overall', {}).get('mean'))} ± {_f(rei.get('overall', {}).get('std'))} | "
            f"{_f(rei.get('per_subset', {}).get('near_identical', {}).get('mean'))} | "
            f"{_f(p5b.get('id_success', [None])[0])} / {_f(p5b.get('ood_success', [None])[0])} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `base` is the existing SmallCNN + tactile MLP pipeline with the existing projections.",
        "- `calibrated` keeps the same frozen encoders and only changes the projection head",
        "  (PCA-whitening on vision/geometry + a residual skip). This isolates the alignment effect.",
        "- `residual_only` and `whiten_only` are the head ablations.",
        "- `dinov2`/`dinov2_vlad`/`clip` replace the vision encoder; `conv_tactile` replaces the tactile encoder.",
        "- The PSS interface (Observation, 128-d public space, entity_token, cost matrices) is unchanged.",
        "- Calibrated AUROC uses per-modality temperature + reliability weights from the validation split.",
        "- CLIP is skipped when its weights are not in the local Hugging Face cache.",
        "",
    ]
    if payload.get("skipped"):
        lines += ["## Skipped", ""]
        for item in payload["skipped"]:
            lines.append(f"- `{item['name']}`: {item['reason']}")
        lines.append("")
    for name, entry in payload["variants"].items():
        if entry.get("calibration"):
            lines.append(f"### `{name}` calibration")
            lines.append("")
            lines.append("| modality | temperature | reliability | val acc |")
            lines.append("|---|---:|---:|---:|")
            for m, c in entry["calibration"].items():
                lines.append(
                    f"| {m} | {_f(c['temperature'])} | {_f(c['reliability'])} | {_f(c['val_accuracy'])} |"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        default="all",
        choices=("all", "projection", "recognition", "reid", "p5b"),
    )
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--p5b-seeds", type=int, default=2)
    ap.add_argument("--p5b-epochs", type=int, default=200)
    ap.add_argument("--allow-download", action="store_true")
    args = ap.parse_args()
    run(
        stage=args.stage,
        only=args.only,
        device=args.device,
        seeds=args.seeds,
        epochs=args.epochs,
        p5b_seeds=args.p5b_seeds,
        p5b_epochs=args.p5b_epochs,
        allow_download=args.allow_download,
    )


if __name__ == "__main__":
    main()
