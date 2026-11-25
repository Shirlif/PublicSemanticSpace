"""Claim-gap E14: per-modality space vs summed/random projection.

The cognitive-runtime claim uses the public semantic space as a per-modality
interface, not as one summed vector.  This experiment compares:

- full per-modality fusion (the PSS interface);
- cross-modal sum (one vector for all modalities);
- random projection (a control with no learned alignment).

It is the ablation that justifies "per-modality projection + prototypes" over a
single shared vector.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from pss._console import ensure_utf8
from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision import load_vision_expert
from pss.eval.baselines import recognition_ablations
from pss.projection.features import load_paired_features
from pss.projection.heads import load_projection
from pss.property.l1 import _project
from pss.prototypes.bank import PrototypeBank

MODALITIES = ("vision", "tactile", "geometry")


def run(
    data_dir: str = "data/expert_v1",
    expert_dir: str = "checkpoints/expert_v1",
    projection_dir: str = "checkpoints/projection_v1",
    prototype_dir: str = "checkpoints/prototypes_v1",
    split: str = "test",
    seed: int = 0,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision = load_vision_expert(Path(expert_dir) / "vision.pt", device=device)
    tactile = load_tactile_expert(Path(expert_dir) / "tactile.pt", device=device)
    feats = load_paired_features(data_dir, vision, tactile)
    heads = {m: load_projection(Path(projection_dir) / f"{m}.pt", device=device) for m in MODALITIES}
    Z = _project(feats, heads, device)
    bank = PrototypeBank.load(Path(prototype_dir) / "prototypes.npz")

    train_idx = [i for i, s in enumerate(feats.splits) if str(s) == "train"]
    test_idx = [i for i, s in enumerate(feats.splits) if str(s) == split]
    classes = list(bank.categories)
    ablations = recognition_ablations(
        feats, Z, bank, test_idx, train_idx, classes, seed=seed
    )
    return {
        "data_dir": data_dir,
        "expert_dir": expert_dir,
        "projection_dir": projection_dir,
        "prototype_dir": prototype_dir,
        "split": split,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "classes": classes,
        "ablations": ablations,
        "chance": 1.0 / max(len(classes), 1),
    }


def _report(m: dict) -> str:
    a = m["ablations"]
    lines = [
        "# Claim-gap E14: per-modality space ablation",
        "",
        f"> data: `{m['data_dir']}`, split: {m['split']}, classes: {len(m['classes'])}, chance: {m['chance']:.3f}",
        "",
        "| variant | accuracy | cross-modal R@1 |",
        "|---|---:|---:|",
        f"| full per-modality | {a['full_accuracy']:.3f} | {a['trained_vision_tactile_r1']:.3f} |",
        f"| cross-modal sum | {a['cross_modal_sum_accuracy']:.3f} | - |",
        f"| random projection | {a['random_projection_accuracy']:.3f} | {a['random_projection_vision_tactile_r1']:.3f} |",
        "",
        "## How to read this",
        "",
        "Closed-set accuracy is invariant across the three variants because the",
        "prototype bank is re-fit in each space; that is not where the learned",
        "alignment shows up. It shows up in cross-modal retrieval: trained",
        "vision-to-tactile R@1 is far above the random-projection control. So the",
        "honest claim is 'per-modality projection provides cross-modal alignment',",
        "not 'per-modality projection improves closed-set accuracy'.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_v1")
    ap.add_argument("--experts", default="checkpoints/expert_v1")
    ap.add_argument("--projections", default="checkpoints/projection_v1")
    ap.add_argument("--prototypes", default="checkpoints/prototypes_v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/claim_gaps/space_ablation")
    args = ap.parse_args()
    metrics = run(
        data_dir=args.data,
        expert_dir=args.experts,
        projection_dir=args.projections,
        prototype_dir=args.prototypes,
        split=args.split,
        seed=args.seed,
    )
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    for name, value in metrics["ablations"].items():
        print(f"{name:42s} {value:.3f}" if isinstance(value, float) else f"{name}: {value}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
