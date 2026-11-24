"""Train a real-image-pretrained backbone (SqueezeNet/ResNet) for the vision expert.

Example:
    python -m pss.encoders.scripts.train_pretrained_expert \
        --data data/expert_v1 --out checkpoints/expert_pretrained_v1 \
        --backbone squeezenet1_1 --pretrained true --epochs 30 --size 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pss._console import ensure_utf8
from pss.encoders.data import build_split, load_category_examples
from pss.encoders.train import evaluate, train_encoder, vision_augment
from pss.encoders.vision.pretrained import (
    ResNet18Encoder,
    SqueezeNetEncoder,
    save_pretrained_expert,
)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_v1")
    ap.add_argument("--out", default="checkpoints/expert_pretrained_v1")
    ap.add_argument("--backbone", choices=["squeezenet1_1", "resnet18"], default="squeezenet1_1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretrained", choices=["true", "false"], default="true")
    args = ap.parse_args()

    examples = load_category_examples(args.data, size=args.size)
    classes = sorted({e.category for e in examples if e.split == "train"})
    Xtr, _, ytr, _ = build_split(examples, "train", classes)
    Xva, _, yva, _ = build_split(examples, "val", classes)
    Xte, _, yte, _ = build_split(examples, "test", classes)

    pretrained = args.pretrained == "true"
    error = None
    factory = SqueezeNetEncoder if args.backbone == "squeezenet1_1" else ResNet18Encoder
    try:
        encoder = factory(out_dim=256, pretrained=pretrained)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        encoder = factory(out_dim=256, pretrained=False)
        pretrained = False

    encoder, head, _info = train_encoder(
        encoder,
        Xtr,
        ytr,
        Xva,
        yva,
        len(classes),
        epochs=args.epochs,
        lr=args.lr,
        device="cpu",
        seed=args.seed,
        augment=vision_augment,
    )
    metrics = {
        "data_dir": args.data,
        "device": "cpu",
        "backbone": args.backbone,
        "pretrained": pretrained,
        "pretrained_error": error,
        "epochs": int(args.epochs),
        "image_size": int(args.size),
        "classes": classes,
        "counts": {"train": len(ytr), "val": len(yva), "test": len(yte)},
        "train_acc": float(evaluate(encoder, head, Xtr, ytr, "cpu")),
        "val_acc": float(evaluate(encoder, head, Xva, yva, "cpu")),
        "test_acc": float(evaluate(encoder, head, Xte, yte, "cpu")),
        "out_dim": int(encoder.out_dim),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_pretrained_expert(
        out / "vision_pretrained.pt", encoder, classes, args.size, args.backbone
    )
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Pretrained vision expert:")
    print(f"  backbone={args.backbone} pretrained={pretrained} error={error}")
    print(f"  train={metrics['train_acc']:.3f} val={metrics['val_acc']:.3f} "
          f"test={metrics['test_acc']:.3f}")
    print(f"  output: {out}")


if __name__ == "__main__":
    main()
