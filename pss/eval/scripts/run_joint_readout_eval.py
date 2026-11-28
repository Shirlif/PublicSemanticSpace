"""Evaluate the joint discriminative readout against the production prototype.

Runs on ``data/expert_v1`` (the production public-space data): closed-set
accuracy, open-set AUROC at a fixed seen-accept operating point, and the cost
of a hot class update.  A second run on the 384-class projected task is already
covered by ``run_projected_fusion_audit``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from pss._console import ensure_utf8
from pss.inference import MODALITIES, PublicSemanticSpace
from pss.projection.features import load_paired_features
from pss.eval.scripts.run_attribution_grid import fused_scores


def _fmt(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _auroc(seen: np.ndarray, unseen: np.ndarray) -> float:
    if not len(seen) or not len(unseen):
        return float("nan")
    y = np.r_[np.ones(len(seen)), np.zeros(len(unseen))]
    return float(roc_auc_score(y, np.r_[seen, unseen]))


def _operating_point(seen, unseen, threshold):
    return {
        "threshold": float(threshold),
        "seen_false_reject": float((seen < threshold).mean()),
        "unseen_reject": float((unseen < threshold).mean()),
    }


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/expert_v1")
    parser.add_argument(
        "--bank", default="eval/results/joint_readout_eval/bank.npz"
    )
    parser.add_argument("--out", default="eval/results/joint_readout_eval")
    parser.add_argument("--device", default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.rebuild:
        Path(args.bank).unlink(missing_ok=True)
        (Path(args.bank).parent / "joint_readout.npz").unlink(missing_ok=True)
    space = PublicSemanticSpace(bank_path=args.bank, device=args.device)
    if space.bank is None or space.joint is None:
        space.build_bank(args.data)
    assert space.bank is not None and space.joint is not None

    feats = load_paired_features(args.data, space.vision, space.tactile)
    with torch.no_grad():
        Z = {
            m: space.heads[m](
                torch.as_tensor(
                    getattr(feats, m), dtype=torch.float32, device=space.device
                )
            )
            .cpu()
            .numpy()
            for m in MODALITIES
        }

    splits = np.asarray(feats.splits)
    cats = np.asarray(feats.categories)
    classes = sorted({str(cats[i]) for i in np.where(splits == "train")[0]})
    cls_to_i = {c: i for i, c in enumerate(classes)}
    y_all = np.array([cls_to_i.get(str(c), -1) for c in cats], dtype=np.int64)
    seen = y_all >= 0
    train_idx = np.where((splits == "train") & seen)[0]
    val_idx = np.where((splits == "val") & seen)[0]
    test_idx = np.where((splits == "test") & seen)[0]
    held_idx = np.where(splits == "test_category")[0]

    # Prototype baseline on the same projected features.
    proto_test = fused_scores(Z, space.bank, test_idx)
    proto_val = fused_scores(Z, space.bank, val_idx)
    proto_held = fused_scores(Z, space.bank, held_idx)
    proto_acc = float(np.mean(proto_test.argmax(1) == y_all[test_idx]))
    proto_threshold = float(np.percentile(proto_val.max(1), 5))

    # Joint readout.
    joint_test = space.joint.scores_batch({m: Z[m][test_idx] for m in MODALITIES})
    joint_val = space.joint.scores_batch({m: Z[m][val_idx] for m in MODALITIES})
    joint_held = space.joint.scores_batch({m: Z[m][held_idx] for m in MODALITIES})
    joint_acc = float(np.mean(joint_test.argmax(1) == y_all[test_idx]))
    joint_threshold = (
        space.joint.threshold
        if space.joint.threshold is not None
        else float(np.percentile(joint_val.max(1), 5))
    )

    # Hot update: add one held-out class and re-score the seen test set.
    seen_before = space.joint.scores_batch(
        {m: Z[m][test_idx] for m in MODALITIES}
    ).argmax(1)
    t0 = time.perf_counter()
    for row in held_idx[:10]:
        label = str(cats[row])
        query = {m: Z[m][row] for m in MODALITIES}
        space.joint.add_class(label, query)
    add_ms = (time.perf_counter() - t0) / max(len(held_idx[:10]), 1) * 1000.0
    seen_after = space.joint.scores_batch(
        {m: Z[m][test_idx] for m in MODALITIES}
    ).argmax(1)
    forgetting = float(np.mean(seen_before != seen_after))

    t0 = time.perf_counter()
    for _ in range(50):
        row = int(test_idx[0])
        query = {m: Z[m][row] for m in MODALITIES}
        space.joint.update_class(str(cats[row]), query)
    update_ms = (time.perf_counter() - t0) / 50 * 1000.0

    payload = {
        "data": args.data,
        "classes": len(classes),
        "readout_choice": space.readout_choice,
        "joint_metadata": space.joint.metadata,
        "counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "held": int(len(held_idx)),
        },
        "closed_set": {
            "prototype": proto_acc,
            "joint": joint_acc,
            "delta": joint_acc - proto_acc,
        },
        "open_set": {
            "prototype": {
                "auroc": _auroc(proto_test.max(1), proto_held.max(1)),
                **_operating_point(
                    proto_test.max(1), proto_held.max(1), proto_threshold
                ),
            },
            "joint": {
                "auroc": _auroc(joint_test.max(1), joint_held.max(1)),
                **_operating_point(
                    joint_test.max(1), joint_held.max(1), joint_threshold
                ),
            },
        },
        "hot_update": {
            "add_ms_per_class": add_ms,
            "update_ms_per_call": update_ms,
            "seen_forgetting_after_add": forgetting,
        },
        "checkpoint": str(args.bank),
    }

    (out / "joint_readout_eval.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Joint discriminative readout vs production prototype",
        "",
        f"> data: `{args.data}`; classes: {len(classes)}; "
        f"train/val/test/held = {len(train_idx)}/{len(val_idx)}/"
        f"{len(test_idx)}/{len(held_idx)}",
        "",
        "## Closed set",
        "",
        "| readout | top-1 |",
        "|---|---:|",
        f"| prototype | {_fmt(proto_acc)} |",
        f"| **joint LDA** | **{_fmt(joint_acc)}** |",
        f"| delta | {_fmt(joint_acc - proto_acc)} |",
        "",
        "## Open set (95% seen accept threshold from val)",
        "",
        "| readout | AUROC | seen false reject | unseen reject |",
        "|---|---:|---:|---:|",
    ]
    for name, row in payload["open_set"].items():
        lines.append(
            f"| {name} | {_fmt(row['auroc'])} | "
            f"{_fmt(row['seen_false_reject'])} | {_fmt(row['unseen_reject'])} |"
        )
    lines += [
        "",
        "## Hot update",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| add class (ms) | {_fmt(add_ms, 3)} |",
        f"| update class (ms) | {_fmt(update_ms, 3)} |",
        f"| seen forgetting after add | {_fmt(forgetting)} |",
        "",
    ]
    (out / "JOINT_READOUT_EVAL.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {out / 'JOINT_READOUT_EVAL.md'}")


if __name__ == "__main__":
    main()
