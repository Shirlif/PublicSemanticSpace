"""Same complementarity audit, but on the production 128-d projection.

The raw-feature audit measures the ceiling of the expert pool.  This script asks
whether that complementarity survives the public-space projection, which is the
deployed bottleneck.  It reuses the A-line projected features from the g audit
so the numbers are directly comparable to ``A_full`` in that report.

Example:
    python -m pss.eval.scripts.run_projected_fusion_audit \
        --out eval/results/projected_fusion_audit
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.encoders.scripts.bench_expert_zoo import _append_fusion, fusion_audit
from pss.eval.scripts.run_g_audit import _load_a_audit_rep


def _meta_from_task(feats: dict, task: dict) -> dict:
    n = len(feats["instances"])
    splits = np.array(["ignore"] * n, dtype=object)
    splits[task["train_idx"]] = "train"
    splits[task["val_idx"]] = "val"
    splits[task["test_idx"]] = "test"
    return {
        "categories": np.asarray(feats["categories"]),
        "instances": np.asarray(feats["instances"]),
        "splits": splits,
    }


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/category_pretrain_instance8")
    parser.add_argument(
        "--experts", default="checkpoints/category_pretrain_instance8_experts"
    )
    parser.add_argument("--projections", default="checkpoints/category_pretrain_instance8_projection")
    parser.add_argument("--feature-cache", default="eval/results/readout_opt")
    parser.add_argument("--regime", choices=["category", "view-instance"], default="view-instance")
    parser.add_argument("--label-key", choices=["category", "instance"], default="instance")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="eval/results/projected_fusion_audit")
    args = parser.parse_args()

    audit_args = Namespace(
        backbone="base",
        data=args.data,
        experts=args.experts,
        projections=args.projections,
        feature_cache=args.feature_cache,
        regime=args.regime,
        device=args.device,
    )
    feats, task, rep = _load_a_audit_rep(audit_args)
    meta = _meta_from_task(feats, task)
    audit = fusion_audit(rep, meta, label_key=args.label_key)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "projected_fusion_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Projected 128-d fusion audit",
        "",
        f"> data: `{args.data}`; projections: `{args.projections}`; "
        f"label: `{args.label_key}`",
        "> Same protocol as `bench_expert_zoo --fusion-audit`, but on the "
        "production public-space projection instead of raw expert features.",
        "",
    ]
    _append_fusion(lines, audit)
    (out / "PROJECTED_FUSION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in (
        "n_classes", "best_single", "best_single_test", "all_concat_test",
        "gap_vs_best_single", "concat_lr_test", "vote_lda_test",
        "weighted_vote_test", "gate_top1_test", "gate_soft_test",
        "mean_error_corr", "mean_disagreement",
    )}, ensure_ascii=False, indent=2))
    print(f"Wrote {out / 'PROJECTED_FUSION_AUDIT.md'}")


if __name__ == "__main__":
    main()
