"""P1-7: split-conformal open-set thresholds.

Compares a marginal 95% threshold with class-conditional conformal thresholds.
Class-conditional conformal targets per-class coverage instead of only the
pooled seen acceptance rate.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from pss._console import ensure_utf8
from pss.eval.scripts.run_p2_openset import (
    MODALITIES,
    _pss_scores,
    _split_indices,
)
from pss.prototypes import PrototypeBank
from pss.prototypes.build import extract_projected_features


def _conformal_quantile(values, alpha=0.05):
    """Split-conformal threshold with finite-sample correction."""
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n == 0:
        return 0.0
    # Lower-tail threshold: scores below it are rejected. This targets
    # alpha rejection on the calibration distribution.
    rank = max(1, min(n, int(math.ceil((n + 1) * alpha))))
    return float(values[rank - 1])


def _evaluate(scores_seen, scores_unseen, threshold):
    accepted = scores_seen >= threshold
    return {
        "threshold": float(threshold),
        "seen_accept": float(np.mean(accepted)),
        "unseen_reject": float(np.mean(scores_unseen < threshold)),
        "auroc": float(
            roc_auc_score(
                [1] * len(scores_seen) + [0] * len(scores_unseen),
                list(scores_seen) + list(scores_unseen),
            )
        ),
        "n_seen": len(scores_seen),
        "n_unseen": len(scores_unseen),
    }


def run(
    data_dir="data/category_pretrain",
    expert_dir="checkpoints/category_pretrain_experts",
    projection_dir="checkpoints/phase45_projection",
    prototype_dir="checkpoints/phase45_prototypes",
    out_dir="eval/results/pss_vs_transformer",
    alpha=0.05,
):
    feats, Z = extract_projected_features(data_dir, expert_dir, projection_dir, "cpu")
    bank = PrototypeBank.load(Path(prototype_dir) / "prototypes.npz")
    weights = {"vision": 1.0, "tactile": 0.5, "geometry": 1.0}
    mp = Path(prototype_dir) / "metrics.json"
    if mp.exists():
        weights = json.loads(mp.read_text(encoding="utf-8")).get("weights", weights)

    val_seen = _split_indices(feats, bank, "val", True)
    test_seen = _split_indices(feats, bank, "test", True)
    test_unseen = _split_indices(feats, bank, "test_category", False)

    val_scores, _ = _pss_scores(feats, Z, bank, val_seen, weights)
    seen_scores, _ = _pss_scores(feats, Z, bank, test_seen, weights)
    unseen_scores, _ = _pss_scores(feats, Z, bank, test_unseen, weights)

    marginal_thr = _conformal_quantile(val_scores, alpha)
    marginal = _evaluate(seen_scores, unseen_scores, marginal_thr)

    # Class-conditional: calibrate each class's own score distribution.
    per_class_thr = {}
    for category in bank.categories:
        vals = [
            val_scores[k]
            for k, i in enumerate(val_seen)
            if feats.categories[i] == category
        ]
        if vals:
            per_class_thr[category] = _conformal_quantile(vals, alpha)
    seen_flags = []
    for k, i in enumerate(test_seen):
        thr = per_class_thr.get(feats.categories[i], marginal_thr)
        seen_flags.append(seen_scores[k] >= thr)
    unseen_flags = [score < marginal_thr for score in unseen_scores]
    by_class = {}
    for category, thr in per_class_thr.items():
        vals = [
            seen_flags[k]
            for k, i in enumerate(test_seen)
            if feats.categories[i] == category
        ]
        if vals:
            by_class[category] = float(np.mean(vals))
    class_cond = {
        "thresholds": per_class_thr,
        "seen_accept": float(np.mean(seen_flags)),
        "min_class_coverage": float(min(by_class.values())) if by_class else 0.0,
        "mean_class_coverage": float(np.mean(list(by_class.values()))) if by_class else 0.0,
        "unseen_reject": float(np.mean(unseen_flags)),
        "per_class_coverage": by_class,
        "alpha": alpha,
    }

    metrics = {
        "data_dir": str(data_dir),
        "alpha": alpha,
        "marginal": marginal,
        "class_conditional": class_cond,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p17_conformal.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p17_conformal.md", metrics)
    return metrics


def _report(path, metrics):
    m = metrics["marginal"]
    c = metrics["class_conditional"]
    lines = [
        "# P1-7: split-conformal open-set thresholds",
        "",
        f"- nominal risk alpha: {metrics['alpha']:.2f}",
        "",
        "| method | threshold | seen accept | unseen reject | AUROC | min class coverage |",
        "|---|---:|---:|---:|---:|---:|",
        f"| marginal conformal | {m['threshold']:.3f} | {m['seen_accept']:.3f} | "
        f"{m['unseen_reject']:.3f} | {m['auroc']:.3f} | - |",
        f"| class-conditional | per-class | {c['seen_accept']:.3f} | "
        f"{c['unseen_reject']:.3f} | - | {c['min_class_coverage']:.3f} |",
        "",
        "Class-conditional thresholds use each class's validation score distribution.",
        "The guarantee is marginal under exchangeability; per-class coverage is empirical.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/category_pretrain")
    ap.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    ap.add_argument("--projections", default="checkpoints/phase45_projection")
    ap.add_argument("--prototypes", default="checkpoints/phase45_prototypes")
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    m = run(args.data, args.experts, args.projections, args.prototypes, args.out, args.alpha)
    print(
        f"marginal: accept {m['marginal']['seen_accept']:.3f}, "
        f"unseen reject {m['marginal']['unseen_reject']:.3f}"
    )
    print(
        f"class-conditional: min coverage {m['class_conditional']['min_class_coverage']:.3f}, "
        f"unseen reject {m['class_conditional']['unseen_reject']:.3f}"
    )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
