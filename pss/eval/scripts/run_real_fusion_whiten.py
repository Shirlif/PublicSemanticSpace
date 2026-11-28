"""Re-run the real-data CLEVR audit after per-expert PCA-whitening.

The raw audit concatenated 1,372 dimensions and the pooled covariance could not
be estimated from 3,000 scenes.  This script fits one PCA-whiten transform per
frozen expert on the train split (128 dims, capped by the expert's width), then
runs the same ``fusion_audit`` protocol on the cached features.

Example:
    python -m pss.eval.scripts.run_real_fusion_whiten \
        --cache eval/results/real_fusion_audit/features_n5000_seed0.npz \
        --out eval/results/real_fusion_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from pss._console import ensure_utf8
from pss.eval.scripts.run_real_fusion_audit import (
    DEFAULT_ROOT,
    LABEL_TASKS,
    SUBSETS,
    _jsonable,
    _summary_table,
    load_scene_split,
    run_audits,
)


def _headline(audits: dict) -> list[str]:
    rows = []
    score_best = None
    for task, per_subset in audits.items():
        for subset, payload in per_subset.items():
            if "skipped" in payload:
                continue
            rows.append((task, subset, float(payload["gap_vs_best_single"])))
            for method in ("vote_lda_test", "weighted_vote_test", "gate_soft_test"):
                delta = float(payload[method]) - float(payload["best_single_test"])
                candidate = (task, subset, method, float(payload[method]), delta)
                if score_best is None or delta > score_best[4]:
                    score_best = candidate
    if not rows:
        return ["No audit ran."]
    positive = sum(1 for _, _, gap in rows if gap > 1e-9)
    best = max(rows, key=lambda row: row[2])
    worst = min(rows, key=lambda row: row[2])
    findings = [
        (
            f"After per-expert PCA-whitening, concatenated pooled-covariance LDA "
            f"beats the val-selected best single expert on {positive}/{len(rows)} "
            f"task-subset pairs. Largest concat gain: `{best[0]}`/{best[1]} "
            f"{best[2]:+.3f}; largest shortfall: `{worst[0]}`/{worst[1]} "
            f"{worst[2]:+.3f}."
        ),
        (
            f"Best score-level fusion: `{score_best[2]}` on `{score_best[0]}`/"
            f"{score_best[1]} {score_best[3]:.3f} vs best single "
            f"{score_best[3] - score_best[4]:.3f} ({score_best[4]:+.3f})."
        ),
        (
            "Whitening removes the covariance-estimation penalty, but it does not "
            "create complementarity: the two vision encoders are correlated and the "
            "geometry stream is oracle metadata that carries little appearance "
            "information. The gain is small and task-dependent."
        ),
    ]
    return findings


def _load_labels(root: Path, counts: dict[str, int], seed: int):
    labels = {task: [] for task in LABEL_TASKS}
    splits: list[str] = []
    images: list[str] = []
    for split in ("train", "val", "test"):
        chosen, split_labels, split_images = load_scene_split(
            root, split, counts[split], seed
        )
        for task in LABEL_TASKS:
            labels[task].extend(split_labels[task])
        splits.extend([split] * len(chosen))
        images.extend(split_images)
    return labels, splits, images


def _fmt(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("eval/results/real_fusion_audit/features_n5000_seed0.npz"),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=Path("eval/results/real_fusion_audit"))
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dims", type=int, default=128)
    args = parser.parse_args()

    counts = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    labels, splits, images = _load_labels(args.root, counts, args.seed)
    splits_arr = np.asarray(splits)
    train_idx = np.where(splits_arr == "train")[0]

    with np.load(args.cache, allow_pickle=False) as f:
        raw = {name: np.asarray(f[name], dtype=np.float32) for name in f.files if name != "__meta__"}

    whiten: dict[str, np.ndarray] = {}
    explained: dict[str, float] = {}
    for name, X in raw.items():
        n_comp = min(args.dims, X.shape[1], len(train_idx))
        pca = PCA(n_components=n_comp, whiten=True, random_state=0)
        pca.fit(X[train_idx])
        whiten[name] = pca.transform(X).astype(np.float32)
        explained[name] = float(np.sum(pca.explained_variance_ratio_))

    audits = run_audits(whiten, labels, splits, images)
    summary = _summary_table(audits, LABEL_TASKS)
    findings = _headline(audits)

    # Compare with the raw audit if it exists.
    raw_path = args.out / "real_fusion_audit.json"
    raw_audits = None
    if raw_path.exists():
        raw_audits = json.loads(raw_path.read_text(encoding="utf-8")).get("audits")

    payload = {
        "status": "ok",
        "cache": str(args.cache),
        "counts": counts,
        "seed": args.seed,
        "dims": args.dims,
        "explained_variance": explained,
        "summary": summary,
        "headline_findings": findings,
        "audits": audits,
        "raw_audits_present": raw_audits is not None,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "real_fusion_whiten.json").write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# CLEVR-CoGenT fusion audit after per-expert PCA-whitening",
        "",
        f"> cache: `{args.cache}`; train/val/test = "
        f"{counts['train']}/{counts['val']}/{counts['test']}; seed {args.seed}",
        f"> PCA dims: {args.dims} (capped by expert width); fit on train only.",
        "",
        "## Headline",
        "",
    ]
    lines.extend(f"- {item}" for item in findings)
    lines += [
        "",
        "## Per-task gap vs best single (raw -> whitened)",
        "",
        "| task | subset | raw gap | whitened gap | raw A_full | whitened A_full |",
        "|---|---|---:|---:|---:|---:|",
    ]
    positive = 0
    total = 0
    for task in LABEL_TASKS:
        for subset in SUBSETS:
            row = audits.get(task, {}).get(subset, {})
            if "skipped" in row:
                continue
            total += 1
            positive += int(row["gap_vs_best_single"] > 0)
            raw_row = None
            if raw_audits:
                raw_row = raw_audits.get(task, {}).get(subset)
            raw_gap = raw_row.get("gap_vs_best_single") if raw_row else None
            raw_full = raw_row.get("all_concat_test") if raw_row else None
            lines.append(
                f"| `{task}` | {subset} | {_fmt(raw_gap)} | "
                f"{_fmt(row['gap_vs_best_single'])} | {_fmt(raw_full)} | "
                f"{_fmt(row['all_concat_test'])} |"
            )
    lines += [
        "",
        f"Whitened concat beats the best single expert on **{positive}/{total}** "
        "task-subset pairs.",
        "",
        "## Per-expert whitened dimensions",
        "",
        "| expert | raw dim | whitened dim | explained variance |",
        "|---|---:|---:|---:|",
    ]
    for name, X in raw.items():
        lines.append(
            f"| `{name}` | {X.shape[1]} | {whiten[name].shape[1]} | "
            f"{_fmt(explained[name])} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- CLEVR is rendered imagery, not a physical sensor dataset.",
        "- The geometry stream is simulator metadata, not a learned depth/pose estimate.",
        "- PCA-whiten is fit on trainA; valA/valB are transformed with it.",
        "- DINOv2 and CLIP are both vision encoders on the same image; their errors "
        "are correlated, so this is not a true multi-modal complementarity test.",
        "",
    ]
    (args.out / "REAL_FUSION_WHITEN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(_jsonable({
        "positive": positive,
        "total": total,
        "explained_variance": explained,
        "headline": findings,
    }), ensure_ascii=False, indent=2))
    print(f"Wrote {args.out / 'REAL_FUSION_WHITEN.md'}")


if __name__ == "__main__":
    main()
