"""Build the Phase 5 prototype bank, tree, recognition evaluation, and
admission simulation.

Everything is category-level because the dataset only provides category-level
pairing. Instance-level prototype validation waits for RD2.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision import load_vision_expert
from pss.projection.features import load_paired_features
from pss.projection.heads import load_projection

from .admission import calibrated_thresholds, pre_review
from .bank import PrototypeBank, build_bank, l2
from .recognize import recognize
from .tree import build_tree, save_tree, tree_candidates

MODALITIES = ("vision", "tactile", "geometry")
WEIGHTS = {"vision": 1.0, "tactile": 0.5, "geometry": 0.95}


def extract_projected_features(
    data_dir: str | Path,
    expert_dir: str | Path,
    projection_dir: str | Path,
    device: str | None = None,
    vision_expert=None,
    tactile_expert=None,
) -> tuple:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    expert_dir = Path(expert_dir)
    projection_dir = Path(projection_dir)
    vision = vision_expert or load_vision_expert(expert_dir / "vision.pt", device=device)
    tactile = tactile_expert or load_tactile_expert(expert_dir / "tactile.pt", device=device)
    feats = load_paired_features(data_dir, vision, tactile)
    X = {"vision": feats.vision, "tactile": feats.tactile, "geometry": feats.geometry}
    Z = {}
    for modality, arr in X.items():
        head = load_projection(projection_dir / f"{modality}.pt", device=device)
        with torch.no_grad():
            z = head(torch.as_tensor(arr, dtype=torch.float32, device=device)).cpu().numpy()
        Z[modality] = z.astype(np.float32)
    return feats, Z


def _query(Z, i: int, modalities=MODALITIES) -> dict[str, np.ndarray]:
    return {m: Z[m][i] for m in modalities}


def _eval_recognition(
    feats,
    Z,
    bank,
    tree,
    idx,
    weights,
    use_tree: bool = True,
    threshold: float | None = None,
    modalities=MODALITIES,
):
    if len(idx) == 0:
        return 0.0, [], 0.0
    correct = 0
    scores = []
    conflicts = 0
    for i in idx:
        q = _query(Z, int(i), modalities)
        res = recognize(
            q,
            bank,
            tree=tree,
            weights={m: weights[m] for m in modalities},
            use_tree=use_tree,
            threshold=threshold,
        )
        if res.ranked and not res.unknown and res.ranked[0][0] == feats.categories[i]:
            correct += 1
        scores.append(res.ranked[0][1] if res.ranked else -1.0)
        conflicts += int(res.conflict)
    n = len(idx)
    return correct / n, scores, conflicts / n


def _variation_index(sample_id: str, instance: str) -> int | None:
    for text in (instance, sample_id):
        match = re.search(r"_v(\d+)$", str(text))
        if match:
            return int(match.group(1))
    return None


def _tune_weights_cv(
    feats,
    Z,
    train_idx,
    classes,
    n_folds: int = 3,
    modalities=MODALITIES,
) -> tuple[dict[str, float], float]:
    """Choose modality weights by cross-validation inside the training split.

    The val split is small (a few variations per category), so tuning on it
    overfits. Folds are grouped by appearance variation, so each fold asks the
    same question as test: "does the weight generalize to an unseen variation?"
    """
    variations = np.array(
        [_variation_index(feats.sample_ids[int(i)], feats.instances[int(i)]) for i in train_idx]
    )
    if any(v is None for v in variations):
        rng = np.random.default_rng(0)
        fold_ids = rng.integers(0, n_folds, size=len(train_idx))
    else:
        fold_ids = np.asarray([int(v) % n_folds for v in variations])

    grid = (0.0, 0.25, 0.5, 1.0)
    combos = [
        {"vision": wv, "tactile": wt, "geometry": wg}
        for wv in grid
        for wt in grid
        for wg in grid
        if wv + wt + wg > 0
    ]
    scores = {i: 0.0 for i in range(len(combos))}
    folds_used = 0
    for fold in range(n_folds):
        val_mask = fold_ids == fold
        fit_idx = train_idx[~val_mask]
        eval_idx = train_idx[val_mask]
        if len(fit_idx) == 0 or len(eval_idx) == 0:
            continue
        bank = build_bank(feats, Z, fit_idx, classes)
        tree = build_tree(bank)
        for i, weights in enumerate(combos):
            acc, _, _ = _eval_recognition(
                feats,
                Z,
                bank,
                tree,
                eval_idx,
                weights,
                use_tree=False,
                modalities=modalities,
            )
            scores[i] += acc
        folds_used += 1

    if folds_used == 0:
        return {m: 1.0 for m in modalities}, -1.0
    best_i = max(
        range(len(combos)),
        key=lambda i: (
            scores[i] / folds_used,
            1e-6 * (combos[i]["vision"] + 0.5 * combos[i]["geometry"]),
        ),
    )
    return combos[best_i], scores[best_i] / folds_used


def _breakdown(feats, Z, bank, tree, idx, weights, use_tree: bool = True) -> tuple[dict, dict]:
    per_category: dict[str, list[int]] = {}
    per_view: dict[int, list[int]] = {}
    for i in idx:
        i = int(i)
        res = recognize(
            _query(Z, i),
            bank,
            tree=tree,
            weights=weights,
            use_tree=use_tree,
        )
        ok = int(
            bool(res.ranked)
            and not res.unknown
            and res.ranked[0][0] == feats.categories[i]
        )
        cat = feats.categories[i]
        per_category.setdefault(cat, [0, 0])
        per_category[cat][0] += ok
        per_category[cat][1] += 1
        view = int(feats.views[i])
        per_view.setdefault(view, [0, 0])
        per_view[view][0] += ok
        per_view[view][1] += 1
    cat_out = {
        c: {"acc": ok / n, "n": n} for c, (ok, n) in sorted(per_category.items())
    }
    view_out = {
        str(v): {"acc": ok / n, "n": n} for v, (ok, n) in sorted(per_view.items())
    }
    return cat_out, view_out


def _open_set(feats, Z, bank, tree, val_idx, test_idx, unseen_idx, weights=None) -> dict:
    weights = dict(weights or WEIGHTS)
    _, val_scores, _ = _eval_recognition(
        feats, Z, bank, tree, val_idx, weights, use_tree=False
    )
    threshold = float(np.percentile(val_scores, 5)) if val_scores else 0.0
    test_acc, test_scores, _ = _eval_recognition(
        feats, Z, bank, tree, test_idx, weights, use_tree=False, threshold=threshold
    )
    _, unseen_scores, _ = _eval_recognition(
        feats, Z, bank, tree, unseen_idx, weights, use_tree=False, threshold=threshold
    )
    auroc = None
    if test_scores and unseen_scores:
        try:
            from sklearn.metrics import roc_auc_score

            y = np.array([1] * len(test_scores) + [0] * len(unseen_scores))
            s = np.array(test_scores + unseen_scores, dtype=float)
            auroc = float(roc_auc_score(y, s))
        except Exception:
            auroc = None
    return {
        "threshold": threshold,
        "val_acc": _eval_recognition(
            feats, Z, bank, tree, val_idx, weights, use_tree=False
        )[0],
        "test_acc_at_threshold": test_acc,
        "seen_false_reject_rate": float(np.mean(np.asarray(test_scores) < threshold))
        if test_scores
        else 0.0,
        "unseen_reject_rate": float(np.mean(np.asarray(unseen_scores) < threshold))
        if unseen_scores
        else 0.0,
        "seen_mean_max_score": float(np.mean(test_scores)) if test_scores else 0.0,
        "unseen_mean_max_score": float(np.mean(unseen_scores)) if unseen_scores else 0.0,
        "auroc": auroc,
    }


def _conflict(feats, Z, bank, tree, test_idx, rng, weights=None) -> dict:
    weights = dict(weights or WEIGHTS)
    n = min(100, len(test_idx))
    if n == 0:
        return {
            "baseline": 0.0,
            "vision_injected": 0.0,
            "geometry_injected": 0.0,
            "injected": 0.0,
            "n": 0,
        }
    by_cat: dict[str, list[int]] = {}
    for i in test_idx:
        by_cat.setdefault(feats.categories[i], []).append(int(i))
    cats = list(by_cat)
    base = vision_inj = geometry_inj = 0
    for k in range(n):
        i = int(test_idx[k])
        res = recognize(_query(Z, i), bank, tree=tree, weights=weights, use_tree=False)
        base += int(res.conflict)
        others = [c for c in cats if c != feats.categories[i]]
        if others:
            c = others[int(rng.integers(0, len(others)))]
            j = by_cat[c][int(rng.integers(0, len(by_cat[c])))]
        else:
            j = i
        # Inject a cross-category signal into a modality that actually carries
        # weight. Tactile has weight 0 in the current fusion, so a tactile
        # injection can never raise the conflict flag.
        q_vision = {
            "vision": Z["vision"][j],
            "geometry": Z["geometry"][i],
            "tactile": Z["tactile"][i],
        }
        res = recognize(q_vision, bank, tree=tree, weights=weights, use_tree=False)
        vision_inj += int(res.conflict)
        q_geometry = {
            "vision": Z["vision"][i],
            "geometry": Z["geometry"][j],
            "tactile": Z["tactile"][i],
        }
        res = recognize(q_geometry, bank, tree=tree, weights=weights, use_tree=False)
        geometry_inj += int(res.conflict)
    return {
        "baseline": base / n,
        "vision_injected": vision_inj / n,
        "geometry_injected": geometry_inj / n,
        "injected": max(vision_inj, geometry_inj) / n,
        "n": n,
    }


def _tree_recall(feats, Z, tree, idx) -> float:
    if len(idx) == 0:
        return 0.0
    correct = 0
    for i in idx:
        cands = tree_candidates(_query(Z, int(i)), tree, top_n=1)
        correct += int(feats.categories[i] in cands)
    return correct / len(idx)


def _admission(feats, Z, bank, unseen_idx, weights=None) -> dict:
    weights = dict(weights or WEIGHTS)
    if len(unseen_idx) == 0:
        return {}
    results: dict[str, dict] = {}
    for cat in sorted({feats.categories[i] for i in unseen_idx}):
        idx = [int(i) for i in unseen_idx if feats.categories[i] == cat]
        half = max(1, len(idx) // 2)
        cand_idx, hold_idx = idx[:half], idx[half:]
        samples = {m: Z[m][cand_idx] for m in MODALITIES}
        calibrated = calibrated_thresholds(bank)["per_modality"]
        thresholds = {
            m: calibrated[m]["distance_threshold"] for m in MODALITIES
        }
        review = pre_review(samples, bank, min_inter_distance=thresholds)
        proto = {m: l2(samples[m].mean(0)) for m in MODALITIES}
        new_cats = list(bank.categories) + [cat]
        new_protos = {
            m: np.vstack([bank.protos[m], proto[m][None, :]]).astype(np.float32)
            for m in MODALITIES
        }
        new_stats = {
            **bank.stats,
            cat: {
                m: {
                    "sample_count": len(cand_idx),
                    "variance": review.metrics[m]["variance"],
                }
                for m in MODALITIES
            },
        }
        temp = PrototypeBank(
            new_cats, new_protos, new_stats, version=f"{bank.version}_admission"
        )
        correct = 0
        for i in hold_idx:
            res = recognize(_query(Z, i), temp, weights=weights, use_tree=False)
            if res.ranked and res.ranked[0][0] == cat:
                correct += 1
        holdout_acc = correct / len(hold_idx) if hold_idx else 0.0
        decision = review.decision
        reasons = list(review.reasons)
        if decision == "accept" and holdout_acc < 0.8:
            decision = "review"
            reasons.append("holdout_acc_low")
        results[cat] = {
            "decision": decision,
            "reasons": reasons,
            "metrics": review.metrics,
            "holdout_acc": holdout_acc,
            "holdout_n": len(hold_idx),
        }
    return results


def _write_report(path: Path, metrics: dict) -> None:
    fusion = metrics["fusion"]
    tree_metrics = metrics["fusion_tree"]
    open_set = metrics["open_set"]
    conflict = metrics["conflict_injection"]
    lines = [
        "# Phase 5 report: prototype bank and recognition (smoke test)",
        "",
        f"- data: `{metrics['data_dir']}`",
        f"- version: `{metrics['version']}`",
        f"- categories ({len(metrics['classes'])}): {', '.join(metrics['classes'])}",
        f"- modality weights (3-fold CV on train variations): {metrics['weights']}",
        f"- tuned CV acc: {metrics['weights_tuned_cv_acc']:.3f}",
        "",
        "## Counts",
        "",
        "| split | examples |",
        "|---|---:|",
    ]
    for k, v in metrics["counts"].items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Recognition top-1",
        "",
        "| query | val | test |",
        "|---|---:|---:|",
        f"| fusion flat (primary) | {fusion['val_acc']:.3f} | {fusion['test_acc']:.3f} |",
        f"| fusion + tree | {tree_metrics['val_acc']:.3f} | {tree_metrics['test_acc']:.3f} |",
    ]
    for m, vals in metrics["single_modality"].items():
        lines.append(f"| {m} only | {vals['val_acc']:.3f} | {vals['test_acc']:.3f} |")
    lines += [
        "",
        "## Per-category test accuracy",
        "",
        "| category | acc | n |",
        "|---|---:|---:|",
    ]
    for cat, row in metrics.get("per_category_test", {}).items():
        lines.append(f"| {cat} | {row['acc']:.3f} | {row['n']} |")
    lines += [
        "",
        "## Per-view test accuracy (camera/light variation)",
        "",
        "| view | acc | n |",
        "|---|---:|---:|",
    ]
    for view, row in metrics.get("per_view_test", {}).items():
        lines.append(f"| {view} | {row['acc']:.3f} | {row['n']} |")
    lines += [
        "",
        "## Open set",
        "",
        f"- threshold (5th percentile of val max score): **{open_set['threshold']:.3f}**",
        f"- seen false-reject rate (test): {open_set['seen_false_reject_rate']:.3f}",
        f"- unseen reject rate (test_category): {open_set['unseen_reject_rate']:.3f}",
        f"- open-set AUROC: {open_set.get('auroc') if open_set.get('auroc') is not None else 'n/a'}",
        f"- mean max score: seen {open_set['seen_mean_max_score']:.3f}, "
        f"unseen {open_set['unseen_mean_max_score']:.3f}",
        "",
        "## Conflict detection",
        "",
        "| condition | conflict rate |",
        "|---|---:|",
        f"| normal test | {conflict['baseline']:.3f} |",
        f"| injected cross-category vision | {conflict.get('vision_injected', conflict['injected']):.3f} |",
        f"| injected cross-category geometry | {conflict.get('geometry_injected', conflict['injected']):.3f} |",
        "",
        "## Tree candidate recall (top-N=1)",
        "",
        f"- val: {metrics['tree_candidate_recall']['val']:.3f}",
        f"- test: {metrics['tree_candidate_recall']['test']:.3f}",
        "",
        "## Admission simulation (unseen categories)",
        "",
        "| category | decision | reasons | holdout acc |",
        "|---|---|---|---:|",
    ]
    for cat, res in metrics["admission"].items():
        lines.append(
            f"| {cat} | {res['decision']} | {', '.join(res['reasons']) or '—'} | "
            f"{res['holdout_acc']:.3f} |"
        )
    lines += [
        "",
        "_Admission requires holdout accuracy ≥ 0.8; lower accuracy is downgraded to review._",
        "",
        "## Admission threshold calibration (diagnostic)",
        "",
        "| scope | min nearest distance | median | 0.5 x min threshold |",
        "|---|---:|---:|---:|",
    ]
    for modality, vals in metrics["calibration"]["per_modality"].items():
        lines.append(
            f"| {modality} | {vals['min_nn']:.3f} | {vals['median_nn']:.3f} | "
            f"{vals['distance_threshold']:.3f} |"
        )
    fused = metrics["calibration"]["fused"]
    lines.append(
        f"| fused | {fused['min_nn']:.3f} | {fused['median_nn']:.3f} | "
        f"{fused['distance_threshold']:.3f} |"
    )
    lines += [
        "",
        "Admission uses per-modality calibrated thresholds (0.5 × min nearest-neighbor distance).",
        "These values are still diagnostics; validate them on held-out categories before production.",
        "",
        "## Caveats",
        "",
        "- Category-level prototypes; no instance-level synchronized data.",
        "- Tactile is synthetic and weak; it receives weight 0.5.",
        "- Thresholds and admission gates are initial values, not calibrated (9.10).",
        "- Tree topology is data-driven; semantic review is an offline gate in the real pipeline.",
        "- Treat all numbers as a pipeline check, not a capability claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase5(
    data_dir: str | Path,
    expert_dir: str | Path,
    projection_dir: str | Path,
    out_dir: str | Path,
    seed: int = 0,
    vision_expert=None,
    tactile_expert=None,
) -> dict:
    rng = np.random.default_rng(seed)
    feats, Z = extract_projected_features(
        data_dir,
        expert_dir,
        projection_dir,
        vision_expert=vision_expert,
        tactile_expert=tactile_expert,
    )
    splits = np.array(feats.splits)
    classes = sorted({feats.categories[i] for i in np.where(splits == "train")[0]})

    def seen(split: str) -> np.ndarray:
        return np.array(
            [
                int(i)
                for i in np.where(splits == split)[0]
                if feats.categories[i] in classes
            ],
            dtype=int,
        )

    train_idx = seen("train")
    val_idx = seen("val")
    test_idx = seen("test")
    unseen_idx = np.where(splits == "test_category")[0].astype(int)

    bank = build_bank(feats, Z, train_idx, classes)
    tree = build_tree(bank)

    weights, tuned_cv_acc = _tune_weights_cv(feats, Z, train_idx, classes)
    val_flat, _, val_conf_flat = _eval_recognition(
        feats, Z, bank, tree, val_idx, weights, use_tree=False
    )
    test_flat, _, test_conf_flat = _eval_recognition(
        feats, Z, bank, tree, test_idx, weights, use_tree=False
    )
    val_tree, _, val_conf_tree = _eval_recognition(
        feats, Z, bank, tree, val_idx, weights, use_tree=True
    )
    test_tree, _, test_conf_tree = _eval_recognition(
        feats, Z, bank, tree, test_idx, weights, use_tree=True
    )
    per_category, per_view = _breakdown(
        feats, Z, bank, tree, test_idx, weights, use_tree=False
    )

    single = {}
    for m in MODALITIES:
        a, _, _ = _eval_recognition(
            feats, Z, bank, tree, val_idx, {m: 1.0}, use_tree=False, modalities=(m,)
        )
        b, _, _ = _eval_recognition(
            feats, Z, bank, tree, test_idx, {m: 1.0}, use_tree=False, modalities=(m,)
        )
        single[m] = {"val_acc": a, "test_acc": b}

    metrics = {
        "data_dir": str(data_dir),
        "experts": str(expert_dir),
        "projections": str(projection_dir),
        "version": bank.version,
        "classes": classes,
        "weights": weights,
        "weights_tuned_cv_acc": tuned_cv_acc,
        "counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "test_category": int(len(unseen_idx)),
        },
        "fusion": {
            "val_acc": val_flat,
            "test_acc": test_flat,
            "val_conflict_rate": val_conf_flat,
            "test_conflict_rate": test_conf_flat,
        },
        "fusion_tree": {
            "val_acc": val_tree,
            "test_acc": test_tree,
            "val_conflict_rate": val_conf_tree,
            "test_conflict_rate": test_conf_tree,
        },
        "single_modality": single,
        "per_category_test": per_category,
        "per_view_test": per_view,
        "open_set": _open_set(feats, Z, bank, tree, val_idx, test_idx, unseen_idx, weights),
        "conflict_injection": _conflict(feats, Z, bank, tree, test_idx, rng, weights),
        "tree_candidate_recall": {
            "val": _tree_recall(feats, Z, tree, val_idx),
            "test": _tree_recall(feats, Z, tree, test_idx),
        },
        "admission": _admission(feats, Z, bank, unseen_idx, weights),
        "calibration": calibrated_thresholds(bank),
        "note": "Category-level prototype smoke test; not a capability claim.",
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bank.save(out / "prototypes.npz")
    save_tree(out / "tree.json", tree)
    digest = hashlib.sha256((out / "prototypes.npz").read_bytes()).hexdigest()
    manifest = {
        "version": bank.version,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "categories": classes,
        "files": ["prototypes.npz", "tree.json"],
        "sha256": digest,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "REPORT.md", metrics)
    return metrics
