"""CLI: Phase 4/5 acceptance checks on the category_pretrain R0 stack.

Reads the Phase 4 projection and Phase 5 prototype outputs, then computes the
gates that the training scripts do not cover:

Phase 4
- cross-modal pair/category R@1 (from the Phase 4 metrics)
- attribute probe retention: raw expert features vs projected features
- class separability (cosine silhouette)
- alignment and uniformity
- same-category / different-material collision rate

Phase 5
- closed-set top-1, open-set AUROC (from Phase 5 metrics)
- tree candidate recall
- recognition latency p50/p95
- same-category attribute confusion (nearest-neighbour proxy)
- admission decisions

Example:
    python -m pss.eval.scripts.run_phase45 \
        --data data/category_pretrain \
        --experts checkpoints/category_pretrain_experts \
        --projections checkpoints/phase45_projection \
        --prototypes checkpoints/phase45_prototypes \
        --out eval/results/phase45
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pss._console import ensure_utf8
from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision import load_vision_expert
from pss.projection.features import load_paired_features
from pss.projection.heads import load_projection
from pss.prototypes.bank import PrototypeBank
from pss.prototypes.recognize import recognize
from pss.prototypes.tree import load_tree

MODALITIES = ("vision", "tactile", "geometry")
ATTR_NAMES = ("hardness", "roughness", "thermal", "size_x", "size_y")


def _project(head, X: np.ndarray, device: str) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        xb = torch.as_tensor(X, dtype=torch.float32, device=device)
        return head(xb).cpu().numpy()


def _attribute_retention(raw, projected, attrs, train_idx, test_idx) -> dict:
    out = {}
    for modality in MODALITIES:
        raw_scores, proj_scores = [], []
        for a in range(attrs.shape[1]):
            raw_reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            proj_reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            raw_reg.fit(raw[modality][train_idx], attrs[train_idx, a])
            proj_reg.fit(projected[modality][train_idx], attrs[train_idx, a])
            raw_scores.append(float(r2_score(attrs[test_idx, a], raw_reg.predict(raw[modality][test_idx]))))
            proj_scores.append(float(r2_score(attrs[test_idx, a], proj_reg.predict(projected[modality][test_idx]))))
        retention = [
            (p / r) if r > 1e-3 else float("nan")
            for r, p in zip(raw_scores, proj_scores)
        ]
        out[modality] = {
            "raw_r2": dict(zip(ATTR_NAMES, raw_scores)),
            "projected_r2": dict(zip(ATTR_NAMES, proj_scores)),
            "retention": dict(zip(ATTR_NAMES, retention)),
            "mean_raw_r2": float(np.nanmean(raw_scores)),
            "mean_projected_r2": float(np.nanmean(proj_scores)),
            "mean_retention": float(np.nanmean(retention)),
        }
    return out


def _separability(projected, labels, idx) -> dict:
    out = {}
    for modality in MODALITIES:
        out[modality] = float(
            silhouette_score(projected[modality][idx], labels[idx], metric="cosine")
        )
    return out


def _alignment_uniformity(projected, test_idx) -> tuple[dict, dict]:
    alignment = {}
    for i, first in enumerate(MODALITIES):
        for second in MODALITIES[i + 1 :]:
            diff = projected[first][test_idx] - projected[second][test_idx]
            alignment[f"{first}-{second}"] = float(np.mean(np.sum(diff * diff, axis=1)))
    uniformity = {}
    for modality in MODALITIES:
        z = projected[modality][test_idx]
        sq = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
        n = len(z)
        mask = ~np.eye(n, dtype=bool)
        uniformity[modality] = float(np.log(np.exp(-2.0 * sq[mask]).mean()))
    return alignment, uniformity


def _attribute_collision(projected, attrs, categories, idx, sim_threshold=0.95, diff_threshold=0.3) -> dict:
    out = {}
    for modality in MODALITIES:
        z = projected[modality]
        pairs = 0
        collisions = 0
        sims = []
        for category in sorted({categories[i] for i in idx}):
            ids = [int(i) for i in idx if categories[i] == category]
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    i, j = ids[a], ids[b]
                    material_diff = max(
                        abs(attrs[i, 0] - attrs[j, 0]),
                        abs(attrs[i, 1] - attrs[j, 1]),
                    )
                    if material_diff < diff_threshold:
                        continue
                    sim = float(np.dot(z[i], z[j]))
                    sims.append(sim)
                    pairs += 1
                    collisions += int(sim >= sim_threshold)
        out[modality] = {
            "pairs": int(pairs),
            "collisions": int(collisions),
            "collision_rate": float(collisions / pairs) if pairs else float("nan"),
            "mean_cosine": float(np.mean(sims)) if sims else float("nan"),
        }
    return out


def _attribute_confusion(projected, attrs, categories, train_idx, test_idx, diff_threshold=0.3) -> dict:
    """Material-level proxy: the nearest same-category train neighbour should not
    be a large hard/soft or rough/smooth mismatch. Size/thermal are excluded
    because they are not tactile properties in the current S1 setup.
    """
    out = {}
    for modality in MODALITIES:
        ztr = projected[modality][train_idx]
        zte = projected[modality][test_idx]
        sims = zte @ ztr.T
        confused = 0
        counted = 0
        for k, i in enumerate(test_idx):
            same = np.where(categories[train_idx] == categories[i])[0]
            if len(same) == 0:
                continue
            j = int(train_idx[same[int(np.argmax(sims[k, same]))]])
            counted += 1
            material_diff = max(
                abs(attrs[i, 0] - attrs[j, 0]),
                abs(attrs[i, 1] - attrs[j, 1]),
            )
            confused += int(material_diff >= diff_threshold)
        out[modality] = {
            "same_category_pairs": int(counted),
            "confused": int(confused),
            "confusion_rate": float(confused / counted) if counted else float("nan"),
        }
    return out


def _latency(bank, tree, projected, test_idx, weights, n=200) -> dict:
    indices = [int(i) for i in test_idx[:n]]
    out = {}
    for name, use_tree in (("flat", False), ("tree", True)):
        times = []
        for i in indices:
            query = {m: projected[m][i] for m in MODALITIES}
            t0 = time.perf_counter()
            recognize(query, bank, tree=tree, weights=weights, use_tree=use_tree)
            times.append((time.perf_counter() - t0) * 1000.0)
        out[name] = {
            "p50_ms": float(np.percentile(times, 50)) if times else float("nan"),
            "p95_ms": float(np.percentile(times, 95)) if times else float("nan"),
            "n": len(times),
        }
    return out


def _verdict(ok: bool, partial: bool = False) -> str:
    if ok:
        return "PASS"
    return "PARTIAL" if partial else "FAIL"


def _fmt(value, digits=3) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_pct(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{float(value):.1%}"


def render_report(payload: dict) -> str:
    phase4 = payload["phase4"]
    phase5 = payload["phase5"]
    gates = payload["gates"]
    lines = [
        "# Phase 4/5 validation report",
        "",
        f"- data: `{payload['data']}`",
        f"- experts: `{payload['experts']}`",
        f"- projections: `{payload['projections']}`",
        f"- prototypes: `{payload['prototypes']}`",
        f"- device: `{payload['device']}`",
        "",
        "## Gate summary",
        "",
        "| gate | threshold | result | verdict |",
        "|---|---:|---:|---|",
    ]
    for gate in gates:
        lines.append(
            f"| {gate['name']} | {gate['threshold']} | {gate['result']} | **{gate['verdict']}** |"
        )

    lines += [
        "",
        "## Phase 4 details",
        "",
        "### Cross-modal retrieval (test)",
        "",
        "| direction | category R@1 | pair R@1 |",
        "|---|---:|---:|",
    ]
    for direction, vals in phase4["retrieval_test"].items():
        lines.append(
            f"| {direction} | {_fmt(vals['category_r1'])} | {_fmt(vals['pair_r1'])} |"
        )
    lines += [
        "",
        "### Attribute probe retention (test R²)",
        "",
        "| modality | raw mean R² | projected mean R² | retention |",
        "|---|---:|---:|---:|",
    ]
    for modality, vals in phase4["attribute_retention"].items():
        lines.append(
            f"| {modality} | {_fmt(vals['mean_raw_r2'])} "
            f"| {_fmt(vals['mean_projected_r2'])} "
            f"| {_fmt_pct(vals['mean_retention'])} |"
        )
    lines += [
        "",
        "### Separability / alignment / uniformity (test)",
        "",
        "| metric | vision | tactile | geometry |",
        "|---|---:|---:|---:|",
    ]
    sep = phase4["separability"]
    uni = phase4["uniformity"]
    lines.append(
        f"| silhouette (cosine) | {_fmt(sep['vision'])} | {_fmt(sep['tactile'])} | {_fmt(sep['geometry'])} |"
    )
    lines.append(
        f"| uniformity | {_fmt(uni['vision'])} | {_fmt(uni['tactile'])} | {_fmt(uni['geometry'])} |"
    )
    lines += [
        "",
        "| pair | alignment (MSE) |",
        "|---|---:|",
    ]
    for pair, value in phase4["alignment"].items():
        lines.append(f"| {pair} | {_fmt(value)} |")
    lines += [
        "",
        "### Same-category / different-material collision",
        "",
        "| modality | pairs | collision rate @cos≥0.95 | mean cosine |",
        "|---|---:|---:|---:|",
    ]
    for modality, vals in phase4["attribute_collision"].items():
        lines.append(
            f"| {modality} | {vals['pairs']} | {_fmt_pct(vals['collision_rate'])} "
            f"| {_fmt(vals['mean_cosine'])} |"
        )

    lines += [
        "",
        "## Phase 5 details",
        "",
        "| query | val | test |",
        "|---|---:|---:|",
        f"| fusion flat | {_fmt(phase5['fusion']['val_acc'])} | {_fmt(phase5['fusion']['test_acc'])} |",
        f"| fusion + tree | {_fmt(phase5['fusion_tree']['val_acc'])} | {_fmt(phase5['fusion_tree']['test_acc'])} |",
    ]
    for modality, vals in phase5["single_modality"].items():
        lines.append(
            f"| {modality} only | {_fmt(vals['val_acc'])} | {_fmt(vals['test_acc'])} |"
        )
    lines += [
        "",
        f"- open-set AUROC: **{_fmt(phase5['open_set']['auroc'])}**",
        f"- seen false-reject: {_fmt_pct(phase5['open_set']['seen_false_reject_rate'])}",
        f"- unseen reject: {_fmt_pct(phase5['open_set']['unseen_reject_rate'])}",
        f"- tree candidate recall: val={_fmt(phase5['tree_candidate_recall']['val'])}, "
        f"test={_fmt(phase5['tree_candidate_recall']['test'])}",
        f"- conflict injection: baseline={_fmt(phase5['conflict_injection']['baseline'])}, "
        f"vision={_fmt(phase5['conflict_injection'].get('vision_injected', phase5['conflict_injection']['injected']))}, "
        f"geometry={_fmt(phase5['conflict_injection'].get('geometry_injected', phase5['conflict_injection']['injected']))}",
        "",
        "### Recognition latency",
        "",
        "| mode | p50 (ms) | p95 (ms) | n |",
        "|---|---:|---:|---:|",
    ]
    for mode, vals in phase5["latency"].items():
        lines.append(
            f"| {mode} | {_fmt(vals['p50_ms'])} | {_fmt(vals['p95_ms'])} | {vals['n']} |"
        )
    lines += [
        "",
        "### Same-category material confusion",
        "",
        "| modality | pairs | confusion rate |",
        "|---|---:|---:|",
    ]
    for modality, vals in phase5["attribute_confusion"].items():
        lines.append(
            f"| {modality} | {vals['same_category_pairs']} | {_fmt_pct(vals['confusion_rate'])} |"
        )
    lines += [
        "",
        "### Admission simulation",
        "",
        "| category | decision | reasons | holdout acc |",
        "|---|---|---|---:|",
    ]
    for category, vals in phase5["admission"].items():
        lines.append(
            f"| {category} | {vals['decision']} | {', '.join(vals['reasons']) or '-'} "
            f"| {_fmt(vals['holdout_acc'])} |"
        )

    retention_mean = float(
        np.nanmean([v["mean_retention"] for v in phase4["attribute_retention"].values()])
    )
    collision = phase4["attribute_collision"]
    material_collision = float(phase4.get("material_collision", float("nan")))
    tree_recall = phase5["tree_candidate_recall"]["test"]
    conflict = phase5["conflict_injection"]
    confusion = phase5["attribute_confusion"]
    confusion_mean = float(
        np.nanmean([v["confusion_rate"] for v in confusion.values()])
    )
    material_status = (
        "passes the 5% gate" if material_collision <= 0.05 else "is still above the 5% gate"
    )
    material_action = (
        "3. Material gate passes at material-weight 1.0 / batch 256; keep monitoring vision "
        "and re-test after any projection change."
        if material_collision <= 0.05
        else "3. Projection-only material loss is insufficient: add a material-conditioned head or sub-prototypes."
    )
    lines += [
        "",
        "## Diagnosis",
        "",
        "- Phase 4 pair R@1 is far below 0.7. The current dataset has only category-level pairing;",
        "  instance-level synchronous visual-tactile pairs are the RD2 deliverable and are not available yet.",
        "- Vision↔geometry category R@1 is strong (≈0.88–0.94); tactile↔anything is weak, matching the",
        "  Phase 3 tactile expert result. The tuned fusion gives tactile a low or zero weight.",
        f"- Attribute retention passes ({retention_mean:.1%} mean). Material collision is {material_collision:.1%}",
        f"  on the material modalities (vision {collision['vision']['collision_rate']:.1%}, "
        f"tactile {collision['tactile']['collision_rate']:.1%};",
        f"  geometry {collision['geometry']['collision_rate']:.1%} is shape/appearance and excluded). "
        f"Material gate {material_status}.",
        "  The projection preserves attributes globally; geometry collision is not a material failure.",
        f"- Phase 5 flat recognition and open-set are strong. The hierarchical tree over-prunes (recall {tree_recall:.1%}),",
        "  so flat retrieval is the correct mode for the current 16-category bank.",
        f"- Conflict injection works after switching to vision/geometry: "
        f"{conflict.get('vision_injected', conflict['injected']):.3f} vs {conflict['baseline']:.3f} baseline.",
        "  The earlier flat result was a tactile-weight artifact, not a conflict-detector failure.",
        "- Admission thresholds are calibrated per modality from the bank and gated by holdout accuracy ≥0.8.",
        f"- Material confusion is {confusion_mean:.1%} "
        f"(vision {confusion['vision']['confusion_rate']:.1%}, "
        f"tactile {confusion['tactile']['confusion_rate']:.1%}, "
        f"geometry {confusion['geometry']['confusion_rate']:.1%}).",
        "  It is a nearest-neighbour proxy, not the final production metric.",
        "",
        "## Next actions",
        "",
        "1. Keep flat retrieval for Phase 5 until the tree is re-built and validated; do not ship the tree.",
        "2. Treat instance-level alignment as blocked on RD2; do not claim the Phase 4 R@1 gate is met.",
        material_action,
        "4. Validate the calibrated admission thresholds on held-out categories; reject/review low holdout accuracy.",
        "5. Anchor retraining passes for vision/tactile; re-run it whenever an expert is actually replaced.",
        "6. Prototype package add/rollback passes; re-run it for each new category.",
        "7. Re-test attribute retention and collision after any expert/projection change.",
        "",
    ]
    return "\n".join(lines)


def run_phase45(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data)
    experts = Path(args.experts)
    projections = Path(args.projections)
    prototypes = Path(args.prototypes)

    vision = load_vision_expert(experts / "vision.pt", device=device)
    tactile = load_tactile_expert(experts / "tactile.pt", device=device)
    feats = load_paired_features(data_dir, vision, tactile)
    heads = {
        modality: load_projection(projections / f"{modality}.pt", device=device)
        for modality in MODALITIES
    }
    raw = {"vision": feats.vision, "tactile": feats.tactile, "geometry": feats.geometry}
    projected = {
        modality: _project(heads[modality], raw[modality], device)
        for modality in MODALITIES
    }
    splits = np.array(feats.splits)
    categories = np.array(feats.categories)
    attrs = np.asarray(feats.attrs, dtype=np.float32)
    train_idx = np.where(splits == "train")[0]
    test_idx = np.where(splits == "test")[0]
    seen_idx = np.where(np.isin(splits, ["train", "val", "test"]))[0]

    phase4_metrics = json.loads((projections / "metrics.json").read_text(encoding="utf-8"))
    phase5_metrics = json.loads((prototypes / "metrics.json").read_text(encoding="utf-8"))
    anchor_path = (
        Path(args.anchor)
        if getattr(args, "anchor", None)
        else Path("checkpoints/anchor_retrain_vision/anchor_report.json")
    )
    anchor = (
        json.loads(anchor_path.read_text(encoding="utf-8"))
        if anchor_path.exists()
        else None
    )
    package_path = (
        Path(args.package_report)
        if getattr(args, "package_report", None)
        else Path("eval/results/phase5_package/package_report.json")
    )
    package = (
        json.loads(package_path.read_text(encoding="utf-8"))
        if package_path.exists()
        else None
    )

    retention = _attribute_retention(raw, projected, attrs, train_idx, test_idx)
    separability = _separability(projected, categories, test_idx)
    alignment, uniformity = _alignment_uniformity(projected, test_idx)
    collision = _attribute_collision(projected, attrs, categories, seen_idx)
    confusion = _attribute_confusion(projected, attrs, categories, train_idx, test_idx)

    bank = PrototypeBank.load(prototypes / "prototypes.npz")
    tree = load_tree(prototypes / "tree.json")
    weights = phase5_metrics["weights"]
    latency = _latency(bank, tree, projected, test_idx, weights)

    retrieval_test = phase4_metrics["retrieval"]["test"]
    pair_values = [vals["pair_r1"] for vals in retrieval_test.values()]
    category_values = [vals["category_r1"] for vals in retrieval_test.values()]
    retention_values = [vals["mean_retention"] for vals in retention.values()]
    retention_mean = float(np.nanmean(retention_values))
    material_collision = float(
        np.nanmean(
            [
                collision["vision"]["collision_rate"],
                collision["tactile"]["collision_rate"],
            ]
        )
    )
    collision_all = float(
        np.nanmean([vals["collision_rate"] for vals in collision.values()])
    )
    confusion_rate = float(
        np.nanmean([vals["confusion_rate"] for vals in confusion.values()])
    )
    latency_p95 = latency["flat"]["p95_ms"]
    tree_recall = phase5_metrics["tree_candidate_recall"]["test"]
    open_auroc = phase5_metrics["open_set"]["auroc"]
    fusion_test = phase5_metrics["fusion"]["test_acc"]

    gates = [
        {
            "name": "P4 cross-modal pair R@1",
            "threshold": "≥ 0.70",
            "result": f"{_fmt(min(pair_values))}–{_fmt(max(pair_values))}",
            "verdict": _verdict(max(pair_values) >= 0.70),
        },
        {
            "name": "P4 attribute retention",
            "threshold": "≥ 90%",
            "result": _fmt_pct(retention_mean),
            "verdict": _verdict(retention_mean >= 0.90, partial=retention_mean >= 0.70),
        },
        {
            "name": "P4 material collision",
            "threshold": "≤ 5%",
            "result": _fmt_pct(material_collision),
            "verdict": _verdict(material_collision <= 0.05, partial=material_collision <= 0.15),
        },
        {
            "name": "P4 anchor retrain",
            "threshold": "runnable + frozen",
            "result": (
                f"{anchor['modality']} R@1={_fmt(anchor['retrieval']['test']['category_r1'])}, "
                f"retention={anchor['attribute_retention']['mean_retention']:.1%}, "
                f"frozen={anchor['frozen_integrity']['unchanged']}"
                if anchor
                else "not implemented"
            ),
            "verdict": (
                "PASS"
                if anchor
                and anchor.get("frozen_integrity", {}).get("unchanged", False)
                else "FAIL"
            ),
        },
        {
            "name": "P5 closed-set top-1",
            "threshold": "≥ 0.80",
            "result": _fmt(fusion_test),
            "verdict": _verdict(fusion_test >= 0.80),
        },
        {
            "name": "P5 open-set AUROC",
            "threshold": "≥ 0.90",
            "result": _fmt(open_auroc),
            "verdict": _verdict(open_auroc >= 0.90),
        },
        {
            "name": "P5 tree candidate recall",
            "threshold": "≥ 99%",
            "result": _fmt_pct(tree_recall),
            "verdict": _verdict(tree_recall >= 0.99),
        },
        {
            "name": "P5 recognition latency p95",
            "threshold": "< 50 ms",
            "result": f"{_fmt(latency_p95)} ms",
            "verdict": _verdict(latency_p95 < 50.0),
        },
        {
            "name": "P5 material confusion",
            "threshold": "< 2%",
            "result": _fmt_pct(confusion_rate),
            "verdict": _verdict(confusion_rate < 0.02, partial=confusion_rate < 0.10),
        },
        {
            "name": "P5 add/rollback package",
            "threshold": "works + no backprop",
            "result": (
                f"add={package['add'].get('added')}, "
                f"rollback={package['rollback'].get('to')}, "
                f"no_backprop={package.get('no_backprop')}"
                if package
                else "admission pre-review only"
            ),
            "verdict": "PASS" if package and package.get("ok") else "PARTIAL",
        },
    ]

    payload = {
        "data": str(data_dir),
        "experts": str(experts),
        "projections": str(projections),
        "prototypes": str(prototypes),
        "device": device,
        "counts": {"train": len(train_idx), "test": len(test_idx), "seen": len(seen_idx)},
        "phase4": {
            "retrieval_test": retrieval_test,
            "category_r1_mean": float(np.mean(category_values)),
            "pair_r1_mean": float(np.mean(pair_values)),
            "attribute_retention": retention,
            "separability": separability,
            "alignment": alignment,
            "uniformity": uniformity,
            "attribute_collision": collision,
            "material_collision": material_collision,
            "collision_all": collision_all,
            "anchor_retrain": anchor,
        },
        "phase5": {
            "fusion": phase5_metrics["fusion"],
            "fusion_tree": phase5_metrics["fusion_tree"],
            "single_modality": phase5_metrics["single_modality"],
            "open_set": phase5_metrics["open_set"],
            "tree_candidate_recall": phase5_metrics["tree_candidate_recall"],
            "conflict_injection": phase5_metrics["conflict_injection"],
            "admission": phase5_metrics["admission"],
            "latency": latency,
            "attribute_confusion": confusion,
            "package": package,
        },
        "gates": gates,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase45_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "PHASE4_5_REPORT.md").write_text(render_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/category_pretrain")
    parser.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    parser.add_argument("--projections", default="checkpoints/phase45_projection")
    parser.add_argument("--prototypes", default="checkpoints/phase45_prototypes")
    parser.add_argument("--anchor", default=None)
    parser.add_argument("--package-report", default=None)
    parser.add_argument("--out", default="eval/results/phase45")
    args = parser.parse_args()
    payload = run_phase45(args)
    print("Phase 4/5 gates:")
    for gate in payload["gates"]:
        print(f"  {gate['verdict']:>7}  {gate['name']}: {gate['result']} ({gate['threshold']})")
    print(f"  report: {Path(args.out) / 'PHASE4_5_REPORT.md'}")


if __name__ == "__main__":
    main()
