"""Gap: calibration robustness pipeline (Wave 1 fusion).

Runs the five-stage pipeline against the pre-registered scenario table and
reports, for each scenario, the state, whether the update was accepted, and the
resulting calibration on the held-out test split.

Usage
-----
    python -m pss.eval.scripts.run_gap_calibration_robust --source synthetic
    python -m pss.eval.scripts.run_gap_calibration_robust --source real \
        --data data/category_pretrain --experts checkpoints/category_pretrain_experts \
        --projections checkpoints/category_pretrain_projection
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss.projection.calibration_pipeline import (
    CalibrationBatch,
    Params,
    RobustCalibrationPipeline,
    State,
    bootstrap_params,
    evaluate_params,
    fit_confidence_maps,
    fit_params,
    fused_probabilities,
    score_samples,
    source_health,
)
from pss.projection.conformal import evaluate_conformal
from pss.projection.calibration_preregistration import check_robustness_preregistration
from pss.projection.online import (
    poison_agree_with_prediction,
    poison_duplicate_inject,
    poison_embeddings_replace,
    poison_labels,
    poison_single_class,
)

SCENARIOS: list[dict] = [
    {"name": "clean", "kind": "none"},
    {"name": "random_label_flip@0.05", "kind": "label_flip", "fraction": 0.05},
    {"name": "random_label_flip@0.10", "kind": "label_flip", "fraction": 0.10},
    {"name": "random_label_flip@0.20", "kind": "label_flip", "fraction": 0.20},
    {"name": "random_label_flip@0.40", "kind": "label_flip", "fraction": 0.40},
    {"name": "embedding_replace@0.10", "kind": "embed", "fraction": 0.10},
    {"name": "embedding_replace@0.20", "kind": "embed", "fraction": 0.20},
    {"name": "embedding_replace@0.40", "kind": "embed", "fraction": 0.40},
    {"name": "duplicate_inject@0.20", "kind": "duplicate", "fraction": 0.20},
    {"name": "duplicate_inject@0.40", "kind": "duplicate", "fraction": 0.40},
    {"name": "targeted_agree@0.10", "kind": "agree", "fraction": 0.10},
    {"name": "targeted_single_class@0.20", "kind": "single", "fraction": 0.20},
    {"name": "source_outlier@0.30", "kind": "source", "fraction": 0.30},
    {"name": "drift_sigma@0.10", "kind": "drift", "sigma": 0.10},
    {"name": "drift_sigma@0.20", "kind": "drift", "sigma": 0.20},
    {"name": "drift_sigma@0.30", "kind": "drift", "sigma": 0.30},
    {"name": "ood_unseen", "kind": "ood"},
    {"name": "label_shift@4of16", "kind": "label_shift", "classes": 4},
]


def _load(args) -> dict:
    if args.source == "synthetic":
        from pss.eval.scripts.run_gap_online_calibration import synthetic_context

        return synthetic_context(seed=args.seed)
    from pss.eval.scripts.run_gap_online_calibration import load_real_context

    return load_real_context(
        args.data,
        args.experts,
        args.projections,
        device=args.device,
        max_per_class=args.max_per_class,
    )


def _split(ctx: dict) -> tuple[tuple, tuple, tuple]:
    """Stratified 50/20/30 split: fit / holdout / gate.

    Stratification matters: an unstratified split gives the small gate set a
    different class mix, which reads as a large accuracy gap and false-flags
    clean data.  The gate set is kept at ~30% so it is large enough to compare
    against (the design suggests 32-64 samples).
    """
    val = np.asarray(ctx["val_idx"], dtype=np.int64)
    y = np.asarray(ctx["y_val"], dtype=np.int64)
    fit_pos: list[int] = []
    hold_pos: list[int] = []
    gate_pos: list[int] = []
    for cls in sorted(set(int(v) for v in y)):
        pos = np.where(y == cls)[0]
        perm = np.random.default_rng(cls).permutation(len(pos))
        pos = pos[perm]
        n = len(pos)
        a = max(1, int(round(n * 0.5)))
        b = max(a + 1, int(round(n * 0.7)))
        fit_pos.extend(pos[:a].tolist())
        hold_pos.extend(pos[a:b].tolist())
        gate_pos.extend(pos[b:].tolist())
    fit_pos = sorted(fit_pos)
    hold_pos = sorted(hold_pos)
    gate_pos = sorted(gate_pos)
    return (
        (val[fit_pos], y[fit_pos]),
        (val[hold_pos], y[hold_pos]),
        (val[gate_pos], y[gate_pos]),
    )


def _batch(ctx: dict, idx, labels, sources=None) -> CalibrationBatch:
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    modalities = ctx["modalities"]
    return CalibrationBatch(
        embeddings={
            m: np.asarray(ctx["Z"][m][idx], dtype=np.float64) for m in modalities
        },
        prototypes={
            m: np.asarray(ctx["prototypes"][m], dtype=np.float64) for m in modalities
        },
        labels=np.asarray(labels, dtype=np.int64),
        sources=list(sources) if sources is not None else ["s0"] * len(idx),
        sample_ids=[f"s{int(i)}" for i in idx],
    )


def _incumbent(ctx: dict, batch: CalibrationBatch) -> Params:
    seed = bootstrap_params(batch)
    return fit_params(batch, score_samples(batch, seed, source_health(batch)))


def _run_scenario(spec: dict, ctx: dict, rng: np.random.Generator) -> dict:
    (fit_idx, fit_y), (hold_idx, hold_y), (gate_idx, gate_y) = _split(ctx)
    modalities = ctx["modalities"]
    n_classes = len(ctx["classes"])
    prototypes = {
        m: np.asarray(ctx["prototypes"][m], dtype=np.float64) for m in modalities
    }

    # Evaluation starts from the clean test split and is mutated only by
    # scenarios that change the input distribution (drift, label shift).
    # Poisoning the calibration set never touches the test set.
    test_idx = np.asarray(ctx["test_idx"], dtype=np.int64)
    test_labels = np.asarray(ctx["y_test"], dtype=np.int64).copy()
    test_emb = {
        m: np.asarray(ctx["Z"][m][test_idx], dtype=np.float64).copy()
        for m in modalities
    }
    evaluation = "clean_test"

    fit_emb = {
        m: np.asarray(ctx["Z"][m][fit_idx], dtype=np.float64).copy() for m in modalities
    }
    labels = fit_y.copy()
    sources = ["s0"] * len(labels)

    hold_emb = {
        m: np.asarray(ctx["Z"][m][hold_idx], dtype=np.float64).copy()
        for m in modalities
    }
    hold_labels = hold_y.copy()

    gate_batch = _batch(ctx, gate_idx, gate_y)
    clean_fit_batch = _batch(ctx, fit_idx, labels)

    kind = spec["kind"]
    if kind == "ood":
        unseen = np.asarray(ctx["unseen_idx"], dtype=np.int64)
        cut = max(1, int(round(len(unseen) * 0.75)))
        fit_idx_o, hold_idx_o = unseen[:cut], unseen[cut:]
        fit_batch = _batch(ctx, fit_idx_o, np.full(len(fit_idx_o), -1))
        hold_batch = _batch(ctx, hold_idx_o, np.full(len(hold_idx_o), -1))
        incumbent = _incumbent(ctx, clean_fit_batch)
    else:
        fraction = float(spec.get("fraction", 0.0))
        if kind == "label_flip":
            labels = poison_labels(labels, fraction, n_classes, rng=rng)
        elif kind == "embed":
            fit_emb = poison_embeddings_replace(fit_emb, labels, fraction, rng=rng)
        elif kind == "duplicate":
            fit_emb, labels = poison_duplicate_inject(fit_emb, labels, fraction, rng=rng)
            sources = ["s0"] * len(labels)
        elif kind == "agree":
            labels = poison_agree_with_prediction(
                fit_emb, prototypes, labels, fraction, rng=rng
            )
            # If the model has no wrong predictions there is nothing to
            # relabel and the attack is infeasible on this stream.  Record that
            # rather than reporting a spurious "not detected".
            if np.array_equal(labels, fit_y):
                return {
                    "scenario": spec["name"],
                    "kind": kind,
                    "state": "infeasible",
                    "detected": None,
                    "abstain": None,
                    "accept_update": None,
                    "suspect_mass": float("nan"),
                    "cross_modal_agreement": float("nan"),
                    "label_consensus_agreement": float("nan"),
                    "drift_score": float("nan"),
                    "gate_reason": "attack infeasible: model has no errors to relabel",
                    "ece": float("nan"),
                    "nll": float("nan"),
                    "accuracy": float("nan"),
                    "mean_confidence": float("nan"),
                    "signed_gap": float("nan"),
                    "temperature": {},
                    "reliability": {},
                }
        elif kind == "single":
            labels = poison_single_class(
                labels, fraction, 0, n_classes, rng=rng
            )
        elif kind == "source":
            bad = np.arange(0, len(labels), max(1, int(round(1.0 / 0.3))))
            sources = ["bad" if i in set(bad.tolist()) else "good" for i in range(len(labels))]
            labels = labels.copy()
            labels[bad] = (labels[bad] + 1) % n_classes
        elif kind == "drift":
            sigma = float(spec.get("sigma", 0.0))
            for m in modalities:
                fit_emb[m] = fit_emb[m] + sigma * rng.normal(size=fit_emb[m].shape)
                hold_emb[m] = hold_emb[m] + sigma * rng.normal(size=hold_emb[m].shape)
                # The evaluation set must live in the same world as the
                # calibration window, otherwise the adaptation failure is
                # invisible: ECE on a clean test set stays perfect.
                test_emb[m] = test_emb[m] + sigma * rng.normal(size=test_emb[m].shape)
            evaluation = "drifted_test"
        elif kind == "label_shift":
            keep = int(spec.get("classes", 4))
            mask = np.isin(labels, np.arange(keep))
            fit_emb = {m: fit_emb[m][mask] for m in modalities}
            labels = labels[mask]
            sources = [s for s, k in zip(sources, mask) if k]
            hold_mask = np.isin(hold_labels, np.arange(keep))
            hold_emb = {m: hold_emb[m][hold_mask] for m in modalities}
            hold_labels = hold_labels[hold_mask]
            test_mask = np.isin(test_labels, np.arange(keep))
            test_emb = {m: test_emb[m][test_mask] for m in modalities}
            test_labels = test_labels[test_mask]
            evaluation = "label_shift_test"

        fit_batch = CalibrationBatch(
            embeddings=fit_emb,
            prototypes=prototypes,
            labels=labels,
            sources=sources,
            sample_ids=[f"f{i}" for i in range(len(labels))],
        )
        hold_batch = CalibrationBatch(
            embeddings=hold_emb,
            prototypes=prototypes,
            labels=hold_labels,
            sources=["s0"] * len(hold_labels),
            sample_ids=[f"h{i}" for i in range(len(hold_labels))],
        )
        incumbent = _incumbent(ctx, clean_fit_batch)

    pipeline = RobustCalibrationPipeline(params=incumbent, gate_set=gate_batch)
    record = pipeline.step(fit_batch, window_holdout=hold_batch, force_gate=True)

    test_batch = CalibrationBatch(
        embeddings=test_emb,
        prototypes=prototypes,
        labels=test_labels,
        sources=["s0"] * len(test_labels),
        sample_ids=[f"t{i}" for i in range(len(test_labels))],
    )
    clean_test_batch = _batch(ctx, ctx["test_idx"], ctx["y_test"])

    # The confidence channel is refitted alongside the calibration, but only on
    # a window the audit accepted.  A quarantined window would poison the map
    # exactly as it would poison the temperature.
    if record["state"] in ("quarantine", "abstain"):
        map_source = clean_fit_batch
    else:
        map_source = fit_batch
    try:
        map_state = State(record["state"])
    except ValueError:
        map_state = State.NORMAL
    map_weights = score_samples(
        map_source,
        pipeline.params,
        source_health(map_source),
        state=map_state,
    )
    confidence_maps = fit_confidence_maps(
        pipeline.params, map_source, weights=map_weights
    )

    metrics = evaluate_params(
        pipeline.params, test_batch, confidence_maps=confidence_maps
    )
    metrics_clean = evaluate_params(
        pipeline.params, clean_test_batch, confidence_maps=confidence_maps
    )
    metrics_no_adapt = evaluate_params(
        incumbent, test_batch, confidence_maps=confidence_maps
    )

    # Conformal channel: calibration window -> test stream, coverage guarantee
    # instead of a fitted confidence.
    calib_probs, calib_labels = fused_probabilities(pipeline.params, map_source)
    test_probs, test_labels = fused_probabilities(pipeline.params, test_batch)
    if len(calib_probs) and len(test_probs):
        conformal = evaluate_conformal(
            calib_probs, calib_labels, test_probs, test_labels,
            alpha_target=0.10, gamma=0.01,
        )
    else:
        conformal = None

    # Detection is relative to this stream's clean flag rate: real embeddings
    # carry a few percent of natural disagreement, so an absolute floor would
    # flag every clean window.
    ref_weights = score_samples(
        clean_fit_batch, incumbent, source_health(clean_fit_batch)
    )
    reference_mass = float(1.0 - np.mean(ref_weights.per_sample))
    duplicate_rate = float(record.get("reason_rates", {}).get("duplicate", 0.0))
    detected = (
        record["state"] != "normal"
        or float(record["suspect_mass"]) >= reference_mass + 0.05
        # Duplicates are reported but not down-weighted, so they never raise the
        # suspect mass; they still count as detected by the audit.
        or duplicate_rate >= 0.10
    )
    return {
        "scenario": spec["name"],
        "kind": kind,
        "state": record["state"],
        "detected": bool(detected),
        "abstain": record["state"] == "abstain",
        "accept_update": bool(record["gate"]["accept"]),
        "suspect_mass": float(record["suspect_mass"]),
        "cross_modal_agreement": float(record["cross_modal_agreement"]),
        "label_consensus_agreement": float(record["label_consensus_agreement"]),
        "drift_score": float(record["drift_score"]),
        "accuracy_inflation": float(record["accuracy_inflation"]),
        "class_agreement_spread": float(record["class_agreement_spread"]),
        "label_coverage": float(record["label_coverage"]),
        "gate_reason": record["gate"]["reason"],
        "ece": float(metrics["ece"]),
        "ece_map": float(metrics["ece_map"]),
        "ece_map_fused": float(metrics["ece_map_fused"]),
        "ece_clean": float(metrics_clean["ece"]),
        "ece_no_adapt": float(metrics_no_adapt["ece"]),
        "evaluation": evaluation,
        "nll": float(metrics["nll"]),
        "ece_map_fused": float(metrics["ece_map_fused"]),
        "signed_gap_map_fused": float(metrics["signed_gap_map_fused"]),
        "fused_accuracy": float(metrics["fused_accuracy"]),
        "conformal_coverage": float(conformal.coverage) if conformal else float("nan"),
        "conformal_mean_size": float(conformal.mean_size) if conformal else float("nan"),
        "conformal_target": float(conformal.target_coverage) if conformal else float("nan"),
        "accuracy": float(metrics["accuracy"]),
        "mean_confidence": float(metrics["mean_confidence"]),
        "signed_gap": float(metrics["signed_gap"]),
        "temperature": dict(record["temperature"]),
        "reliability": dict(record["reliability"]),
        "pinned": list(record.get("pinned", [])),
    }


def _report(metrics: dict) -> str:
    rows = metrics["scenarios"]
    verdict = metrics["preregistration"]
    lines = [
        "# Gap: calibration robustness pipeline (Wave 1)",
        "",
        f"> generated: {metrics['generated']}",
        f"> source: {metrics['source']}",
        f"> scenarios: {len(rows)}",
        "",
        "## Pre-registration",
        "",
        f"- ok: **{verdict['ok']}**",
    ]
    for mismatch in verdict["mismatches"]:
        lines.append(f"  - {mismatch}")
    lines += [
        "",
        "## Scenarios",
        "",
        "`ECE` is measured on the scenario's own test set; for drift and label-shift",
        "scenarios that set is transformed the same way as the calibration window, so",
        "an adaptation failure is visible instead of hidden by a clean test set.",
        "`ECE(no adapt)` is the incumbent calibration on the same set: when the two are",
        "equal, the pipeline did not adapt at all.",
        "",
        "| scenario | eval | state | detected | accept | ECE(T) | ECE(map) | coverage | set size | ECE(no adapt) | reason |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['scenario']} | {r['evaluation']} | {r['state']} | {r['detected']} | "
            f"{r['accept_update']} | {r['ece']:.4f} | {r['ece_map_fused']:.4f} | "
            f"{r['conformal_coverage']:.3f} | {r['conformal_mean_size']:.2f} | "
            f"{r['ece_no_adapt']:.4f} | {r['gate_reason']} |"
        )
    lines += [
        "",
        "## How to read",
        "",
        "- `detected` = the audit flagged mass above the clean floor, or the state is not NORMAL.",
        "- `accept_update` is the gate decision. Low poison fractions may accept because the",
        "  weighted fit already neutralised the rows; the guarantee there is `ECE` staying flat.",
        "- `signed gap = mean_confidence - accuracy`: negative is underconfident.",
        "",
    ]
    return "\n".join(lines)


def run(args) -> dict:
    ctx = _load(args)
    # Per-scenario RNG: results must not depend on scenario order, otherwise
    # reordering or dropping a scenario silently changes every later number.
    rows = [
        _run_scenario(spec, ctx, np.random.default_rng([int(args.seed), i]))
        for i, spec in enumerate(SCENARIOS)
    ]
    observed = {}
    for r in rows:
        entry = {
            key: r[key]
            for key in (
                "detected",
                "abstain",
                "accept_update",
                "ece_map_fused",
                "signed_gap_map_fused",
                "fused_accuracy",
            )
            if r.get(key) is not None
        }
        observed[r["scenario"]] = entry
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": ctx.get("kind", "unknown"),
        "counts": ctx.get("counts", {}),
        "scenarios": rows,
        "preregistration": check_robustness_preregistration(observed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--data", default="data/category_pretrain")
    parser.add_argument("--experts", default="checkpoints/category_pretrain_experts")
    parser.add_argument("--projections", default="checkpoints/category_pretrain_projection")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/claim_gaps/calibration_robust")
    args = parser.parse_args()

    metrics = run(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    print(_report(metrics))


if __name__ == "__main__":
    main()
