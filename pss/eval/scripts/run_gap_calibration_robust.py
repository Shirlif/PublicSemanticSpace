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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss.projection.calibration_pipeline import (
    CalibrationBatch,
    ConfidenceGateThresholds,
    Params,
    RobustCalibrationPipeline,
    SampleWeights,
    State,
    TransitionThresholds,
    bootstrap_params,
    calibrate_confidence_thresholds,
    calibrate_thresholds_from_batches,
    classify_state,
    compare_confidence_maps,
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


def _batch_from(ctx: dict, embeddings: dict, labels) -> CalibrationBatch:
    """Batch from explicit embeddings (a rolling window, possibly noised)."""
    modalities = ctx["modalities"]
    labels = np.asarray(labels, dtype=np.int64)
    return CalibrationBatch(
        embeddings={
            m: np.asarray(embeddings[m], dtype=np.float64) for m in modalities
        },
        prototypes={
            m: np.asarray(ctx["prototypes"][m], dtype=np.float64) for m in modalities
        },
        labels=labels,
        sources=["s0"] * len(labels),
        sample_ids=[f"w{i}" for i in range(len(labels))],
    )


def _balanced_order(labels, seed: int) -> np.ndarray:
    """Round-robin class order so a rolling window stays class-balanced."""
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    by_class = {
        int(c): list(np.where(labels == c)[0]) for c in sorted(set(labels.tolist()))
    }
    for c in by_class:
        rng.shuffle(by_class[c])
    order: list[int] = []
    while any(by_class.values()):
        for c in sorted(by_class):
            if by_class[c]:
                order.append(by_class[c].pop())
    return np.asarray(order, dtype=np.int64)


def _clean_batches(
    ctx: dict,
    fit_idx,
    fit_y,
    *,
    seed: int,
    n_windows: int = 50,
) -> list[CalibrationBatch]:
    """Bootstrap clean windows; shared by every clean-stream calibration."""
    idx = np.asarray(fit_idx, dtype=np.int64)
    y = np.asarray(fit_y, dtype=np.int64)
    rng = np.random.default_rng([int(seed), 0xC1])
    batches = []
    for _ in range(int(n_windows)):
        pick = rng.integers(0, len(idx), size=len(idx))
        batches.append(_batch(ctx, idx[pick], y[pick]))
    return batches


def _calibrated_thresholds(
    ctx: dict,
    params: Params,
    gate_batch: CalibrationBatch,
    fit_idx,
    fit_y,
    *,
    seed: int,
    n_windows: int = 50,
) -> TransitionThresholds:
    """Read the state thresholds off bootstrap clean windows (design §6)."""
    return calibrate_thresholds_from_batches(
        params,
        gate_batch,
        _clean_batches(ctx, fit_idx, fit_y, seed=seed, n_windows=n_windows),
        quantile=0.99,
    )


def _clean_stream(
    ctx: dict,
    params: Params,
    gate_batch: CalibrationBatch,
    fit_idx,
    fit_y,
    thresholds: TransitionThresholds,
    *,
    seed: int,
    windows: int = 100,
) -> dict:
    """Clean-stream false-alarm rates for the gate and the state machine.

    Every window is forced through the gate, which is stricter than production
    (which gates every 32 steps).  A rejection here is a false positive on
    clean data, the cost the design caps at 5%.
    """
    idx = np.asarray(fit_idx, dtype=np.int64)
    y = np.asarray(fit_y, dtype=np.int64)
    pipeline = RobustCalibrationPipeline(
        params=params, gate_set=gate_batch, thresholds=thresholds
    )
    rng = np.random.default_rng([int(seed), 0xC2])
    rejects = 0
    alarms = 0
    reasons: dict[str, int] = {}
    for _ in range(int(windows)):
        pick = rng.integers(0, len(idx), size=len(idx))
        batch = _batch(ctx, idx[pick], y[pick])
        record = pipeline.step(batch, force_gate=True)
        if not record["gate"]["accept"]:
            rejects += 1
            reasons[record["gate"]["reason"]] = (
                reasons.get(record["gate"]["reason"], 0) + 1
            )
        if record["state"] != State.NORMAL.value:
            alarms += 1
    return {
        "windows": int(windows),
        "reject_rate": float(rejects / max(1, windows)),
        "reject_target": 0.05,
        "reject_ok": bool(rejects / max(1, windows) < 0.05),
        "state_alarm_rate": float(alarms / max(1, windows)),
        "reject_reasons": reasons,
    }


def _recovery_latency(
    ctx: dict,
    spec: dict,
    gate_batch: CalibrationBatch,
    params: Params,
    thresholds: TransitionThresholds,
    *,
    seed: int,
    index: int,
    window_size: int = 96,
    warmup: int = 96,
    n_seeds: int = 5,
) -> dict:
    """Samples until the state leaves NORMAL after a clean warm-up.

    This is the F6-comparable detection latency: one sample is appended at a
    time to a rolling window and the state machine is probed on that window.
    No parameters are updated, so the measurement isolates detection, exactly
    as F6's latency did.

    The window is 96 samples (6/class), the same size the thresholds were
    calibrated on.  Probing a smaller or half-filled window lets sample-
    composition noise in the median prototype fit cross the drift threshold:
    measured latency then collapses to the same value for every sigma, which
    measures the protocol, not the detector.
    """
    modalities = ctx["modalities"]
    val_idx = np.asarray(ctx["val_idx"], dtype=np.int64)
    val_y = np.asarray(ctx["y_val"], dtype=np.int64)
    kind = spec["kind"]
    sigma = float(spec.get("sigma", 0.0))
    keep = int(spec.get("classes", 0))
    probe = RobustCalibrationPipeline(
        params=params, gate_set=gate_batch, thresholds=thresholds
    )

    latencies: list[int] = []
    first_states: list[str] = []
    for run in range(int(n_seeds)):
        rng = np.random.default_rng([int(seed), int(index), run, 0xD1])
        order = _balanced_order(val_y, seed=int(seed) + 1000 + run)
        warm_pos = order[: int(warmup)]
        rest_pos = order[int(warmup) :]

        if kind == "drift":
            stream_pos = rest_pos
        elif kind == "label_shift":
            stream_pos = rest_pos[val_y[rest_pos] < keep]
            if len(stream_pos) == 0:
                latencies.append(-1)
                first_states.append("infeasible")
                continue
            need = min(len(rest_pos), max(2 * int(window_size), len(stream_pos)))
            stream_pos = np.tile(
                stream_pos, int(np.ceil(need / len(stream_pos)))
            )[:need]
        else:
            stream_pos = rest_pos

        buffer_pos = list(warm_pos[-int(window_size) :])
        buffer_emb = {
            m: [
                np.asarray(ctx["Z"][m][val_idx[p]], dtype=np.float64)
                for p in buffer_pos
            ]
            for m in modalities
        }
        labels = [int(val_y[p]) for p in buffer_pos]

        latency = None
        for k, pos in enumerate(stream_pos):
            gidx = int(val_idx[pos])
            for m in modalities:
                value = np.asarray(ctx["Z"][m][gidx], dtype=np.float64)
                if kind == "drift":
                    value = value + sigma * rng.normal(size=value.shape)
                buffer_emb[m].append(value)
                if len(buffer_emb[m]) > int(window_size):
                    buffer_emb[m].pop(0)
            labels.append(int(val_y[pos]))
            if len(labels) > int(window_size):
                labels.pop(0)
            # Probe only once the rolling window is full: the thresholds were
            # calibrated on full-size windows, and a half-filled window has
            # enough sampling noise in the median prototype fit to cross the
            # drift threshold on its own.
            if len(labels) < int(window_size):
                continue
            window = _batch_from(
                ctx,
                {m: np.stack(buffer_emb[m], axis=0) for m in modalities},
                np.asarray(labels, dtype=np.int64),
            )
            evidence = probe.evidence(
                window, SampleWeights(per_sample=np.ones(len(labels)))
            )
            state = classify_state(evidence, thresholds)
            if state is not State.NORMAL:
                latency = int(k + 1)
                first_states.append(state.value)
                break
        latencies.append(latency if latency is not None else -1)
        if latency is None:
            first_states.append("not_detected")

    detected = [v for v in latencies if v > 0]
    return {
        "recovery_samples": float(np.mean(detected)) if detected else None,
        "detected_seeds": int(len(detected)),
        "n_seeds": int(n_seeds),
        "latencies": [int(v) for v in latencies],
        "first_states": first_states,
        "window_size": int(window_size),
        "warmup": int(warmup),
    }


def _run_scenario(
    spec: dict,
    ctx: dict,
    rng: np.random.Generator,
    *,
    thresholds: TransitionThresholds,
    seed: int,
    index: int,
    enforce_confidence_gate: bool = True,
    confidence_thresholds: ConfidenceGateThresholds | None = None,
) -> dict:
    if confidence_thresholds is None:
        confidence_thresholds = ConfidenceGateThresholds()

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
                    "recovery_samples": None,
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

    # The confidence channel's standard map comes from the clean window; the
    # pipeline re-derives it under the current parameters and gates every
    # candidate against it.  The gate set is never used for fitting.
    confidence_reference = clean_fit_batch
    incumbent_maps = fit_confidence_maps(
        incumbent,
        confidence_reference,
        weights=score_samples(
            confidence_reference, incumbent, source_health(confidence_reference)
        ),
    )
    pipeline = RobustCalibrationPipeline(
        params=incumbent,
        gate_set=gate_batch,
        thresholds=thresholds,
        confidence_maps=incumbent_maps,
        confidence_reference=confidence_reference,
        confidence_thresholds=confidence_thresholds,
        confidence_enforce=bool(enforce_confidence_gate),
    )
    record = pipeline.step(fit_batch, window_holdout=hold_batch, force_gate=True)
    confidence_maps = pipeline.confidence_maps
    confidence_record = record.get("confidence") or {}

    # Streaming detection latency, in samples.  Only drift-like scenarios make
    # this claim; poison scenarios are single-window claims.
    recovery = None
    if kind in ("drift", "label_shift"):
        recovery = _recovery_latency(
            ctx,
            spec,
            gate_batch,
            incumbent,
            thresholds,
            seed=seed,
            index=index,
        )

    test_batch = CalibrationBatch(
        embeddings=test_emb,
        prototypes=prototypes,
        labels=test_labels,
        sources=["s0"] * len(test_labels),
        sample_ids=[f"t{i}" for i in range(len(test_labels))],
    )
    clean_test_batch = _batch(ctx, ctx["test_idx"], ctx["y_test"])

    # Conformal channel still uses the accepted window: a quarantined window
    # would poison it exactly as it would poison the map.
    map_source = (
        clean_fit_batch
        if record["state"] in ("quarantine", "abstain")
        else fit_batch
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
        "confidence_gate": confidence_record.get("integrity", {}),
        "confidence_adapt": confidence_record.get("adaptation", {}),
        # Reject only when the candidate is too far from the frozen reference.
        # "Adaptation gain" is kept as a diagnostic, not a requirement: a
        # harmless no-op update on a clean stream must not be rejected.
        "confidence_map_integrity_ok": bool(
            (confidence_record.get("integrity") or {}).get("not_worse", False)
        ),
        "confidence_map_adapt_ok": bool(
            (confidence_record.get("adaptation") or {}).get("better", False)
        ),
        # Strict rule: reject whenever the frozen reference degrades.  It is
        # too brittle on its own: a legitimate drift map can be slightly worse
        # on the stale reference while being much better on the new world.
        "confidence_map_strict_accept": bool(
            (confidence_record.get("integrity") or {}).get("not_worse", False)
        ),
        # Recommended: keep a candidate that is either harmless on the frozen
        # reference or demonstrably better on the recent trusted slice.
        "confidence_map_accept": bool(confidence_record.get("accepted", False)),
        "confidence_map_used": confidence_record.get("used", "candidate"),
        "confidence_gate_enforced": bool(enforce_confidence_gate),
        "accuracy": float(metrics["accuracy"]),
        "mean_confidence": float(metrics["mean_confidence"]),
        "signed_gap": float(metrics["signed_gap"]),
        "temperature": dict(record["temperature"]),
        "reliability": dict(record["reliability"]),
        "pinned": list(record.get("pinned", [])),
        "recovery_samples": (
            recovery["recovery_samples"] if recovery is not None else None
        ),
        "recovery_detail": recovery,
    }


def _delta(value) -> str:
    """Signed difference for the report; an unmeasured value stays visible."""
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):+.4f}"


def _num(value) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):.4f}"


def _z(value) -> str:
    if value is None:
        return "—"
    v = float(value)
    if not np.isfinite(v):
        return "inf" if v > 0 else "—"
    return f"{v:+.1f}"


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

    thresholds = metrics.get("thresholds", {})
    clean = metrics.get("clean_stream", {})
    conf = metrics.get("confidence_thresholds", {})
    lines += [
        "",
        "## State thresholds (calibrated on the clean stream)",
        "",
        "| threshold | value |",
        "|---|---:|",
    ]
    for key in sorted(thresholds):
        lines.append(f"| `{key}` | {thresholds[key]} |")
    lines += [
        "",
        "## Clean-stream false alarms",
        "",
        f"- windows: **{clean.get('windows', 0)}** (gate forced on every window)",
        f"- gate rejection rate: **{clean.get('reject_rate', float('nan')):.3f}** "
        f"(target < 0.05 -> {'PASS' if clean.get('reject_ok') else 'FAIL'})",
        f"- state alarm rate: **{clean.get('state_alarm_rate', float('nan')):.3f}**",
        f"- reasons: {clean.get('reject_reasons', {})}",
    ]
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
        "| scenario | eval | state | detected | accept | ECE(T) | ECE(map) | coverage | set size | ECE(no adapt) | recovery | reason |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        recovery = r.get("recovery_samples")
        recovery_text = "—" if recovery is None else f"{float(recovery):.0f}"
        lines.append(
            f"| {r['scenario']} | {r['evaluation']} | {r['state']} | {r['detected']} | "
            f"{r['accept_update']} | {r['ece']:.4f} | {r['ece_map_fused']:.4f} | "
            f"{r['conformal_coverage']:.3f} | {r['conformal_mean_size']:.2f} | "
            f"{r['ece_no_adapt']:.4f} | {recovery_text} | {r['gate_reason']} |"
        )
    lines += [
        "",
        "## Confidence-channel gate (P0)",
        "",
        "The gate set is the confidence channel's standard answer; the",
        "tolerances are calibrated on clean windows.",
        f"`gap_tolerance` = {conf.get('gap_tolerance', float('nan')):.4f},",
        f"`ece_tolerance` = {conf.get('ece_tolerance', float('nan')):.4f},",
        f"`z` = {conf.get('z', 2.0):.1f}.",
        "`gap` is mean confidence minus accuracy; `dGap(gate)` is |gap|",
        "candidate minus incumbent; `z` is that difference over its paired",
        "standard error. `gate ok` = the candidate is within the calibrated",
        "band on the frozen reference; `improve(hold)` = |gap| reduction on the",
        "recent trusted slice; `adapt ok` = that reduction is significant.",
        "`would accept` = keep when `gate ok` OR `adapt ok` (a real drift update",
        "must not be blocked by the stale reference). `map used` shows the",
        "enforced choice; `--report-only-confidence-gate` disables rejection.",
        "",
        "| scenario | n | gap(inc) | gap(cand) | dGap(gate) | z | gate ok | improve(hold) | adapt ok | strict | would accept | map used |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        gate = r.get("confidence_gate", {})
        adapt = r.get("confidence_adapt", {})
        lines.append(
            f"| {r['scenario']} | {gate.get('n', 0)} | "
            f"{_num(gate.get('gap_incumbent'))} | {_num(gate.get('gap_candidate'))} | "
            f"{_delta(gate.get('gap_delta'))} | {_z(gate.get('gap_z'))} | "
            f"{gate.get('not_worse', False)} | "
            f"{_delta(adapt.get('gap_improvement'))} | {adapt.get('better', False)} | "
            f"{r.get('confidence_map_strict_accept', False)} | "
            f"{r.get('confidence_map_accept', False)} | "
            f"{r.get('confidence_map_used', 'candidate')} |"
        )
    lines += [
        "",
        "## How to read",
        "",
        "- `detected` = the audit flagged mass above the clean floor, or the state is not NORMAL.",
        "- `accept_update` is the gate decision. Low poison fractions may accept because the",
        "  weighted fit already neutralised the rows; the guarantee there is `ECE` staying flat.",
        "- `recovery` is the number of streamed samples after a clean warm-up until the",
        "  state leaves NORMAL (F6-comparable detection latency). `—` means not asserted.",
        "- `signed gap = mean_confidence - accuracy`: negative is underconfident.",
        "- `dGap` in the confidence-channel gate is |mean confidence - accuracy|",
        "  candidate minus incumbent on the same samples: negative means the",
        "  candidate is closer to the truth.",
        "",
    ]
    return "\n".join(lines)


def run(args) -> dict:
    ctx = _load(args)
    # Thresholds are read off the clean stream, not hand-picked (design §6).
    # Every scenario shares them so a scenario cannot tune its own detector.
    (fit_idx, fit_y), _hold, (gate_idx, gate_y) = _split(ctx)
    gate_batch = _batch(ctx, gate_idx, gate_y)
    clean_fit_batch = _batch(ctx, fit_idx, fit_y)
    incumbent = _incumbent(ctx, clean_fit_batch)
    clean_batches = _clean_batches(
        ctx, fit_idx, fit_y, seed=int(args.seed), n_windows=50
    )
    thresholds = calibrate_thresholds_from_batches(
        incumbent, gate_batch, clean_batches, quantile=0.99
    )
    # The confidence channel's tolerances come from the same clean windows, so
    # a scenario cannot tune its own gate either.
    reference_maps = fit_confidence_maps(incumbent, clean_fit_batch)
    confidence_thresholds = calibrate_confidence_thresholds(
        incumbent, reference_maps, clean_batches, quantile=0.99
    )
    clean_stream = _clean_stream(
        ctx, incumbent, gate_batch, fit_idx, fit_y, thresholds, seed=int(args.seed)
    )
    # Per-scenario RNG: results must not depend on scenario order, otherwise
    # reordering or dropping a scenario silently changes every later number.
    rows = [
        _run_scenario(
            spec,
            ctx,
            np.random.default_rng([int(args.seed), i]),
            thresholds=thresholds,
            seed=int(args.seed),
            index=i,
            enforce_confidence_gate=not bool(
                getattr(args, "report_only_confidence_gate", False)
            ),
            confidence_thresholds=confidence_thresholds,
        )
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
                "confidence_map_accept",
                "ece_map_fused",
                "signed_gap_map_fused",
                "fused_accuracy",
            )
        }
        # Keep the key even when None: an unmeasured or undetected latency must
        # be visible to the pre-registration checker, not silently dropped.
        entry["recovery_samples"] = r.get("recovery_samples")
        observed[r["scenario"]] = entry
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": ctx.get("kind", "unknown"),
        "counts": ctx.get("counts", {}),
        "thresholds": asdict(thresholds),
        "confidence_thresholds": asdict(confidence_thresholds),
        "threshold_source": {
            "quantile": 0.99,
            "source": "bootstrap clean windows (50)",
        },
        "clean_stream": clean_stream,
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
    parser.add_argument(
        "--report-only-confidence-gate",
        action="store_true",
        help="Compute the confidence-map gate but keep the candidate map "
        "instead of rejecting it (the decision is still reported).",
    )
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
