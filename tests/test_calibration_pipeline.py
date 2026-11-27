"""Behaviour tests for the five-stage calibration robustness pipeline.

Synthetic data only: one-hot class centres with controlled noise, so the
consensus is known and the effect of each stage is attributable.
"""

from __future__ import annotations

import numpy as np
import pytest

from pss.projection.calibration_pipeline import (
    CalibrationBatch,
    Evidence,
    GateThresholds,
    Params,
    RobustCalibrationPipeline,
    State,
    bound_step,
    classify_state,
    evaluate_params,
    fit_params,
    gate,
    score_samples,
    source_health,
)
from pss.projection.online import (
    poison_agree_with_prediction,
    poison_single_class,
)

MODALITIES = ("vision", "geometry", "tactile")


def _data(
    *,
    n: int = 90,
    k: int = 3,
    d: int = 8,
    noise: float = 0.15,
    seed: int = 0,
    flip: float = 0.0,
    sources=None,
):
    """One-hot centres; returns (batch, true_labels)."""
    rng = np.random.default_rng(seed)
    centres = np.eye(k, d)
    truth = np.repeat(np.arange(k), n // k)
    truth = np.concatenate([truth, np.arange(n - len(truth))])
    truth = rng.permutation(truth)

    embeddings = {
        m: centres[truth] + rng.normal(scale=noise, size=(n, d)) for m in MODALITIES
    }
    prototypes = {m: centres.copy() for m in MODALITIES}

    labels = truth.copy()
    if flip > 0:
        idx = rng.choice(n, size=int(round(flip * n)), replace=False)
        labels[idx] = (truth[idx] + 1) % k

    srcs = list(sources) if sources is not None else ["s0"] * n
    batch = CalibrationBatch(
        embeddings=embeddings,
        prototypes=prototypes,
        labels=labels,
        sources=srcs,
        sample_ids=[f"i{i}" for i in range(n)],
    )
    return batch, truth


def _clean_params(rel: float = 0.9) -> Params:
    return Params(
        temperature={m: 1.0 for m in MODALITIES},
        reliability={m: rel for m in MODALITIES},
    )


# ── stage 2: consensus screening ────────────────────────────────────────────


def test_clean_data_has_no_mismatch() -> None:
    batch, _ = _data()
    weights = score_samples(batch, _clean_params(), {})
    assert np.mean(weights.per_sample) > 0.9
    assert weights.reasons["label_consensus_mismatch"].sum() == 0


def test_strong_consensus_mismatch_gets_near_zero_weight() -> None:
    batch, truth = _data(flip=0.4)
    weights = score_samples(batch, _clean_params(), {})
    labels = np.asarray(batch.labels)
    flipped = labels != truth
    assert flipped.sum() > 0
    # The flipped rows are the ones the consensus disagrees with.
    assert weights.per_sample[flipped].mean() < 0.1
    assert weights.per_sample[~flipped].mean() > 0.7


def test_suspect_source_is_downweighted() -> None:
    n = 90
    truth = np.repeat(np.arange(3), n // 3)
    sources = ["good"] * n
    labels = truth.copy()
    # A quarter of the batch comes from a source with badly wrong labels.
    bad_idx = np.arange(0, n, 4)
    sources = ["bad" if i in set(bad_idx.tolist()) else "good" for i in range(n)]
    labels[bad_idx] = (truth[bad_idx] + 1) % 3
    batch, _ = _data(sources=sources)
    batch = CalibrationBatch(
        embeddings=batch.embeddings,
        prototypes=batch.prototypes,
        labels=labels,
        sources=sources,
        sample_ids=list(batch.sample_ids),
    )
    trust = source_health(batch)
    assert trust["bad"] < trust["good"]


def test_small_source_is_not_fully_trusted() -> None:
    batch, _ = _data(n=30)
    sources = ["tiny"] * 2 + ["main"] * 28
    batch = CalibrationBatch(
        embeddings=batch.embeddings,
        prototypes=batch.prototypes,
        labels=batch.labels,
        sources=sources,
        sample_ids=list(batch.sample_ids),
    )
    trust = source_health(batch, min_source_n=8)
    assert trust["tiny"] < 1.0


def test_drift_state_stops_penalising_label_mismatch() -> None:
    """The two-pass fix: under drift, labels are trusted, experts are wrong."""
    batch, _ = _data(flip=0.4)
    params = _clean_params()

    suspicious = score_samples(batch, params, {}, state=State.NORMAL)
    drifting = score_samples(batch, params, {}, state=State.RECALIBRATE)

    assert suspicious.per_sample.mean() < 0.7
    assert drifting.per_sample.mean() > 0.9
    # The audit still *reports* the mismatch under drift; it just does not act
    # on it, so the signal stays observable.
    assert drifting.reasons["label_consensus_mismatch"].sum() > 0


def test_pipeline_adapts_under_drift() -> None:
    clean, _ = _data(noise=0.15)
    incumbent = _fitted(clean)
    pipeline = RobustCalibrationPipeline(params=incumbent, gate_set=clean)

    drifted, _ = _data(noise=0.55, seed=3)
    record = pipeline.step(drifted, force_gate=True)

    assert record["state"] == State.RECALIBRATE.value
    # The fit must be allowed to move: before the two-pass fix the screening
    # zeroed the model's own errors and the temperature never changed.
    for modality in MODALITIES:
        assert record["temperature"][modality] != incumbent.temperature[modality]


def test_evaluate_excludes_weak_modality() -> None:
    """A bad expert must not swamp the aggregate.

    Unweighted, tactile's near-chance ECE dominates and a change that helps
    vision/geometry can read as a regression.
    """
    base, _ = _data(n=90)
    rng = np.random.default_rng(7)
    noisy_tactile = rng.normal(size=base.embeddings["tactile"].shape)
    batch = CalibrationBatch(
        embeddings={
            "vision": base.embeddings["vision"],
            "geometry": base.embeddings["geometry"],
            "tactile": noisy_tactile,
        },
        prototypes=base.prototypes,
        labels=base.labels,
        sources=list(base.sources),
        sample_ids=list(base.sample_ids),
    )
    fitted = _fitted(batch)
    params = Params(
        temperature={**dict(fitted.temperature), "tactile": 0.005},
        reliability={"vision": 0.95, "geometry": 0.90, "tactile": 0.05},
    )
    metrics = evaluate_params(params, batch)
    assert metrics["ece"] < metrics["ece_unweighted"] - 0.01


def test_vote_weights_sum_to_one_and_drop_the_weak() -> None:
    from pss.projection.calibration_pipeline import _vote_weights

    params = Params(
        temperature={m: 1.0 for m in MODALITIES},
        reliability={"vision": 0.95, "geometry": 0.90, "tactile": 0.05},
    )
    weights = _vote_weights(params, MODALITIES)
    assert "tactile" not in weights
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_pinned_detection_and_widened_ranking_bounds() -> None:
    """The old (-3, 3) box pinned the fit; the ranking channel does not."""
    from pss.projection.calibration import (
        RANKING_LOG_BOUNDS,
        TEMPERATURE_LOG_BOUNDS,
        fit_temperature_weighted,
        temperature_at_bound,
    )
    from pss.projection.calibration_pipeline import _logits

    batch, _ = _data()
    logits = _logits(batch, "vision")
    labels = np.asarray(batch.labels)
    weights = np.ones(len(labels))

    old = fit_temperature_weighted(
        logits, labels, weights, log_bounds=TEMPERATURE_LOG_BOUNDS
    )
    new = fit_temperature_weighted(
        logits, labels, weights, log_bounds=RANKING_LOG_BOUNDS
    )
    assert temperature_at_bound(old, log_bounds=TEMPERATURE_LOG_BOUNDS)
    assert not temperature_at_bound(new, log_bounds=RANKING_LOG_BOUNDS)
    assert new < old

    # The pipeline keeps the bounded search: widening was measured to hurt
    # fused accuracy and calibration, so the pinned value is the deployed one.
    params = _fitted(batch)
    assert params.pinned != frozenset()


def test_abstain_auto_releases_after_clean_windows() -> None:
    clean, _ = _data()
    pipeline = RobustCalibrationPipeline(params=_fitted(clean), gate_set=clean)
    pipeline.state = State.ABSTAIN
    for _ in range(int(pipeline.thresholds.abstain_release_windows)):
        pipeline.step(clean)
    assert pipeline.state is State.NORMAL


def test_accuracy_cusum_is_fed_by_the_pipeline() -> None:
    from pss.projection.cusum import CusumDetector

    clean, _ = _data()
    det = CusumDetector(reference=0.90, slack=0.01, threshold=0.05, direction=-1.0)
    pipeline = RobustCalibrationPipeline(
        params=_fitted(clean), gate_set=clean, accuracy_cusum=det
    )
    rng = np.random.default_rng(0)
    bad = CalibrationBatch(
        embeddings={
            m: rng.normal(size=clean.embeddings[m].shape) for m in MODALITIES
        },
        prototypes=clean.prototypes,
        labels=clean.labels,
        sources=list(clean.sources),
        sample_ids=list(clean.sample_ids),
    )
    record = pipeline.step(bad)
    assert record["accuracy_alarm"] is True


# ── stage 3: robust fit ─────────────────────────────────────────────────────


def test_weighted_fit_keeps_reliability_under_label_flip() -> None:
    batch, _ = _data(flip=0.4)
    weights = score_samples(batch, _clean_params(), {})
    robust = fit_params(batch, weights)
    naive = fit_params(
        batch,
        type(weights)(per_sample=np.ones(len(batch.labels))),
    )
    for m in MODALITIES:
        assert robust.reliability[m] > 0.7
        assert naive.reliability[m] < robust.reliability[m]


def test_weighted_fit_on_clean_data_matches_naive() -> None:
    batch, _ = _data()
    weights = score_samples(batch, _clean_params(), {})
    robust = fit_params(batch, weights)
    naive = fit_params(batch, type(weights)(per_sample=np.ones(len(batch.labels))))
    for m in MODALITIES:
        assert abs(robust.reliability[m] - naive.reliability[m]) < 0.15


def test_all_zero_weights_fall_back() -> None:
    batch, _ = _data()
    weights = score_samples(batch, _clean_params(), {})
    zeroed = type(weights)(per_sample=np.zeros(len(batch.labels)))
    params = fit_params(batch, zeroed)
    for m in MODALITIES:
        assert params.temperature[m] == 1.0
        assert params.reliability[m] == 0.0


# ── stage 4: bounded step ───────────────────────────────────────────────────


def test_bound_step_caps_normal_movement() -> None:
    old = _clean_params()
    new = Params(
        temperature={m: 10.0 for m in MODALITIES},
        reliability={m: 0.95 for m in MODALITIES},
    )
    evidence = Evidence(0.99, 0.99, 0.0, 0.0)
    bounded = bound_step(old, new, evidence)
    for m in MODALITIES:
        assert bounded.temperature[m] == pytest.approx(1.2, rel=1e-6)


def test_bound_step_allows_drift_movement() -> None:
    old = _clean_params()
    new = Params(
        temperature={m: 10.0 for m in MODALITIES},
        reliability={m: 0.95 for m in MODALITIES},
    )
    evidence = Evidence(0.2, 0.99, 0.0, 0.0)
    bounded = bound_step(old, new, evidence)
    for m in MODALITIES:
        assert bounded.temperature[m] == pytest.approx(3.0, rel=1e-6)


def test_bound_step_freezes_when_labels_are_suspect() -> None:
    old = Params(
        temperature={m: 0.5 for m in MODALITIES},
        reliability={m: 0.8 for m in MODALITIES},
    )
    new = Params(
        temperature={m: 9.0 for m in MODALITIES},
        reliability={m: 0.1 for m in MODALITIES},
    )
    evidence = Evidence(0.99, 0.2, 0.0, 0.4)
    bounded = bound_step(old, new, evidence)
    assert dict(bounded.temperature) == dict(old.temperature)
    assert dict(bounded.reliability) == dict(old.reliability)


# ── state machine ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cross,label,expected",
    [
        (0.95, 0.95, State.NORMAL),
        (0.95, 0.40, State.QUARANTINE),
        (0.40, 0.95, State.RECALIBRATE),
        (0.40, 0.40, State.ABSTAIN),
    ],
)
def test_classify_state_matrix(cross, label, expected) -> None:
    evidence = Evidence(cross, label, 0.0, 0.0)
    assert classify_state(evidence) is expected


def test_high_drift_score_triggers_recalibrate() -> None:
    evidence = Evidence(0.95, 0.95, 0.30, 0.0)
    assert classify_state(evidence) is State.RECALIBRATE


# ── stage 5: gate ───────────────────────────────────────────────────────────


def _fitted(batch):
    weights = score_samples(batch, _clean_params(), {})
    return fit_params(batch, weights)


def _mixed_batch(n: int = 90, seed: int = 7) -> CalibrationBatch:
    """Two informative modalities and one near-random one."""
    base, _ = _data(n=n)
    rng = np.random.default_rng(seed)
    return CalibrationBatch(
        embeddings={
            "vision": base.embeddings["vision"],
            "geometry": base.embeddings["geometry"],
            "tactile": rng.normal(size=base.embeddings["tactile"].shape),
        },
        prototypes=base.prototypes,
        labels=base.labels,
        sources=list(base.sources),
        sample_ids=list(base.sample_ids),
    )


def test_gate_accepts_the_fitted_candidate() -> None:
    batch, _ = _data()
    bad = Params(
        temperature={m: 5.0 for m in MODALITIES},
        reliability={m: 0.9 for m in MODALITIES},
    )
    good = _fitted(batch)
    decision = gate(bad, good, batch, batch)
    assert decision.accept
    assert decision.integrity_ok


def test_gate_rejects_a_regression() -> None:
    """A candidate that lets the useless expert drive the decision."""
    batch = _mixed_batch()
    good = Params(
        temperature={m: 1.0 for m in MODALITIES},
        reliability={"vision": 0.95, "geometry": 0.90, "tactile": 0.05},
    )
    bad = Params(
        temperature={m: 1.0 for m in MODALITIES},
        reliability={"vision": 0.03, "geometry": 0.03, "tactile": 0.95},
    )
    decision = gate(good, bad, batch, batch)
    assert not decision.accept
    assert not decision.adaptation_ok


def test_drift_override_relaxes_integrity_but_not_adaptation() -> None:
    gate_batch, _ = _data(seed=1)
    drift_batch, _ = _data(seed=2, noise=0.6)
    old = _fitted(gate_batch)
    new = _fitted(drift_batch)
    strict = gate(old, new, gate_batch, drift_batch)
    relaxed = gate(
        old,
        new,
        gate_batch,
        drift_batch,
        thresholds=GateThresholds(drift_override=True),
    )
    # The override never invents adaptation gain; it only removes the block.
    assert relaxed.adaptation_ok == strict.adaptation_ok
    assert relaxed.accept or not relaxed.adaptation_ok


# ── driver ──────────────────────────────────────────────────────────────────


def test_pipeline_freezes_parameters_when_labels_are_poisoned() -> None:
    clean, _ = _data()
    params = _fitted(clean)
    pipeline = RobustCalibrationPipeline(params=params, gate_set=clean)

    poisoned, _ = _data(flip=0.4)
    record = pipeline.step(poisoned)

    assert record["state"] == State.QUARANTINE.value
    assert record["temperature"] == dict(params.temperature)
    assert record["reliability"] == dict(params.reliability)


def test_pipeline_adapts_on_clean_data() -> None:
    clean, _ = _data()
    params = _fitted(clean)
    pipeline = RobustCalibrationPipeline(params=params, gate_set=clean, gate_every=1)
    record = pipeline.step(clean, force_gate=True)
    assert record["gate"]["accept"]
    assert record["state"] == State.NORMAL.value


def test_abstain_is_sticky_until_released() -> None:
    clean, _ = _data()
    pipeline = RobustCalibrationPipeline(params=_fitted(clean), gate_set=clean)
    pipeline.state = State.ABSTAIN
    before = dict(pipeline.params.temperature)

    both_bad, _ = _data(flip=0.4, noise=0.9)
    record = pipeline.step(both_bad)
    assert record["state"] == State.ABSTAIN.value
    assert record["temperature"] == before

    pipeline.release_abstain(reason="human review")
    assert pipeline.state is State.NORMAL


# ── track E: targeted poison generators ─────────────────────────────────────


def test_poison_agree_with_prediction_apparent_accuracy_rises() -> None:
    batch, truth = _data(noise=0.8, n=90)
    preds = {}
    for m in MODALITIES:
        e = batch.embeddings[m] / np.linalg.norm(batch.embeddings[m], axis=1, keepdims=True)
        p = batch.prototypes[m] / np.linalg.norm(batch.prototypes[m], axis=1, keepdims=True)
        preds[m] = (e @ p.T).argmax(axis=1)
    before = float(np.mean(preds["vision"] == truth))

    poisoned = poison_agree_with_prediction(
        dict(batch.embeddings),
        dict(batch.prototypes),
        truth,
        fraction=0.5,
        modality="vision",
        rng=np.random.default_rng(0),
    )
    after = float(np.mean(preds["vision"] == poisoned))
    assert after >= before
    assert after > before  # at least one wrong prediction was relabelled


def test_poison_single_class_only_touches_the_target() -> None:
    labels = np.repeat(np.arange(3), 10)
    poisoned = poison_single_class(
        labels, fraction=0.5, target_class=0, n_classes=3, rng=np.random.default_rng(0)
    )
    assert np.array_equal(poisoned[10:], labels[10:])
    assert np.sum(poisoned[:10] != 0) == 5
