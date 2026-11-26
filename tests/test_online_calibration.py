"""Tests for streaming calibration, drift detection and calibration-set audit (F6)."""

from __future__ import annotations

import numpy as np

from pss.projection.calibration import calibrate_modality
from pss.projection.online import (
    DriftThresholds,
    PoisonThresholds,
    StreamingCalibrator,
    StreamingModalityCalibration,
    audit_calibration_set,
    calibrate_drift_thresholds,
    calibrated_test_metrics,
    poison_duplicate_inject,
    poison_embeddings_replace,
    poison_labels,
    select_balanced_subset,
)


def _toy(seed: int = 0, n_classes: int = 6, dim: int = 16, n: int = 60):
    rng = np.random.default_rng(seed)
    protos = rng.normal(size=(n_classes, dim))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    labels = rng.integers(0, n_classes, size=n)
    query = protos[labels] + 0.15 * rng.normal(size=(n, dim))
    return query, protos, labels


def test_ece_zero_for_perfectly_calibrated_bins():
    from pss.projection.calibration import expected_calibration_error

    conf = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8])
    ok = np.array([1, 0, 0, 0, 0, 1, 1, 1, 1, 0])
    assert expected_calibration_error(conf, ok, n_bins=10) < 1e-9


def test_streaming_matches_one_shot_in_one_batch():
    query, protos, labels = _toy()
    one = calibrate_modality(query, protos, labels, "m")
    stream = StreamingModalityCalibration("m", len(protos), window_size=len(labels))
    stream.update_queries(query, protos, labels)
    assert np.isclose(one.temperature, stream.temperature, rtol=1e-6, atol=1e-6)
    assert np.isclose(one.reliability, stream.reliability, rtol=1e-6, atol=1e-6)
    assert np.isclose(one.val_accuracy, stream.accuracy, rtol=1e-6, atol=1e-6)


def test_streaming_per_sample_matches_batch():
    query, protos, labels = _toy(seed=1)
    batch = StreamingModalityCalibration("m", len(protos), window_size=len(labels))
    batch.update_queries(query, protos, labels)
    per = StreamingModalityCalibration("m", len(protos), window_size=len(labels))
    for i in range(len(labels)):
        per.update_queries(query[i], protos, labels[i : i + 1])
    assert np.isclose(batch.temperature, per.temperature, rtol=1e-6, atol=1e-6)
    assert np.isclose(batch.reliability, per.reliability, rtol=1e-6, atol=1e-6)
    assert np.isclose(batch.accuracy, per.accuracy, rtol=1e-6, atol=1e-6)


def test_window_is_bounded():
    query, protos, labels = _toy(seed=2, n=40)
    stream = StreamingModalityCalibration("m", len(protos), window_size=8)
    stream.update_queries(query, protos, labels)
    assert stream.window_n == 8
    assert stream.total_seen == 40


def test_clean_stream_does_not_drift_but_noise_does():
    query, protos, labels = _toy(seed=3, n=80)
    clean = StreamingCalibrator(["m"], window_size=64)
    clean.update({"m": query[:40]}, {"m": protos}, labels[:40])
    clean.set_reference()
    for i in range(40, 80):
        clean.update({"m": query[i]}, {"m": protos}, labels[i : i + 1])
    clean_report = clean.drift_report()
    assert clean_report["action"] == "continue"

    noisy = StreamingCalibrator(["m"], window_size=64)
    noisy.update({"m": query[:40]}, {"m": protos}, labels[:40])
    noisy.set_reference()
    rng = np.random.default_rng(4)
    for i in range(40, 80):
        q = query[i] + 2.0 * rng.normal(size=query.shape[1])
        noisy.update({"m": q}, {"m": protos}, labels[i : i + 1])
    noisy_report = noisy.drift_report()
    assert noisy_report["drift"]
    assert noisy_report["action"] in {"recalibrate", "abstain"}
    assert noisy_report["score"] >= 1.0


def test_unknown_labels_count_as_errors_and_trigger_abstention():
    query, protos, labels = _toy(seed=5, n=80)
    cal = StreamingCalibrator(["m"], window_size=64)
    cal.update({"m": query[:40]}, {"m": protos}, labels[:40])
    cal.set_reference()
    unknown = np.full(40, -1, dtype=np.int64)
    for i in range(40, 80):
        cal.update({"m": query[i]}, {"m": protos}, unknown[i - 40 : i - 39])
    report = cal.drift_report()
    assert report["abstain"]
    assert report["action"] == "abstain"


def test_calibrate_drift_thresholds_is_floored():
    deviations = [
        {"accuracy_drop": 0.01, "ece_increase": 0.01, "reliability_drop": 0.01, "coverage_drop": 0.0}
    ]
    thresholds = calibrate_drift_thresholds(deviations)
    assert thresholds.accuracy_drop >= DriftThresholds().accuracy_drop
    assert thresholds.ece_increase >= DriftThresholds().ece_increase


def test_poison_audit_detects_label_flips():
    query, protos, labels = _toy(seed=6, n=120)
    clean = audit_calibration_set(query, protos, labels)
    assert clean.decision == "clean"
    poisoned = poison_labels(labels, 0.35, len(protos), rng=np.random.default_rng(7))
    audit = audit_calibration_set(query, protos, poisoned, baseline=clean)
    assert audit.detected
    assert audit.decision in {"recalibrate", "abstain"}
    assert audit.excess["label_mismatch"] > 0.1


def test_poison_audit_detects_embedding_replacement():
    query, protos, labels = _toy(seed=8, n=120)
    clean = audit_calibration_set(query, protos, labels)
    replaced = poison_embeddings_replace(
        {"m": query}, labels, 0.35, rng=np.random.default_rng(9)
    )["m"]
    audit = audit_calibration_set(replaced, protos, labels, baseline=clean)
    assert audit.detected
    assert audit.decision in {"recalibrate", "abstain"}


def test_poison_audit_detects_duplicate_injection():
    query, protos, labels = _toy(seed=10, n=120)
    clean = audit_calibration_set(query, protos, labels)
    emb, lab = poison_duplicate_inject(
        {"m": query}, labels, 0.35, rng=np.random.default_rng(11)
    )
    audit = audit_calibration_set(emb["m"], protos, lab, baseline=clean)
    assert audit.detected
    assert audit.excess["duplicate"] > 0.1


def test_poison_audit_abstains_on_heavy_label_poisoning():
    query, protos, labels = _toy(seed=12, n=120)
    clean = audit_calibration_set(query, protos, labels)
    poisoned = poison_labels(labels, 0.8, len(protos), rng=np.random.default_rng(13))
    audit = audit_calibration_set(query, protos, poisoned, baseline=clean)
    assert audit.detected
    assert audit.decision == "abstain"


def test_select_balanced_subset_covers_classes():
    labels = np.repeat(np.arange(8), 10)
    idx = select_balanced_subset(labels, 8, seed=0)
    assert len(idx) == 8
    assert set(labels[idx].tolist()) == set(range(8))
    idx16 = select_balanced_subset(labels, 16, seed=0)
    counts = np.bincount(labels[idx16], minlength=8)
    assert counts.min() == 2 and counts.max() == 2


def test_calibrated_test_metrics_shape():
    query, protos, labels = _toy(seed=14, n=80)
    cal = {"m": calibrate_modality(query[:40], protos, labels[:40], "m")}
    metrics = calibrated_test_metrics({"m": query[40:]}, {"m": protos}, labels[40:], cal)
    assert "fused" in metrics and "m" in metrics["per_modality"]
    assert 0.0 <= metrics["fused"]["accuracy"] <= 1.0
    assert np.isfinite(metrics["fused"]["nll"])
