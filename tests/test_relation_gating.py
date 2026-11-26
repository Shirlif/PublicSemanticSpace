"""Tests for the F8 relation gate and the gating experiment helpers."""

from __future__ import annotations

import numpy as np
import torch

from pss.eval.relational_bench import NAMES, REGIONS
from pss.eval.scripts.run_gap_relation_gating import (
    COORDS,
    INDEX,
    _choose_expert,
    _feature_names,
    _fit_ood,
    _gate_features,
    _hull_distance,
    _ood_features,
    _ood_scores,
    _priority_bias,
    run_primary,
)
from pss.memory.typed_relations import SpatialRelationGate


def test_spatial_gate_forward_and_correctness():
    gate = SpatialRelationGate(in_dim=8, n_experts=3, hidden=8)
    x = torch.randn(5, 8)
    logits = gate(x)
    assert logits.shape == (5, 3)
    probs = gate.correctness(x)
    assert probs.shape == (5, 3)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_gate_priority_prefers_geometry_on_ties():
    bias = _priority_bias((0.25,))
    # Equal scores: geometry(0.25) wins over learned.
    chosen = _choose_expert(np.array([[0.5, 0.5], [0.4, 0.6]]), bias)
    assert chosen.tolist() == [0, 1]


def test_gate_features_and_noise_ablation_dimensions():
    pa = np.array([[0.1, 0.0]], dtype=np.float32)
    pb = np.array([[0.9, 0.1]], dtype=np.float32)
    lp = np.array([[0.1, 0.9, 0.2, 0.8]], dtype=np.float32)
    full = _gate_features(pa, pb, lp, COORDS, include_noise=True)
    no_noise = _gate_features(pa, pb, lp, COORDS, include_noise=False)
    assert full.shape[1] == len(_feature_names(True))
    assert no_noise.shape[1] == len(_feature_names(False))
    assert full.shape[1] == no_noise.shape[1] + 3
    # The noise estimate is positive and the ratio features are finite.
    assert full[0, 4] > 0.0
    assert np.isfinite(full[0, 5]) and np.isfinite(full[0, 6])


def test_hull_distance_separates_extrapolation_from_interpolation():
    train_centers = COORDS[[INDEX[n] for n in ("A", "B", "C")]]
    # G is inside the A/B/C triangle; D is outside it.
    g = np.array([[0.0, 0.0]])
    d = np.array([[0.0, -1.0]])
    assert float(_hull_distance(g, train_centers)[0]) == 0.0
    assert float(_hull_distance(d, train_centers)[0]) > 0.5


def test_ood_detector_flags_unseen_region():
    rng = np.random.default_rng(0)
    centers = COORDS
    train_centers = COORDS[[INDEX[n] for n in ("A", "B", "C")]]
    train_rows = []
    for a, b in (("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "C"), ("C", "B")):
        ca, cb = COORDS[INDEX[a]], COORDS[INDEX[b]]
        pa = ca + rng.normal(scale=0.05, size=(20, 2)).astype(np.float32)
        pb = cb + rng.normal(scale=0.05, size=(20, 2)).astype(np.float32)
        lp = np.full((20, 4), 0.5, dtype=np.float32)
        train_rows.append(_ood_features(pa, pb, lp, centers, train_centers))
    X_train = np.concatenate(train_rows, axis=0)
    stats = _fit_ood(X_train, coverage_col=X_train.shape[1] - 1)
    assert stats.coverage_threshold > 0.0

    # D and G are outside / far from the A/B/C training coverage.
    pa = np.array([[0.0, -1.0]], dtype=np.float32)  # D
    pb = np.array([[0.0, 0.0]], dtype=np.float32)   # G
    lp = np.full((1, 4), 0.5, dtype=np.float32)
    X_test = _ood_features(pa, pb, lp, centers, train_centers)
    assert X_test[0, -1] > stats.coverage_threshold
    _, flagged = _ood_scores(X_test, stats)
    assert bool(flagged[0])


def test_run_primary_smoke():
    per_seed = run_primary(
        seeds=1, n_train=2, n_test=2, head_epochs=1, gate_epochs=1, device="cpu"
    )
    assert len(per_seed) == 1
    run = per_seed[0]
    assert set(run["by_noise"]) == {"0.0", "0.03", "0.08", "0.15"}
    row = run["by_noise"]["0.15"]
    for method in ("gate_binary", "gate_binary_ood", "gate_multi", "gate_ablation"):
        assert method in row["correct"]
        assert row["correct"][method].dtype == bool
    assert row["fallback"].dtype == bool
    assert "binary" in row["chosen"] and "multi" in row["chosen"]
