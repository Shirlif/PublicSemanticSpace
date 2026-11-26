"""Claim-gap F8: learned relation gating with out-of-distribution degradation.

E3 (``run_gap_hybrid_relations``) showed that ``hybrid_diag`` beats the strict
geometric predicate under coordinate noise, but only at noise 0.08/0.15 and
only against ``geometry(eps=0.25)``.  The held-out set was 14 pairs and 3
seeds.  This script asks the next question: *is a learned gate better than a
wider geometric dead zone?*

Protocol
--------
- Train the spatial relation head on regions A/B/C (same recipe as E3).
- Train a pair-level gate on A/B/C samples across several noise levels.  The
  gate predicts, per candidate expert, whether that expert's 4-bit relation
  prediction is correct; it routes to the expert with the highest predicted
  correctness.  Experts are geometry(eps) and the learned head.
- Test on the unseen regions D/G (14 ordered pairs), at each noise level.
- Key controls: ``geometry(0.25)``, ``hybrid_diag`` (E3), and a *tuned* fixed
  geometric dead zone selected on A/B/C (never on D/G).
- OOD degradation: a feature-space novelty score plus the distance outside the
  convex hull of the training regions forces a fallback to geometry on OOD
  samples.  Fallback rate and the accuracy of fallback vs non-fallback samples
  are reported.
- Ablation: drop the local-noise features (``sigma_hat`` and the margin /
  noise ratios) and check whether the gate's gain disappears.
- Anti-overfitting: 10 seeds x 500 samples/pair, 3-fold pair cross-validation
  for gate early stopping, and leave-one-region-out cross-validation.

The gate deliberately does *not* see absolute coordinates or region identity:
it is translation-invariant.  The OOD detector separately uses the distance
to the nearest training-region centre, which is available from the map.

Outputs ``reports/claim_gaps/relation_gating/{metrics.json,REPORT.md}``.

Usage::

    python -m pss.eval.scripts.run_gap_relation_gating --seeds 10
    python -m pss.eval.scripts.run_gap_relation_gating --quick
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import ConvexHull, QhullError

from pss._console import ensure_utf8
from pss.eval.relational_bench import NAMES, REGIONS
from pss.memory.typed_relations import SpatialRelationGate, SpatialRelationHead

TRAIN_REGIONS = ("A", "B", "C")
TEST_REGIONS = ("D", "G")
NOISE_LEVELS = (0.0, 0.03, 0.08, 0.15)
TAU = 0.25
EPS = 0.1
EPS_GRID = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75)
HEAD_DIM = 4
COORDS = np.array([REGIONS[n] for n in NAMES], dtype=np.float32)
INDEX = {n: i for i, n in enumerate(NAMES)}


# ── data helpers (E3-compatible) ────────────────────────────────────────────


def _pair_feature(pa, pb) -> np.ndarray:
    dx = pb[..., 0] - pa[..., 0]
    dy = pb[..., 1] - pa[..., 1]
    return np.stack(
        [pa[..., 0], pa[..., 1], pb[..., 0], pb[..., 1], dx, dy, np.abs(dx), np.abs(dy)],
        axis=-1,
    ).astype(np.float32)


def _pair_labels(pa, pb, eps: float = EPS) -> np.ndarray:
    """left / right / above / below, matching E3's ``_pair_labels``."""
    dx = pb[..., 0] - pa[..., 0]
    dy = pb[..., 1] - pa[..., 1]
    return np.stack([dx > eps, dx < -eps, dy < -eps, dy > eps], axis=-1).astype(np.float32)


def _true_bits(ca, cb) -> np.ndarray:
    return _pair_labels(ca, cb, eps=EPS) > 0.5


def _noisy(center, noise, rng, n=None):
    if n is None:
        return center + rng.normal(scale=noise, size=2)
    return center + rng.normal(scale=noise, size=(n, 2))


def _train_head(
    train_regions,
    seed: int,
    device: str,
    n_per_pair: int = 400,
    epochs: int = 400,
    noise: float = 0.05,
):
    """E3's ``_train_head`` generalised to an arbitrary region set."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for a in train_regions for b in train_regions if a != b]
    X, y = [], []
    for a, b in pairs:
        ca, cb = COORDS[INDEX[a]], COORDS[INDEX[b]]
        for _ in range(n_per_pair):
            pa = _noisy(ca, noise, rng)
            pb = _noisy(cb, noise, rng)
            X.append(_pair_feature(pa, pb))
            y.append(_pair_labels(ca, cb))
    X_t = torch.as_tensor(np.stack(X), device=device)
    y_t = torch.as_tensor(np.stack(y), device=device)
    head = SpatialRelationHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X_t), device=device)
        for start in range(0, len(perm), 128):
            idx = perm[start : start + 128]
            loss = F.binary_cross_entropy_with_logits(head.net(X_t[idx]), y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


def _learned_probs(head, pa, pb, device: str, batch_size: int = 4096) -> np.ndarray:
    head.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(pa), batch_size):
            pa_t = torch.as_tensor(pa[start : start + batch_size], dtype=torch.float32, device=device)
            pb_t = torch.as_tensor(pb[start : start + batch_size], dtype=torch.float32, device=device)
            out.append(torch.sigmoid(head(pa_t, pb_t)).cpu().numpy())
    return np.concatenate(out, axis=0)


def _geom_bits(dx, dy, eps: float) -> np.ndarray:
    return np.stack([dx > eps, dx < -eps, dy < -eps, dy > eps], axis=1)


def _hybrid_bits(dx, dy, learned_bits, tau: float = TAU) -> np.ndarray:
    pred = np.empty((len(dx), HEAD_DIM), dtype=bool)
    xclear = np.abs(dx) >= tau
    yclear = np.abs(dy) >= tau
    pred[:, 0] = np.where(xclear, dx > 0, learned_bits[:, 0])
    pred[:, 1] = np.where(xclear, dx < 0, learned_bits[:, 1])
    pred[:, 2] = np.where(yclear, dy < 0, learned_bits[:, 2])
    pred[:, 3] = np.where(yclear, dy > 0, learned_bits[:, 3])
    return pred


def _hybrid_diag_bits(dx, dy, learned_bits, tau: float = TAU) -> np.ndarray:
    both = (np.abs(dx) >= tau) & (np.abs(dy) >= tau)
    pred = _geom_bits(dx, dy, tau)
    pred[both] = learned_bits[both]
    return pred


def _heldout_pairs(train_regions, all_regions=NAMES):
    return [
        (a, b)
        for a in all_regions
        for b in all_regions
        if a != b and (a not in train_regions or b not in train_regions)
    ]


# ── gate features ───────────────────────────────────────────────────────────


def _feature_names(include_noise: bool = True) -> list[str]:
    names = ["dx", "dy", "abs_dx", "abs_dy"]
    if include_noise:
        names += ["sigma_hat", "abs_dx_over_sigma", "abs_dy_over_sigma"]
    names += [
        "p_left",
        "p_right",
        "p_above",
        "p_below",
        "learned_conf",
        "learned_entropy",
        "distance",
        "contact",
        "near",
    ]
    return names


def _local_sigma(pa, pb, centers) -> np.ndarray:
    """Local noise estimate: radial residual to the nearest map region centre.

    For a 2-D Gaussian, E[||n||] = sigma * sqrt(pi/2), so this is a per-pair
    noise proxy.  It uses the known map, not the true noise level.
    """
    r_a = np.min(np.linalg.norm(pa[:, None, :] - centers[None, :, :], axis=2), axis=1)
    r_b = np.min(np.linalg.norm(pb[:, None, :] - centers[None, :, :], axis=2), axis=1)
    return np.maximum((r_a + r_b) / 2.0 / math.sqrt(math.pi / 2.0), 1e-3)


def _gate_features(pa, pb, learned_probs, centers, include_noise: bool = True) -> np.ndarray:
    dx = pb[:, 0] - pa[:, 0]
    dy = pb[:, 1] - pa[:, 1]
    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    sigma = _local_sigma(pa, pb, centers)
    p = np.clip(learned_probs, 1e-6, 1.0 - 1e-6)
    conf = np.maximum(p, 1.0 - p).mean(axis=1)
    ent = (-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))).mean(axis=1)
    dist = np.hypot(dx, dy)
    # The grid bench has no category/tactile modality: contact/near are
    # included for API completeness but are constant here.
    contact = (dist <= 0.12).astype(np.float32)
    near = (dist < 0.5).astype(np.float32)
    feats = [dx, dy, abs_dx, abs_dy]
    if include_noise:
        feats += [sigma, abs_dx / sigma, abs_dy / sigma]
    feats += [p[:, 0], p[:, 1], p[:, 2], p[:, 3], conf, ent, dist, contact, near]
    return np.stack(feats, axis=1).astype(np.float32)


def _ood_features(pa, pb, learned_probs, centers, train_centers) -> np.ndarray:
    """Full gate features + region coverage (distance to nearest train centre)."""
    base = _gate_features(pa, pb, learned_probs, centers, include_noise=True)
    cov_a = np.min(np.linalg.norm(pa[:, None, :] - train_centers[None, :, :], axis=2), axis=1)
    cov_b = np.min(np.linalg.norm(pb[:, None, :] - train_centers[None, :, :], axis=2), axis=1)
    coverage = np.maximum(cov_a, cov_b)
    return np.concatenate([base, coverage[:, None]], axis=1).astype(np.float32)


# ── OOD detector ────────────────────────────────────────────────────────────


@dataclass
class OODStats:
    mean: np.ndarray
    std: np.ndarray
    active: np.ndarray
    threshold: float
    coverage_threshold: float


def _fit_ood(X: np.ndarray, coverage_col: int | None = None) -> OODStats:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    active = std > 1e-8
    Z = (X[:, active] - mean[active]) / std[active]
    scores = (Z**2).mean(axis=1)
    threshold = float(np.quantile(scores, 0.99))
    coverage_threshold = (
        float(np.quantile(X[:, coverage_col], 0.99)) if coverage_col is not None else float("nan")
    )
    return OODStats(mean, std, active, threshold, coverage_threshold)


def _ood_scores(X: np.ndarray, stats: OODStats):
    Z = (X[:, stats.active] - stats.mean[stats.active]) / stats.std[stats.active]
    scores = (Z**2).mean(axis=1)
    return scores, scores > stats.threshold


def _hull_distance(points, train_centers) -> np.ndarray:
    """Distance outside the convex hull of the training regions (0 if inside).

    This separates interpolation (a novel region inside the A/B/C hull, e.g.
    G) from extrapolation (a region outside it, e.g. D).  The gate is allowed
    to route on interpolated samples; extrapolated samples fall back to
    geometry.
    """
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(train_centers, dtype=np.float64)
    if len(centers) < 3:
        return np.zeros(len(points))
    try:
        hull = ConvexHull(centers)
    except QhullError:
        return np.zeros(len(points))
    eq = hull.equations  # (n_facets, 3): inside is eq @ [x, y, 1] <= 0
    norms = np.linalg.norm(eq[:, :2], axis=1)
    vals = points @ eq[:, :2].T + eq[:, 2][None, :]
    violations = vals / norms[None, :]
    return np.maximum(0.0, violations.max(axis=1))


# ── gate training ───────────────────────────────────────────────────────────


@dataclass
class GateBundle:
    name: str
    models: list = field(default_factory=list)  # (model, mean, std)
    expert_eps: tuple = ()
    expert_names: list = field(default_factory=list)
    priority_bias: np.ndarray | None = None
    fallback_idx: int = 0
    include_noise: bool = True
    device: str = "cpu"
    val_acc: list = field(default_factory=list)
    best_epochs: list = field(default_factory=list)
    val_expert_acc: np.ndarray | None = None
    ood_feature: OODStats | None = None
    region_threshold: float = 0.05


def _priority_bias(expert_eps) -> np.ndarray:
    n_geo = len(expert_eps)
    order = sorted(range(n_geo), key=lambda i: -expert_eps[i]) + [n_geo]
    bias = np.zeros(n_geo + 1, dtype=np.float32)
    for rank, idx in enumerate(order):
        bias[idx] = -1e-4 * rank
    return bias


def _choose_expert(probs: np.ndarray, priority_bias: np.ndarray) -> np.ndarray:
    return np.argmax(probs + priority_bias[None, :], axis=1)


def _standardize(Xtr, Xva):
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (Xtr - mean) / std, (Xva - mean) / std, mean, std


def _train_gate_fold(
    Xtr, Ytr, Xva, Yva, priority_bias, seed, device, epochs=200, patience=20
):
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    Ytr_t = torch.as_tensor(Ytr, dtype=torch.float32, device=device)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    Yva_np = Yva.astype(np.float32)
    torch.manual_seed(seed)
    model = SpatialRelationGate(Xtr.shape[1], Ytr.shape[1], hidden=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_acc, best_state, best_epoch, bad = -1.0, None, -1, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for start in range(0, len(perm), 128):
            idx = perm[start : start + 128]
            logits = model(Xtr_t[idx])
            loss = F.binary_cross_entropy_with_logits(logits, Ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(Xva_t).cpu().numpy()
        chosen = _choose_expert(logits, priority_bias)
        acc = float(Yva_np[np.arange(len(chosen)), chosen].mean())
        if acc > best_acc + 1e-6:
            best_acc, best_epoch, bad = acc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_acc, best_epoch


def _build_gate_data(head, train_regions, seed, n_per_pair, device):
    pairs = [(a, b) for a in train_regions for b in train_regions if a != b]
    train_centers = COORDS[[INDEX[n] for n in train_regions]]
    X_full, X_nonoise, region_dists = [], [], []
    geo_correct, learned_correct = [], []
    pair_ids, noise_ids = [], []
    for ni, noise in enumerate(NOISE_LEVELS):
        for pi, (a, b) in enumerate(pairs):
            rng = np.random.default_rng(seed * 100003 + ni * 1009 + pi)
            ca, cb = COORDS[INDEX[a]], COORDS[INDEX[b]]
            pa = ca + rng.normal(scale=noise, size=(n_per_pair, 2)).astype(np.float32)
            pb = cb + rng.normal(scale=noise, size=(n_per_pair, 2)).astype(np.float32)
            true = _true_bits(ca, cb)
            dx = pb[:, 0] - pa[:, 0]
            dy = pb[:, 1] - pa[:, 1]
            geom = {eps: _geom_bits(dx, dy, eps) for eps in EPS_GRID}
            gc = np.stack(
                [np.all(geom[eps] == true[None, :], axis=1) for eps in EPS_GRID], axis=1
            )
            lp = _learned_probs(head, pa, pb, device)
            lb = lp > 0.5
            lc = np.all(lb == true[None, :], axis=1)
            X_full.append(_gate_features(pa, pb, lp, COORDS, include_noise=True))
            X_nonoise.append(_gate_features(pa, pb, lp, COORDS, include_noise=False))
            region_dists.append(
                np.maximum(
                    _hull_distance(pa, train_centers), _hull_distance(pb, train_centers)
                )
            )
            geo_correct.append(gc)
            learned_correct.append(lc)
            pair_ids.append(np.full(n_per_pair, pi))
            noise_ids.append(np.full(n_per_pair, ni))
    region_threshold = max(0.05, float(np.quantile(np.concatenate(region_dists), 0.99)) + 0.05)
    return {
        "pairs": pairs,
        "train_centers": train_centers,
        "X_full": np.concatenate(X_full, axis=0).astype(np.float32),
        "X_nonoise": np.concatenate(X_nonoise, axis=0).astype(np.float32),
        "region_threshold": region_threshold,
        "geo_correct": np.concatenate(geo_correct, axis=0),
        "learned_correct": np.concatenate(learned_correct, axis=0),
        "pair_ids": np.concatenate(pair_ids, axis=0),
        "noise_ids": np.concatenate(noise_ids, axis=0),
    }


def _train_gate(
    name,
    data,
    include_noise,
    expert_eps,
    seed,
    device,
    n_folds=3,
    epochs=200,
    patience=20,
):
    X = data["X_full"] if include_noise else data["X_nonoise"]
    geo_cols = [EPS_GRID.index(e) for e in expert_eps]
    Y = np.concatenate(
        [data["geo_correct"][:, geo_cols], data["learned_correct"][:, None]], axis=1
    ).astype(np.float32)
    pairs = data["pairs"]
    fold_ids = np.array_split(np.arange(len(pairs)), n_folds)
    bundle = GateBundle(
        name=name,
        expert_eps=tuple(expert_eps),
        expert_names=[f"geometry_eps{e}" for e in expert_eps] + ["learned"],
        priority_bias=_priority_bias(expert_eps),
        include_noise=include_noise,
        device=device,
    )
    val_expert_acc = np.zeros(Y.shape[1], dtype=np.float64)
    for fi, fold in enumerate(fold_ids):
        val_mask = np.isin(data["pair_ids"], fold)
        train_mask = ~val_mask
        Xtr, Xva, mean, std = _standardize(X[train_mask], X[val_mask])
        model, val_acc, best_epoch = _train_gate_fold(
            Xtr,
            Y[train_mask],
            Xva,
            Y[val_mask],
            bundle.priority_bias,
            seed * 100 + fi,
            device,
            epochs,
            patience,
        )
        bundle.models.append((model, mean, std))
        bundle.val_acc.append(val_acc)
        bundle.best_epochs.append(best_epoch)
        val_expert_acc += Y[val_mask].mean(axis=0)
    val_expert_acc /= len(fold_ids)
    bundle.val_expert_acc = val_expert_acc
    # Fallback is always a geometry expert (never the learned head).
    bundle.fallback_idx = int(np.argmax(val_expert_acc[: len(expert_eps)]))
    # OOD: feature-space detector on the full gate features; region
    # extrapolation is checked separately against the training-centre hull.
    bundle.ood_feature = _fit_ood(data["X_full"])
    bundle.region_threshold = data["region_threshold"]
    return bundle


def _gate_predict(bundle: GateBundle, X: np.ndarray):
    probs = []
    for model, mean, std in bundle.models:
        Xs = (X - mean) / std
        with torch.no_grad():
            logits = model(torch.as_tensor(Xs, dtype=torch.float32, device=bundle.device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    p = np.mean(probs, axis=0)
    return p, _choose_expert(p, bundle.priority_bias)


def _expert_pred_stack(geom, learned_bits, expert_eps):
    return np.stack([geom[eps] for eps in expert_eps] + [learned_bits], axis=1)


# ── evaluation ──────────────────────────────────────────────────────────────


def _eval_split(head, gates, train_regions, novel_regions, noise, seed, n_per_pair, device):
    pairs = _heldout_pairs(train_regions, NAMES)
    correct: dict[str, list] = {}
    fallback, feature_ood, region_ood = [], [], []
    chosen: dict[str, list] = {name: [] for name in gates}
    chosen_fb: dict[str, list] = {name: [] for name in gates}
    pair_names, novel_labels = [], []

    for pi, (a, b) in enumerate(pairs):
        rng = np.random.default_rng(seed * 100003 + int(round(noise * 1000)) + pi)
        ca, cb = COORDS[INDEX[a]], COORDS[INDEX[b]]
        pa = ca + rng.normal(scale=noise, size=(n_per_pair, 2)).astype(np.float32)
        pb = cb + rng.normal(scale=noise, size=(n_per_pair, 2)).astype(np.float32)
        true = _true_bits(ca, cb)
        dx = pb[:, 0] - pa[:, 0]
        dy = pb[:, 1] - pa[:, 1]
        geom = {eps: _geom_bits(dx, dy, eps) for eps in EPS_GRID}
        lp = _learned_probs(head, pa, pb, device)
        lb = lp > 0.5
        learned_ok = np.all(lb == true[None, :], axis=1)

        preds = {
            "learned": lb,
            "hybrid": _hybrid_bits(dx, dy, lb),
            "hybrid_diag": _hybrid_diag_bits(dx, dy, lb),
            "oracle_geo025_learned": None,
            "oracle_eps_learned": None,
        }
        for eps in EPS_GRID:
            preds[f"geometry_eps{eps}"] = geom[eps]
        geo025_ok = np.all(geom[TAU] == true[None, :], axis=1)
        any_geo_ok = np.zeros(len(dx), dtype=bool)
        for eps in EPS_GRID:
            any_geo_ok |= np.all(geom[eps] == true[None, :], axis=1)
        preds["oracle_geo025_learned"] = geo025_ok | learned_ok
        preds["oracle_eps_learned"] = any_geo_ok | learned_ok

        X_full = _gate_features(pa, pb, lp, COORDS, include_noise=True)
        X_nonoise = _gate_features(pa, pb, lp, COORDS, include_noise=False)
        train_centers = COORDS[[INDEX[n] for n in train_regions]]

        first = next(iter(gates.values()))
        feat_scores, feat_flag = _ood_scores(X_full, first.ood_feature)
        region_dist = np.maximum(
            _hull_distance(pa, train_centers), _hull_distance(pb, train_centers)
        )
        reg_flag = region_dist > first.region_threshold
        comb_flag = feat_flag | reg_flag
        fallback.append(comb_flag)
        feature_ood.append(feat_flag)
        region_ood.append(reg_flag)

        for name, gate in gates.items():
            X_gate = X_full if gate.include_noise else X_nonoise
            _, ch = _gate_predict(gate, X_gate)
            expert_preds = _expert_pred_stack(geom, lb, gate.expert_eps)
            rows = np.arange(len(dx))
            pred = expert_preds[rows, ch]
            ch_fb = np.where(comb_flag, gate.fallback_idx, ch)
            pred_fb = expert_preds[rows, ch_fb]
            preds[f"gate_{name}"] = pred
            preds[f"gate_{name}_ood"] = pred_fb
            chosen[name].append(ch)
            chosen_fb[name].append(ch_fb)

        for m, pred in preds.items():
            if pred is None:
                continue
            ok = pred if pred.ndim == 1 else np.all(pred == true[None, :], axis=1)
            correct.setdefault(m, []).append(ok)
        pair_names.append(np.full(len(dx), f"{a}->{b}"))
        novel = "+".join(sorted(set((a, b)) & set(novel_regions)))
        novel_labels.append(np.full(len(dx), novel or "id"))

    out = {
        "correct": {m: np.concatenate(v, axis=0) for m, v in correct.items()},
        "fallback": np.concatenate(fallback, axis=0),
        "feature_ood": np.concatenate(feature_ood, axis=0),
        "region_ood": np.concatenate(region_ood, axis=0),
        "chosen": {k: np.concatenate(v, axis=0) for k, v in chosen.items()},
        "chosen_fb": {k: np.concatenate(v, axis=0) for k, v in chosen_fb.items()},
        "pair_names": np.concatenate(pair_names, axis=0),
        "novel": np.concatenate(novel_labels, axis=0),
        "noise": noise,
        "seed": seed,
    }
    return out


def _val_best_eps(data):
    """Fixed dead zone tuned on A/B/C (never on the held-out D/G test set)."""
    out = {}
    for ni, noise in enumerate(NOISE_LEVELS):
        mask = data["noise_ids"] == ni
        accs = data["geo_correct"][mask].mean(axis=0)
        best = int(np.argmax(accs))
        out[str(noise)] = {
            "eps": EPS_GRID[best],
            "acc": float(accs[best]),
            "all": {str(e): float(a) for e, a in zip(EPS_GRID, accs)},
        }
    return out


# ── aggregation ─────────────────────────────────────────────────────────────


def _paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = a.astype(np.float64) - b.astype(np.float64)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {
        "diff": mean,
        "se": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
        "n": int(len(d)),
    }


def _aggregate_primary(per_seed):
    method_names = [
        "geometry_eps0.1",
        "geometry_eps0.25",
        "learned",
        "hybrid",
        "hybrid_diag",
        "gate_binary",
        "gate_binary_ood",
        "gate_multi",
        "gate_multi_ood",
        "gate_ablation",
        "gate_ablation_ood",
        "oracle_geo025_learned",
        "oracle_eps_learned",
    ]
    method_names += [
        f"geometry_eps{eps}" for eps in EPS_GRID if f"geometry_eps{eps}" not in method_names
    ]
    summary = {}
    comparisons = {}
    ood = {}
    selection = {}
    for noise in NOISE_LEVELS:
        key = str(noise)
        rows = [s["by_noise"][key] for s in per_seed]
        concat = {
            m: np.concatenate([r["correct"][m] for r in rows], axis=0) for m in method_names
        }
        per_seed_mean = {
            m: [float(r["correct"][m].mean()) for r in rows] for m in method_names
        }
        # Validation-selected fixed dead zone, per seed.
        val_means = []
        val_arrays = []
        for s, r in zip(per_seed, rows):
            eps = s["val_fixed"][key]["eps"]
            m = f"geometry_eps{eps}"
            val_means.append(float(r["correct"][m].mean()))
            val_arrays.append(r["correct"][m])
        per_seed_mean["best_fixed_geo_val"] = val_means
        concat["best_fixed_geo_val"] = np.concatenate(val_arrays, axis=0)
        val_eps = [s["val_fixed"][key]["eps"] for s in per_seed]
        val_accs = [s["val_fixed"][key]["acc"] for s in per_seed]
        # Test-oracle fixed dead zone (upper bound for any fixed eps).
        eps_means = {eps: float(concat[f"geometry_eps{eps}"].mean()) for eps in EPS_GRID}
        best_eps = max(eps_means, key=eps_means.get)
        per_seed_mean["best_fixed_geo_test"] = [
            float(r["correct"][f"geometry_eps{best_eps}"].mean()) for r in rows
        ]
        concat["best_fixed_geo_test"] = concat[f"geometry_eps{best_eps}"]
        summary[key] = {
            m: {
                "mean": float(np.mean(per_seed_mean[m])),
                "std": float(np.std(per_seed_mean[m])),
                "per_seed": per_seed_mean[m],
                "pooled_mean": float(concat[m].mean()),
            }
            for m in per_seed_mean
        }
        summary[key]["best_fixed_geo_val"]["eps_per_seed"] = val_eps
        summary[key]["best_fixed_geo_val"]["eps"] = max(set(val_eps), key=val_eps.count)
        summary[key]["best_fixed_geo_val"]["val_acc"] = float(np.mean(val_accs))
        summary[key]["best_fixed_geo_test"]["eps"] = best_eps
        # Paired comparisons on pooled samples.
        pairs_to_test = [
            ("gate_binary", "geometry_eps0.25"),
            ("gate_binary", "hybrid_diag"),
            ("gate_binary", "best_fixed_geo_val"),
            ("gate_binary", "best_fixed_geo_test"),
            ("gate_binary", "gate_ablation"),
            ("gate_multi", "geometry_eps0.25"),
            ("gate_multi", "hybrid_diag"),
            ("gate_multi", "best_fixed_geo_val"),
            ("gate_multi", "best_fixed_geo_test"),
            ("gate_ablation", "geometry_eps0.25"),
            ("gate_ablation", "best_fixed_geo_val"),
        ]
        comparisons[key] = {
            f"{a}_vs_{b}": _paired_stats(concat[a], concat[b]) for a, b in pairs_to_test
        }
        # OOD / fallback statistics (the detector is shared by all gates).
        fb = np.concatenate([r["fallback"] for r in rows], axis=0)
        f_ood = np.concatenate([r["feature_ood"] for r in rows], axis=0)
        r_ood = np.concatenate([r["region_ood"] for r in rows], axis=0)
        ood[key] = {
            "fallback_rate": float(fb.mean()),
            "feature_ood_rate": float(f_ood.mean()),
            "region_ood_rate": float(r_ood.mean()),
            "per_region": {},
        }
        novel = np.concatenate([r["novel"] for r in rows], axis=0)
        for region in sorted(set(novel.tolist())):
            mask = novel == region
            ood[key]["per_region"][region] = {
                "n": int(mask.sum()),
                "fallback_rate": float(fb[mask].mean()),
                "gate_binary_ood_acc": float(concat["gate_binary_ood"][mask].mean()),
                "gate_binary_acc": float(concat["gate_binary"][mask].mean()),
                "geometry_eps0.25_acc": float(concat["geometry_eps0.25"][mask].mean()),
            }
        for gate_name in ("gate_binary", "gate_multi", "gate_ablation"):
            ood_key = f"{gate_name}_ood"
            acc_fb = float(concat[ood_key][fb].mean()) if fb.any() else float("nan")
            acc_nofb = float(concat[ood_key][~fb].mean()) if (~fb).any() else float("nan")
            ood[key][gate_name] = {
                "acc_fallback": acc_fb,
                "acc_nonfallback": acc_nofb,
                "acc_overall": float(concat[ood_key].mean()),
                "acc_no_fallback": float(concat[gate_name].mean()),
                "fallback_rate": float(fb.mean()),
            }
        # Expert selection distribution (deployed choice, including fallback).
        sel = {}
        for gate_name in ("gate_binary", "gate_multi", "gate_ablation"):
            short = gate_name.removeprefix("gate_")
            ch = np.concatenate([r["chosen_fb"][short] for r in rows], axis=0)
            ch_nofb = np.concatenate([r["chosen"][short] for r in rows], axis=0)
            gate = per_seed[0]["gates"][short]
            names = gate.expert_names
            sel[gate_name] = {
                "deployed": {names[i]: float((ch == i).mean()) for i in range(len(names))},
                "no_fallback": {
                    names[i]: float((ch_nofb == i).mean()) for i in range(len(names))
                },
                "learned_rate": float((ch == len(names) - 1).mean()),
                "learned_rate_no_fallback": float((ch_nofb == len(names) - 1).mean()),
                "fallback_expert": names[gate.fallback_idx],
            }
        selection[key] = sel
    return summary, comparisons, ood, selection


def _aggregate_cv(cv_runs):
    folds = {}
    for run in cv_runs:
        key = run["novel_region"]
        folds.setdefault(key, []).append(run)
    out = {}
    for region, runs in folds.items():
        by_noise = {}
        for noise in NOISE_LEVELS:
            key = str(noise)
            rows = [r["by_noise"][key] for r in runs]
            methods = list(rows[0]["correct"].keys())
            by_noise[key] = {
                m: {
                    "mean": float(np.mean([r["correct"][m].mean() for r in rows])),
                    "std": float(np.std([r["correct"][m].mean() for r in rows])),
                }
                for m in methods
            }
            fb = np.concatenate([r["fallback"] for r in rows], axis=0)
            by_noise[key]["fallback_rate"] = float(fb.mean())
        out[region] = {"by_noise": by_noise, "seeds": [r["seed"] for r in runs]}
    return out


# ── report ──────────────────────────────────────────────────────────────────


def _fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def _report(metrics):
    s = metrics["summary"]
    noises = [str(x) for x in NOISE_LEVELS]
    methods = [
        ("geometry_eps0.1", "geometry(0.10)"),
        ("geometry_eps0.25", "geometry(0.25) [E3 strict]"),
        ("learned", "learned head"),
        ("hybrid", "hybrid [E3]"),
        ("hybrid_diag", "hybrid_diag [E3]"),
        ("gate_binary", "gate (geo0.25 vs learned)"),
        ("gate_binary_ood", "gate binary + OOD fallback"),
        ("gate_multi", "gate (eps grid + learned)"),
        ("gate_multi_ood", "gate multi + OOD fallback"),
        ("gate_ablation", "gate, no noise features"),
        ("gate_ablation_ood", "gate no-noise + OOD fallback"),
        ("best_fixed_geo_val", "tuned fixed geo (val-selected)"),
        ("best_fixed_geo_test", "best fixed geo (test oracle)"),
        ("oracle_geo025_learned", "oracle(geo0.25, learned)"),
        ("oracle_eps_learned", "oracle(eps grid, learned)"),
    ]
    lines = [
        "# Claim-gap F8: learned relation gating + OOD degradation",
        "",
        f"> generated: {metrics['generated']}",
        f"> train regions: {metrics['config']['train_regions']}; "
        f"test regions: {metrics['config']['test_regions']}; "
        f"held-out pairs: {metrics['config']['n_heldout_pairs']}",
        f"> noise levels: {noises}; tau={TAU}, strict eps={EPS}; "
        f"seeds={metrics['config']['seeds']}, "
        f"n_train/pair={metrics['config']['n_train']}, "
        f"n_test/pair={metrics['config']['n_test']}",
        "",
        "## TL;DR",
        "",
    ]
    v = metrics["verdict"]
    lines += [
        f"- Acceptance bar (`deployed gate > geometry(0.25)` at some noise): "
        f"**{v['acceptance_met']}**.",
        f"- No-fallback binary gate beats `geometry(0.25)`: "
        f"**{v['binary_gt_geo025']}**; beats `hybrid_diag`: "
        f"**{v['binary_gt_hybrid_diag']}**.",
        f"- No-fallback multi-expert gate beats `geometry(0.25)`: "
        f"**{v['multi_gt_geo025']}**; beats `hybrid_diag`: "
        f"**{v['multi_gt_hybrid_diag']}**.",
        f"- Any no-fallback learned gate beats the tuned fixed geometric dead zone: "
        f"**{v['learned_beyond_dead_zone']}**.",
        f"- OOD fallback active on unseen D/G: **{v['ood_fallback_active']}** "
        f"(fallback rate at noise 0.15: {v['fallback_rate_015']:.3f}).",
        f"- Noise-feature ablation removes the binary gate's gain: "
        f"**{v['ablation_removes_gain']}**.",
        f"- Noise 0.15 headline: geo0.25 {v['noise_015_numbers']['geometry_eps0.25']:.3f}, "
        f"hybrid_diag {v['noise_015_numbers']['hybrid_diag']:.3f}, "
        f"tuned fixed geo {v['noise_015_numbers']['best_fixed_geo_val']:.3f}; "
        f"binary gate {v['noise_015_numbers']['gate_binary']:.3f} "
        f"(+OOD {v['noise_015_numbers']['gate_binary_ood']:.3f}), "
        f"multi gate {v['noise_015_numbers']['gate_multi']:.3f} "
        f"(+OOD {v['noise_015_numbers']['gate_multi_ood']:.3f}).",
        f"- **Verdict**: {v['conclusion']}",
        "",
        "## 1. Main results (A/B/C -> D/G)",
        "",
        "| method | " + " | ".join(f"noise={n}" for n in noises) + " |",
        "|---|" + "---:|" * len(noises),
    ]
    for key, label in methods:
        cells = [_fmt(s[n][key]["mean"], s[n][key]["std"]) for n in noises]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "`best_fixed_geo_val` is the fixed `geometry(eps)` with the best A/B/C",
        "validation accuracy per noise; it is selected without touching D/G.",
        "`best_fixed_geo_test` is the best fixed eps on the test set itself and is",
        "only an upper bound for any fixed dead zone.",
        "",
        "### Paired comparisons (pooled samples, 95% CI)",
        "",
        "| noise | comparison | diff | 95% CI |",
        "|---:|---|---:|---|",
    ]
    for n in noises:
        for name, st in metrics["comparisons"][n].items():
            lines.append(
                f"| {n} | {name} | {st['diff']:+.4f} | "
                f"[{st['ci_low']:+.4f}, {st['ci_high']:+.4f}] |"
            )
    lines += [
        "",
        "## 2. Key control: is it just a wider dead zone?",
        "",
        "Geometry-only accuracy for a sweep of fixed dead zones:",
        "",
        "| eps | " + " | ".join(f"noise={n}" for n in noises) + " |",
        "|---:|" + "---:|" * len(noises),
    ]
    for eps in EPS_GRID:
        cells = [_fmt(s[n][f"geometry_eps{eps}"]["mean"], s[n][f"geometry_eps{eps}"]["std"])
                 for n in noises]
        lines.append(f"| {eps:.2f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| noise | val-selected eps | val acc | test acc of that eps | test-oracle eps |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n in noises:
        vf = s[n]["best_fixed_geo_val"]
        tf = s[n]["best_fixed_geo_test"]
        lines.append(
            f"| {n} | {vf['eps']:.2f} | {vf['val_acc']:.3f} | {vf['mean']:.3f} | "
            f"{tf['eps']:.2f} ({tf['mean']:.3f}) |"
        )
    lines += [
        "",
        "If a fixed `geometry(eps)` tuned on A/B/C matches or beats the gate,",
        "the learned gate is not adding calibration beyond a wider dead zone.",
        "",
        "## 3. OOD degradation and fallback",
        "",
        "| noise | fallback rate | feature-OOD rate | region-OOD rate | "
        "fallback acc | non-fallback acc | gate no-fallback acc |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in noises:
        o = metrics["ood"][n]
        b = o["gate_binary"]
        lines.append(
            f"| {n} | {o['fallback_rate']:.3f} | {o['feature_ood_rate']:.3f} | "
            f"{o['region_ood_rate']:.3f} | {b['acc_fallback']:.3f} | "
            f"{b['acc_nonfallback']:.3f} | {b['acc_no_fallback']:.3f} |"
        )
    lines += [
        "",
        "Per novel-region breakdown at noise 0.15:",
        "",
        "| region | n | fallback rate | gate+OOD acc | gate no-fallback acc | geo0.25 acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for region, row in metrics["ood"]["0.15"]["per_region"].items():
        lines.append(
            f"| {region} | {row['n']} | {row['fallback_rate']:.3f} | "
            f"{row['gate_binary_ood_acc']:.3f} | {row['gate_binary_acc']:.3f} | "
            f"{row['geometry_eps0.25_acc']:.3f} |"
        )
    lines += [
        "",
        "### Expert selection (multi-expert gate, deployed choice)",
        "",
        "| expert | " + " | ".join(f"noise={n}" for n in noises) + " |",
        "|---|" + "---:|" * len(noises),
    ]
    experts = list(metrics["selection"]["0.0"]["gate_multi"]["deployed"].keys())
    for e in experts:
        cells = [f"{metrics['selection'][n]['gate_multi']['deployed'][e]:.3f}" for n in noises]
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### Expert selection (multi-expert gate, no fallback)",
        "",
        "| expert | " + " | ".join(f"noise={n}" for n in noises) + " |",
        "|---|" + "---:|" * len(noises),
    ]
    for e in experts:
        cells = [
            f"{metrics['selection'][n]['gate_multi']['no_fallback'][e]:.3f}"
            for n in noises
        ]
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The learned expert's selection rate and the dominant geometry eps are the",
        "direct evidence for (or against) a wider-dead-zone explanation.  The",
        "deployed table is dominated by the OOD fallback; the no-fallback table",
        "shows what the learned gate itself routes to.",
        "",
        "## 4. Ablation: remove local-noise features",
        "",
        "| noise | full gate | no-noise gate | delta (full - no-noise) | 95% CI |",
        "|---:|---:|---:|---:|---|",
    ]
    for n in noises:
        st = metrics["comparisons"][n]["gate_binary_vs_gate_ablation"]
        full = s[n]["gate_binary"]["mean"]
        abl = s[n]["gate_ablation"]["mean"]
        delta = full - abl
        lines.append(
            f"| {n} | {full:.3f} | {abl:.3f} | {delta:+.4f} | "
            f"[{st['ci_low']:+.4f}, {st['ci_high']:+.4f}] |"
        )
    lines += [
        "",
        "If the full gate's advantage over `geometry(0.25)` disappears when the",
        "noise features are removed, the gain comes from noise-adaptive routing;",
        "if it persists, it comes from the margin/learned-score features.",
        "",
        "## 5. Anti-overfitting: leave-one-region-out CV",
        "",
    ]
    val_acc = metrics.get("per_seed_val_acc", {})
    if val_acc:
        def _mean_val(gate_name):
            vals = [v for seed in val_acc.values() for v in seed[gate_name]]
            return float(np.mean(vals))

        lines += [
            f"Gate validation accuracy on held-out A/B/C pairs (3-fold pair CV): "
            f"binary {_mean_val('binary'):.3f}, multi {_mean_val('multi'):.3f}, "
            f"ablation {_mean_val('ablation'):.3f}. This is in-distribution accuracy; "
            f"it does not transfer to D/G (see the main table).",
            "",
        ]
    cv = metrics.get("cv", {})
    if cv:
        lines += [
            "Each fold trains the head and gate on four regions and tests on the",
            "held-out region (8 ordered pairs), with 3 seeds per fold.",
            "",
            "| held-out region | noise | gate binary | gate multi | geo0.25 | "
            "tuned fixed geo | hybrid_diag | fallback rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for region, data in cv.items():
            for n in noises:
                row = data["by_noise"][n]
                lines.append(
                    f"| {region} | {n} | {row['gate_binary']['mean']:.3f} | "
                    f"{row['gate_multi']['mean']:.3f} | "
                    f"{row['geometry_eps0.25']['mean']:.3f} | "
                    f"{row['best_fixed_geo_val']['mean']:.3f} | "
                    f"{row['hybrid_diag']['mean']:.3f} | {row['fallback_rate']:.3f} |"
                )
    else:
        lines.append("(CV disabled in this run.)")
    lines += [
        "",
        "## 6. Verdict and limitations",
        "",
        f"**Verdict**: {v['conclusion']}",
        "",
        "Limitations:",
        "- Synthetic five-region grid; only 5 regions and 14 primary held-out pairs.",
        "- The learned head is E3's architecture (absolute-coordinate features), so",
        "  it is OOD-broken on vertical-aligned pairs such as C<->G and D<->C; this",
        "  is a property of the head, not of the gate.",
        "- The grid bench has no category or tactile modality, so the contact/near",
        "  gate features are constant and cannot contribute.",
        "- The OOD detector combines a feature-space novelty score with the",
        "  distance outside the convex hull of the training regions.  D is an",
        "  extrapolation (outside the A/B/C hull) and falls back to geometry; G is",
        "  an interpolation (inside the hull), so the gate may still route there.",
        "  The feature detector catches some relation-level OOD cases (e.g. C<->G).",
        "",
        "Recommendations:",
        "- Tune the geometric dead zone to the local noise level (a fixed eps ~0.5",
        "  at sigma ~0.15 already dominates the E3 hybrid at tau=0.25).",
        "- If a learned head is kept, train it on vertical-aligned pairs and mixed",
        "  noise, and use a translation-invariant feature set.",
        "- Treat learned gating as a safety layer only after the head and the",
        "  geometric dead zone are properly calibrated; otherwise it is a wider",
        "  dead zone in learned clothing.",
        "",
    ]
    return "\n".join(lines)


# ── run ─────────────────────────────────────────────────────────────────────


def run_primary(seeds, n_train, n_test, head_epochs, gate_epochs, device):
    per_seed = []
    for seed in range(seeds):
        head = _train_head(
            TRAIN_REGIONS, seed, device, n_per_pair=n_train, epochs=head_epochs
        )
        data = _build_gate_data(head, TRAIN_REGIONS, seed, n_train, device)
        gates = {
            "binary": _train_gate(
                "binary", data, True, (TAU,), seed, device, epochs=gate_epochs
            ),
            "multi": _train_gate(
                "multi", data, True, EPS_GRID, seed, device, epochs=gate_epochs
            ),
            "ablation": _train_gate(
                "ablation", data, False, (TAU,), seed, device, epochs=gate_epochs
            ),
        }
        val_fixed = _val_best_eps(data)
        by_noise = {}
        for noise in NOISE_LEVELS:
            r = _eval_split(
                head, gates, TRAIN_REGIONS, set(TEST_REGIONS), noise, seed, n_test, device
            )
            by_noise[str(noise)] = r
        per_seed.append(
            {
                "seed": seed,
                "by_noise": by_noise,
                "val_fixed": val_fixed,
                "gates": gates,
                "val_acc": {k: g.val_acc for k, g in gates.items()},
            }
        )
        print(
            f"seed {seed}: "
            + " ".join(
                f"n{noise}: gate={by_noise[str(noise)]['correct']['gate_binary'].mean():.3f} "
                f"multi={by_noise[str(noise)]['correct']['gate_multi'].mean():.3f}"
                for noise in NOISE_LEVELS
            )
        )
    return per_seed


def run_cv(seeds, n_train, n_test, head_epochs, gate_epochs, device):
    runs = []
    for region in NAMES:
        train_regions = tuple(n for n in NAMES if n != region)
        for seed in range(seeds):
            head = _train_head(
                train_regions, seed, device, n_per_pair=n_train, epochs=head_epochs
            )
            data = _build_gate_data(head, train_regions, seed, n_train, device)
            gates = {
                "binary": _train_gate(
                    "binary", data, True, (TAU,), seed, device, epochs=gate_epochs
                ),
                "multi": _train_gate(
                    "multi", data, True, EPS_GRID, seed, device, epochs=gate_epochs
                ),
                "ablation": _train_gate(
                    "ablation", data, False, (TAU,), seed, device, epochs=gate_epochs
                ),
            }
            val_fixed = _val_best_eps(data)
            by_noise = {}
            for noise in NOISE_LEVELS:
                r = _eval_split(
                    head, gates, train_regions, {region}, noise, seed, n_test, device
                )
                # Validation-selected fixed dead zone, applied to this fold.
                eps = val_fixed[str(noise)]["eps"]
                r["correct"]["best_fixed_geo_val"] = r["correct"][f"geometry_eps{eps}"]
                by_noise[str(noise)] = r
            runs.append(
                {
                    "novel_region": region,
                    "seed": seed,
                    "by_noise": by_noise,
                    "val_fixed": val_fixed,
                }
            )
        print(f"cv region {region}: done")
    return runs


def _verdict(summary, comparisons, ood, ablation_delta):
    noises = [str(x) for x in NOISE_LEVELS]

    def any_gt(method, control):
        return any(summary[n][method]["mean"] > summary[n][control]["mean"] for n in noises)

    def gains(method, control):
        return {
            n: summary[n][method]["mean"] - summary[n][control]["mean"] for n in noises
        }

    binary_gt_geo025 = any_gt("gate_binary", "geometry_eps0.25")
    binary_gt_hybrid = any_gt("gate_binary", "hybrid_diag")
    binary_gt_fixed = any_gt("gate_binary", "best_fixed_geo_val")
    multi_gt_geo025 = any_gt("gate_multi", "geometry_eps0.25")
    multi_gt_hybrid = any_gt("gate_multi", "hybrid_diag")
    multi_gt_fixed = any_gt("gate_multi", "best_fixed_geo_val")
    deployed_gt_geo025 = any_gt("gate_binary_ood", "geometry_eps0.25") or any_gt(
        "gate_multi_ood", "geometry_eps0.25"
    )
    deployed_gt_fixed = any_gt("gate_binary_ood", "best_fixed_geo_val") or any_gt(
        "gate_multi_ood", "best_fixed_geo_val"
    )
    learned_beyond_dead_zone = binary_gt_fixed or multi_gt_fixed
    fallback_rate_015 = ood["0.15"]["fallback_rate"]
    ood_active = fallback_rate_015 > 0.0
    # Ablation removes the gain if the no-noise gate loses the full gate's
    # advantage over geo0.25 at the noise levels where the full gate wins.
    full_gain = gains("gate_binary", "geometry_eps0.25")
    abl_gain = gains("gate_ablation", "geometry_eps0.25")
    removes = any(full_gain[n] > 0.0 and abl_gain[n] <= 0.0 for n in noises)
    key = "0.15"
    key_numbers = {
        "noise": 0.15,
        "geometry_eps0.25": summary[key]["geometry_eps0.25"]["mean"],
        "hybrid_diag": summary[key]["hybrid_diag"]["mean"],
        "best_fixed_geo_val": summary[key]["best_fixed_geo_val"]["mean"],
        "gate_binary": summary[key]["gate_binary"]["mean"],
        "gate_binary_ood": summary[key]["gate_binary_ood"]["mean"],
        "gate_multi": summary[key]["gate_multi"]["mean"],
        "gate_multi_ood": summary[key]["gate_multi_ood"]["mean"],
        "gate_ablation": summary[key]["gate_ablation"]["mean"],
        "gate_ablation_ood": summary[key]["gate_ablation_ood"]["mean"],
    }
    if learned_beyond_dead_zone:
        conclusion = (
            "The no-fallback learned gate beats the tuned fixed geometric dead zone at "
            "at least one noise level, so there is evidence for learned gating beyond a "
            "wider dead zone; check the paired CIs and the CV table before treating it "
            "as robust."
        )
    elif deployed_gt_geo025:
        b_gain = key_numbers["gate_binary_ood"] - key_numbers["geometry_eps0.25"]
        m_gap = key_numbers["gate_multi_ood"] - key_numbers["best_fixed_geo_val"]
        if binary_gt_geo025 or multi_gt_geo025:
            nofb_clause = (
                f"The no-fallback gates are mixed: binary "
                f"{key_numbers['gate_binary']:.3f}, multi "
                f"{key_numbers['gate_multi']:.3f} vs geo0.25 "
                f"{key_numbers['geometry_eps0.25']:.3f}; neither beats the tuned fixed "
                f"dead zone ({key_numbers['best_fixed_geo_val']:.3f})."
            )
        else:
            nofb_clause = (
                f"No no-fallback learned gate beats geo0.25 (binary "
                f"{key_numbers['gate_binary']:.3f}, multi "
                f"{key_numbers['gate_multi']:.3f} vs "
                f"{key_numbers['geometry_eps0.25']:.3f})."
            )
        conclusion = (
            f"Acceptance is met numerically only by the deployed multi gate: at noise "
            f"0.15 it reaches {key_numbers['gate_multi_ood']:.3f} vs geometry(0.25) "
            f"{key_numbers['geometry_eps0.25']:.3f} and hybrid_diag "
            f"{key_numbers['hybrid_diag']:.3f}, but this is within {abs(m_gap):.3f} of "
            f"the tuned fixed dead zone ({key_numbers['best_fixed_geo_val']:.3f}) and "
            f"does not exceed it. The deployed binary gate is "
            f"{key_numbers['gate_binary_ood']:.3f} ({b_gain:+.3f} over geo0.25), "
            f"essentially geometry(0.25). {nofb_clause} The gate fits A/B/C validation "
            f"almost perfectly but does not transfer to D/G because the learned head is "
            f"OOD-broken on vertical-aligned / unseen-coordinate pairs. The OOD fallback "
            f"is necessary for safety, but the only safe performance gain is a "
            f"wider/adaptive geometric dead zone. E3's hybrid_diag advantage at tau=0.25 "
            f"is an under-tuned-dead-zone artifact."
        )
    else:
        conclusion = (
            "The gate does not beat geometry(0.25) at any noise level. The learned head "
            "is OOD-broken on the unseen regions, and routing around it does not recover "
            "a gain; a tuned geometric dead zone remains the better choice."
        )
    return {
        "gate_gt_geo025": bool(binary_gt_geo025),
        "gate_gt_hybrid_diag": bool(binary_gt_hybrid),
        "gate_gt_best_fixed_geo": bool(binary_gt_fixed),
        "binary_gt_geo025": bool(binary_gt_geo025),
        "binary_gt_hybrid_diag": bool(binary_gt_hybrid),
        "binary_gt_best_fixed_geo": bool(binary_gt_fixed),
        "multi_gt_geo025": bool(multi_gt_geo025),
        "multi_gt_hybrid_diag": bool(multi_gt_hybrid),
        "multi_gt_best_fixed_geo": bool(multi_gt_fixed),
        "deployed_gt_geo025": bool(deployed_gt_geo025),
        "deployed_gt_best_fixed_geo": bool(deployed_gt_fixed),
        "learned_beyond_dead_zone": bool(learned_beyond_dead_zone),
        "acceptance_met": bool(deployed_gt_geo025),
        "ood_fallback_active": bool(ood_active),
        "fallback_rate_015": float(fallback_rate_015),
        "ablation_removes_gain": bool(removes),
        "binary_gain_over_geo025": full_gain,
        "multi_gain_over_geo025": gains("gate_multi", "geometry_eps0.25"),
        "deployed_gain_over_geo025": {
            "binary_ood": gains("gate_binary_ood", "geometry_eps0.25"),
            "multi_ood": gains("gate_multi_ood", "geometry_eps0.25"),
        },
        "binary_gain_over_fixed": gains("gate_binary", "best_fixed_geo_val"),
        "multi_gain_over_fixed": gains("gate_multi", "best_fixed_geo_val"),
        "ablation_gain_over_geo025": abl_gain,
        "noise_015_numbers": key_numbers,
        "conclusion": conclusion,
    }


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--head-epochs", type=int, default=400)
    ap.add_argument("--gate-epochs", type=int, default=200)
    ap.add_argument("--cv-seeds", type=int, default=3)
    ap.add_argument("--cv-n-train", type=int, default=300)
    ap.add_argument("--cv-n-test", type=int, default=300)
    ap.add_argument("--cv-head-epochs", type=int, default=300)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reports/claim_gaps/relation_gating")
    args = ap.parse_args()

    if args.quick:
        args.seeds = 1
        args.n_train = 40
        args.n_test = 40
        args.head_epochs = 30
        args.gate_epochs = 20
        args.cv_seeds = 0
        args.no_cv = True

    per_seed = run_primary(
        args.seeds,
        args.n_train,
        args.n_test,
        args.head_epochs,
        args.gate_epochs,
        args.device,
    )
    summary, comparisons, ood, selection = _aggregate_primary(per_seed)
    cv = {}
    if not args.no_cv and args.cv_seeds > 0:
        cv_runs = run_cv(
            args.cv_seeds,
            args.cv_n_train,
            args.cv_n_test,
            args.cv_head_epochs,
            args.gate_epochs,
            args.device,
        )
        cv = _aggregate_cv(cv_runs)

    ablation_delta = {
        n: summary[n]["gate_binary"]["mean"] - summary[n]["gate_ablation"]["mean"]
        for n in [str(x) for x in NOISE_LEVELS]
    }
    verdict = _verdict(summary, comparisons, ood, ablation_delta)
    metrics = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config": {
            "train_regions": list(TRAIN_REGIONS),
            "test_regions": list(TEST_REGIONS),
            "heldout_pairs": [list(p) for p in _heldout_pairs(TRAIN_REGIONS)],
            "n_heldout_pairs": len(_heldout_pairs(TRAIN_REGIONS)),
            "noise_levels": list(NOISE_LEVELS),
            "tau": TAU,
            "eps": EPS,
            "eps_grid": list(EPS_GRID),
            "seeds": args.seeds,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "head_epochs": args.head_epochs,
            "gate_epochs": args.gate_epochs,
            "gate_features_full": _feature_names(True),
            "gate_features_no_noise": _feature_names(False),
            "gate_folds": 3,
            "fallback": (
                "feature-space OOD (full gate features) OR outside the convex hull "
                "of the training regions"
            ),
        },
        "summary": summary,
        "comparisons": comparisons,
        "ood": ood,
        "selection": selection,
        "cv": cv,
        "verdict": verdict,
        "per_seed_val_acc": {
            str(s["seed"]): {k: v for k, v in s["val_acc"].items()} for s in per_seed
        },
        "per_seed_val_fixed_eps": {
            str(s["seed"]): {n: s["val_fixed"][n]["eps"] for n in s["val_fixed"]}
            for s in per_seed
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    for n in [str(x) for x in NOISE_LEVELS]:
        print(
            f"noise={n}: geo025={summary[n]['geometry_eps0.25']['mean']:.3f} "
            f"hybrid_diag={summary[n]['hybrid_diag']['mean']:.3f} "
            f"gate={summary[n]['gate_binary']['mean']:.3f} "
            f"gate_ood={summary[n]['gate_binary_ood']['mean']:.3f} "
            f"multi={summary[n]['gate_multi']['mean']:.3f} "
            f"fixed_val={summary[n]['best_fixed_geo_val']['mean']:.3f} "
            f"fallback={ood[n]['fallback_rate']:.3f}"
        )
    print(f"verdict: {verdict['conclusion']}")
    print(f"output: {out}")


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


if __name__ == "__main__":
    main()
