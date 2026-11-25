"""Claim-gap E3: hybrid relation (geometry + learned) under coordinate noise.

P5b uses a five-region grid. Train the spatial relation head on regions
A/B/C; test on held-out regions D/G. Labels come from the true region centers;
predictions see noisy coordinates. Comparing:

- geometry(eps=0.1): the strict predicate used in PSS;
- geometry(eps=0.25): a wider dead zone, to separate "wider dead zone" from
  "learned calibration";
- learned: SpatialRelationHead trained on noisy A/B/C pairs;
- hybrid: per-axis, use geometry when |delta| >= tau, else the learned head.

Noise 0 means clean coordinates (the learned head still sees its training
noise; the geometry sees no noise).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.eval.relational_bench import NAMES, REGIONS
from pss.memory.typed_relations import SpatialRelationHead

TRAIN_REGIONS = ("A", "B", "C")
NOISE_LEVELS = (0.0, 0.03, 0.08, 0.15)
TAU = 0.25
EPS = 0.1
HEAD_DIM = 4


def _pair_feature(pa, pb) -> np.ndarray:
    dx = pb[0] - pa[0]
    dy = pb[1] - pa[1]
    return np.array(
        [pa[0], pa[1], pb[0], pb[1], dx, dy, abs(dx), abs(dy)], dtype=np.float32
    )


def _pair_labels(pa, pb, eps: float = EPS) -> np.ndarray:
    """left / right / above / below, matching ``_rel(a, b, predicate)``."""
    dx = pb[0] - pa[0]
    dy = pb[1] - pa[1]
    return np.array([dx > eps, dx < -eps, dy < -eps, dy > eps], dtype=np.float32)


def _noisy(center: np.ndarray, noise: float, rng) -> np.ndarray:
    return center + rng.normal(scale=noise, size=2)


def _train_head(seed: int, device: str, n_per_pair: int = 400, epochs: int = 400, noise: float = 0.05):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    coords = np.array([REGIONS[n] for n in NAMES], dtype=np.float32)
    index = {n: i for i, n in enumerate(NAMES)}
    pairs = [
        (a, b)
        for a in TRAIN_REGIONS
        for b in TRAIN_REGIONS
        if a != b
    ]
    X, y = [], []
    for a, b in pairs:
        for _ in range(n_per_pair):
            pa = _noisy(coords[index[a]], noise, rng)
            pb = _noisy(coords[index[b]], noise, rng)
            X.append(_pair_feature(pa, pb))
            # Label from the true region relation, not from noisy coords:
            # the head learns to denoise, not to imitate the noise.
            y.append(_pair_labels(coords[index[a]], coords[index[b]]))
    X = torch.as_tensor(np.stack(X), device=device)
    y = torch.as_tensor(np.stack(y), device=device)
    head = SpatialRelationHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device)
        for start in range(0, len(perm), 128):
            idx = perm[start : start + 128]
            loss = F.binary_cross_entropy_with_logits(head.net(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


def _heldout_pairs():
    return [
        (a, b)
        for a in NAMES
        for b in NAMES
        if a != b and (a not in TRAIN_REGIONS or b not in TRAIN_REGIONS)
    ]


def _predict_geometry(pa, pb, eps: float) -> np.ndarray:
    return _pair_labels(pa, pb, eps=eps)


def _predict_learned(head, pa, pb, device: str) -> np.ndarray:
    x = torch.as_tensor(_pair_feature(pa, pb)[None], device=device)
    with torch.no_grad():
        probs = torch.sigmoid(head.net(x))[0].cpu().numpy()
    return (probs > 0.5).astype(np.float32)


def _predict_hybrid(head, pa, pb, device: str, tau: float = TAU) -> np.ndarray:
    dx = pb[0] - pa[0]
    dy = pb[1] - pa[1]
    learned = _predict_learned(head, pa, pb, device)
    pred = np.zeros(HEAD_DIM, dtype=np.float32)
    if abs(dx) >= tau:
        pred[0] = float(dx > 0)
        pred[1] = float(dx < 0)
    else:
        pred[0:2] = learned[0:2]
    if abs(dy) >= tau:
        pred[2] = float(dy < 0)
        pred[3] = float(dy > 0)
    else:
        pred[2:4] = learned[2:4]
    return pred


def _predict_hybrid_diag(head, pa, pb, device: str, tau: float = TAU) -> np.ndarray:
    """Use learned only in the regime it was trained on (both axes clear).

    The head never saw an aligned axis (dx=0 or dy=0) in A/B/C training, so
    asking it to resolve an aligned pair is out of distribution.  Geometry
    owns the aligned/dead-zone cases; learned owns the diagonal cases, where
    it was trained and can denoise.
    """
    dx = pb[0] - pa[0]
    dy = pb[1] - pa[1]
    if abs(dx) >= tau and abs(dy) >= tau:
        return _predict_learned(head, pa, pb, device)
    return _predict_geometry(pa, pb, eps=tau)


def _evaluate(head, noise: float, seed: int, n_per_pair: int = 200, device: str = "cpu"):
    rng = np.random.default_rng(seed + 1000)
    coords = np.array([REGIONS[n] for n in NAMES], dtype=np.float32)
    index = {n: i for i, n in enumerate(NAMES)}
    methods = ("geometry_eps0.1", "geometry_eps0.25", "learned", "hybrid", "hybrid_diag")
    hits = {m: [] for m in methods}
    for a, b in _heldout_pairs():
        true = _pair_labels(coords[index[a]], coords[index[b]])
        for _ in range(n_per_pair):
            pa = _noisy(coords[index[a]], noise, rng)
            pb = _noisy(coords[index[b]], noise, rng)
            preds = {
                "geometry_eps0.1": _predict_geometry(pa, pb, 0.1),
                "geometry_eps0.25": _predict_geometry(pa, pb, 0.25),
                "learned": _predict_learned(head, pa, pb, device),
                "hybrid": _predict_hybrid(head, pa, pb, device),
                "hybrid_diag": _predict_hybrid_diag(head, pa, pb, device),
            }
            for name, pred in preds.items():
                hits[name].append(float(np.array_equal(pred, true)))
    return {name: float(np.mean(vals)) for name, vals in hits.items()}


def run(seeds: int = 3, device: str = "cpu") -> dict:
    per_seed = []
    for seed in range(seeds):
        head = _train_head(seed, device)
        by_noise = {}
        for noise in NOISE_LEVELS:
            by_noise[str(noise)] = _evaluate(head, noise, seed, device=device)
        per_seed.append({"seed": seed, "by_noise": by_noise})

    summary = {}
    for noise in NOISE_LEVELS:
        key = str(noise)
        summary[key] = {}
        for method in ("geometry_eps0.1", "geometry_eps0.25", "learned", "hybrid", "hybrid_diag"):
            vals = [r["by_noise"][key][method] for r in per_seed]
            summary[key][method] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    verdict = {}
    for noise in NOISE_LEVELS:
        key = str(noise)
        row = summary[key]
        verdict[key] = {
            "hybrid_ge_geometry": row["hybrid"]["mean"] >= row["geometry_eps0.1"]["mean"],
            "hybrid_ge_learned": row["hybrid"]["mean"] >= row["learned"]["mean"],
            "hybrid_gt_geometry": row["hybrid"]["mean"] > row["geometry_eps0.1"]["mean"],
            "hybrid_ge_geometry_tau": row["hybrid"]["mean"] >= row["geometry_eps0.25"]["mean"],
            "hybrid_diag_ge_geometry_tau": row["hybrid_diag"]["mean"] >= row["geometry_eps0.25"]["mean"],
            "hybrid_diag_gt_geometry_tau": row["hybrid_diag"]["mean"] > row["geometry_eps0.25"]["mean"],
        }
    return {
        "train_regions": list(TRAIN_REGIONS),
        "heldout_pairs": _heldout_pairs(),
        "noise_levels": list(NOISE_LEVELS),
        "tau": TAU,
        "eps": EPS,
        "seeds": seeds,
        "per_seed": per_seed,
        "summary": summary,
        "verdict": verdict,
    }


def _report(m: dict) -> str:
    noises = [str(x) for x in m["noise_levels"]]
    lines = [
        "# Claim-gap E3: hybrid relation under coordinate noise",
        "",
        f"> train regions: {m['train_regions']}; held-out pairs: {len(m['heldout_pairs'])}",
        f"> tau={m['tau']}, strict eps={m['eps']}, seeds={m['seeds']}",
        "",
        "| method | " + " | ".join(f"noise={n}" for n in noises) + " |",
        "|---|" + "---:|" * len(noises),
    ]
    for method in ("geometry_eps0.1", "geometry_eps0.25", "learned", "hybrid", "hybrid_diag"):
        cells = [
            f"{m['summary'][n][method]['mean']:.3f} ± {m['summary'][n][method]['std']:.3f}"
            for n in noises
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Verdict",
        "",
        "| noise | hybrid ≥ geometry | hybrid > geometry | hybrid ≥ learned | hybrid ≥ geometry(tau) | hybrid_diag > geometry(tau) |",
        "|---:|---|---|---|---|---|",
    ]
    for n in noises:
        v = m["verdict"][n]
        lines.append(
            f"| {n} | {v['hybrid_ge_geometry']} | {v['hybrid_gt_geometry']} | "
            f"{v['hybrid_ge_learned']} | {v['hybrid_ge_geometry_tau']} | "
            f"{v['hybrid_diag_gt_geometry_tau']} |"
        )
    lines += [
        "",
        "`geometry_eps0.25` is the key control: if hybrid matches a wider geometric dead zone,",
        "the gain comes from the dead zone, not from the learned head. A hybrid claim is only",
        "supported if `hybrid > geometry(tau)` at some noise level.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reports/claim_gaps/hybrid_relations")
    args = ap.parse_args()
    metrics = run(seeds=args.seeds, device=args.device)
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    for n in [str(x) for x in metrics["noise_levels"]]:
        row = metrics["summary"][n]
        print(
            f"noise={n}: geo={row['geometry_eps0.1']['mean']:.3f} "
            f"geo_tau={row['geometry_eps0.25']['mean']:.3f} "
            f"learned={row['learned']['mean']:.3f} "
            f"hybrid={row['hybrid']['mean']:.3f} "
            f"hybrid_diag={row['hybrid_diag']['mean']:.3f}"
        )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
