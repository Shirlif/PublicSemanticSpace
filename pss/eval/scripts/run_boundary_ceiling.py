"""Information ceiling for the noise-boundary contract.

Triggered by the boundary benchmark's NO-GO (D5 stop rule).  The question is
not "can a better model do it" but "is the contract target reachable at all".

Given clean labels and noisy observations, the Bayes decision is computed with
an empirical prior over true coordinate pairs and a Gaussian noise model:

    p(true_pair | observed) ∝ prior(true_pair) * N(observed; true_pair, sigma)

The prior is the train-pair distribution; the Bayes 3-way state per axis is the
weighted vote of the prior samples' states.  The resulting exact-4 is the
information ceiling for any predictor that sees only the noisy coordinates.

Example
-------
python -u -m pss.eval.scripts.run_boundary_ceiling \
  --out eval/results/boundary_bench/ceiling.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.scripts.run_attribution_grid import _write_json
from pss.eval.scripts.run_upper_bound_u3 import _load_cache, _normalise, _pairs
from pss.eval.upper_bound import pair_labels

EPS = 0.02
LEVELS = (0.005, 0.01, 0.02, 0.04)


def _states(d: np.ndarray, eps: float = EPS) -> np.ndarray:
    """0 = positive, 1 = negative, 2 = neither."""

    out = np.full(d.shape, 2, dtype=np.int8)
    out[d > eps] = 0
    out[d < -eps] = 1
    return out


def _pair_matrix(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    return np.stack([pb[:, 0] - pa[:, 0], pb[:, 1] - pa[:, 1]], axis=1)


def _bits_from_states(states: np.ndarray) -> np.ndarray:
    # states: (N, 2) -> [left, right, above, below]
    sx, sy = states[:, 0], states[:, 1]
    return np.stack(
        [
            sx == 0,  # left
            sx == 1,  # right
            sy == 1,  # above
            sy == 0,  # below
        ],
        axis=1,
    )


def _bayes_exact4(
    prior: np.ndarray,
    prior_states: np.ndarray,
    observed: np.ndarray,
    true_bits: np.ndarray,
    sigma: float,
    chunk: int = 500,
) -> float:
    correct = []
    two_sigma2 = 2.0 * float(sigma) ** 2
    for start in range(0, len(observed), chunk):
        obs = observed[start : start + chunk]
        # (chunk, K) squared distances
        d2 = (
            (obs[:, None, 0] - prior[None, :, 0]) ** 2
            + (obs[:, None, 1] - prior[None, :, 1]) ** 2
        )
        w = np.exp(-d2 / two_sigma2)
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        # Weighted vote per axis and state.
        votes = np.zeros((len(obs), 2, 3), dtype=np.float64)
        for state in range(3):
            for axis in range(2):
                votes[:, axis, state] = (
                    w * (prior_states[None, :, axis] == state)
                ).sum(axis=1)
        decided = votes.argmax(axis=2)
        bits = _bits_from_states(decided)
        correct.append(np.all(bits == true_bits[start : start + chunk], axis=1))
    return float(np.concatenate(correct).mean())


def run(args) -> dict:
    root = Path(args.clevr_root)
    train = _load_cache(root, args.train_split)
    test = _load_cache(root, args.test_split)
    pa_tr, pb_tr, _, _ = _pairs(train, args.train_pairs, args.seed)
    pa_te, pb_te, _, _ = _pairs(test, args.test_pairs, args.seed + 1)
    lo = np.minimum(pa_tr.min(0), pb_tr.min(0))
    hi = np.maximum(pa_tr.max(0), pb_tr.max(0))
    pa_tr, pb_tr = _normalise(pa_tr, lo, hi), _normalise(pb_tr, lo, hi)
    pa_te, pb_te = _normalise(pa_te, lo, hi), _normalise(pb_te, lo, hi)

    prior = _pair_matrix(pa_tr, pb_tr)
    if len(prior) > args.prior_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(prior), size=args.prior_samples, replace=False)
        prior = prior[idx]
    prior_states = np.stack(
        [_states(prior[:, 0]), _states(prior[:, 1])], axis=1
    )

    observed_clean = _pair_matrix(pa_te, pb_te)
    true_bits = pair_labels(pa_te, pb_te, eps=EPS).astype(bool)
    rng = np.random.default_rng(args.seed + 7)

    rows = []
    for sigma in LEVELS:
        scores = []
        for _ in range(args.repeats):
            observed = observed_clean + rng.normal(
                0.0, sigma, size=observed_clean.shape
            )
            scores.append(
                _bayes_exact4(prior, prior_states, observed, true_bits, sigma)
            )
        rows.append(
            {
                "sigma": float(sigma),
                "sigma_over_eps": float(sigma / EPS),
                "bayes_exact4": float(np.mean(scores)),
                "bayes_std": float(np.std(scores)),
            }
        )
        print(
            f"[sigma {sigma:.3f}] bayes exact-4 {np.mean(scores):.3f} "
            f"(sigma/eps {sigma/EPS:.2f})",
            flush=True,
        )

    payload = {
        "stage": "boundary_ceiling",
        "clevr_root": str(root),
        "train_pairs": int(len(pa_tr)),
        "test_pairs": int(len(pa_te)),
        "prior_samples": int(len(prior)),
        "eps": EPS,
        "levels": rows,
        "note": (
            "Bayes ceiling with an empirical train-pair prior and Gaussian "
            "coordinate noise; this is the best any predictor can do from the "
            "noisy coordinates alone."
        ),
    }
    _write_json(Path(args.out), payload)
    return payload


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--clevr-root",
        default="data/public/clevr_cogent/CLEVR_CoGenT_v1.0",
    )
    ap.add_argument("--train-split", default="traina")
    ap.add_argument("--test-split", default="vala")
    ap.add_argument("--train-pairs", type=int, default=30000)
    ap.add_argument("--test-pairs", type=int, default=10000)
    ap.add_argument("--prior-samples", type=int, default=3000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", default="eval/results/boundary_bench/ceiling.json"
    )
    args = ap.parse_args()
    payload = run(args)
    print(json.dumps(payload["levels"], indent=2))


if __name__ == "__main__":
    main()
