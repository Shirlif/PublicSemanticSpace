"""Binding stability benchmark (Phase 0).

Measures three things that decide the representation contract:

1. Round-trip error under sensor noise and repeated bind/unbind cycles.
2. Superposition capacity: how many facts can share one memory vector.
3. Slot orthogonality for the chosen slot basis.

Usage:
    python bench/binding_stability.py
    python bench/binding_stability.py --quick

Outputs:
    bench/results/binding_stability.md
    bench/results/binding_stability.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow `python bench/binding_stability.py` from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pss._console import ensure_utf8
from pss.binding import bind, make_slots, slot_orthogonality, unbind

SCHEMES = ("hadamard", "rademacher", "gaussian")


def _unit(rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


def roundtrip_stats(scheme: str, d: int, noise: float, cycles: int, trials: int, rng):
    """Mean/p95 round-trip error and norm drift over ``trials`` random features."""
    errs = []
    drifts = []
    for _ in range(trials):
        slot = make_slots(scheme, d, 1, seed=int(rng.integers(0, 2**31)))[0]
        x0 = _unit(rng, d)
        x = x0.copy()
        for _ in range(cycles):
            b = bind(x, slot)
            if noise > 0:
                b = b + noise * (np.linalg.norm(b) / math.sqrt(d)) * rng.normal(size=d)
            x = unbind(b, slot)
        errs.append(float(np.linalg.norm(x - x0) / np.linalg.norm(x0)))
        drifts.append(float(abs(np.linalg.norm(x) - 1.0)))
    return (
        float(np.mean(errs)),
        float(np.percentile(errs, 95)),
        float(np.mean(drifts)),
        float(np.percentile(drifts, 95)),
    )


def capacity_accuracy(scheme: str, d: int, k: int, noise: float, trials: int, rng):
    """Top-1 retrieval accuracy for ``k`` facts superposed in one memory vector."""
    accs = []
    for _ in range(trials):
        slots = make_slots(scheme, d, k, seed=int(rng.integers(0, 2**31)))
        feats = np.stack([_unit(rng, d) for _ in range(k)])
        memory = np.zeros(d)
        for i in range(k):
            memory += bind(feats[i], slots[i])
        if noise > 0:
            memory = memory + noise * (np.linalg.norm(memory) / math.sqrt(d)) * rng.normal(size=d)
        correct = 0
        for i in range(k):
            q = unbind(memory, slots[i])
            sims = feats @ q / (np.linalg.norm(feats, axis=1) * np.linalg.norm(q))
            correct += int(np.argmax(sims) == i)
        accs.append(correct / k)
    return float(np.mean(accs))


def _fmt_err(x: float) -> str:
    if x == 0.0:
        return "0"
    if x < 1e-6 or x > 1e3:
        return f"{x:.2e}"
    return f"{x:.4f}"


def build_report(roundtrip, capacity, crosstalk, meta) -> str:
    lines = [
        "# Binding stability report (Phase 0)",
        "",
        f"- generated: {meta['generated']}",
        f"- trials: roundtrip={meta['trials']}, capacity={meta['cap_trials']}",
        f"- runtime: {meta['runtime_s']:.1f}s",
        "",
        "## 1. Slot orthogonality (K=4)",
        "",
        "Max absolute off-diagonal entry of the slot Gram matrix. "
        "Hadamard slots are exactly orthogonal; Gaussian slots are only approximately so.",
        "",
        "| scheme | D | max off-diagonal |",
        "|---|---:|---:|",
    ]
    for row in crosstalk:
        lines.append(f"| {row['scheme']} | {row['D']} | {_fmt_err(row['max_offdiag'])} |")

    lines += [
        "",
        "## 2. Round-trip error under noise and repeated cycles",
        "",
        "Relative error `||unbind(bind(x,s),s) - x|| / ||x||`, with Gaussian noise "
        "added to the bound vector each cycle at `noise * RMS(bound)`.",
        "",
        "| scheme | D | noise | cycles | mean err | p95 err | mean norm drift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in roundtrip:
        lines.append(
            f"| {row['scheme']} | {row['D']} | {row['noise']:.2f} | {row['cycles']} | "
            f"{_fmt_err(row['mean'])} | {_fmt_err(row['p95'])} | "
            f"{_fmt_err(row['norm_mean'])} |"
        )

    lines += [
        "",
        "## 3. Superposition capacity (top-1 retrieval)",
        "",
        "`K` facts share one memory vector: `memory = sum_i bind(f_i, s_i)`. "
        "Each fact is retrieved by `unbind(memory, s_i)` and ranked by cosine.",
        "",
        "| scheme | D | K | noise | top-1 accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in capacity:
        lines.append(
            f"| {row['scheme']} | {row['D']} | {row['K']} | {row['noise']:.2f} | "
            f"{row['accuracy']:.3f} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Hadamard and Rademacher slots have `|s_i| = 1/sqrt(D)` for every component, "
        "so division never amplifies near-zero components.",
        "- Gaussian unit vectors contain near-zero components; those components amplify "
        "noise, which is why the Gaussian row degrades fastest.",
        "- Capacity is approximate by construction: cross-talk grows with `K`, and "
        "orthogonal slots do not make element-wise binding orthogonal.",
        "- Contract v0 therefore uses Hadamard slots with a power-of-two `D` (128/256/512).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grid for CI")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--out", type=str, default="bench/results")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(0)

    if args.quick:
        dims = [128]
        noises = [0.0, 0.05]
        cycles = [1, 10]
        cap_ks = [5, 20]
        cap_noises = [0.0, 0.05]
        trials = args.trials or 8
        cap_trials = max(4, trials // 2)
    else:
        dims = [128, 256, 512]
        noises = [0.0, 0.01, 0.05, 0.10]
        cycles = [1, 10, 100]
        cap_ks = [5, 20, 50, 100]
        cap_noises = [0.0, 0.05]
        trials = args.trials or 24
        cap_trials = max(4, trials // 3)

    crosstalk = []
    for scheme in SCHEMES:
        for d in dims:
            slots = make_slots(scheme, d, 4, seed=1)
            crosstalk.append(
                {"scheme": scheme, "D": d, "max_offdiag": slot_orthogonality(slots)}
            )

    roundtrip = []
    for scheme in SCHEMES:
        for d in dims:
            for noise in noises:
                for cyc in cycles:
                    mean, p95, norm_mean, norm_p95 = roundtrip_stats(
                        scheme, d, noise, cyc, trials, rng
                    )
                    roundtrip.append(
                        {
                            "scheme": scheme,
                            "D": d,
                            "noise": noise,
                            "cycles": cyc,
                            "mean": mean,
                            "p95": p95,
                            "norm_mean": norm_mean,
                            "norm_p95": norm_p95,
                        }
                    )

    capacity = []
    for scheme in SCHEMES:
        for d in dims:
            for k in cap_ks:
                if k > d:
                    continue
                for noise in cap_noises:
                    acc = capacity_accuracy(scheme, d, k, noise, cap_trials, rng)
                    capacity.append(
                        {"scheme": scheme, "D": d, "K": k, "noise": noise, "accuracy": acc}
                    )

    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "trials": trials,
        "cap_trials": cap_trials,
        "runtime_s": time.time() - t0,
    }
    report = build_report(roundtrip, capacity, crosstalk, meta)

    out = Path(args.out)
    if args.quick and args.out == "bench/results":
        out = out / "quick"
    out.mkdir(parents=True, exist_ok=True)
    (out / "binding_stability.md").write_text(report, encoding="utf-8")
    (out / "binding_stability.json").write_text(
        json.dumps(
            {"meta": meta, "crosstalk": crosstalk, "roundtrip": roundtrip, "capacity": capacity},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(report)
    print(f"\nWrote {out / 'binding_stability.md'} and .json")


if __name__ == "__main__":
    main()
