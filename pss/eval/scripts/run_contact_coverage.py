"""ContactEvent coverage grid: force degradation x geometry mismatch.

Design
------
Physics is expensive and force ``sigma`` / ``dropout`` / ``delay`` only change
the sensor post-processing.  The design therefore assigns several noise
settings to each *physical* scene ``(kind, n_objects, occlusion,
geometry_mismatch, seed)``; one worker builds/captures/GTs that scene once and
sweeps the noise settings on the cached frames.  Clean force, geometry and
fusion scores are memoised per ``(frames, pair)``.

Two stages:

- **coarse**: 24 cells over the full factor space, 2 seeds;
- **refine**: 24 cells in the expected crossover region
  (sigma 10-30%, mismatch 0.5-1.5 x EPS), 3 seeds.

A cell is a ``(physical config, noise setting)`` combination; seeds are
replicates.  The fusion weights are the C2 calibration restricted to
``force + geometry`` (see ``C2_FUSION``), frozen before this grid.  Only the
fusion event threshold is selected, and only on a designated coarse cal
subset; test cells are report-only.

Strict matching: same pair + temporal IoU >= 0.5 + onset <= 50 ms +
offset <= 100 ms.

Example
-------
python -u -m pss.eval.scripts.run_contact_coverage \
  --out eval/results/contact_coverage --jobs 8
"""

from __future__ import annotations

import os

# Keep MuJoCo / BLAS single-threaded inside workers.  Set before importing the
# scientific stack; spawn re-imports this module in each worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.events import ContactConfig, evaluate_events, run_channel
from pss.events.bench_api import ChannelExtractor
from pss.events.extractors import ForceExtractor, GeometryExtractor
from pss.events.scenarios_grid import EPS, GridSpec, build_grid_scenario
from pss.events.sensor_models import NoisyForceExtractor

# ---------------------------------------------------------------------------
# Frozen C2 calibration restricted to force + geometry.
#
# Produced by ``pss.events.fusion.calibrate`` with channels
# ("force", "geometry") on drop/collide/near_miss, train seed 0, cal seed 1,
# noisy force sigma=0.10, delay=1+seed%2, quantization=0.1 N, dropout=0.05
# (the C2 protocol).  These values are *not* fitted on the coverage grid.
C2_FUSION = {
    "channels": ["force", "geometry"],
    "mean": [0.14204717399811834, 0.31192886047269797],
    "std": [0.22564275741116693, 0.4529886683355066],
    "coef": [0.18737663027277832, -0.10603978248010358],
    "intercept": -1.0997470520209012,
    "source": "C2 noisy calibration restricted to force+geometry",
}

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90]
STRICT = {
    "iou_threshold": 0.5,
    "onset_tolerance": 0.05,
    "offset_tolerance": 0.1,
    "strict": True,
}

COARSE_SIGMA = [0.0, 0.05, 0.10, 0.20, 0.40]
REFINE_SIGMA = [0.10, 0.15, 0.20, 0.25, 0.30]
DROPOUT = [0.0, 0.05, 0.20]
DELAY = [0, 1, 2, 4]
COARSE_MISMATCH = [0.0, 0.5, 1.0, 2.0]
REFINE_MISMATCH = [0.5, 0.75, 1.0, 1.25, 1.5]
KINDS = ["drop", "collide", "near_miss"]


# ---------------------------------------------------------------------------
# Cached channel wrappers


class _CachedForce(ForceExtractor):
    """Memoise clean force evidence per (frames, pair)."""

    def __init__(self):
        super().__init__()
        self._score_cache: dict = {}
        self._intensity_cache: dict = {}
        self._location_cache: dict = {}

    def pair_score(self, frames, pair):
        key = (id(frames), pair)
        if key not in self._score_cache:
            self._score_cache[key] = super().pair_score(frames, pair)
        return self._score_cache[key]

    def pair_intensity(self, frames, pair):
        key = (id(frames), pair)
        if key not in self._intensity_cache:
            self._intensity_cache[key] = super().pair_intensity(frames, pair)
        return self._intensity_cache[key]

    def pair_location(self, frames, pair):
        key = (id(frames), pair)
        if key not in self._location_cache:
            self._location_cache[key] = super().pair_location(frames, pair)
        return self._location_cache[key]


class _CachedNoisyForce(NoisyForceExtractor):
    """Memoise the degraded series; the clean source is a shared cache."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._series_cache: dict = {}

    def _series(self, frames, pair):
        key = (id(frames), pair)
        if key not in self._series_cache:
            self._series_cache[key] = super()._series(frames, pair)
        return self._series_cache[key]


class _CachedGeometry(GeometryExtractor):
    def __init__(self):
        super().__init__()
        self._score_cache: dict = {}
        self._intensity_cache: dict = {}

    def pair_score(self, frames, pair):
        key = (id(frames), pair)
        if key not in self._score_cache:
            self._score_cache[key] = super().pair_score(frames, pair)
        return self._score_cache[key]

    def pair_intensity(self, frames, pair):
        key = (id(frames), pair)
        if key not in self._intensity_cache:
            self._intensity_cache[key] = super().pair_intensity(frames, pair)
        return self._intensity_cache[key]


def _logit(scores: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(s / (1.0 - s))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class FusionExtractor(ChannelExtractor):
    """Logistic fusion of a noisy force and a geometry channel.

    ``logit p = b + w_f * z_f + w_g * z_g`` with C2 standardisation.  The
    weights are frozen; only the downstream segmentation threshold is chosen
    (on cal cells).
    """

    name = "fusion"

    def __init__(self, force: ChannelExtractor, geometry: ChannelExtractor, params: dict):
        self.force = force
        self.geometry = geometry
        self.mean = np.asarray(params["mean"], dtype=np.float64)
        self.std = np.asarray(params["std"], dtype=np.float64)
        self.coef = np.asarray(params["coef"], dtype=np.float64)
        self.intercept = float(params["intercept"])
        self._score_cache: dict = {}

    def pair_score(self, frames, pair):
        key = (id(frames), pair)
        if key in self._score_cache:
            return self._score_cache[key]
        sf = np.asarray(self.force.pair_score(frames, pair), dtype=np.float64)
        sg = np.asarray(self.geometry.pair_score(frames, pair), dtype=np.float64)
        zf = (_logit(sf) - self.mean[0]) / self.std[0]
        zg = (_logit(sg) - self.mean[1]) / self.std[1]
        out = _sigmoid(self.intercept + self.coef[0] * zf + self.coef[1] * zg)
        self._score_cache[key] = out
        return out

    def pair_intensity(self, frames, pair):
        return self.force.pair_intensity(frames, pair)

    def pair_location(self, frames, pair):
        return self.force.pair_location(frames, pair)


# ---------------------------------------------------------------------------
# Design


def _balanced(levels, n: int, rng: np.random.Generator) -> list:
    """Repeat ``levels`` to length ``n`` with balanced counts, then shuffle."""
    q, r = divmod(int(n), len(levels))
    values = []
    for i, level in enumerate(levels):
        values.extend([level] * (q + (1 if i < r else 0)))
    rng.shuffle(values)
    return values


def _noise_rows(sigmas, n: int, rng: np.random.Generator) -> list[dict]:
    sigma = _balanced(sigmas, n, rng)
    dropout = _balanced(DROPOUT, n, rng)
    delay = _balanced(DELAY, n, rng)
    rows = [
        {"sigma": float(s), "dropout": float(d), "delay": int(dl)}
        for s, d, dl in zip(sigma, dropout, delay)
    ]
    rng.shuffle(rows)
    return rows


def _balanced_object_modes(rng: np.random.Generator, n: int) -> list[tuple[int, bool]]:
    modes = [(2, False), (4, False), (2, True), (4, True)]
    q, r = divmod(n, len(modes))
    values = modes * q + modes[:r]
    rng.shuffle(values)
    return values


def _physical_configs(stage: str, rng: np.random.Generator) -> list[dict]:
    configs: list[dict] = []
    if stage == "coarse":
        for kind in KINDS:
            for mismatch in COARSE_MISMATCH:
                configs.append({"kind": kind, "mismatch": float(mismatch)})
    else:
        # Four mismatch levels per kind; rotate so all five refine levels
        # appear at least twice across the 12 physical configs.
        rotation = {
            "drop": [0.5, 0.75, 1.0, 1.25],
            "collide": [0.75, 1.0, 1.25, 1.5],
            "near_miss": [0.5, 1.0, 1.25, 1.5],
        }
        for kind in KINDS:
            for mismatch in rotation[kind]:
                configs.append({"kind": kind, "mismatch": float(mismatch)})
    modes = _balanced_object_modes(rng, len(configs))
    for cfg, (n_objects, occlusion) in zip(configs, modes):
        cfg["n_objects"] = int(n_objects)
        cfg["occlusion"] = bool(occlusion)
    rng.shuffle(configs)
    return configs


def build_design(seed: int = 0) -> list[dict]:
    """Return the 24 coarse + 24 refine cells with stage and cell_id."""
    rng = np.random.default_rng(int(seed))
    cells: list[dict] = []

    phys = _physical_configs("coarse", rng)
    noise = _noise_rows(COARSE_SIGMA, len(phys) * 2, rng)
    for i, cfg in enumerate(phys):
        for j in range(2):
            cell = dict(cfg)
            cell.update(noise[2 * i + j])
            cell["stage"] = "coarse"
            cell["cell_id"] = len(cells)
            cells.append(cell)

    phys = _physical_configs("refine", rng)
    noise = _noise_rows(REFINE_SIGMA, len(phys) * 2, rng)
    for i, cfg in enumerate(phys):
        for j in range(2):
            cell = dict(cfg)
            cell.update(noise[2 * i + j])
            cell["stage"] = "refine"
            cell["cell_id"] = len(cells)
            cells.append(cell)
    return cells


def stage_seeds(stage: str) -> list[int]:
    return [0, 1] if stage == "coarse" else [0, 1, 2]


def select_cal_cells(cells: list[dict]) -> list[int]:
    """Eight coarse cells: mismatch 0 and 2 for each kind, plus 2 high-sigma.

    The selection is deterministic and only uses the *design*, not results.
    """
    coarse = [c for c in cells if c["stage"] == "coarse"]
    cal: list[int] = []
    for kind in KINDS:
        group = [c for c in coarse if c["kind"] == kind]
        for mismatch in (0.0, 2.0):
            candidates = [c for c in group if abs(c["mismatch"] - mismatch) < 1e-9]
            if candidates:
                cal.append(int(candidates[0]["cell_id"]))
    # Two extra high-sigma cells from any kind.
    extra = [
        c for c in coarse
        if c["cell_id"] not in cal and c["sigma"] >= 0.40
    ]
    for cell in extra[:2]:
        cal.append(int(cell["cell_id"]))
    return sorted(set(cal))


def group_cells(cells: list[dict]) -> list[dict]:
    """Group cell replicates by physical scene + seed for one build each."""
    groups: dict[tuple, dict] = {}
    for cell in cells:
        for seed in stage_seeds(cell["stage"]):
            key = (
                cell["kind"],
                int(cell["n_objects"]),
                bool(cell["occlusion"]),
                float(cell["mismatch"]),
                int(seed),
            )
            group = groups.get(key)
            if group is None:
                group = {
                    "physical": {
                        "kind": cell["kind"],
                        "n_objects": int(cell["n_objects"]),
                        "occlusion": bool(cell["occlusion"]),
                        "mismatch": float(cell["mismatch"]),
                        "seed": int(seed),
                    },
                    "cells": [],
                }
                groups[key] = group
            group["cells"].append(
                {
                    "cell_id": int(cell["cell_id"]),
                    "stage": cell["stage"],
                    "sigma": float(cell["sigma"]),
                    "dropout": float(cell["dropout"]),
                    "delay": int(cell["delay"]),
                }
            )
    return list(groups.values())


# ---------------------------------------------------------------------------
# Worker


def _row_base(cell, physical, frames, scenario):
    return {
        "cell_id": int(cell["cell_id"]),
        "stage": cell["stage"],
        "seed": int(physical["seed"]),
        "kind": physical["kind"],
        "n_objects": int(physical["n_objects"]),
        "occlusion": bool(physical["occlusion"]),
        "mismatch": float(physical["mismatch"]),
        "geom_mismatch_m": float(physical["mismatch"]) * EPS,
        "sigma": float(cell["sigma"]),
        "dropout": float(cell["dropout"]),
        "delay": int(cell["delay"]),
        "frames": int(len(frames)),
        "gt_events": int(len(scenario.gt_events)),
    }


def _evaluate(predicted, scenario):
    return evaluate_events(predicted, scenario.gt_events, **STRICT)


def _run_group(spec: dict) -> list[dict]:
    """Build one physical scene, then sweep all assigned noise settings."""
    physical = spec["physical"]
    config = ContactConfig(**spec["config"])
    grid = GridSpec(
        kind=physical["kind"],
        n_objects=int(physical["n_objects"]),
        occlusion=bool(physical["occlusion"]),
        geometry_mismatch=float(physical["mismatch"]),
        seed=int(physical["seed"]),
    )
    scenario = build_grid_scenario(grid, config)
    frames = scenario.frames
    rows: list[dict] = []

    geometry = _CachedGeometry()
    clean_force = _CachedForce()
    base = _row_base(spec["cells"][0], physical, frames, scenario)

    # Geometry is noise-independent: one evaluation, copied per cell.
    cfg_geom = replace(config, threshold=0.5)
    geom_pred = run_channel(scenario, cfg_geom, channel=geometry)
    geom_eval = _evaluate(geom_pred, scenario)

    # Unique noise settings in this group (coarse/refine may overlap).
    unique: dict[tuple, dict] = {}
    for cell in spec["cells"]:
        unique.setdefault(
            (cell["sigma"], cell["dropout"], cell["delay"]), cell
        )

    for (sigma, dropout, delay), cell in unique.items():
        noisy = _CachedNoisyForce(
            seed=int(physical["seed"]),
            sigma=float(sigma),
            delay=int(delay),
            dropout=float(dropout),
        )
        # Share the clean force cache across all noise settings.
        noisy._clean = clean_force
        cfg_force = replace(config, threshold=0.5)
        force_pred = run_channel(scenario, cfg_force, channel=noisy)
        force_eval = _evaluate(force_pred, scenario)

        fusion = FusionExtractor(noisy, geometry, spec["fusion"])
        fusion_evals = {}
        for threshold in spec["thresholds"]:
            cfg_fusion = replace(config, threshold=float(threshold))
            pred = run_channel(scenario, cfg_fusion, channel=fusion)
            fusion_evals[float(threshold)] = _evaluate(pred, scenario)

        # Emit one row per cell that uses this noise setting.
        for cell in spec["cells"]:
            if (
                float(cell["sigma"]) != float(sigma)
                or float(cell["dropout"]) != float(dropout)
                or int(cell["delay"]) != int(delay)
            ):
                continue
            row = dict(base)
            row.update(
                {
                    "cell_id": int(cell["cell_id"]),
                    "stage": cell["stage"],
                    "sigma": float(sigma),
                    "dropout": float(dropout),
                    "delay": int(delay),
                }
            )
            for arm, ev, thr in (
                ("force_noisy", force_eval, 0.5),
                ("geometry", geom_eval, 0.5),
            ):
                r = dict(row)
                r.update({"arm": arm, "threshold": thr, **ev})
                rows.append(r)
            for threshold, ev in fusion_evals.items():
                r = dict(row)
                r.update({"arm": "fusion", "threshold": float(threshold), **ev})
                rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Aggregation


def _micro(rows: list[dict]) -> dict:
    tp = int(sum(r["tp"] for r in rows))
    fp = int(sum(r["fp"] for r in rows))
    fn = int(sum(r["fn"] for r in rows))
    n_gt = int(sum(r["n_gt"] for r in rows))
    n_pred = int(sum(r["n_pred"] for r in rows))
    p = tp / (tp + fp) if (tp + fp) > 0 else None
    r = tp / (tp + fn) if (tp + fn) > 0 else None
    if tp == 0 and (fp > 0 or fn > 0):
        f1 = 0.0
    elif p is not None and r is not None and (p + r) > 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = None
    return {
        "precision": None if p is None else float(p),
        "recall": None if r is None else float(r),
        "f1": None if f1 is None else float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "runs": int(len(rows)),
    }


def _strict_score(m: dict) -> float:
    """F1 on GT cells; 0/1 correctness on no-GT cells (documented convention)."""
    if m["n_gt"] > 0:
        return float(m["f1"] or 0.0)
    return 0.0 if m["fp"] > 0 else 1.0


def _latency(rows: list[dict], key: str) -> float | None:
    num = 0.0
    den = 0
    for r in rows:
        value = r.get(key)
        if value is None:
            continue
        num += float(value) * int(r["tp"])
        den += int(r["tp"])
    return float(num / den) if den else None


def _mean_or_none(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _select_threshold(rows: list[dict], cal_ids: list[int], thresholds: list[float]) -> dict:
    cal = set(cal_ids)
    selection = []
    for threshold in thresholds:
        sel = [
            r for r in rows
            if r["arm"] == "fusion"
            and r["stage"] == "coarse"
            and r["cell_id"] in cal
            and abs(float(r["threshold"]) - float(threshold)) < 1e-9
        ]
        m = _micro(sel)
        selection.append({"threshold": float(threshold), **m})
    # Primary: strict micro F1 on cal (FP in no-GT cal cells is already in the
    # denominator).  Tie-break: fewer FP, then closer to the C2 threshold 0.15.
    def key(item):
        f1 = item["f1"] if item["f1"] is not None else -1.0
        return (f1, -item["fp"], -abs(item["threshold"] - 0.15))

    best = max(selection, key=key)
    return {"selected": float(best["threshold"]), "selection": selection}


def _group_by(rows: list[dict], keys: tuple[str, ...]) -> dict:
    out: dict[tuple, list[dict]] = {}
    for row in rows:
        k = tuple(row[key] for key in keys)
        out.setdefault(k, []).append(row)
    return out


def _arm_rows(rows: list[dict], arm: str) -> list[dict]:
    return [r for r in rows if r["arm"] == arm]


def _arm_summary(rows: list[dict]) -> dict:
    arms = {}
    for arm in ("force_noisy", "geometry", "fusion"):
        sel = _arm_rows(rows, arm)
        m = _micro(sel)
        m["onset_latency"] = _latency(sel, "onset_latency")
        m["offset_latency"] = _latency(sel, "offset_latency")
        arms[arm] = m
    return arms


def _false_alarm(rows: list[dict], arm: str) -> dict:
    sel = [r for r in _arm_rows(rows, arm) if int(r["n_gt"]) == 0]
    runs = len(sel)
    fp = int(sum(r["fp"] for r in sel))
    runs_with_fp = int(sum(1 for r in sel if r["fp"] > 0))
    pred = int(sum(r["n_pred"] for r in sel))
    return {
        "runs": runs,
        "fp_events": fp,
        "pred_events": pred,
        "runs_with_fp": runs_with_fp,
        "run_fa_rate": float(runs_with_fp / runs) if runs else None,
        "event_fa_rate": float(fp / pred) if pred else None,
    }


def _factor_table(rows: list[dict], factor: str, levels: list) -> list[dict]:
    """Per-factor micro F1 (GT cells) and false alarm (no-GT cells)."""
    gt = [r for r in rows if int(r["n_gt"]) > 0]
    no_gt = [r for r in rows if int(r["n_gt"]) == 0]
    out = []
    for level in levels:
        gt_sel = [r for r in gt if r[factor] == level]
        ng_sel = [r for r in no_gt if r[factor] == level]
        if not gt_sel and not ng_sel:
            continue
        entry = {"level": level, "n_cells": len({r["cell_id"] for r in gt_sel + ng_sel})}
        for arm in ("force_noisy", "geometry", "fusion"):
            m = _micro([r for r in gt_sel if r["arm"] == arm])
            fa = _false_alarm(gt_sel + ng_sel, arm)
            entry[arm] = {
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "n_gt": m["n_gt"],
                "fa_runs": fa["runs"],
                "fa_fp_events": fa["fp_events"],
                "fa_run_rate": fa["run_fa_rate"],
                "onset_latency": _latency(
                    [r for r in gt_sel if r["arm"] == arm], "onset_latency"
                ),
                "offset_latency": _latency(
                    [r for r in gt_sel if r["arm"] == arm], "offset_latency"
                ),
            }
        entry["delta_f1_fusion_minus_force"] = _delta(
            entry["fusion"]["f1"], entry["force_noisy"]["f1"]
        )
        entry["delta_f1_fusion_minus_geometry"] = _delta(
            entry["fusion"]["f1"], entry["geometry"]["f1"]
        )
        out.append(entry)
    return out


def _delta(a, b) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def _sigma_mismatch_table(rows: list[dict]) -> list[dict]:
    gt = [r for r in rows if int(r["n_gt"]) > 0]
    no_gt = [r for r in rows if int(r["n_gt"]) == 0]
    combos = sorted({(float(r["sigma"]), float(r["mismatch"])) for r in rows})
    out = []
    for sigma, mismatch in combos:
        g = [r for r in gt if r["sigma"] == sigma and r["mismatch"] == mismatch]
        n = [r for r in no_gt if r["sigma"] == sigma and r["mismatch"] == mismatch]
        if not g and not n:
            continue
        entry = {
            "sigma": sigma,
            "mismatch": mismatch,
            "mismatch_m": mismatch * EPS,
            "n_cells": len({r["cell_id"] for r in g + n}),
        }
        for arm in ("force_noisy", "geometry", "fusion"):
            m = _micro([r for r in g if r["arm"] == arm])
            entry[arm] = {
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "n_gt": m["n_gt"],
            }
            entry[arm]["fa"] = _false_alarm(g + n, arm)
        entry["delta_f1_fusion_minus_force"] = _delta(
            entry["fusion"]["f1"], entry["force_noisy"]["f1"]
        )
        entry["delta_f1_fusion_minus_geometry"] = _delta(
            entry["fusion"]["f1"], entry["geometry"]["f1"]
        )
        out.append(entry)
    return out


def _per_cell_table(rows: list[dict]) -> list[dict]:
    out = []
    for cell_id, sel in sorted(_group_by(rows, ("cell_id",)).items()):
        sel = list(sel)
        first = sel[0]
        entry = {
            "cell_id": int(cell_id[0]),
            "stage": first["stage"],
            "kind": first["kind"],
            "n_objects": first["n_objects"],
            "occlusion": first["occlusion"],
            "mismatch": first["mismatch"],
            "sigma": first["sigma"],
            "dropout": first["dropout"],
            "delay": first["delay"],
            "seeds": sorted({int(r["seed"]) for r in sel}),
            "frames": sorted({int(r["frames"]) for r in sel}),
            "gt_events": sorted({int(r["gt_events"]) for r in sel}),
        }
        for arm in ("force_noisy", "geometry", "fusion"):
            m = _micro([r for r in sel if r["arm"] == arm])
            entry[arm] = {
                "f1": m["f1"],
                "strict_score": _strict_score(m),
                "precision": m["precision"],
                "recall": m["recall"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "n_gt": m["n_gt"],
                "n_pred": m["n_pred"],
            }
        entry["delta_f1_fusion_minus_force"] = _delta(
            entry["fusion"]["strict_score"], entry["force_noisy"]["strict_score"]
        )
        entry["delta_f1_fusion_minus_geometry"] = _delta(
            entry["fusion"]["strict_score"], entry["geometry"]["strict_score"]
        )
        out.append(entry)
    return out


def _crossover(per_cell: list[dict], sigma_mismatch: list[dict]) -> dict:
    gains_force = [
        c for c in per_cell
        if c["delta_f1_fusion_minus_force"] is not None
        and c["delta_f1_fusion_minus_force"] >= 0.05
    ]
    gains_geometry = [
        c for c in per_cell
        if c["delta_f1_fusion_minus_geometry"] is not None
        and c["delta_f1_fusion_minus_geometry"] >= 0.05
    ]
    deltas_force = [
        c["delta_f1_fusion_minus_force"] for c in per_cell
        if c["delta_f1_fusion_minus_force"] is not None
    ]
    deltas_geometry = [
        c["delta_f1_fusion_minus_geometry"] for c in per_cell
        if c["delta_f1_fusion_minus_geometry"] is not None
    ]
    sm_gain_force = [
        {
            "sigma": e["sigma"],
            "mismatch": e["mismatch"],
            "delta": e["delta_f1_fusion_minus_force"],
        }
        for e in sigma_mismatch
        if e["delta_f1_fusion_minus_force"] is not None
        and e["delta_f1_fusion_minus_force"] >= 0.05
    ]
    return {
        "n_cells": len(per_cell),
        "cells_with_gain_over_force": len(gains_force),
        "cells_with_gain_over_geometry": len(gains_geometry),
        "max_delta_f1_fusion_minus_force": max(deltas_force) if deltas_force else None,
        "min_delta_f1_fusion_minus_force": min(deltas_force) if deltas_force else None,
        "max_delta_f1_fusion_minus_geometry": max(deltas_geometry) if deltas_geometry else None,
        "min_delta_f1_fusion_minus_geometry": min(deltas_geometry) if deltas_geometry else None,
        "sigma_mismatch_with_gain_over_force": sm_gain_force,
        "gain_cells_over_force": [
            {
                "cell_id": c["cell_id"],
                "kind": c["kind"],
                "sigma": c["sigma"],
                "mismatch": c["mismatch"],
                "n_objects": c["n_objects"],
                "occlusion": c["occlusion"],
                "dropout": c["dropout"],
                "delay": c["delay"],
                "delta": c["delta_f1_fusion_minus_force"],
                "force_f1": c["force_noisy"]["f1"],
                "fusion_f1": c["fusion"]["f1"],
            }
            for c in gains_force
        ],
    }


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _render(payload: dict) -> str:
    cfg = payload["config"]
    lines = [
        "# PSS ContactEvent Coverage（自动生成）",
        "",
        f"- stages: coarse {cfg['n_coarse_cells']} cells x {cfg['coarse_seeds']} seeds; "
        f"refine {cfg['n_refine_cells']} cells x {cfg['refine_seeds']} seeds",
        f"- frames: max {cfg['max_steps']} steps x {cfg['capture_every']} "
        f"(dt {cfg['frame_dt']:.3f} s), early-stop cooldown {cfg['cooldown_frames']} frames",
        f"- strict: IoU>={STRICT['iou_threshold']}, onset<={STRICT['onset_tolerance']} s, "
        f"offset<={STRICT['offset_tolerance']} s",
        f"- fusion threshold selected on cal coarse cells {payload['cal_cells']}: "
        f"{payload['fusion_threshold']}",
        "",
        "## Overall (strict, micro)",
        "",
        "| arm | P | R | strict F1 | tp/fp/fn | onset | offset |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for arm, m in payload["overall"].items():
        lines.append(
            f"| `{arm}` | {_fmt(m['precision'])} | {_fmt(m['recall'])} | "
            f"{_fmt(m['f1'])} | {m['tp']}/{m['fp']}/{m['fn']} | "
            f"{_fmt(m['onset_latency'], 4)} | {_fmt(m['offset_latency'], 4)} |"
        )
    lines += [
        "",
        "## Sigma x mismatch (GT cells drop/collide)",
        "",
        "| sigma | mismatch | n | force F1 | geometry F1 | fusion F1 | "
        "dF1(fusion-force) | dF1(fusion-geom) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for e in payload["sigma_mismatch"]:
        lines.append(
            f"| {e['sigma']:.2f} | {e['mismatch']:.2f} | {e['n_cells']} | "
            f"{_fmt(e['force_noisy']['f1'])} | {_fmt(e['geometry']['f1'])} | "
            f"{_fmt(e['fusion']['f1'])} | "
            f"{_fmt(e['delta_f1_fusion_minus_force'])} | "
            f"{_fmt(e['delta_f1_fusion_minus_geometry'])} |"
        )
    lines += [
        "",
        "## False alarm (near-miss runs)",
        "",
        "| arm | runs | FP events | runs with FP | run FA rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, fa in payload["false_alarm"].items():
        lines.append(
            f"| `{arm}` | {fa['runs']} | {fa['fp_events']} | "
            f"{fa['runs_with_fp']} | {_fmt(fa['run_fa_rate'])} |"
        )
    co = payload["crossover"]
    lines += [
        "",
        "## Crossover",
        "",
        f"- cells with fusion-force >= +0.05: {co['cells_with_gain_over_force']}/{co['n_cells']}",
        f"- max/min dF1(fusion-force): {_fmt(co['max_delta_f1_fusion_minus_force'])} / "
        f"{_fmt(co['min_delta_f1_fusion_minus_force'])}",
        f"- max/min dF1(fusion-geometry): {_fmt(co['max_delta_f1_fusion_minus_geometry'])} / "
        f"{_fmt(co['min_delta_f1_fusion_minus_geometry'])}",
        "",
        "## 口径",
        "",
        "- GT = MuJoCo contact force >= 1 N; clean force shares the GT signal.",
        "- strict = same pair + IoU>=0.5 + onset<=50 ms + offset<=100 ms.",
        "- Geometry mismatch displaces only `frame.scene_model`, never physics/GT.",
        "- Fusion = C2 force+geometry logistic calibration, frozen; threshold on cal only.",
        "- No-GT cells: F1 undefined; the per-cell `strict_score` is 1 if no FP else 0.",
        "- MuJoCo proxy, no real sensors; no sim2real claim.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main


def run(args) -> dict:
    cells = build_design(seed=args.design_seed)
    cal_cells = select_cal_cells(cells)
    config = ContactConfig(
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
        capture_every=args.capture_every,
        render=False,
        threshold=0.5,
        gt_threshold=args.gt_threshold,
        hysteresis=args.hysteresis,
        min_duration=args.min_duration,
        max_gap=args.max_gap,
        seed=0,
    )
    config_dict = {
        "width": int(config.width),
        "height": int(config.height),
        "max_steps": int(config.max_steps),
        "capture_every": int(config.capture_every),
        "render": False,
        "threshold": 0.5,
        "gt_threshold": float(config.gt_threshold),
        "hysteresis": float(config.hysteresis),
        "min_duration": float(config.min_duration),
        "max_gap": float(config.max_gap),
        "eps_contact": float(config.eps_contact),
        "strict": True,
        "iou_threshold": float(STRICT["iou_threshold"]),
        "onset_tolerance": float(STRICT["onset_tolerance"]),
        "offset_tolerance": float(STRICT["offset_tolerance"]),
        "seed": 0,
    }

    groups = group_cells(cells)
    for group in groups:
        group["config"] = config_dict
        group["fusion"] = C2_FUSION
        group["thresholds"] = THRESHOLDS
        group.setdefault("cells", [])
        # sort for deterministic tie-breaking
        group["cells"].sort(key=lambda c: (c["cell_id"], c["stage"]))

    jobs = max(1, int(args.jobs))
    if jobs == 1:
        all_rows = []
        for i, group in enumerate(groups, start=1):
            all_rows.extend(_run_group(group))
            print(f"[group {i}/{len(groups)}] done", flush=True)
    else:
        ctx = mp.get_context("spawn")
        all_rows = []
        with ctx.Pool(processes=jobs) as pool:
            for i, rows in enumerate(pool.imap_unordered(_run_group, groups), start=1):
                all_rows.extend(rows)
                print(f"[group {i}/{len(groups)}] done", flush=True)

    all_rows.sort(
        key=lambda r: (
            r["stage"], r["cell_id"], r["seed"], r["arm"], r["threshold"]
        )
    )

    selection = _select_threshold(all_rows, cal_cells, THRESHOLDS)
    fusion_threshold = float(selection["selected"])
    final_rows = [
        r for r in all_rows
        if r["arm"] != "fusion" or abs(float(r["threshold"]) - fusion_threshold) < 1e-9
    ]

    overall = _arm_summary(final_rows)
    false_alarm = {arm: _false_alarm(final_rows, arm) for arm in ("force_noisy", "geometry", "fusion")}
    sigma_mismatch = _sigma_mismatch_table(final_rows)
    per_cell = _per_cell_table(final_rows)
    crossover = _crossover(per_cell, sigma_mismatch)

    def by_factor(factor, levels):
        return _factor_table(final_rows, factor, levels)

    payload = {
        "stage": "contact_event_coverage",
        "config": {
            "width": int(config.width),
            "height": int(config.height),
            "max_steps": int(config.max_steps),
            "capture_every": int(config.capture_every),
            "frame_dt": float(config.capture_every * 0.002),
            "cooldown_frames": 25,
            "gt_threshold": float(config.gt_threshold),
            "min_duration": float(config.min_duration),
            "max_gap": float(config.max_gap),
            "strict": STRICT,
            "n_coarse_cells": int(sum(1 for c in cells if c["stage"] == "coarse")),
            "n_refine_cells": int(sum(1 for c in cells if c["stage"] == "refine")),
            "coarse_seeds": stage_seeds("coarse"),
            "refine_seeds": stage_seeds("refine"),
            "eps_m": EPS,
            "design_seed": int(args.design_seed),
            "jobs": jobs,
        },
        "fusion": C2_FUSION,
        "thresholds": THRESHOLDS,
        "cal_cells": cal_cells,
        "fusion_threshold": fusion_threshold,
        "threshold_selection": selection["selection"],
        "design": cells,
        "groups": [
            {
                "physical": g["physical"],
                "cells": g["cells"],
            }
            for g in groups
        ],
        "overall": overall,
        "false_alarm": false_alarm,
        "sigma_mismatch": sigma_mismatch,
        "by_sigma": by_factor("sigma", sorted({c["sigma"] for c in cells})),
        "by_mismatch": by_factor("mismatch", sorted({c["mismatch"] for c in cells})),
        "by_kind": by_factor("kind", KINDS),
        "by_occlusion": by_factor("occlusion", [False, True]),
        "by_n_objects": by_factor("n_objects", [2, 4]),
        "by_dropout": by_factor("dropout", DROPOUT),
        "by_delay": by_factor("delay", DELAY),
        "by_stage": by_factor("stage", ["coarse", "refine"]),
        "per_cell": per_cell,
        "crossover": crossover,
        "rows": all_rows,
        "note": (
            "GT = MuJoCo contact force. Strict = same pair + IoU>=0.5 + "
            "onset<=50 ms + offset<=100 ms. Fusion weights are the frozen C2 "
            "force+geometry calibration; only the event threshold is selected, "
            "on designated coarse cal cells. Physics frames are cached per "
            "(kind,n_objects,occlusion,mismatch,seed); sigma/dropout/delay are "
            "post-processing sweeps. Geometry mismatch displaces only the "
            "scene-model pose, never physics or GT."
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "COVERAGE_REPORT.md").write_text(_render(payload), encoding="utf-8")
    return payload


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="eval/results/contact_coverage")
    ap.add_argument("--design-seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--height", type=int, default=72)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--capture-every", type=int, default=4)
    ap.add_argument("--gt-threshold", type=float, default=1.0)
    ap.add_argument("--hysteresis", type=float, default=0.0)
    ap.add_argument("--min-duration", type=float, default=0.02)
    ap.add_argument("--max-gap", type=float, default=0.05)
    args = ap.parse_args()
    payload = run(args)

    print(f"cells: {len(payload['design'])} (coarse+refine)")
    print(f"fusion threshold (cal): {payload['fusion_threshold']}")
    for arm, m in payload["overall"].items():
        print(
            f"{arm:<12} strict F1 {_fmt(m['f1'])} "
            f"P {_fmt(m['precision'])} R {_fmt(m['recall'])} "
            f"tp/fp/fn {m['tp']}/{m['fp']}/{m['fn']}"
        )
    co = payload["crossover"]
    print(
        "crossover: cells with fusion-force >= +0.05: "
        f"{co['cells_with_gain_over_force']}/{co['n_cells']}; "
        f"max dF1 {_fmt(co['max_delta_f1_fusion_minus_force'])}; "
        f"min dF1 {_fmt(co['min_delta_f1_fusion_minus_force'])}"
    )
    print(f"wrote {Path(args.out) / 'coverage.json'}")


if __name__ == "__main__":
    main()
