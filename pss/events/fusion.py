"""Multi-channel fusion and calibration for contact events.

The single-channel extractors return *evidence scores* in ``[0, 1]``; they are
not calibrated probabilities.  This module turns a per-pair score matrix into
a calibrated contact probability and then into ``ContactEvent`` objects.

Fusion model
------------
A L2-regularised logistic model over per-channel logits::

    p(contact | s_1..s_C) = sigmoid(b + sum_c w_c * (logit(s_c) - mu_c) / sd_c)

The model is fitted on a *train* seed, the ridge strength is selected on a
*cal* seed, and the event decision threshold (plus an optional abstain margin)
is selected on the same cal seed.  The test seed is only evaluated, never used
for fitting or selection.

The Bayesian log-likelihood-ratio form is mathematically the same model with
``w_c = 1`` and ``b = prior log-odds``; the fitted weights are the calibrated
version of the per-channel reliability.  We keep a per-channel ``reliability``
in the parameters for provenance and for ``ContactEvidence``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .bench_api import (
    ContactConfig,
    ContactScenario,
    all_pairs,
    build_scenario,
    evaluate_events,
)
from .extractors import ChannelExtractor, make_channel
from .model import ContactEvent, ContactEvidence
from .segmenter import Interval, segment_signal, temporal_iou

DEFAULT_CHANNELS = ("force", "tactile", "geometry", "vision", "audio")
DEFAULT_PARAMS_PATH = (
    Path(__file__).resolve().parents[2]
    / "eval"
    / "results"
    / "contact_event"
    / "c2"
    / "fusion_params.json"
)

# Logit clipping.  Raw channel scores include exact 0 (no evidence) and 1
# (saturated tactile maps); clip so the logit stays finite.
_SCORE_EPS = 1e-4
_LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _logit(scores: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=np.float64), _SCORE_EPS, 1.0 - _SCORE_EPS)
    return np.log(s / (1.0 - s))


@dataclass
class FusionParams:
    """Serialisable fusion parameters.

    ``mean``/``std`` standardise the per-channel logits; ``coef`` are the
    logistic weights on the standardised features; ``intercept`` is the bias
    (prior log-odds in the Bayesian reading).  The remaining fields are the
    calibration decisions made on the cal split.
    """

    channels: tuple[str, ...] = DEFAULT_CHANNELS
    mean: list[float] = field(default_factory=lambda: [0.0] * len(DEFAULT_CHANNELS))
    std: list[float] = field(default_factory=lambda: [1.0] * len(DEFAULT_CHANNELS))
    coef: list[float] = field(
        default_factory=lambda: [1.0] + [0.0] * (len(DEFAULT_CHANNELS) - 1)
    )
    intercept: float = 0.0
    threshold: float = 0.5
    abstain_threshold: float = 0.5
    lam: float = 1.0
    method: str = "logistic"
    reliability: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = asdict(self)
        out["channels"] = list(self.channels)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "FusionParams":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PARAMS_PATH) -> "FusionParams":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def save(self, path: str | Path = DEFAULT_PARAMS_PATH) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @property
    def decision_threshold(self) -> float:
        return float(self.threshold)


# ---------------------------------------------------------------------------
# Per-pair series


def pair_data(
    scenario: ContactScenario,
    pair: tuple[str, str],
    channels: tuple[str, ...] = DEFAULT_CHANNELS,
    *,
    force_extractor: ChannelExtractor | None = None,
) -> dict:
    """Collect raw channel series for one pair over the scenario frame grid.

    ``force_extractor`` replaces the clean force channel (e.g. with
    ``NoisyForceExtractor``); the channel key stays ``"force"`` so the same
    fusion model and evidence schema apply.
    """
    times = np.asarray([float(f.t) for f in scenario.frames], dtype=np.float64)
    extractors = {name: make_channel(name) for name in channels}
    if force_extractor is not None and "force" in extractors:
        extractors["force"] = force_extractor
    scores = np.zeros((len(scenario.frames), len(channels)), dtype=np.float64)
    for j, name in enumerate(channels):
        scores[:, j] = extractors[name].pair_score(scenario.frames, pair)
    force_intensity = np.zeros(len(scenario.frames), dtype=np.float64)
    tactile_intensity = np.zeros(len(scenario.frames), dtype=np.float64)
    locations: list[np.ndarray | None] = [None] * len(scenario.frames)
    if "force" in channels:
        force_intensity = extractors["force"].pair_intensity(scenario.frames, pair)
        locations = extractors["force"].pair_location(scenario.frames, pair)
    if "tactile" in channels:
        tactile_intensity = extractors["tactile"].pair_intensity(scenario.frames, pair)
    return {
        "times": times,
        "scores": scores,
        "force_intensity": force_intensity,
        "tactile_intensity": tactile_intensity,
        "locations": locations,
    }


def _labels(times: np.ndarray, gt_events: list[ContactEvent], pair: tuple[str, str]) -> np.ndarray:
    y = np.zeros(len(times), dtype=np.float64)
    for event in gt_events:
        if event.pair != pair:
            continue
        end = event.offset if event.offset is not None else event.onset
        y[(times >= event.onset - 1e-9) & (times <= end + 1e-9)] = 1.0
    return y


def _design(scores: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (_logit(scores) - mean) / std
    return np.hstack([np.ones((len(z), 1)), z])


def predict_proba(scores: np.ndarray, params: FusionParams) -> np.ndarray:
    mean = np.asarray(params.mean, dtype=np.float64)
    std = np.asarray(params.std, dtype=np.float64)
    std = np.where(std <= 0, 1.0, std)
    coef = np.asarray(params.coef, dtype=np.float64)
    logits = _design(scores, mean, std) @ np.concatenate([[params.intercept], coef])
    return _sigmoid(logits)


# ---------------------------------------------------------------------------
# Fitting


def _fit_logistic(Z: np.ndarray, y: np.ndarray, lam: float, n_iter: int = 200) -> np.ndarray:
    """IRLS logistic regression with an unpenalised intercept."""
    w = np.zeros(Z.shape[1], dtype=np.float64)
    penalty = lam * np.eye(Z.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(n_iter):
        p = _sigmoid(Z @ w)
        weights = p * (1.0 - p) + 1e-9
        grad = Z.T @ (y - p) - penalty @ w
        hess = (Z * weights[:, None]).T @ Z + penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        w += step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return w


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _collect(
    scenarios: list[ContactScenario],
    channels: tuple[str, ...],
    *,
    force_extractor_factory=None,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for scenario in scenarios:
        force_extractor = (
            force_extractor_factory(scenario) if force_extractor_factory else None
        )
        for pair in all_pairs(scenario.frames[0]):
            data = pair_data(
                scenario, pair, channels, force_extractor=force_extractor
            )
            xs.append(data["scores"])
            ys.append(_labels(data["times"], scenario.gt_events, pair))
    if not xs:
        return np.zeros((0, len(channels))), np.zeros(0)
    return np.vstack(xs), np.concatenate(ys)


def reliability_metrics(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = len(y)
    if n == 0:
        return {"n": 0, "log_loss": None, "brier": None, "ece": None, "positive_rate": None}
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return {
        "n": int(n),
        "log_loss": _log_loss(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": float(ece),
        "positive_rate": float(y.mean()),
    }


def _offset_errors(
    predicted: list[ContactEvent],
    ground_truth: list[ContactEvent],
    *,
    iou_threshold: float = 0.3,
    onset_tolerance: float = 0.05,
) -> list[float]:
    """Absolute offset errors for matches under the benchmark's own rule."""
    by_pair_gt: dict[tuple[str, str], list[ContactEvent]] = {}
    by_pair_pred: dict[tuple[str, str], list[ContactEvent]] = {}
    for e in ground_truth:
        by_pair_gt.setdefault(e.pair, []).append(e)
    for e in predicted:
        by_pair_pred.setdefault(e.pair, []).append(e)
    errors: list[float] = []
    for pair in set(by_pair_gt) | set(by_pair_pred):
        gts = list(by_pair_gt.get(pair, []))
        preds = list(by_pair_pred.get(pair, []))
        used = [False] * len(preds)
        for gt in gts:
            gi_start, gi_end = gt.onset, gt.offset if gt.offset is not None else gt.onset
            best = None
            best_iou = 0.0
            for i, pred in enumerate(preds):
                if used[i]:
                    continue
                pi_end = pred.offset if pred.offset is not None else pred.onset
                iou = temporal_iou(
                    Interval(gi_start, gi_end),
                    Interval(pred.onset, pi_end),
                )
                onset_ok = abs(pred.onset - gt.onset) <= onset_tolerance
                if (iou >= iou_threshold or onset_ok) and iou >= best_iou:
                    best, best_iou = i, iou
            if best is None:
                continue
            used[best] = True
            pred = preds[best]
            pred_offset = pred.offset if pred.offset is not None else pred.onset
            gt_offset = gt.offset if gt.offset is not None else gt.onset
            errors.append(abs(float(pred_offset) - float(gt_offset)))
    return errors


def _sweep_threshold(
    scenarios: list[ContactScenario],
    config: ContactConfig,
    params: FusionParams,
    grid: np.ndarray,
    *,
    vary: str = "threshold",
    force_extractor_factory=None,
    tiebreak: str = "f1",
) -> tuple[float, dict]:
    """Sweep the event decision threshold or the abstain threshold on cal."""
    if vary not in ("threshold", "abstain"):
        raise ValueError(f"unknown vary={vary!r}")
    if tiebreak not in ("f1", "f1_offset"):
        raise ValueError(f"unknown tiebreak={tiebreak!r}")
    best_t = 0.5
    best_key = (-float("inf"), -float("inf"))
    best_row: dict = {}
    for t in grid:
        if vary == "threshold":
            candidate = replace(params, threshold=float(t))
            abstain = False
        else:
            candidate = replace(params, abstain_threshold=float(t))
            abstain = True
        rows = []
        offset_errors: list[float] = []
        for scenario in scenarios:
            force_extractor = (
                force_extractor_factory(scenario) if force_extractor_factory else None
            )
            predicted = fuse_events(
                scenario,
                config,
                candidate,
                abstain=abstain,
                force_extractor=force_extractor,
            )
            rows.append(evaluate_events(predicted, scenario.gt_events))
            offset_errors.extend(_offset_errors(predicted, scenario.gt_events))
        f1 = float(np.mean([r["f1"] for r in rows])) if rows else 0.0
        offset_error = float(np.mean(offset_errors)) if offset_errors else 0.0
        key = (f1, -offset_error) if tiebreak == "f1_offset" else (f1, 0.0)
        if key > best_key:
            best_key = key
            best_t = float(t)
            best_row = {
                "f1_mean": f1,
                "precision_mean": float(np.mean([r["precision"] for r in rows])),
                "recall_mean": float(np.mean([r["recall"] for r in rows])),
                "offset_error_mean": offset_error,
            }
    return best_t, best_row


def calibrate(
    train_scenarios: list[ContactScenario],
    cal_scenarios: list[ContactScenario],
    config: ContactConfig,
    *,
    channels: tuple[str, ...] = DEFAULT_CHANNELS,
    lambdas: tuple[float, ...] = _LAMBDAS,
    threshold_grid: tuple[float, ...] = tuple(np.round(np.arange(0.05, 1.0, 0.05), 2)),
    metadata: dict | None = None,
    force_extractor_factory=None,
    threshold_tiebreak: str = "f1",
) -> FusionParams:
    """Fit on train, select ridge + threshold on cal.  Test is not used."""
    x_train, y_train = _collect(
        train_scenarios, channels, force_extractor_factory=force_extractor_factory
    )
    x_cal, y_cal = _collect(
        cal_scenarios, channels, force_extractor_factory=force_extractor_factory
    )
    if len(x_train) == 0:
        raise ValueError("no training rows")

    mean = x_train.mean(axis=0) if len(x_train) else np.zeros(len(channels))
    std = x_train.std(axis=0)
    std = np.where(std <= 1e-9, 1.0, std)

    z_train = (_logit(x_train) - mean) / std
    z_cal = (_logit(x_cal) - mean) / std
    z_train_full = np.hstack([np.ones((len(z_train), 1)), z_train])
    z_cal_full = np.hstack([np.ones((len(z_cal), 1)), z_cal])

    selection = []
    best_lam = float(lambdas[0])
    best_loss = float("inf")
    for lam in lambdas:
        w = _fit_logistic(z_train_full, y_train, lam)
        if len(z_cal_full):
            loss = _log_loss(y_cal, _sigmoid(z_cal_full @ w))
        else:
            loss = _log_loss(y_train, _sigmoid(z_train_full @ w))
        selection.append({"lambda": float(lam), "cal_log_loss": loss})
        if loss < best_loss - 1e-12:
            best_loss, best_lam = loss, float(lam)

    # Refit on train + cal with the selected ridge strength.
    z_all = np.vstack([z_train_full, z_cal_full]) if len(z_cal_full) else z_train_full
    y_all = np.concatenate([y_train, y_cal]) if len(y_cal) else y_train
    w = _fit_logistic(z_all, y_all, best_lam)

    abs_coef = np.abs(w[1:])
    scale = float(abs_coef.max()) if abs_coef.size and abs_coef.max() > 0 else 1.0
    reliability = {
        name: float(abs_coef[i] / scale) for i, name in enumerate(channels)
    }

    base = FusionParams(
        channels=tuple(channels),
        mean=[float(v) for v in mean],
        std=[float(v) for v in std],
        coef=[float(v) for v in w[1:]],
        intercept=float(w[0]),
        lam=best_lam,
        method="logistic",
        reliability=reliability,
    )

    threshold, cal_event = _sweep_threshold(
        cal_scenarios,
        config,
        base,
        np.asarray(threshold_grid),
        force_extractor_factory=force_extractor_factory,
        tiebreak=threshold_tiebreak,
    )
    base = replace(base, threshold=threshold)

    abstain, abstain_event = _sweep_threshold(
        cal_scenarios,
        config,
        replace(base, abstain_threshold=float(threshold)),
        np.asarray([t for t in threshold_grid if t >= threshold] or [threshold]),
        vary="abstain",
        force_extractor_factory=force_extractor_factory,
        tiebreak=threshold_tiebreak,
    ) if threshold < 0.99 else (threshold, {})
    base = replace(base, abstain_threshold=float(max(abstain, threshold)))

    cal_metrics = reliability_metrics(y_cal, predict_proba(x_cal, base)) if len(x_cal) else {}
    train_metrics = reliability_metrics(y_train, predict_proba(x_train, base))
    base.calibration = {
        "train": train_metrics,
        "cal": cal_metrics,
        "lambda_selection": selection,
        "selected_lambda": best_lam,
        "threshold_selection": cal_event,
        "abstain_selection": abstain_event,
        "metadata": metadata or {},
    }
    return base


# ---------------------------------------------------------------------------
# Event construction


def fuse_events(
    scenario: ContactScenario,
    config: ContactConfig,
    params: FusionParams,
    *,
    abstain: bool = False,
    force_extractor: ChannelExtractor | None = None,
) -> list[ContactEvent]:
    """Fuse channel series into ``ContactEvent`` objects for one scenario."""
    channels = tuple(params.channels)
    events: list[ContactEvent] = []
    for pair in all_pairs(scenario.frames[0]):
        data = pair_data(
            scenario, pair, channels, force_extractor=force_extractor
        )
        times = data["times"]
        scores = data["scores"]
        probs = predict_proba(scores, params)
        intervals = segment_signal(
            times,
            probs,
            threshold=params.decision_threshold,
            hysteresis=config.hysteresis,
            min_duration=config.min_duration,
            max_gap=config.max_gap,
        )
        for k, interval in enumerate(intervals):
            mask = (times >= interval.start - 1e-9) & (times <= interval.end + 1e-9)
            if not mask.any():
                continue
            window_idx = np.where(mask)[0]
            peak = int(window_idx[int(np.argmax(probs[window_idx]))])
            confidence = float(probs[peak])
            if abstain and confidence < params.abstain_threshold:
                continue

            raw = {name: float(scores[peak, j]) for j, name in enumerate(channels)}
            force_score = raw.get("force", 0.0)
            tactile_score = raw.get("tactile", 0.0)
            status = "confirmed" if max(force_score, tactile_score) >= 0.5 else "predicted"

            intensity = None
            if data["force_intensity"][peak] > 0:
                intensity = float(data["force_intensity"][peak])
            elif data["tactile_intensity"][peak] > 0:
                intensity = float(data["tactile_intensity"][peak])
            location = data["locations"][peak]

            evidence: list[ContactEvidence] = []
            for j, name in enumerate(channels):
                intensity_j = None
                if name == "force":
                    intensity_j = float(data["force_intensity"][peak])
                elif name == "tactile":
                    intensity_j = float(data["tactile_intensity"][peak])
                evidence.append(
                    ContactEvidence(
                        source=name,
                        timestamp=float(times[peak]),
                        score=float(scores[peak, j]),
                        intensity=intensity_j,
                        location=location if name == "force" else None,
                        reliability=float(params.reliability.get(name, 0.0)),
                        metadata={"logit": float(_logit(scores[peak, j]))},
                    )
                )

            events.append(
                ContactEvent(
                    event_id=(
                        f"{scenario.scenario_id}-fusion-{pair[0]}-{pair[1]}-"
                        f"{interval.start:.4f}-{k}"
                    ),
                    subject=pair[0],
                    object=pair[1],
                    onset=float(interval.start),
                    offset=float(interval.end),
                    location=None if location is None else np.asarray(location, dtype=np.float64),
                    intensity=intensity,
                    confidence=confidence,
                    status=status,
                    evidence=evidence,
                    provenance={
                        "arm": "fusion_abstain" if abstain else "fusion",
                        "method": params.method,
                        "threshold": float(params.decision_threshold),
                        "abstain_threshold": float(params.abstain_threshold) if abstain else None,
                    },
                )
            )
    return sorted(events, key=lambda e: (e.onset, e.subject, e.object))


# ---------------------------------------------------------------------------
# CLI: calibrate and write params


def _build_scenarios(seeds: list[int], config: ContactConfig, names: list[str]) -> list[ContactScenario]:
    out = []
    for seed in seeds:
        cfg = replace(config, seed=seed)
        for name in names:
            out.append(build_scenario(name, cfg))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate contact-event fusion on train/cal seeds")
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--cal-seed", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+", default=["drop", "collide", "near_miss"])
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--capture-every", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gt-threshold", type=float, default=1.0)
    parser.add_argument("--hysteresis", type=float, default=0.0)
    parser.add_argument("--min-duration", type=float, default=0.02)
    parser.add_argument("--max-gap", type=float, default=0.05)
    parser.add_argument("--test-seed", type=int, default=None, help="report-only frame metrics")
    parser.add_argument("--force-model", choices=("clean", "noisy"), default="clean")
    parser.add_argument("--noisy-sigma", type=float, default=0.10)
    parser.add_argument("--noisy-delay", type=int, default=None,
                        help="frame delay; default 1 + seed %% 2")
    parser.add_argument("--noisy-quantization", type=float, default=0.1)
    parser.add_argument("--noisy-dropout", type=float, default=0.05)
    parser.add_argument("--threshold-tiebreak", choices=("f1", "f1_offset"), default="f1",
                        help="cal tie-break when event F1 is flat across thresholds")
    parser.add_argument("--out", default=str(DEFAULT_PARAMS_PATH))
    args = parser.parse_args(argv)

    force_extractor_factory = None
    sensor_meta = {"force_model": args.force_model}
    if args.force_model == "noisy":
        from .sensor_models import NoisyForceExtractor

        def force_extractor_factory(scenario):
            seed = int(scenario.metadata.get("seed", 0))
            return NoisyForceExtractor(
                seed=seed,
                sigma=args.noisy_sigma,
                delay=args.noisy_delay,
                quantization=args.noisy_quantization,
                dropout=args.noisy_dropout,
            )

        sensor_meta.update(
            {
                "sigma": args.noisy_sigma,
                "delay": "1 + seed % 2" if args.noisy_delay is None else args.noisy_delay,
                "quantization_n": args.noisy_quantization,
                "dropout": args.noisy_dropout,
            }
        )

    config = ContactConfig(
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
        capture_every=args.capture_every,
        render=False,
        threshold=args.threshold,
        gt_threshold=args.gt_threshold,
        hysteresis=args.hysteresis,
        min_duration=args.min_duration,
        max_gap=args.max_gap,
        seed=args.train_seeds[0],
    )
    channels = tuple(args.channels)
    train = _build_scenarios(args.train_seeds, config, args.scenarios)
    cal = _build_scenarios([args.cal_seed], config, args.scenarios)
    params = calibrate(
        train,
        cal,
        config,
        channels=channels,
        force_extractor_factory=force_extractor_factory,
        threshold_tiebreak=args.threshold_tiebreak,
        metadata={
            "train_seeds": list(args.train_seeds),
            "cal_seed": args.cal_seed,
            "scenarios": list(args.scenarios),
            "sensor": sensor_meta,
        },
    )

    report = {"params": params.as_dict(), "calibration": params.calibration}
    if args.test_seed is not None:
        test = _build_scenarios([args.test_seed], config, args.scenarios)
        x_test, y_test = _collect(
            test, channels, force_extractor_factory=force_extractor_factory
        )
        report["test_frame_metrics"] = (
            reliability_metrics(y_test, predict_proba(x_test, params)) if len(x_test) else {}
        )
        report["test_seed"] = args.test_seed
    params.save(args.out)
    data_path = Path(args.out).with_name("fusion_calibration.json")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "coef": params.coef,
        "intercept": params.intercept,
        "threshold": params.threshold,
        "abstain_threshold": params.abstain_threshold,
        "lambda": params.lam,
        "reliability": params.reliability,
        "cal": params.calibration.get("cal"),
    }, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {data_path}")


__all__ = [
    "DEFAULT_CHANNELS",
    "DEFAULT_PARAMS_PATH",
    "FusionParams",
    "calibrate",
    "fuse_events",
    "main",
    "pair_data",
    "predict_proba",
    "reliability_metrics",
]
