"""Benchmark API for contact events: scenarios, channel arms, evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .extractors import ChannelExtractor, make_channel
from .model import ContactEvent, ContactFrame
from .segmenter import Interval, segment_signal, temporal_iou


@dataclass
class ContactConfig:
    width: int = 96
    height: int = 72
    max_steps: int = 400
    capture_every: int = 2
    render: bool = False
    threshold: float = 0.5
    gt_threshold: float = 1.0
    hysteresis: float = 0.0
    min_duration: float = 0.02
    max_gap: float = 0.05
    eps_contact: float = 0.02
    strict: bool = False
    iou_threshold: float = 0.3
    onset_tolerance: float = 0.05
    offset_tolerance: float = 0.1
    seed: int = 0


@dataclass
class ContactScenario:
    scenario_id: str
    frames: list[ContactFrame]
    gt_events: list[ContactEvent]
    metadata: dict = field(default_factory=dict)


ScenarioBuilder = Callable[[ContactConfig], ContactScenario]
ArmFn = Callable[[ContactScenario, ContactConfig], list[ContactEvent]]

SCENARIO_BUILDERS: dict[str, ScenarioBuilder] = {}
ARMS: dict[str, ArmFn] = {}


def register_scenario(name: str):
    def _wrap(fn: ScenarioBuilder) -> ScenarioBuilder:
        SCENARIO_BUILDERS[name] = fn
        return fn

    return _wrap


def register_arm(name: str):
    def _wrap(fn: ArmFn) -> ArmFn:
        ARMS[name] = fn
        return fn

    return _wrap


def build_scenario(name: str, config: ContactConfig) -> ContactScenario:
    if name not in SCENARIO_BUILDERS:
        raise KeyError(f"unknown scenario {name!r}; known: {sorted(SCENARIO_BUILDERS)}")
    return SCENARIO_BUILDERS[name](config)


def all_pairs(frame: ContactFrame) -> list[tuple[str, str]]:
    ids = sorted(frame.poses)
    return [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]]


def _event_from_interval(
    scenario_id: str,
    pair: tuple[str, str],
    interval: Interval,
    scores: np.ndarray,
    times: np.ndarray,
    intensity: np.ndarray,
    location: list[np.ndarray | None],
    channel: str,
    k: int,
) -> ContactEvent:
    mask = (times >= interval.start) & (times <= interval.end)
    if mask.any():
        idx = int(np.argmax(scores[mask]))
        peak_idx = int(np.where(mask)[0][idx])
        conf = float(scores[peak_idx])
        intensity_value = float(intensity[peak_idx]) if peak_idx < len(intensity) else None
        loc = location[peak_idx] if peak_idx < len(location) else None
    else:
        conf = 0.0
        intensity_value = None
        loc = None
    return ContactEvent(
        event_id=f"{scenario_id}-{channel}-{pair[0]}-{pair[1]}-{interval.start:.4f}-{k}",
        subject=pair[0],
        object=pair[1],
        onset=float(interval.start),
        offset=float(interval.end),
        location=loc,
        intensity=intensity_value,
        confidence=conf,
        status="predicted",
        provenance={"channel": channel},
    )


def run_channel(
    scenario: ContactScenario,
    config: ContactConfig,
    *,
    channel: str | ChannelExtractor = "force",
) -> list[ContactEvent]:
    extractor = make_channel(channel) if isinstance(channel, str) else channel
    times = np.asarray([frame.t for frame in scenario.frames], dtype=np.float64)
    events: list[ContactEvent] = []
    for pair in all_pairs(scenario.frames[0]):
        scores = extractor.pair_score(scenario.frames, pair)
        intensity = extractor.pair_intensity(scenario.frames, pair)
        location = extractor.pair_location(scenario.frames, pair)
        intervals = segment_signal(
            times, scores,
            threshold=config.threshold,
            hysteresis=config.hysteresis,
            min_duration=config.min_duration,
            max_gap=config.max_gap,
        )
        for k, interval in enumerate(intervals):
            events.append(
                _event_from_interval(
                    scenario.scenario_id, pair, interval, scores, times,
                    intensity, location, extractor.name, k,
                )
            )
    return sorted(events, key=lambda e: (e.onset, e.subject, e.object))


def evaluate_events(
    predicted: list[ContactEvent],
    ground_truth: list[ContactEvent],
    *,
    iou_threshold: float = 0.3,
    onset_tolerance: float = 0.05,
    offset_tolerance: float = 0.1,
    strict: bool = False,
) -> dict:
    by_pair_gt: dict[tuple[str, str], list[ContactEvent]] = {}
    by_pair_pred: dict[tuple[str, str], list[ContactEvent]] = {}
    for e in ground_truth:
        by_pair_gt.setdefault(e.pair, []).append(e)
    for e in predicted:
        by_pair_pred.setdefault(e.pair, []).append(e)

    pairs = set(by_pair_gt) | set(by_pair_pred)
    tp = fp = fn = 0
    latencies: list[float] = []
    offset_latencies: list[float] = []
    intensity_errors: list[float] = []
    for pair in pairs:
        gts = list(by_pair_gt.get(pair, []))
        preds = list(by_pair_pred.get(pair, []))
        used = [False] * len(preds)
        for gt in gts:
            gi = Interval(gt.onset, gt.offset or gt.onset)
            best = None
            best_iou = 0.0
            for i, pred in enumerate(preds):
                if used[i]:
                    continue
                pi = Interval(pred.onset, pred.offset or pred.onset)
                iou = temporal_iou(gi, pi)
                onset_ok = abs(pred.onset - gt.onset) <= onset_tolerance
                offset_ok = True
                if offset_tolerance is not None:
                    offset_ok = abs(
                        (pred.offset or pred.onset) - (gt.offset or gt.onset)
                    ) <= offset_tolerance
                if strict:
                    matched = iou >= iou_threshold and onset_ok and offset_ok
                else:
                    matched = iou >= iou_threshold or onset_ok
                if matched and iou >= best_iou:
                    best, best_iou = i, iou
            if best is None:
                fn += 1
                continue
            used[best] = True
            tp += 1
            pred = preds[best]
            latencies.append(abs(pred.onset - gt.onset))
            offset_latencies.append(
                abs((pred.offset or pred.onset) - (gt.offset or gt.onset))
            )
            if gt.intensity and pred.intensity is not None:
                intensity_errors.append(
                    abs(pred.intensity - gt.intensity) / max(abs(gt.intensity), 1e-9)
                )
        fp += sum(1 for u in used if not u)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "n_gt": int(len(ground_truth)),
        "n_pred": int(len(predicted)),
        "onset_latency": float(np.mean(latencies)) if latencies else None,
        "offset_latency": float(np.mean(offset_latencies)) if offset_latencies else None,
        "intensity_mae": float(np.mean(intensity_errors)) if intensity_errors else None,
    }


def run_bench(
    arm_names: list[str],
    scenarios: list[ContactScenario],
    config: ContactConfig,
) -> dict:
    from . import extractors  # noqa: F401  (ensures channels exist)

    results: dict[str, list[dict]] = {name: [] for name in arm_names}
    for scenario in scenarios:
        for name in arm_names:
            if name in ARMS:
                predicted = ARMS[name](scenario, config)
            else:
                predicted = run_channel(scenario, config, channel=name)
            results[name].append(
                evaluate_events(
                    predicted,
                    scenario.gt_events,
                    iou_threshold=config.iou_threshold,
                    onset_tolerance=config.onset_tolerance,
                    offset_tolerance=config.offset_tolerance,
                    strict=config.strict,
                )
            )
    aggregated = {}
    for name, rows in results.items():
        keys = ("precision", "recall", "f1", "onset_latency", "offset_latency", "intensity_mae")
        out = {}
        for key in keys:
            vals = [r[key] for r in rows if r.get(key) is not None]
            out[key] = {
                "mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals)) if vals else None,
                "n": len(vals),
            }
        out["tp"] = int(sum(r["tp"] for r in rows))
        out["fp"] = int(sum(r["fp"] for r in rows))
        out["fn"] = int(sum(r["fn"] for r in rows))
        out["n_gt"] = int(sum(r["n_gt"] for r in rows))
        aggregated[name] = out
    return aggregated


__all__ = [
    "ARMS",
    "SCENARIO_BUILDERS",
    "ContactConfig",
    "ContactScenario",
    "all_pairs",
    "build_scenario",
    "evaluate_events",
    "register_arm",
    "register_scenario",
    "run_bench",
    "run_channel",
]
