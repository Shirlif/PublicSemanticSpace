"""Pre-registered expectations for the open-category stream plan.

Executable form of ``docs/plans/PSS-开放类别流实验计划.md`` §3.  Bars are fixed
before the run; a failed arm is reported, never relaxed after seeing results.

Two kinds of checks:

- **absolute** bars on one arm (e.g. ``active_cluster.novel_acc >= 0.80``);
- **relative** bars on a paired comparison (e.g. ``random_query`` must trail
  ``active_cluster`` by 0.10 at the same budget).  Relative bars take a
  separate mapping because they cannot be expressed as one arm's absolute
  value without knowing the other arm.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "EXPECTED",
    "RelativeRule",
    "check_open_stream_preregistration",
]


@dataclass(frozen=True)
class StreamExpectation:
    novel_acc_min: float | None = None
    novel_acc_max: float | None = None
    old_forgetting_max: float | None = None
    old_forgetting_min: float | None = None
    unknown_auroc_min: float | None = None
    cluster_ari_min: float | None = None
    top1_match_min: float | None = None
    spearman_min: float | None = None
    ece_max: float | None = None
    update_false_accept_max: float | None = None
    rollback_rate_max: float | None = None
    review_rate_max: float | None = None
    label_efficiency_max: float | None = None


EXPECTED: dict[str, StreamExpectation] = {
    # Main arm: limited labels must reach the new classes without forgetting.
    "active_cluster": StreamExpectation(
        novel_acc_min=0.80,
        old_forgetting_max=0.030,
    ),
    # Positive control: random query must be materially worse at the same budget.
    # The paired rule is the primary check; this absolute cap is a fallback.
    "random_query": StreamExpectation(novel_acc_max=0.70),
    # No learning: new classes cannot be classified.
    "frozen": StreamExpectation(novel_acc_max=0.20),
    # Negative control: no replay must forget, or fail to reach the new classes.
    "active_no_replay": StreamExpectation(old_forgetting_min=0.100),
    # Tier C is allowed to be harder, but the novelty signal must still work.
    "tier_c": StreamExpectation(
        novel_acc_min=0.60,
        unknown_auroc_min=0.80,
    ),
    # Reference arms (no bar): oracle budget, self-training, model-cell audit.
    "oracle_full": StreamExpectation(),
    "self_train": StreamExpectation(),
    "active_replay": StreamExpectation(),
    "active_replay_guard": StreamExpectation(),
    "model_cells": StreamExpectation(),
}


@dataclass(frozen=True)
class RelativeRule:
    """One paired check: ``left_metric - right_metric`` must satisfy a bound."""

    left: str
    right: str
    metric: str
    op: str  # ">=" or "<="
    bound: float
    description: str


RELATIVE_RULES: dict[str, RelativeRule] = {
    "random_query_vs_active": RelativeRule(
        left="random_query",
        right="active_cluster",
        metric="novel_acc",
        op="<=",
        bound=-0.10,
        description="random query must trail active by at least 0.10 at the same budget",
    ),
    "no_replay_vs_replay": RelativeRule(
        left="active_no_replay",
        right="active_replay",
        metric="old_forgetting",
        op=">=",
        bound=0.07,
        description="no replay must forget at least 0.07 more than the replay arm",
    ),
}


_UPPER = {
    "novel_acc_max": "novel_acc",
    "old_forgetting_max": "old_forgetting",
    "ece_max": "ece",
    "update_false_accept_max": "update_false_accept",
    "rollback_rate_max": "rollback_rate",
    "review_rate_max": "review_rate",
    "label_efficiency_max": "label_efficiency",
}
_LOWER = {
    "novel_acc_min": "novel_acc",
    "old_forgetting_min": "old_forgetting",
    "unknown_auroc_min": "unknown_auroc",
    "cluster_ari_min": "cluster_ari",
    "top1_match_min": "top1_match",
    "spearman_min": "spearman",
}


def _check_absolute(observed: Mapping[str, Mapping[str, object]]) -> tuple[list[str], int]:
    mismatches: list[str] = []
    checked = 0
    for arm, expected in EXPECTED.items():
        actual = observed.get(arm)
        if actual is None:
            continue
        checked += 1
        for field_name, metric in _UPPER.items():
            cap = getattr(expected, field_name)
            if cap is None:
                continue
            if metric not in actual or actual[metric] is None:
                mismatches.append(
                    f"{arm}.{metric}: expected <= {cap}, got none (not measured)"
                )
                continue
            value = float(actual[metric])
            if value > float(cap):
                mismatches.append(
                    f"{arm}.{metric}: expected <= {cap}, got {value:.4f}"
                )
        for field_name, metric in _LOWER.items():
            floor = getattr(expected, field_name)
            if floor is None:
                continue
            if metric not in actual or actual[metric] is None:
                mismatches.append(
                    f"{arm}.{metric}: expected >= {floor}, got none (not measured)"
                )
                continue
            value = float(actual[metric])
            if value < float(floor):
                mismatches.append(
                    f"{arm}.{metric}: expected >= {floor}, got {value:.4f}"
                )
    return mismatches, checked


def _check_relative(
    observed: Mapping[str, Mapping[str, object]],
    relative: Mapping[str, float] | None,
) -> tuple[list[str], int]:
    if not relative:
        return [], 0
    mismatches: list[str] = []
    checked = 0
    for key, rule in RELATIVE_RULES.items():
        if key not in relative:
            continue
        checked += 1
        value = float(relative[key])
        ok = value >= rule.bound if rule.op == ">=" else value <= rule.bound
        if not ok:
            mismatches.append(
                f"{key}: {rule.description}; got {value:+.4f}"
            )

    # OR rules need custom handling.
    if "self_train_old_forgetting" in relative or "self_train_novel_delta" in relative:
        checked += 1
        forget = float(relative.get("self_train_old_forgetting", float("nan")))
        delta = float(relative.get("self_train_novel_delta", float("nan")))
        if not ((not np_isnan(forget) and forget >= 0.10)
                or (not np_isnan(delta) and delta <= -0.10)):
            mismatches.append(
                "self_train: negative control must forget (>=0.10) or trail "
                f"active by >=0.10; got forgetting={forget:+.4f}, delta={delta:+.4f}"
            )
    if "model_cells_top1_match" in relative or "model_cells_spearman" in relative:
        checked += 1
        top1 = float(relative.get("model_cells_top1_match", float("nan")))
        spearman = float(relative.get("model_cells_spearman", float("nan")))
        if not ((not np_isnan(top1) and top1 >= 0.60)
                or (not np_isnan(spearman) and spearman >= 0.70)):
            mismatches.append(
                "model_cells: top1 must be >=0.60 or Spearman >=0.70; got "
                f"top1={top1:+.4f}, spearman={spearman:+.4f}"
            )
    return mismatches, checked


def np_isnan(value: float) -> bool:
    return value != value


def check_open_stream_preregistration(
    observed: Mapping[str, Mapping[str, object]],
    relative: Mapping[str, float] | None = None,
) -> dict:
    """Compare observed arms against the frozen open-stream expectations."""

    mismatches, checked = _check_absolute(observed)
    rel_mismatches, rel_checked = _check_relative(observed, relative)
    return {
        "ok": not mismatches and not rel_mismatches,
        "mismatches": mismatches + rel_mismatches,
        "checked": checked + rel_checked,
        "absolute_checked": checked,
        "relative_checked": rel_checked,
    }
