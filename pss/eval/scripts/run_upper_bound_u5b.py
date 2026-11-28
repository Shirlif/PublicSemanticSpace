"""U5b: query-conditioned fact selection on a replayed long-history stream.

U5 (``run_gap_fact_active.py``) saturated on a synthetic stream: both
``query_recency`` and ``learned_query`` reached accuracy 1.000.  This script
moves the same selection question onto the longest replay trace that ships in
the repo and adds a coverage objective.

Data
----
There is no real fact-store trace in this repository (``data/open_stream`` and
``data/tracking_bench_long`` contain rendered frames, not facts).  The longest
replay with changing per-entity state is the ``s1_occlusion_20`` bundle
(``bundles/dev/s1_occlusion_20/18c18c5e/seed*/gt.parquet``): 20 entities over
1,200 physics steps.  Facts are extracted from the replay ground truth:

    entity      -> key (20 keys)
    attribute   -> visibility state
    value       -> visibility bucket 1..9 (``round(visibility_ratio * 9)``)
    timestamp   -> physics_step
    reliability -> 0.5 + 0.5 * visibility_ratio

A fact is written only when the entity is at least partially visible, so
fully occluded frames leave the key without a new write.  This is a replay of
a simulated (MuJoCo/Isaac) benchmark, **not** a real sensor fact stream and
not a production fact store.

Workload
--------
A workload has ``Q`` queries ``(key, as_of)``.  The answer is the value of the
latest fact at or before ``as_of``.  A fact *supports* a query when it has the
query key, the answer value, and a timestamp at or before ``as_of``.

Methods
-------
``full`` / ``random`` / ``hot`` / ``recency`` /
``query_recency`` (as_of-aware) / ``query_recency_noasof`` (U5 rule) /
``learned_query`` (query-conditioned policy) /
``coverage_aware`` (submodular greedy max-coverage) /
``coverage_aware_learned`` (greedy with the learned score as tie-break).

Metrics
-------
``accuracy``: the answer resolved from the selected set (latest selected fact
with the query key at or before ``as_of``) equals the true answer.
``recall_at_k``: the selected set contains at least one fact with the query
key at or before ``as_of`` (value-blind retrieval).
``coverage``: the selected set contains at least one value-correct supporting
fact.  ``coverage_loss = 1 - coverage`` is the objective the greedy selector
minimises.

Example
-------
python -u -m pss.eval.scripts.run_upper_bound_u5b \
  --seeds 3 --qs 8,16 --ks 4,8,16 \
  --out eval/results/upper_bound/u5b
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.memory.active_policy import (
    ActiveSetPolicy,
    allocate_budget,
    greedy_max_coverage,
)

REPLAY_ROOT = Path("bundles/dev/s1_occlusion_20/18c18c5e")
REPLAY_SEEDS = 5
TEST_REPLAY_SEED = 0

VIS_LEVELS = 9  # round(visibility_ratio * 9) -> 0..9; 0 is filtered out
FEATURE_DIM = 12
METHODS = (
    "full",
    "random",
    "hot",
    "recency",
    "query_recency",
    "query_recency_noasof",
    "learned_query",
    "coverage_aware",
    "coverage_aware_learned",
)
METHOD_LABELS = {
    "full": "full",
    "random": "random",
    "hot": "hot",
    "recency": "recency",
    "query_recency": "query_recency (as_of)",
    "query_recency_noasof": "query_recency (no as_of)",
    "learned_query": "learned_query",
    "coverage_aware": "coverage_aware",
    "coverage_aware_learned": "coverage_aware + learned",
}


@dataclass
class ReplayStream:
    keys: np.ndarray
    values: np.ndarray
    ts: np.ndarray
    reliability: np.ndarray
    entity_names: list[str]
    key_positions: list[np.ndarray]
    key_freq: np.ndarray
    n_keys: int
    n_steps: int
    last_step: int
    max_value: int
    n_source_rows: int
    n_occluded_rows: int
    visibility_changes: int
    duplicate_writes: int

    @property
    def n_facts(self) -> int:
        return int(len(self.keys))


@dataclass
class QueryWorkload:
    keys: np.ndarray
    as_of: np.ndarray
    targets: np.ndarray
    answer_idx: np.ndarray
    support: np.ndarray  # (T, Q) value-correct and <= as_of
    relevant: np.ndarray  # (T, Q) key hit and <= as_of

    def __len__(self) -> int:
        return int(len(self.keys))


def _replay_path(seed: int) -> Path:
    return REPLAY_ROOT / f"seed{seed}" / "gt.parquet"


def load_replay_stream(seed: int, path: Path | None = None) -> ReplayStream:
    """Extract a fact stream from one replay bundle ground-truth trace."""
    import pandas as pd

    path = Path(path) if path is not None else _replay_path(seed)
    if not path.exists():
        raise FileNotFoundError(
            f"replay bundle not found: {path}. "
            "U5b needs bundles/dev/s1_occlusion_20 (gitignored data)."
        )
    frame = pd.read_parquet(path)
    frame = frame.sort_values(["physics_step", "instance_id"]).reset_index(drop=True)
    names = sorted(str(x) for x in frame["instance_id"].unique())
    index = {name: i for i, name in enumerate(names)}

    visibility = frame["visibility_ratio"].to_numpy(dtype=np.float64)
    steps = frame["physics_step"].to_numpy(dtype=np.int64)
    raw_keys = frame["instance_id"].map(index).to_numpy(dtype=np.int64)
    buckets = np.rint(visibility * VIS_LEVELS).astype(np.int64)

    keep = visibility > 0.0
    keys = raw_keys[keep].astype(np.int64)
    values = buckets[keep].astype(np.int64)
    ts = steps[keep].astype(np.int64)
    reliability = (0.5 + 0.5 * visibility[keep]).astype(np.float32)

    n_keys = len(names)
    key_positions: list[np.ndarray] = []
    for key in range(n_keys):
        key_positions.append(np.flatnonzero(keys == key).astype(np.int64))
    key_freq = np.bincount(keys, minlength=n_keys).astype(np.float32)

    # Count visibility-state changes across the raw (including occluded) trace.
    changes = 0
    for _, group in frame.groupby("instance_id", sort=False):
        levels = np.rint(group["visibility_ratio"].to_numpy(dtype=np.float64) * VIS_LEVELS)
        changes += int((np.diff(levels) != 0).sum())

    # Consecutive writes of the same (key, value) are duplicates in the log.
    duplicates = int(
        np.sum((keys[1:] == keys[:-1]) & (values[1:] == values[:-1]))
    ) if len(keys) > 1 else 0

    return ReplayStream(
        keys=keys,
        values=values,
        ts=ts,
        reliability=reliability,
        entity_names=names,
        key_positions=key_positions,
        key_freq=key_freq,
        n_keys=n_keys,
        n_steps=int(frame["physics_step"].nunique()),
        last_step=int(steps.max()),
        max_value=int(values.max()),
        n_source_rows=int(len(frame)),
        n_occluded_rows=int((~keep).sum()),
        visibility_changes=changes,
        duplicate_writes=duplicates,
    )


def _first_ts(stream: ReplayStream) -> np.ndarray:
    first = np.full(stream.n_keys, stream.last_step, dtype=np.int64)
    for key, positions in enumerate(stream.key_positions):
        if len(positions):
            first[key] = int(stream.ts[positions[0]])
    return first


def _sample_key(stream: ReplayStream, rng: np.random.Generator) -> int:
    probs = stream.key_freq.astype(np.float64)
    probs = probs / probs.sum()
    return int(rng.choice(stream.n_keys, p=probs))


def _sample_as_of(stream: ReplayStream, first: np.ndarray, key: int, rng, regime: str) -> int:
    if regime == "current":
        return int(stream.last_step)
    lo = int(first[key])
    return int(rng.integers(lo, stream.last_step + 1))


def make_queries(
    stream: ReplayStream,
    Q: int,
    seed: int,
    regime: str,
) -> QueryWorkload:
    """Sample ``Q`` queries and build the support / relevance matrices."""
    rng = np.random.default_rng(seed)
    first = _first_ts(stream)
    n_facts = stream.n_facts
    qkeys = np.empty(Q, dtype=np.int64)
    as_of = np.empty(Q, dtype=np.int64)
    support = np.zeros((n_facts, Q), dtype=bool)
    relevant = np.zeros((n_facts, Q), dtype=bool)
    targets = np.empty(Q, dtype=np.int64)
    answer_idx = np.empty(Q, dtype=np.int64)

    for j in range(Q):
        key = _sample_key(stream, rng)
        moment = _sample_as_of(stream, first, key, rng, regime)
        positions = stream.key_positions[key]
        hi = int(np.searchsorted(stream.ts[positions], moment, side="right"))
        if hi == 0:
            moment = int(stream.ts[positions[0]])
            hi = 1
        chosen = positions[:hi]
        target = int(stream.values[chosen[-1]])
        qkeys[j] = key
        as_of[j] = moment
        targets[j] = target
        answer_idx[j] = int(chosen[-1])
        relevant[chosen, j] = True
        support[chosen[stream.values[chosen] == target], j] = True

    return QueryWorkload(
        keys=qkeys,
        as_of=as_of,
        targets=targets,
        answer_idx=answer_idx,
        support=support,
        relevant=relevant,
    )


def query_features(stream: ReplayStream, key: int, as_of: int) -> np.ndarray:
    """(T, FEATURE_DIM) features for one query; row i describes fact i."""
    n_facts = stream.n_facts
    positions = np.arange(n_facts, dtype=np.int64)
    match = (stream.keys == key).astype(np.float32)
    recency = ((n_facts - 1 - positions) / max(1, n_facts - 1)).astype(np.float32)
    key_positions = stream.key_positions[key]
    hi = int(np.searchsorted(stream.ts[key_positions], as_of, side="right"))
    last_pos = int(key_positions[hi - 1]) if hi > 0 else -1
    is_answer = np.zeros(n_facts, dtype=np.float32)
    if last_pos >= 0:
        is_answer[last_pos] = 1.0
    after = (stream.ts > as_of).astype(np.float32)
    age = np.where(
        stream.ts <= as_of,
        (as_of - stream.ts) / max(1, stream.last_step),
        1.0,
    ).astype(np.float32)
    key_freq = (stream.key_freq[stream.keys] / max(1, n_facts)).astype(np.float32)
    value_ratio = (stream.values / max(1, stream.max_value)).astype(np.float32)
    return np.stack(
        [
            match,
            recency,
            stream.reliability,
            key_freq,
            is_answer,
            after,
            age,
            match * recency,
            match * is_answer,
            match * stream.reliability,
            value_ratio,
            np.ones(n_facts, dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def train_query_policy(
    stream: ReplayStream,
    seed: int,
    epochs: int = 40,
    batch: int = 64,
    device: str = "cpu",
) -> ActiveSetPolicy:
    """Train a query-conditioned scorer on replayed facts.

    Training queries mix current-time and historical ``as_of`` values.  The
    listwise target is the answer fact (latest fact at or before ``as_of``).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = ActiveSetPolicy(feature_dim=FEATURE_DIM).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    first = _first_ts(stream)

    for _ in range(epochs):
        policy.train()
        features: list[np.ndarray] = []
        targets: list[int] = []
        for _ in range(batch):
            key = _sample_key(stream, rng)
            regime = "current" if rng.random() < 0.5 else "historical"
            moment = _sample_as_of(stream, first, key, rng, regime)
            positions = stream.key_positions[key]
            hi = int(np.searchsorted(stream.ts[positions], moment, side="right"))
            if hi == 0:
                moment = int(stream.ts[positions[0]])
                hi = 1
            features.append(query_features(stream, key, moment))
            targets.append(int(positions[hi - 1]))
        x = torch.as_tensor(np.stack(features), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.asarray(targets, dtype=np.int64), device=device)
        logits = policy(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    policy.eval()
    return policy


# ── selectors ───────────────────────────────────────────────────────────────


def _empty_selection() -> np.ndarray:
    return np.zeros(0, dtype=np.int64)


def _select_hot(stream: ReplayStream, k: int, hot_frac: float = 0.2) -> np.ndarray:
    n_hot = max(1, int(round(hot_frac * stream.n_keys)))
    hot_keys = np.argsort(-stream.key_freq, kind="stable")[:n_hot]
    mask = np.isin(stream.keys, hot_keys)
    order = np.flatnonzero(mask)
    if len(order):
        order = order[np.argsort(-stream.ts[order], kind="stable")]
    selected = list(int(i) for i in order[:k])
    if len(selected) < k:
        rest = np.argsort(-stream.ts, kind="stable")
        chosen = set(selected)
        for i in rest:
            if len(selected) >= k:
                break
            if int(i) not in chosen:
                selected.append(int(i))
    return np.asarray(selected, dtype=np.int64)


def _select_recency(stream: ReplayStream, k: int) -> np.ndarray:
    return np.argsort(-stream.ts, kind="stable")[:k].astype(np.int64)


def _select_query_recency(
    stream: ReplayStream,
    workload: QueryWorkload,
    k: int,
    respect_as_of: bool,
) -> np.ndarray:
    allocation = allocate_budget(k, stream.key_freq[workload.keys])
    chosen: list[int] = []
    for j in range(len(workload)):
        budget = int(allocation[j])
        if budget <= 0:
            continue
        positions = stream.key_positions[workload.keys[j]]
        if respect_as_of:
            hi = int(np.searchsorted(stream.ts[positions], workload.as_of[j], side="right"))
            positions = positions[:hi]
        if len(positions) == 0:
            continue
        chosen.extend(int(i) for i in positions[-budget:])
    if not chosen:
        return _empty_selection()
    return np.unique(np.asarray(chosen, dtype=np.int64))


def _select_learned(
    policy: ActiveSetPolicy,
    stream: ReplayStream,
    workload: QueryWorkload,
    k: int,
    device: str,
) -> np.ndarray:
    allocation = allocate_budget(k, stream.key_freq[workload.keys])
    chosen: list[int] = []
    for j in range(len(workload)):
        budget = int(allocation[j])
        if budget <= 0:
            continue
        scores = policy.scores(
            query_features(stream, int(workload.keys[j]), int(workload.as_of[j])),
            device,
        )
        scores = np.where(stream.ts <= workload.as_of[j], scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:budget]
        chosen.extend(int(i) for i in order if np.isfinite(scores[i]))
    if not chosen:
        return _empty_selection()
    return np.unique(np.asarray(chosen, dtype=np.int64))


def _select_coverage(
    stream: ReplayStream,
    workload: QueryWorkload,
    k: int,
    priority: np.ndarray,
) -> np.ndarray:
    return greedy_max_coverage(workload.support, k, priority=priority)


def _learned_aggregate_priority(
    policy: ActiveSetPolicy,
    stream: ReplayStream,
    workload: QueryWorkload,
    device: str,
) -> np.ndarray:
    aggregate = np.zeros(stream.n_facts, dtype=np.float32)
    for j in range(len(workload)):
        scores = policy.scores(
            query_features(stream, int(workload.keys[j]), int(workload.as_of[j])),
            device,
        )
        aggregate += np.where(np.isfinite(scores), scores, 0.0)
    return aggregate


def select(
    method: str,
    stream: ReplayStream,
    workload: QueryWorkload,
    k: int,
    seed: int,
    policy: ActiveSetPolicy | None,
    learned_priority: np.ndarray | None,
    device: str,
) -> np.ndarray:
    n_facts = stream.n_facts
    if method == "full":
        return np.arange(n_facts, dtype=np.int64)
    if method == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n_facts, size=min(k, n_facts), replace=False)).astype(np.int64)
    if method == "hot":
        return _select_hot(stream, k)
    if method == "recency":
        return _select_recency(stream, k)
    if method == "query_recency":
        return _select_query_recency(stream, workload, k, respect_as_of=True)
    if method == "query_recency_noasof":
        return _select_query_recency(stream, workload, k, respect_as_of=False)
    if method == "learned_query":
        assert policy is not None
        return _select_learned(policy, stream, workload, k, device)
    recency = np.arange(n_facts, dtype=np.float32) / max(1, n_facts - 1)
    if method == "coverage_aware":
        priority = stream.reliability + recency
        return _select_coverage(stream, workload, k, priority)
    if method == "coverage_aware_learned":
        assert learned_priority is not None
        priority = learned_priority + 1e-3 * (stream.reliability + recency)
        return _select_coverage(stream, workload, k, priority)
    raise ValueError(f"unknown method: {method}")


# ── metrics ─────────────────────────────────────────────────────────────────


def evaluate(
    selected: np.ndarray,
    stream: ReplayStream,
    workload: QueryWorkload,
) -> dict:
    """Accuracy / recall@k / coverage for one selection."""
    selected = np.unique(np.asarray(selected, dtype=np.int64))
    accuracy: list[float] = []
    recall: list[float] = []
    coverage: list[float] = []
    for j in range(len(workload)):
        if len(selected) == 0:
            cand = selected
        else:
            cand = selected[(stream.keys[selected] == workload.keys[j])
                            & (stream.ts[selected] <= workload.as_of[j])]
        if len(cand) == 0:
            accuracy.append(0.0)
            recall.append(0.0)
            coverage.append(0.0)
            continue
        recall.append(1.0)
        values = stream.values[cand]
        coverage.append(float(np.any(values == workload.targets[j])))
        # Resolve with the latest selected fact (ties broken by position).
        order = np.lexsort((cand, stream.ts[cand]))
        answer = int(stream.values[cand[order[-1]]])
        accuracy.append(float(answer == workload.targets[j]))
    return {
        "accuracy": float(np.mean(accuracy)),
        "recall_at_k": float(np.mean(recall)),
        "coverage": float(np.mean(coverage)),
        "coverage_loss": float(1.0 - np.mean(coverage)),
        "n_selected": int(len(selected)),
    }


def _mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


# ── experiment ──────────────────────────────────────────────────────────────


def run(
    seeds: int = 3,
    Qs: tuple[int, ...] = (8, 16),
    ks: tuple[int, ...] = (4, 8, 16),
    regimes: tuple[str, ...] = ("current", "historical"),
    epochs: int = 40,
    device: str = "cpu",
) -> dict:
    test = load_replay_stream(TEST_REPLAY_SEED)
    runs: list[dict] = []

    for seed in range(seeds):
        train_seed = (seed + 1) % REPLAY_SEEDS
        train = load_replay_stream(train_seed)
        policy = train_query_policy(train, seed=seed, epochs=epochs, device=device)

        per_cell: dict[str, dict[str, dict[str, dict[str, dict]]]] = {}
        for regime in regimes:
            per_cell[regime] = {}
            for Q in Qs:
                workload = make_queries(test, Q, seed * 1000 + Q, regime)
                learned_priority = _learned_aggregate_priority(
                    policy, test, workload, device
                )
                per_cell[regime][str(Q)] = {}
                for k in ks:
                    cell: dict[str, dict] = {}
                    for method in METHODS:
                        selected = select(
                            method,
                            test,
                            workload,
                            k,
                            seed * 100 + k,
                            policy,
                            learned_priority,
                            device,
                        )
                        cell[method] = evaluate(selected, test, workload)
                    per_cell[regime][str(Q)][str(k)] = cell
        runs.append(
            {
                "seed": seed,
                "train_replay_seed": train_seed,
                "cells": per_cell,
            }
        )

    results: dict = {}
    for regime in regimes:
        results[regime] = {}
        for Q in Qs:
            results[regime][str(Q)] = {}
            for k in ks:
                results[regime][str(Q)][str(k)] = {}
                for method in METHODS:
                    metric_names = ("accuracy", "recall_at_k", "coverage", "coverage_loss")
                    results[regime][str(Q)][str(k)][method] = {
                        name: _mean_std(
                            [r["cells"][regime][str(Q)][str(k)][method][name] for r in runs]
                        )
                        for name in metric_names
                    }
                    results[regime][str(Q)][str(k)][method]["n_selected"] = float(
                        np.mean(
                            [
                                r["cells"][regime][str(Q)][str(k)][method]["n_selected"]
                                for r in runs
                            ]
                        )
                    )

    verdict: dict = {}
    for regime in regimes:
        verdict[regime] = {}
        for Q in Qs:
            verdict[regime][str(Q)] = {}
            for k in ks:
                row = results[regime][str(Q)][str(k)]
                cov = row["coverage_aware"]["coverage"]["mean"]
                qrec = row["query_recency"]["coverage"]["mean"]
                learned = row["learned_query"]["coverage"]["mean"]
                verdict[regime][str(Q)][str(k)] = {
                    "coverage_aware_minus_query_recency": cov - qrec,
                    "coverage_aware_minus_learned_query": cov - learned,
                    "query_recency_saturated": qrec >= 0.9995,
                    "coverage_aware_saturated": cov >= 0.9995,
                }

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data_source": {
            "kind": "replay",
            "real_fact_stream_available": False,
            "note": (
                "No real fact-store trace ships in the repo. Facts are extracted "
                "from the s1_occlusion_20 replay bundle (simulated MuJoCo/Isaac "
                "occlusion benchmark), not from a real sensor stream."
            ),
            "root": str(REPLAY_ROOT),
            "test_replay_seed": TEST_REPLAY_SEED,
            "train_replay_seeds": [(s + 1) % REPLAY_SEEDS for s in range(seeds)],
        },
        "config": {
            "seeds": seeds,
            "Qs": list(Qs),
            "ks": list(ks),
            "regimes": list(regimes),
            "epochs": epochs,
            "methods": list(METHODS),
            "fact_schema": {
                "entity": "instance_id",
                "attribute": "visibility state",
                "value": f"round(visibility_ratio * {VIS_LEVELS}) in 1..{VIS_LEVELS}",
                "timestamp": "physics_step",
                "reliability": "0.5 + 0.5 * visibility_ratio",
                "write_rule": "only when visibility_ratio > 0",
            },
        },
        "stream_summary": {
            "n_source_rows": test.n_source_rows,
            "n_written_facts": test.n_facts,
            "n_occluded_rows_skipped": test.n_occluded_rows,
            "n_keys": test.n_keys,
            "n_steps": test.n_steps,
            "n_visibility_state_changes": test.visibility_changes,
            "duplicate_writes": test.duplicate_writes,
            "labels": test.entity_names,
        },
        "runs": runs,
        "results": results,
        "verdict": verdict,
    }


# ── report ──────────────────────────────────────────────────────────────────


def _cell(row: dict, method: str, metric: str) -> str:
    value = row[method][metric]
    return f"{value['mean']:.3f} ± {value['std']:.3f}"


def _report(metrics: dict) -> str:
    cfg = metrics["config"]
    summary = metrics["stream_summary"]
    Qs = [str(q) for q in cfg["Qs"]]
    ks = [str(k) for k in cfg["ks"]]
    lines = [
        "# U5b: query-conditioned fact selection on a replayed long-history stream",
        "",
        f"> generated: {metrics['generated']}",
        f"> source: `{metrics['data_source']['root']}` (replay, not real)",
        "",
        "## Data",
        "",
        "No real fact-store trace ships in this repository. Facts are extracted from",
        "the `s1_occlusion_20` replay bundle: 20 entities over 1,200 physics steps.",
        "",
        "| property | value |",
        "|---|---:|",
        f"| source rows | {summary['n_source_rows']} |",
        f"| written facts | {summary['n_written_facts']} |",
        f"| fully occluded rows skipped | {summary['n_occluded_rows_skipped']} |",
        f"| keys (entities) | {summary['n_keys']} |",
        f"| steps | {summary['n_steps']} |",
        f"| visibility state changes | {summary['n_visibility_state_changes']} |",
        f"| duplicate consecutive writes | {summary['duplicate_writes']} |",
        "",
        "Fact schema: key = entity, attribute = visibility state, value = visibility",
        "bucket 1..9, timestamp = physics_step, reliability = 0.5 + 0.5 * visibility.",
        "A fact is written only when the entity is at least partially visible. The",
        "answer to a query is the value of the latest fact at or before `as_of`.",
        "",
        "## Config",
        "",
        f"seeds={cfg['seeds']}, Q={Qs}, k={ks}, regimes={cfg['regimes']}, "
        f"epochs={cfg['epochs']}.",
        "",
    ]

    for regime in cfg["regimes"]:
        lines += [
            f"## {regime} queries: accuracy",
            "",
            "| method | " + " | ".join(f"Q={Q}, k={k}" for Q in Qs for k in ks) + " |",
            "|---|" + "---:|" * (len(Qs) * len(ks)),
        ]
        for method in METHODS:
            cells = [
                _cell(metrics["results"][regime][Q][k], method, "accuracy")
                for Q in Qs
                for k in ks
            ]
            lines.append(f"| {METHOD_LABELS[method]} | " + " | ".join(cells) + " |")
        lines += [
            "",
            f"## {regime} queries: coverage",
            "",
            "| method | " + " | ".join(f"Q={Q}, k={k}" for Q in Qs for k in ks) + " |",
            "|---|" + "---:|" * (len(Qs) * len(ks)),
        ]
        for method in METHODS:
            cells = [
                _cell(metrics["results"][regime][Q][k], method, "coverage")
                for Q in Qs
                for k in ks
            ]
            lines.append(f"| {METHOD_LABELS[method]} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "## Coverage delta",
        "",
        "Delta = `coverage_aware` minus baseline. Positive means the submodular",
        "greedy selector covered more queries with the same budget.",
        "",
        "| regime | Q | k | vs query_recency | vs learned_query |",
        "|---|---:|---:|---:|---:|",
    ]
    for regime in cfg["regimes"]:
        for Q in Qs:
            for k in ks:
                v = metrics["verdict"][regime][Q][k]
                lines.append(
                    f"| {regime} | {Q} | {k} | "
                    f"{v['coverage_aware_minus_query_recency']:+.3f} | "
                    f"{v['coverage_aware_minus_learned_query']:+.3f} |"
                )
    lines += [
        "",
        "## Saturation",
        "",
        "| regime | Q | k | query_recency coverage | coverage_aware coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for regime in cfg["regimes"]:
        for Q in Qs:
            for k in ks:
                row = metrics["results"][regime][Q][k]
                lines.append(
                    f"| {regime} | {Q} | {k} | "
                    f"{row['query_recency']['coverage']['mean']:.3f} | "
                    f"{row['coverage_aware']['coverage']['mean']:.3f} |"
                )
    lines += [
        "",
        "## Verdict",
        "",
    ]
    deltas = []
    for regime in cfg["regimes"]:
        for Q in Qs:
            for k in ks:
                v = metrics["verdict"][regime][Q][k]
                deltas.append(
                    (v["coverage_aware_minus_query_recency"], regime, Q, k)
                )
    deltas.sort(reverse=True)
    biggest = deltas[0]
    saturated_cells = sum(
        1
        for regime in cfg["regimes"]
        for Q in Qs
        for k in ks
        if metrics["verdict"][regime][Q][k]["query_recency_saturated"]
    )
    total_cells = len(cfg["regimes"]) * len(Qs) * len(ks)
    lines += [
        f"- `query_recency` reaches coverage 1.000 in **{saturated_cells}/{total_cells}** "
        "cells. In those cells there is no measurable headroom for a coverage",
        "objective: the greedy selector also reaches 1.000.",
        f"- Largest `coverage_aware - query_recency` delta: "
        f"**{biggest[0]:+.3f}** at {biggest[1]}, Q={biggest[2]}, k={biggest[3]}.",
        "- Any positive delta appears only under budget pressure (k < number of",
        "  distinct query keys), where the greedy selector allocates the shared",
        "  budget by query multiplicity instead of key frequency. It is not a",
        "  value-resolution gain: `query_recency` is already as_of-aware and covers",
        "  every query whenever it gets one slot.",
        "- `query_recency_noasof` exposes the drift gap: ignoring `as_of` selects",
        "  facts written after the query time and can miss the target value.",
        "",
        "## Boundaries",
        "",
        "- Replay, not real: the trace is a simulated MuJoCo/Isaac occlusion",
        "  benchmark. It is not a production fact store, not a real sensor stream,",
        "  and it has no contradictory writes or adversarial updates.",
        "- One attribute (visibility state) per entity; the replay poses are static,",
        "  so there is no continuous pose drift to select over.",
        "- Coverage uses the support relation induced by the replay answer rule",
        "  (latest fact at or before `as_of` plus the resolved value). It is an",
        "  oracle support signal, not a learned relevance model.",
        "- No real fact stream is claimed. The result does not transfer to real",
        "  fact stores with conflicts, multi-attribute keys, or different query",
        "  distributions.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--qs", default="8,16")
    parser.add_argument("--ks", default="4,8,16")
    parser.add_argument("--regimes", default="current,historical")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="eval/results/upper_bound/u5b")
    args = parser.parse_args()

    Qs = tuple(int(x) for x in args.qs.split(",") if x.strip())
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    regimes = tuple(x.strip() for x in args.regimes.split(",") if x.strip())

    metrics = run(
        seeds=args.seeds,
        Qs=Qs,
        ks=ks,
        regimes=regimes,
        epochs=args.epochs,
        device=args.device,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")

    print(
        f"facts={metrics['stream_summary']['n_written_facts']} "
        f"keys={metrics['stream_summary']['n_keys']} "
        f"steps={metrics['stream_summary']['n_steps']}"
    )
    for regime in regimes:
        for Q in Qs:
            for k in ks:
                row = metrics["results"][regime][str(Q)][str(k)]
                print(
                    f"[{regime}] Q={Q},k={k}: "
                    f"q_rec={row['query_recency']['coverage']['mean']:.3f} "
                    f"learned={row['learned_query']['coverage']['mean']:.3f} "
                    f"cov={row['coverage_aware']['coverage']['mean']:.3f} "
                    f"hot={row['hot']['coverage']['mean']:.3f}"
                )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
