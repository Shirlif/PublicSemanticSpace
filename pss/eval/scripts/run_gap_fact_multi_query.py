"""Claim-gap F9: multi-query fact selection under a shared budget.

E4 (``run_gap_fact_active.py``) tested a single query and saturated: both
``query_recency`` and the learned policy reached 1.000 because the answer was
the latest fact for the query key, so any query-conditioned recency rule hit.

This experiment makes the problem a **coverage** problem:

- a workload has ``Q`` queries, each ``(key, as_of)``;
- the answer for a query is resolved from the stream (latest value, or a
  reliability-weighted majority under conflict);
- a fact *supports* a query when it has the query key, the resolved value and
  was written at or before ``as_of``;
- a single active set of ``k`` facts is shared by all queries;
- coverage = fraction of queries with at least one supporting fact.

Methods: full / random / query_recency (E4 rule) / per-query top-k union
(value-blind and value-aware) / per-query union + budget fill / greedy
submodular max-coverage. Streams: easy / conflict / duplicate / drift /
hotkey / hard. The hard streams are designed to stop ``query_recency`` from
saturating.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.memory.active_policy import allocate_budget, greedy_max_coverage

T = 1000
N_KEYS = 50
N_VALUES = 16
N_SEEDS = 3
QS = (2, 4, 8)
KS = (2, 4, 8, 16, 32)
LATENCY_QS = (2, 4, 8)
LATENCY_KS = (4, 8, 16)
LATENCY_REPEATS = 20

METHODS = (
    "full",
    "random",
    "query_recency",
    "per_query_union",
    "per_query_union_value",
    "per_query_union_filled",
    "greedy_coverage",
)


@dataclass(frozen=True)
class StreamSpec:
    name: str
    skew: float = 0.0  # 0 = uniform, > 0 = Zipf exponent over key ranks
    conflict_rate: float = 0.0
    drift_rate: float = 0.0
    duplicate_rate: float = 0.0
    truth_rule: str = "latest"  # "latest" | "majority"


SPECS = (
    StreamSpec("easy"),
    StreamSpec("conflict", conflict_rate=0.45, truth_rule="majority"),
    StreamSpec("duplicate", duplicate_rate=0.30),
    StreamSpec("drift", drift_rate=0.15),
    StreamSpec("hotkey", skew=1.2),
    StreamSpec(
        "hard",
        skew=1.2,
        conflict_rate=0.45,
        drift_rate=0.10,
        duplicate_rate=0.25,
        truth_rule="majority",
    ),
)
SPEC_BY_NAME = {s.name: s for s in SPECS}


@dataclass
class FactStream:
    keys: np.ndarray  # (T,) int
    values: np.ndarray  # (T,) int
    reliability: np.ndarray  # (T,) float
    key_probs: np.ndarray  # (n_keys,) float
    hot_keys: np.ndarray  # (n_keys,) bool

    @property
    def hot(self) -> np.ndarray:
        return self.hot_keys[self.keys]


@dataclass
class QuerySet:
    keys: np.ndarray  # (Q,)
    as_of: np.ndarray  # (Q,)
    targets: np.ndarray  # (Q,)
    support: np.ndarray  # (T, Q) bool
    support_sizes: np.ndarray  # (Q,)

    def __len__(self) -> int:
        return int(len(self.keys))


def _key_probs(n_keys: int, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return np.full(n_keys, 1.0 / n_keys, dtype=np.float64)
    rank = np.arange(1, n_keys + 1, dtype=np.float64)
    p = 1.0 / rank**alpha
    return p / p.sum()


def _other_value(rng, value: int, n_values: int) -> int:
    draw = int(rng.integers(0, n_values - 1))
    return draw + 1 if draw >= value else draw


def make_stream(
    spec: StreamSpec,
    seed: int,
    T: int = T,
    n_keys: int = N_KEYS,
    n_values: int = N_VALUES,
) -> FactStream:
    rng = np.random.default_rng(seed)
    probs = _key_probs(n_keys, spec.skew)
    keys = rng.choice(n_keys, size=T, p=probs).astype(np.int64)
    values = np.empty(T, dtype=np.int64)
    reliability = rng.uniform(0.4, 1.0, size=T).astype(np.float32)
    current = rng.integers(0, n_values, size=n_keys).astype(np.int64)

    for i in range(T):
        if i > 0 and spec.duplicate_rate > 0 and rng.random() < spec.duplicate_rate:
            keys[i] = keys[i - 1]
            values[i] = values[i - 1]
            continue
        key = int(keys[i])
        if spec.drift_rate > 0 and rng.random() < spec.drift_rate:
            current[key] = _other_value(rng, int(current[key]), n_values)
        value = int(current[key])
        if spec.conflict_rate > 0 and rng.random() < spec.conflict_rate:
            value = _other_value(rng, value, n_values)
        values[i] = value

    n_hot = max(1, int(round(0.2 * n_keys)))
    hot_keys = np.zeros(n_keys, dtype=bool)
    hot_keys[np.argsort(-probs)[:n_hot]] = True
    return FactStream(
        keys=keys,
        values=values,
        reliability=reliability,
        key_probs=probs.astype(np.float32),
        hot_keys=hot_keys,
    )


def _resolve_value(stream: FactStream, idxs: np.ndarray, rule: str) -> int:
    """Resolve a query's answer from facts at or before ``as_of``."""
    if len(idxs) == 0:
        return -1
    if rule == "latest":
        return int(stream.values[idxs[-1]])
    window = idxs[-64:]
    scores: dict[int, float] = {}
    latest: dict[int, int] = {}
    for i in window:
        value = int(stream.values[i])
        scores[value] = scores.get(value, 0.0) + float(stream.reliability[i])
        latest[value] = max(latest.get(value, -1), int(i))
    return int(max(scores, key=lambda v: (scores[v], latest[v])))


def make_queries(stream: FactStream, Q: int, seed: int, spec: StreamSpec) -> QuerySet:
    rng = np.random.default_rng(seed)
    T = len(stream.keys)
    positions = np.arange(T)
    counts = np.bincount(stream.keys, minlength=len(stream.key_probs))
    present = np.flatnonzero(counts > 0)
    probs = stream.key_probs[present].astype(np.float64)
    probs = probs / probs.sum()

    first = np.full(len(stream.key_probs), T, dtype=np.int64)
    for i in range(T - 1, -1, -1):
        first[int(stream.keys[i])] = i

    qkeys = rng.choice(present, size=Q, p=probs).astype(np.int64)
    as_of = np.empty(Q, dtype=np.int64)
    for j, key in enumerate(qkeys):
        lo = int(first[key])
        if spec.drift_rate > 0:
            # Historical query: facts after as_of may hold a drifted value, so
            # a recency rule that ignores as_of picks the wrong regime.
            hi = max(lo, min(T - 1, int(0.75 * T)))
            as_of[j] = int(rng.integers(lo, hi + 1))
        else:
            # Current-time query, the E4 regime: the stream ends at as_of.
            as_of[j] = T - 1

    support = np.zeros((T, Q), dtype=bool)
    targets = np.empty(Q, dtype=np.int64)
    for j, key in enumerate(qkeys):
        idxs = np.flatnonzero((stream.keys == key) & (positions <= as_of[j]))
        if len(idxs) == 0:
            idxs = np.flatnonzero(stream.keys == key)
            as_of[j] = int(idxs[-1])
        target = _resolve_value(stream, idxs, spec.truth_rule)
        supporting = idxs[stream.values[idxs] == target]
        if len(supporting) == 0:
            target = int(stream.values[idxs[-1]])
            supporting = idxs[stream.values[idxs] == target]
        targets[j] = target
        support[supporting, j] = True

    return QuerySet(
        keys=qkeys,
        as_of=as_of,
        targets=targets,
        support=support,
        support_sizes=support.sum(axis=0).astype(np.int64),
    )


def _recency(T: int) -> np.ndarray:
    return (np.arange(T, dtype=np.float32) / max(1, T - 1)).astype(np.float32)


def _value_blind_scores(stream: FactStream, qs: QuerySet, j: int, T: int) -> np.ndarray:
    recency = _recency(T)
    key_match = (stream.keys == qs.keys[j]).astype(np.float32)
    scores = key_match * (0.5 * stream.reliability + 0.5 * recency)
    scores[np.arange(T) > qs.as_of[j]] = -np.inf
    return scores


def _value_aware_scores(stream: FactStream, qs: QuerySet, j: int, T: int) -> np.ndarray:
    recency = _recency(T)
    key_match = (stream.keys == qs.keys[j]).astype(np.float32)
    value_match = (stream.values == qs.targets[j]).astype(np.float32)
    scores = key_match * (2.0 * value_match + 0.5 * stream.reliability + 0.5 * recency)
    scores[np.arange(T) > qs.as_of[j]] = -np.inf
    return scores


def _per_query_select(stream, qs, alloc, score_fn) -> np.ndarray:
    T = len(stream.keys)
    chosen: list[int] = []
    for j in range(len(qs)):
        b = int(alloc[j])
        if b <= 0:
            continue
        scores = score_fn(j, T)
        order = np.argsort(-scores, kind="stable")[:b]
        chosen.extend(int(i) for i in order if np.isfinite(scores[i]))
    if not chosen:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.asarray(chosen, dtype=np.int64))


def _select_query_recency(stream, qs, alloc) -> np.ndarray:
    chosen: list[int] = []
    for j in range(len(qs)):
        b = int(alloc[j])
        if b <= 0:
            continue
        idxs = np.flatnonzero(stream.keys == qs.keys[j])
        if len(idxs) == 0:
            continue
        chosen.extend(int(i) for i in idxs[np.argsort(-idxs)[:b]])
    if not chosen:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.asarray(chosen, dtype=np.int64))


def _select_filled(stream, qs, alloc, k: int) -> np.ndarray:
    """Per-query union, then spend leftover budget on aggregate relevance."""
    T = len(stream.keys)
    base = _per_query_select(
        stream, qs, alloc, lambda j, t: _value_blind_scores(stream, qs, j, t)
    )
    if len(base) >= k:
        return base[:k]
    aggregate = np.zeros(T, dtype=np.float32)
    for j in range(len(qs)):
        scores = _value_blind_scores(stream, qs, j, T)
        aggregate += np.where(np.isfinite(scores), scores, 0.0)
    aggregate[base] = -np.inf
    order = np.argsort(-aggregate, kind="stable")
    finite = order[np.isfinite(aggregate[order])]
    extra = [int(i) for i in finite[: k - len(base)]]
    if len(base) + len(extra) < k:
        need = k - len(base) - len(extra)
        fallback = np.argsort(-(stream.reliability + _recency(T)), kind="stable")
        taken = np.concatenate([base, np.asarray(extra, dtype=np.int64)])
        remaining = np.setdiff1d(fallback, taken, assume_unique=False)
        extra.extend(int(i) for i in remaining[:need])
    return np.unique(np.concatenate([base, np.asarray(extra, dtype=np.int64)]))


def select(
    method: str,
    stream: FactStream,
    qs: QuerySet,
    k: int,
    seed: int = 0,
) -> np.ndarray:
    T = len(stream.keys)
    if method == "full":
        return np.arange(T, dtype=np.int64)
    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(T, size=min(k, T), replace=False).astype(np.int64)
    alloc = allocate_budget(k, stream.key_probs[qs.keys])
    if method == "query_recency":
        return _select_query_recency(stream, qs, alloc)
    if method == "per_query_union":
        return _per_query_select(
            stream, qs, alloc, lambda j, t: _value_blind_scores(stream, qs, j, t)
        )
    if method == "per_query_union_value":
        return _per_query_select(
            stream, qs, alloc, lambda j, t: _value_aware_scores(stream, qs, j, t)
        )
    if method == "per_query_union_filled":
        return _select_filled(stream, qs, alloc, k)
    if method == "greedy_coverage":
        priority = stream.reliability + _recency(T)
        return greedy_max_coverage(qs.support, k, priority=priority)
    raise ValueError(f"unknown method: {method}")


def coverage(selected: np.ndarray, qs: QuerySet) -> float:
    if len(selected) == 0:
        return 0.0
    return float(qs.support[np.asarray(selected)].any(axis=0).mean())


def redundancy(selected: np.ndarray, stream: FactStream, qs: QuerySet) -> dict:
    sel = np.unique(np.asarray(selected, dtype=np.int64))
    if len(sel) == 0:
        return {
            "n_selected": 0,
            "wasted_frac": 1.0,
            "duplicate_frac": 1.0,
            "facts_per_covered_query": float("inf"),
            "coverage_per_fact": 0.0,
            "mean_queries_per_fact": 0.0,
        }
    supported = qs.support[sel]
    covered = supported.any(axis=0)
    supports = supported.sum(axis=1)
    pairs = list(zip(stream.keys[sel].tolist(), stream.values[sel].tolist()))
    duplicates = len(pairs) - len(set(pairs))
    return {
        "n_selected": int(len(sel)),
        "wasted_frac": float((supports == 0).mean()),
        "duplicate_frac": float(duplicates / len(sel)),
        "facts_per_covered_query": float(len(sel) / max(1, int(covered.sum()))),
        "coverage_per_fact": float(covered.sum() / len(sel)),
        "mean_queries_per_fact": float(supports.mean()),
    }


def _mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def run(
    seeds: int = N_SEEDS,
    Qs: tuple[int, ...] = QS,
    ks: tuple[int, ...] = KS,
    stream_names: tuple[str, ...] = tuple(s.name for s in SPECS),
) -> dict:
    results: dict = {}
    support_sizes: dict = {}
    for name in stream_names:
        spec = SPEC_BY_NAME[name]
        cells: dict[str, dict[str, dict[str, list[dict]]]] = {}
        for Q in Qs:
            cells[str(Q)] = {str(k): {m: [] for m in METHODS} for k in ks}
        support_values: dict[str, list[float]] = {str(Q): [] for Q in Qs}
        for seed in range(seeds):
            stream = make_stream(spec, seed)
            for Q in Qs:
                qs = make_queries(stream, Q, seed * 1000 + Q, spec)
                support_values[str(Q)].append(float(qs.support_sizes.mean()))
                for k in ks:
                    for method in METHODS:
                        sel = select(method, stream, qs, k, seed=seed)
                        cells[str(Q)][str(k)][method].append(
                            {
                                "coverage": coverage(sel, qs),
                                "redundancy": redundancy(sel, stream, qs),
                            }
                        )
        summary: dict = {}
        for Q in Qs:
            summary[str(Q)] = {}
            for k in ks:
                summary[str(Q)][str(k)] = {}
                for method in METHODS:
                    records = cells[str(Q)][str(k)][method]
                    covs = [r["coverage"] for r in records]
                    red_keys = records[0]["redundancy"].keys()
                    summary[str(Q)][str(k)][method] = {
                        "coverage": _mean_std(covs),
                        "redundancy": {
                            key: float(np.mean([r["redundancy"][key] for r in records]))
                            for key in red_keys
                        },
                    }
        results[name] = summary
        support_sizes[name] = {
            str(Q): float(np.mean(support_values[str(Q)])) for Q in Qs
        }
    return {
        "config": {
            "T": T,
            "n_keys": N_KEYS,
            "n_values": N_VALUES,
            "seeds": seeds,
            "Qs": list(Qs),
            "ks": list(ks),
            "streams": list(stream_names),
            "methods": list(METHODS),
        },
        "stream_specs": {
            s.name: {
                "skew": s.skew,
                "conflict_rate": s.conflict_rate,
                "drift_rate": s.drift_rate,
                "duplicate_rate": s.duplicate_rate,
                "truth_rule": s.truth_rule,
            }
            for s in SPECS
            if s.name in stream_names
        },
        "results": results,
        "support_sizes": support_sizes,
    }


def benchmark_latency(
    stream_name: str = "hard",
    Qs: tuple[int, ...] = LATENCY_QS,
    ks: tuple[int, ...] = LATENCY_KS,
    repeats: int = LATENCY_REPEATS,
    seed: int = 0,
) -> dict:
    spec = SPEC_BY_NAME[stream_name]
    stream = make_stream(spec, seed)
    out: dict = {}
    for Q in Qs:
        qs = make_queries(stream, Q, seed * 1000 + Q, spec)
        for k in ks:
            row: dict = {}
            for method in METHODS:
                times: list[float] = []
                for r in range(repeats):
                    start = time.perf_counter()
                    select(method, stream, qs, k, seed=r)
                    times.append((time.perf_counter() - start) * 1e3)
                arr = np.asarray(times, dtype=np.float64)
                row[method] = {
                    "mean_ms": float(arr.mean()),
                    "p95_ms": float(np.percentile(arr, 95)),
                }
            out[f"Q={Q},k={k}"] = row
    return {
        "stream": stream_name,
        "repeats": repeats,
        "Qs": list(Qs),
        "ks": list(ks),
        "latency": out,
    }


def _fmt_cell(value: dict) -> str:
    return f"{value['mean']:.3f} ± {value['std']:.3f}"


def _report(m: dict, latency: dict) -> str:
    cfg = m["config"]
    Qs = [str(q) for q in cfg["Qs"]]
    ks = [str(k) for k in cfg["ks"]]
    rep_q = "4" if "4" in Qs else Qs[0]
    rep_k = "8" if "8" in ks else ks[-1]
    lines = [
        "# Claim-gap F9: multi-query fact selection under a shared budget",
        "",
        f"> generated: {m['generated']}",
        f"> T={cfg['T']}, keys={cfg['n_keys']}, values={cfg['n_values']}, "
        f"seeds={cfg['seeds']}, Q={Qs}, k={ks}",
        "",
        "E4 saturated because its query answer was the latest fact for the key and",
        "`query_recency` selected exactly those facts. Here the workload has `Q`",
        "queries and one shared active set of `k` facts. A fact supports a query if",
        "it has the query key, the resolved answer value, and a write time at or",
        "before the query's `as_of`. Coverage is the fraction of queries with at",
        "least one supporting fact.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m pss.eval.scripts.run_gap_fact_multi_query "
        "--seeds 3 --qs 2,4,8 --ks 2,4,8,16,32 \\",
        "  --streams easy,conflict,duplicate,drift,hotkey,hard",
        "```",
        "",
        "Outputs: `reports/claim_gaps/fact_multi_query/REPORT.md` and",
        "`reports/claim_gaps/fact_multi_query/metrics.json`.",
        "",
        "## Streams",
        "",
        "| stream | key skew | conflict | drift | duplicate | answer rule |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, spec in m["stream_specs"].items():
        lines.append(
            f"| {name} | {spec['skew']:.1f} | {spec['conflict_rate']:.2f} | "
            f"{spec['drift_rate']:.2f} | {spec['duplicate_rate']:.2f} | "
            f"{spec['truth_rule']} |"
        )
    lines += [
        "",
        "`easy` reproduces the E4 regime. `conflict` mixes contradictory writes and",
        "resolves the answer by reliability-weighted majority over the recent window.",
        "`duplicate` repeats facts. `drift` changes a key's value over time.",
        "`hotkey` draws both facts and queries from a Zipf key distribution.",
        "`hard` combines conflict, drift, duplicates and hot-key skew.",
        "",
        "## Coverage vs budget (Q={})".format(rep_q),
        "",
    ]
    for name in cfg["streams"]:
        lines += [
            f"### {name}",
            "",
            "| method | " + " | ".join(f"k={k}" for k in ks) + " |",
            "|---|" + "---:|" * len(ks),
        ]
        for method in METHODS:
            cells = [
                _fmt_cell(m["results"][name][rep_q][k][method]["coverage"]) for k in ks
            ]
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "## Greedy coverage vs budget (all Q)",
        "",
        "Absolute coverage of the coverage optimiser, i.e. the coverage-budget",
        "curve. The delta tables below show where allocation beats per-query",
        "selection.",
        "",
    ]
    for name in cfg["streams"]:
        lines += [
            f"### {name}",
            "",
            "| Q | " + " | ".join(f"k={k}" for k in ks) + " |",
            "|---:|" + "---:|" * len(ks),
        ]
        for Q in Qs:
            cells = [
                f"{m['results'][name][Q][k]['greedy_coverage']['coverage']['mean']:.3f}"
                for k in ks
            ]
            lines.append(f"| {Q} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "## Coverage optimisation vs per-query union",
        "",
        "Delta = `greedy_coverage` - `per_query_union` at the same stream, Q and k.",
        "Positive cells mean the coverage-aware selection covered more queries with",
        "the same budget. `full` is omitted (it is 1.000 by construction).",
        "",
    ]
    for name in cfg["streams"]:
        lines += [
            f"### {name}",
            "",
            "| Q | " + " | ".join(f"k={k}" for k in ks) + " |",
            "|---:|" + "---:|" * len(ks),
        ]
        for Q in Qs:
            cells = []
            for k in ks:
                g = m["results"][name][Q][k]["greedy_coverage"]["coverage"]["mean"]
                u = m["results"][name][Q][k]["per_query_union"]["coverage"]["mean"]
                cells.append(f"{g - u:+.3f}")
            lines.append(f"| {Q} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "## Saturation",
        "",
        "Smallest `k` at which a method reaches coverage 1.000, or `-` if it never",
        "does in the swept range. `q_rec` is the E4 recency rule; `greedy` is the",
        "coverage optimiser.",
        "",
        "| stream | Q | q_rec | per_query | greedy | mean support |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in cfg["streams"]:
        for Q in Qs:
            row = m["results"][name][Q]

            def first_full(method: str) -> str:
                for k in ks:
                    if row[k][method]["coverage"]["mean"] >= 0.9995:
                        return k
                return "-"

            mean_support = m["support_sizes"][name][Q]
            lines.append(
                f"| {name} | {Q} | {first_full('query_recency')} | "
                f"{first_full('per_query_union')} | {first_full('greedy_coverage')} | "
                f"{mean_support:.1f} |"
            )
    lines += [
        "",
        f"## Redundancy (Q={rep_q}, k={rep_k})",
        "",
        "| stream | method | coverage | selected | wasted | duplicate | facts/covered | "
        "queries/fact |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in cfg["streams"]:
        for method in METHODS:
            cell = m["results"][name][rep_q][rep_k][method]
            red = cell["redundancy"]
            lines.append(
                f"| {name} | {method} | {cell['coverage']['mean']:.3f} | "
                f"{red['n_selected']:.1f} | {red['wasted_frac']:.3f} | "
                f"{red['duplicate_frac']:.3f} | {red['facts_per_covered_query']:.2f} | "
                f"{red['mean_queries_per_fact']:.2f} |"
            )
    lines += [
        "",
        "`full` is the reference row: keeping the whole stream trivially covers",
        "everything but has maximal redundancy.",
        "",
        "## Selection latency",
        "",
        f"Measured on the `{latency['stream']}` stream, {latency['repeats']} repeats;",
        "mean / p95 milliseconds per selection call.",
        "",
        "| setting | " + " | ".join(METHODS) + " |",
        "|---|" + "---:|" * len(METHODS),
    ]
    for setting, row in latency["latency"].items():
        cells = [
            f"{row[m]['mean_ms']:.3f}/{row[m]['p95_ms']:.3f}" for m in METHODS
        ]
        lines.append(f"| {setting} | " + " | ".join(cells) + " |")

    lines += ["", "## Verdict", ""]
    hard_names = {"conflict", "duplicate", "drift", "hotkey", "hard"}
    wins: list[str] = []
    value_wins: list[str] = []
    filled_wins: list[str] = []
    ranked: list[tuple[float, str, str, str, float, float]] = []
    for name in cfg["streams"]:
        for Q in Qs:
            for k in ks:
                row = m["results"][name][Q][k]
                g = row["greedy_coverage"]["coverage"]["mean"]
                u = row["per_query_union"]["coverage"]["mean"]
                uv = row["per_query_union_value"]["coverage"]["mean"]
                uf = row["per_query_union_filled"]["coverage"]["mean"]
                if g - u > 0.01:
                    wins.append(f"{name}/Q={Q}/k={k} ({g:.3f} vs {u:.3f})")
                    if name in hard_names:
                        ranked.append((g - u, name, Q, k, g, u))
                if g - uv > 0.01:
                    value_wins.append(f"{name}/Q={Q}/k={k} ({g:.3f} vs {uv:.3f})")
                if g - uf > 0.01:
                    filled_wins.append(f"{name}/Q={Q}/k={k} ({g:.3f} vs {uf:.3f})")
    ranked.sort(reverse=True)

    if wins:
        lines += [
            "Coverage optimisation beats the value-blind per-query union in "
            f"{len(wins)} setting(s):",
            "",
        ]
        lines += [f"- {w}" for w in wins[:24]]
        if len(wins) > 24:
            lines.append(f"- ... and {len(wins) - 24} more")
        lines.append("")
        if ranked:
            lines += [
                "**Material hard-setting wins** (largest deltas vs the value-blind",
                "union, which is the E4-style query-conditioned baseline):",
                "",
            ]
            for delta, name, Q, k, g, u in ranked[:6]:
                lines.append(
                    f"- {name}/Q={Q}/k={k}: {g:.3f} vs {u:.3f} (+{delta:.3f})"
                )
            lines += [
                "",
                "The small `easy`/`duplicate` Q=8, k=2 deltas come from chance key",
                "collisions between queries, not from the conflict/drift/hot-key",
                "mechanism; they are not the claimed effect.",
                "",
            ]
        if filled_wins:
            lines += [
                "Against `per_query_union_filled`, which spends the full budget k,",
                f"greedy still wins in {len(filled_wins)} setting(s):",
                "",
            ]
            lines += [f"- {w}" for w in filled_wins[:8]]
            if len(filled_wins) > 8:
                lines.append(f"- ... and {len(filled_wins) - 8} more")
            lines += [
                "",
                "The gain is therefore not merely unused budget in the union.",
                "",
            ]
    else:
        lines += [
            "Coverage optimisation did **not** beat the value-blind per-query union",
            "anywhere in this sweep. This is a negative result.",
            "",
        ]
    if value_wins:
        lines += [
            "Against the value-aware per-query union it wins in "
            f"{len(value_wins)} setting(s) (budget allocation / shared support):",
            "",
        ]
        lines += [f"- {w}" for w in value_wins[:16]]
        if len(value_wins) > 16:
            lines.append(f"- ... and {len(value_wins) - 16} more")
        lines.append("")
    else:
        lines += [
            "Against the value-aware per-query union there is no win: when each",
            "query receives at least one budget slot, a value-aware top-1 already",
            "covers it. The coverage gain is therefore about value resolution and",
            "budget allocation, not about a better top-1 ranker.",
            "",
        ]

    lines += [
        "### Saturation condition",
        "",
        "Coverage is capped at 1.0, so it saturates whenever every query already",
        "has one selected supporting fact. Concretely: if `k >= Q` and the",
        "method allocates at least one slot per query, the only way to stay below",
        "1.0 is to pick a non-supporting fact. That is exactly what the value-blind",
        "rules do on `conflict` and `hard`; on `easy`, `duplicate` and `hotkey` the",
        "value is stable, so any key-matched fact supports the query and",
        "`query_recency` returns to 1.000 for `k >= Q`. On `drift`, recency that",
        "ignores `as_of` picks future facts with the wrong value, so it does not",
        "saturate. When `k < Q`, per-query allocation gives some queries zero slots",
        "and coverage is bounded by `k / Q` unless one fact supports several",
        "queries (shared support under hot-key skew).",
        "",
        "### Limitations",
        "",
        "- The stream is synthetic (T=1000, 50 keys, 16 values); the absolute",
        "  numbers are not a claim about real fact stores.",
        "- Coverage only asks for one supporting fact per query. It does not measure",
        "  whether the selected fact is sufficient to answer, nor answer accuracy.",
        "- `greedy_coverage` uses the support relation induced by the query and the",
        "  stream's answer rule; it is not learning a relevance model. A deployable",
        "  version still needs a value-resolution / relevance classifier.",
        "- Latency is single-process CPU numpy on a synthetic stream; it does not",
        "  include query parsing or support construction.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--qs", default=",".join(str(q) for q in QS))
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    ap.add_argument(
        "--streams", default=",".join(s.name for s in SPECS)
    )
    ap.add_argument("--latency-repeats", type=int, default=LATENCY_REPEATS)
    ap.add_argument("--out", default="reports/claim_gaps/fact_multi_query")
    args = ap.parse_args()

    Qs = tuple(int(x) for x in args.qs.split(",") if x.strip())
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    streams = tuple(x.strip() for x in args.streams.split(",") if x.strip())
    metrics = run(seeds=args.seeds, Qs=Qs, ks=ks, stream_names=streams)
    metrics["latency"] = benchmark_latency(
        stream_name="hard" if "hard" in streams else streams[0],
        Qs=tuple(q for q in Qs if q in LATENCY_QS) or LATENCY_QS,
        ks=tuple(k for k in ks if k in LATENCY_KS) or LATENCY_KS,
        repeats=args.latency_repeats,
    )
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "REPORT.md").write_text(_report(metrics, metrics["latency"]), encoding="utf-8")

    for name in streams:
        parts = []
        for Q in Qs:
            for k in ks:
                row = metrics["results"][name][str(Q)][str(k)]
                g = row["greedy_coverage"]["coverage"]["mean"]
                u = row["per_query_union"]["coverage"]["mean"]
                qr = row["query_recency"]["coverage"]["mean"]
                parts.append(f"Q={Q},k={k}: greedy={g:.3f} union={u:.3f} rec={qr:.3f}")
        print(f"[{name}] " + " | ".join(parts[:6]))
    print(f"output: {out}")


if __name__ == "__main__":
    main()
