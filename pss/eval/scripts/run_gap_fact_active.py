"""Claim-gap E4: query-conditioned fact active set.

The existing fact active-set result is negative (learned 0.372 < hot 0.480)
because the scorer is query-independent: one top-k set is shared by every
query. This experiment keeps the same stream and adds query-conditioned
features, then selects a separate top-k set per query.

Baselines: full / recency / hot / random / learned (query-independent) /
query-recency heuristic / learned-query.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pss._console import ensure_utf8
from pss.memory.active_policy import ActiveSetPolicy

N_KEYS = 50
N_VALUES = 16
T = 1000
K = 16


def _make_facts(T: int, rng, n_keys: int = N_KEYS, n_values: int = N_VALUES, hot_frac: float = 0.2):
    keys = rng.integers(0, n_keys, size=T)
    values = rng.integers(0, n_values, size=T)
    reliability = rng.uniform(0.4, 1.0, size=T)
    hot_key = rng.random(n_keys) < hot_frac
    hot = hot_key[keys]
    return keys, values, reliability, hot


def _latest_answers(keys, values, reliability, T):
    best = {}
    for i in range(T):
        key = int(keys[i])
        if key not in best or (i, reliability[i]) > best[key][:2]:
            best[key] = (i, reliability[i], int(values[i]))
    return {k: v[2] for k, v in best.items()}


def _sample_query(keys, hot, T, rng):
    hot_keys = np.unique(keys[hot])
    if len(hot_keys) and rng.random() < 0.7:
        return int(rng.choice(hot_keys))
    p = np.linspace(1.0, 0.2, T)
    p /= p.sum()
    return int(rng.choice(keys, p=p))


def _query_features(keys, values, reliability, hot, T, q, n_keys=N_KEYS):
    """(T, F) features for one query; row i describes fact i w.r.t. query q."""
    key_freq = np.bincount(keys, minlength=n_keys).astype(np.float32)
    positions = np.arange(T)
    # index of the latest fact for each key
    latest = {}
    for i in range(T):
        latest[int(keys[i])] = i
    latest_idx = np.array([latest[int(k)] for k in keys], dtype=np.int64)
    is_latest = (positions == latest_idx).astype(np.float32)
    later_same = np.zeros(T, dtype=np.float32)
    seen: dict[int, int] = {}
    for i in range(T - 1, -1, -1):
        key = int(keys[i])
        later_same[i] = seen.get(key, 0)
        seen[key] = seen.get(key, 0) + 1
    match = (keys == q).astype(np.float32)
    q_freq = float(np.mean(keys == q))
    return np.stack(
        [
            match,
            (T - 1 - positions) / max(1, T - 1),
            hot.astype(np.float32),
            reliability.astype(np.float32),
            key_freq[keys] / T,
            values.astype(np.float32) / N_VALUES,
            is_latest,
            later_same / T,
            np.full(T, q_freq, dtype=np.float32),
            match * ((T - 1 - positions) / max(1, T - 1)),
            match * reliability.astype(np.float32),
            np.ones(T, dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def _old_features(keys, values, reliability, hot, T, n_keys=N_KEYS):
    key_freq = np.bincount(keys, minlength=n_keys).astype(np.float32)
    out = []
    for i in range(T):
        out.append(
            [
                (T - 1 - i) / (T - 1),
                float(hot[i]),
                float(reliability[i]),
                key_freq[int(keys[i])] / T,
                float(values[i]) / N_VALUES,
                i / (T - 1),
                1.0,
                1.0,
            ]
        )
    return np.asarray(out, dtype=np.float32)


def _train_query_policy(keys, values, reliability, hot, T, seed=0, epochs=40, device="cpu"):
    rng = np.random.default_rng(seed)
    feat_dim = _query_features(keys, values, reliability, hot, T, 0).shape[1]
    policy = ActiveSetPolicy(feature_dim=feat_dim).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    latest = {}
    for i in range(T):
        latest[int(keys[i])] = i
    for _ in range(epochs):
        policy.train()
        qs = np.array([_sample_query(keys, hot, T, rng) for _ in range(64)])
        feats = np.stack(
            [_query_features(keys, values, reliability, hot, T, q) for q in qs]
        )
        x = torch.as_tensor(feats, device=device)
        logits = policy(x)
        targets = []
        for q in qs:
            idxs = np.where(keys == q)[0]
            targets.append(
                int(idxs[np.lexsort((reliability[idxs], idxs))[-1]])
                if len(idxs)
                else 0
            )
        loss = F.cross_entropy(logits, torch.as_tensor(targets, device=device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    policy.eval()
    return policy


def _train_old_policy(keys, values, reliability, hot, T, seed=0, epochs=40, device="cpu"):
    rng = np.random.default_rng(seed)
    feats = _old_features(keys, values, reliability, hot, T)
    policy = ActiveSetPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    x = torch.as_tensor(feats, device=device)
    latest = {}
    for i in range(T):
        latest[int(keys[i])] = i
    for _ in range(epochs):
        policy.train()
        qs = np.array([_sample_query(keys, hot, T, rng) for _ in range(64)])
        targets = []
        for q in qs:
            idxs = np.where(keys == q)[0]
            targets.append(
                int(idxs[np.lexsort((reliability[idxs], idxs))[-1]])
                if len(idxs)
                else 0
            )
        logits = policy(x)[None].expand(64, -1)
        loss = F.cross_entropy(logits, torch.as_tensor(targets, device=device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    policy.eval()
    return policy


def _diverse_topk(scores, k, keys):
    order = np.argsort(-scores)
    selected, seen = [], set()
    for i in order:
        if len(selected) >= k:
            break
        if int(keys[i]) in seen:
            continue
        selected.append(int(i))
        seen.add(int(keys[i]))
    if len(selected) < k:
        for i in order:
            if len(selected) >= k:
                break
            if int(i) not in selected:
                selected.append(int(i))
    return selected


def _accuracy_fixed(selected, keys, values, reliability, answers, hot, n=500, seed=1):
    rng = np.random.default_rng(seed)
    selected = set(int(i) for i in selected)
    hits = []
    for _ in range(n):
        q = _sample_query(keys, hot, T, rng)
        cand = [i for i in selected if keys[i] == q]
        hits.append(
            float(
                bool(cand)
                and values[max(cand, key=lambda i: (i, reliability[i]))] == answers[q]
            )
        )
    return float(np.mean(hits))


def _accuracy_query(policy, keys, values, reliability, hot, answers, k, n=500, seed=1):
    rng = np.random.default_rng(seed)
    hits = []
    for _ in range(n):
        q = _sample_query(keys, hot, T, rng)
        feats = _query_features(keys, values, reliability, hot, T, q)
        scores = policy.scores(feats)
        order = np.argsort(-scores)[:k]
        cand = [int(i) for i in order if keys[i] == q]
        hits.append(
            float(bool(cand) and values[max(cand, key=lambda i: (i, reliability[i]))] == answers[q])
        )
    return float(np.mean(hits))


def _accuracy_query_recency(keys, values, reliability, hot, answers, k, n=500, seed=1):
    rng = np.random.default_rng(seed)
    hits = []
    for _ in range(n):
        q = _sample_query(keys, hot, T, rng)
        idxs = np.where(keys == q)[0]
        sel = idxs[np.argsort(-idxs)][:k]
        hits.append(
            float(bool(len(sel)) and values[max(sel, key=lambda i: (i, reliability[i]))] == answers[q])
        )
    return float(np.mean(hits))


def run(seeds: int = 3, ks: tuple[int, ...] = (4, 8, 16), device: str = "cpu") -> dict:
    test_rng = np.random.default_rng(99)
    keys, values, reliability, hot = _make_facts(T, test_rng)
    answers = _latest_answers(keys, values, reliability, T)

    runs = []
    for seed in range(seeds):
        train_rng = np.random.default_rng(seed)
        tkeys, tvalues, trel, thot = _make_facts(T, train_rng)
        query_policy = _train_query_policy(tkeys, tvalues, trel, thot, T, seed=seed, device=device)
        old_policy = _train_old_policy(tkeys, tvalues, trel, thot, T, seed=seed, device=device)

        old_scores = old_policy.scores(_old_features(keys, values, reliability, hot, T), device=device)
        hot_order = np.lexsort((-np.arange(T), -hot.astype(int)))
        rand = np.random.default_rng(seed).choice(T, max(ks), replace=False)

        per_k = {}
        for k in ks:
            old_sel = _diverse_topk(old_scores, k, keys)
            recency = list(range(T - k, T))
            per_k[str(k)] = {
                "full": _accuracy_fixed(
                    list(range(T)), keys, values, reliability, answers, hot, seed=seed + 1
                ),
                "recency": _accuracy_fixed(
                    recency, keys, values, reliability, answers, hot, seed=seed + 1
                ),
                "hot": _accuracy_fixed(
                    list(hot_order[:k]), keys, values, reliability, answers, hot, seed=seed + 1
                ),
                "random": _accuracy_fixed(
                    list(rand[:k]), keys, values, reliability, answers, hot, seed=seed + 1
                ),
                "learned_old": _accuracy_fixed(
                    old_sel, keys, values, reliability, answers, hot, seed=seed + 1
                ),
                "query_recency": _accuracy_query_recency(
                    keys, values, reliability, hot, answers, k, seed=seed + 1
                ),
                "learned_query": _accuracy_query(
                    query_policy, keys, values, reliability, hot, answers, k, seed=seed + 1
                ),
            }
        runs.append({"seed": seed, "per_k": per_k})

    methods = (
        "full",
        "recency",
        "hot",
        "random",
        "learned_old",
        "query_recency",
        "learned_query",
    )
    summary = {}
    for k in ks:
        summary[str(k)] = {
            method: {
                "mean": float(np.mean([r["per_k"][str(k)][method] for r in runs])),
                "std": float(np.std([r["per_k"][str(k)][method] for r in runs])),
            }
            for method in methods
        }
    verdict = {}
    for k in ks:
        row = summary[str(k)]
        verdict[str(k)] = {
            "learned_query_beats_hot": row["learned_query"]["mean"] > row["hot"]["mean"],
            "learned_query_beats_old": row["learned_query"]["mean"] > row["learned_old"]["mean"],
            "query_recency_beats_hot": row["query_recency"]["mean"] > row["hot"]["mean"],
        }
    return {
        "T": T,
        "ks": list(ks),
        "n_keys": N_KEYS,
        "n_values": N_VALUES,
        "seeds": seeds,
        "runs": runs,
        "summary": summary,
        "verdict": verdict,
    }


def _report(m: dict) -> str:
    ks = [str(k) for k in m["ks"]]
    lines = [
        "# Claim-gap E4: query-conditioned fact active set",
        "",
        f"> T={m['T']}, keys={m['n_keys']}, values={m['n_values']}, seeds={m['seeds']}",
        "",
        "| method | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|---|" + "---:|" * len(ks),
    ]
    for method in m["summary"][ks[0]]:
        cells = [
            f"{m['summary'][k][method]['mean']:.3f} ± {m['summary'][k][method]['std']:.3f}"
            for k in ks
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Verdict",
        "",
        "| k | learned_query > hot | learned_query > old | query_recency > hot |",
        "|---:|---|---|---|",
    ]
    for k in ks:
        v = m["verdict"][k]
        lines.append(
            f"| {k} | {v['learned_query_beats_hot']} | {v['learned_query_beats_old']} | "
            f"{v['query_recency_beats_hot']} |"
        )
    lines += [
        "",
        "The old negative result came from a query-independent scorer: one top-k set was",
        "shared by every query. Query-conditioned selection is compared against the hot",
        "heuristic and a query-recency rule; the latter attributes the gain to conditioning",
        "rather than to learning. If query_recency equals learned_query, the honest",
        "conclusion is 'the missing ingredient was query conditioning', not 'the learned",
        "model is better'.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reports/claim_gaps/fact_active")
    args = ap.parse_args()
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    metrics = run(seeds=args.seeds, ks=ks, device=args.device)
    metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    for k in [str(x) for x in ks]:
        row = metrics["summary"][k]
        print(
            f"k={k}: hot={row['hot']['mean']:.3f} old={row['learned_old']['mean']:.3f} "
            f"q_rec={row['query_recency']['mean']:.3f} q_learn={row['learned_query']['mean']:.3f}"
        )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
