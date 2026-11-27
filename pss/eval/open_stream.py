"""Open-category stream simulator.

The stream protocol is deliberately small and inspectable: one projected
representation, one pooled-covariance scorer, a novelty threshold calibrated
on the base validation split, and a query/update/verify loop.

What is *not* here
------------------
- No self-supervised labels.  The oracle labels are the generator's ground
  truth; the point is to measure the query/update loop, not to solve labelling.
- No encoder training.  The projection is frozen; if a new tier is invisible
  after projection, that is a real finding (the encoder is the bottleneck).

Arms are defined by ``policy`` (how rows are queried) plus two flags:
``replay`` (refit the pooled covariance on a replay buffer) and ``guard``
(roll back an update that hurts the held-out probe).
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, roc_auc_score

from pss.encoders.data import load_index
from pss.eval.scripts.run_attribution_grid import MODALITIES, load_features, task_split
from pss.projection.heads import load_projection
from pss.prototypes.scorers import MahalanobisScorer

__all__ = [
    "StreamData",
    "calibrate_threshold",
    "cluster_features",
    "concat",
    "evaluate_probe",
    "load_base",
    "load_stream",
    "select_query",
    "simulate",
]


def concat(rep: dict[str, np.ndarray], rows) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    return np.concatenate([rep[m][rows] for m in MODALITIES], axis=1)


def _project(feats: dict, projections: str | Path, device: str) -> dict[str, np.ndarray]:
    heads = {
        m: load_projection(Path(projections) / f"{m}.pt", device=device)
        for m in MODALITIES
    }
    rep: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for m in MODALITIES:
            heads[m].eval()
            rep[m] = (
                heads[m](torch.as_tensor(feats[m], dtype=torch.float32, device=device))
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    return rep


def load_base(
    base_data: str | Path,
    experts: str | Path,
    projections: str | Path,
    *,
    regime: str = "category",
    device: str = "cpu",
    cache_dir: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Load the offline base model's representation and split."""

    cache = Path(cache_dir or "eval/results/open_stream/_base_cache")
    feats = load_features(str(base_data), str(experts), cache)
    task = task_split(feats, regime)
    rep = _project(feats, projections, device)
    return rep, task


@dataclass
class StreamData:
    rep: dict[str, np.ndarray]
    labels: np.ndarray
    episodes: np.ndarray
    is_novel: np.ndarray
    drift: np.ndarray
    tier: str
    update_mask: np.ndarray
    probe_mask: np.ndarray
    unseen_mask: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.labels))


def load_stream(
    stream_data: str | Path,
    experts: str | Path,
    projections: str | Path,
    *,
    label_mode: str = "category",
    device: str = "cpu",
    cache_dir: str | Path | None = None,
    split_seed: int = 0,
) -> StreamData:
    """Load a generated stream and split each label into update/probe halves.

    The probe half is never queried, so ``novel_acc`` is not inflated by the
    rows the loop just labelled.
    """

    stream_data = Path(stream_data)
    cache = Path(cache_dir or stream_data / "_feature_cache")
    feats = load_features(str(stream_data), str(experts), cache)
    index = load_index(stream_data)
    by_id = {
        str(entry["id"]): entry
        for entry in index["entries"]
        if entry.get("kind") == "category"
    }
    entries = [by_id[str(sample_id)] for sample_id in feats["sample_ids"]]
    labels = np.asarray(
        [
            str(e["label"]) if label_mode == "instance" else str(e["category"])
            for e in entries
        ]
    )
    episodes = np.asarray([int(e["episode"]) for e in entries], dtype=np.int64)
    is_novel = np.asarray([bool(e["is_novel"]) for e in entries], dtype=bool)
    drift = np.asarray([bool(e["drift"]) for e in entries], dtype=bool)
    tier = str(index.get("config", {}).get("tier", "?"))

    rep = _project(feats, projections, device)

    rng = np.random.default_rng(int(split_seed))
    update_mask = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        rows = np.where(labels == label)[0]
        rng.shuffle(rows)
        take = max(1, len(rows) // 2)
        update_mask[rows[:take]] = True
    probe_mask = ~update_mask

    # Reserve ~20% of the novel labels as "never introduced" so unknown AUROC
    # always has a positive class.  These rows are never queried.
    novel_labels = sorted({str(x) for x in labels[is_novel]})
    unseen_mask = np.zeros(len(labels), dtype=bool)
    if novel_labels:
        rng_unseen = np.random.default_rng(int(split_seed) + 1)
        n_unseen = max(1, int(round(0.2 * len(novel_labels))))
        unseen_labels = set(
            str(x)
            for x in rng_unseen.choice(novel_labels, size=n_unseen, replace=False)
        )
        unseen_mask = is_novel & np.isin(labels, list(unseen_labels))

    return StreamData(
        rep=rep,
        labels=labels,
        episodes=episodes,
        is_novel=is_novel,
        drift=drift,
        tier=tier,
        update_mask=update_mask,
        probe_mask=probe_mask,
        unseen_mask=unseen_mask,
    )


def calibrate_threshold(
    scorer: MahalanobisScorer, X: np.ndarray, acceptance: float = 0.95
) -> float:
    """Score threshold that accepts ``acceptance`` of the known validation rows."""

    scores = scorer.scores(np.asarray(X)).max(axis=1)
    return float(np.quantile(scores, 1.0 - float(acceptance)))


def cluster_features(X: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:
    """Cluster projected features; returns ``-1`` for noise."""

    X = np.asarray(X, dtype=np.float64)
    if len(X) < max(2, int(min_cluster_size)):
        return np.zeros(len(X), dtype=np.int64)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    Z = X / np.clip(norm, 1e-12, None)

    try:  # sklearn >= 1.3
        from sklearn.cluster import HDBSCAN

        model = HDBSCAN(min_cluster_size=int(min_cluster_size), metric="euclidean", copy=True)
        return model.fit_predict(Z).astype(np.int64)
    except Exception:
        pass

    k = min(5, len(Z) - 1)
    distances = np.sort(
        np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1), axis=1
    )[:, k]
    eps = max(float(np.median(distances)) * 1.5, 1e-3)
    return DBSCAN(eps=eps, min_samples=2).fit_predict(Z).astype(np.int64)


def _cluster_query(
    rep: dict[str, np.ndarray],
    rows: np.ndarray,
    scores: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One query per *novel* cluster, then the most uncertain rows.

    Looking at every row lets a low-score seen class consume the budget.  We
    first restrict to the ``3 * budget`` most uncertain rows (where novel rows
    live), cluster those, and allocate one query per cluster.  That keeps both
    cluster coverage and novelty focus.
    """

    n = len(rows)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    X = concat(rep, rows)
    clusters = cluster_features(X)

    # Treat noise as singleton clusters: a novel class often fails to form a
    # dense HDBSCAN cluster, and dropping those points loses exactly the novel
    # rows we are trying to find.
    labels = clusters.copy()
    next_label = int(labels.max(initial=-1)) + 1
    for i in np.where(labels < 0)[0]:
        labels[i] = next_label
        next_label += 1

    unique = [int(c) for c in np.unique(labels)]
    unique.sort(key=lambda c: float(np.mean(scores[labels == c])))
    members = {}
    for c in unique:
        idx = np.where(labels == c)[0]
        members[c] = idx[np.argsort(scores[idx])].tolist()

    selected: list[int] = []
    # Round 1: one query per cluster, most novel clusters first.
    for c in unique:
        if len(selected) >= budget:
            break
        if members[c]:
            selected.append(int(members[c].pop(0)))
    # Round 2: round-robin over clusters so the budget does not collapse onto
    # a single hard-but-known class.
    while len(selected) < budget and any(members.values()):
        for c in unique:
            if len(selected) >= budget:
                break
            if members[c]:
                selected.append(int(members[c].pop(0)))
    if not selected:
        selected = list(rng.choice(n, size=min(budget, n), replace=False))
    return rows[np.asarray(selected[: int(budget)], dtype=np.int64)]


def select_query(
    rep: dict[str, np.ndarray],
    rows: np.ndarray,
    scores: np.ndarray,
    budget: int,
    policy: str,
    rng: np.random.Generator,
    threshold: float,
) -> np.ndarray:
    """Choose rows to label (or pseudo-label, for ``self_train``)."""

    rows = np.asarray(rows, dtype=np.int64)
    n = len(rows)
    if n == 0:
        return rows
    budget = min(int(budget), n)
    if policy == "random":
        return rows[rng.choice(n, size=budget, replace=False)]
    if policy == "novelty":
        return rows[np.argsort(scores)[:budget]]
    if policy == "self_train":
        return rows[np.argsort(scores)[::-1][:budget]]

    unknown = np.where(scores < threshold)[0]
    pool = unknown if len(unknown) >= budget else np.arange(n)
    if policy == "active_no_cluster":
        return rows[pool[np.argsort(scores[pool])[:budget]]]
    if policy in ("active_cluster",):
        return _cluster_query(rep, rows[pool], scores[pool], budget, rng)
    raise ValueError(f"unknown policy {policy!r}")


def _ece(scores: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    z = np.asarray(scores, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    total = max(len(y), 1)
    out = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        out += abs(correct[mask].mean() - conf[mask].mean()) * mask.sum() / total
    return float(out)


def evaluate_probe(
    scorer: MahalanobisScorer,
    classes: list[str],
    base_X: np.ndarray,
    base_y: np.ndarray,
    novel_X: np.ndarray,
    novel_y: np.ndarray,
    unseen_X: np.ndarray,
) -> dict:
    """Closed-set accuracy plus unknown AUROC on a fixed probe."""

    def _predict(X):
        scores = scorer.scores(np.asarray(X))
        return np.asarray(classes)[scorer.classes_[scores.argmax(axis=1)]], scores

    old_pred, old_scores = _predict(base_X)
    old_acc = float(np.mean(old_pred == base_y)) if len(base_y) else float("nan")

    if len(novel_y):
        novel_pred, _ = _predict(novel_X)
        novel_acc = float(np.mean(novel_pred == novel_y))
    else:
        novel_acc = float("nan")

    if len(unseen_X):
        known_max = old_scores.max(axis=1)
        unseen_max = scorer.scores(np.asarray(unseen_X)).max(axis=1)
        y = np.concatenate([np.zeros(len(known_max)), np.ones(len(unseen_max))])
        score = -np.concatenate([known_max, unseen_max])
        unknown_auroc = float(roc_auc_score(y, score))
    else:
        unknown_auroc = float("nan")

    all_X = np.vstack([base_X, novel_X]) if len(novel_X) else base_X
    all_y = np.concatenate([base_y, novel_y]) if len(novel_y) else base_y
    _, all_scores = _predict(all_X)
    return {
        "old_acc": old_acc,
        "novel_acc": novel_acc,
        "unknown_auroc": unknown_auroc,
        "ece": _ece(all_scores, all_y),
        "n_old": int(len(base_y)),
        "n_novel": int(len(novel_y)),
    }


def _apply_labels(
    scorer: MahalanobisScorer,
    X: np.ndarray,
    labels: np.ndarray,
    cls_to_i: dict[str, int],
    buffers: dict[int, list[np.ndarray]],
) -> None:
    """Update the scorer with **cumulative** per-class buffers.

    ``MahalanobisScorer.update_class`` sets the mean to the mean of the rows it
    receives.  Calling it once per single row would overwrite the class mean
    with the latest sample; keeping a buffer and passing the whole class batch
    is what makes repeated labels accumulate.
    """

    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for pos, label in enumerate(labels):
        groups[str(label)].append(np.asarray(X[int(pos)]))
    existing = {int(c) for c in scorer.classes_}
    for label, rows in groups.items():
        idx = int(cls_to_i[label])
        buffers.setdefault(idx, []).extend(rows)
        Z = np.vstack(buffers[idx])
        if idx in existing:
            scorer.update_class(idx, Z)
        else:
            scorer.add_class(idx, Z)
            existing.add(idx)


def simulate(
    base_rep: dict[str, np.ndarray],
    base_task: dict,
    stream: StreamData,
    *,
    policy: str,
    budget: int = 20,
    replay: bool = False,
    guard: bool = False,
    seed: int = 0,
    acceptance: float = 0.95,
) -> dict:
    """Run one arm over the stream and return the final metrics + history."""

    rng = np.random.default_rng(int(seed))
    classes = sorted(set(str(c) for c in base_task["classes"]) | set(map(str, stream.labels)))
    cls_to_i = {c: i for i, c in enumerate(classes)}

    base_X = concat(base_rep, base_task["train_idx"])
    base_y = np.asarray(
        [cls_to_i[str(base_task["labels"][i])] for i in base_task["train_idx"]],
        dtype=np.int64,
    )
    val_X = concat(base_rep, base_task["val_idx"])
    test_X = concat(base_rep, base_task["test_idx"])
    test_y = np.asarray(
        [str(base_task["labels"][i]) for i in base_task["test_idx"]]
    )
    scorer = MahalanobisScorer().fit(base_X, base_y)
    threshold = calibrate_threshold(scorer, val_X, acceptance=acceptance)

    initial = evaluate_probe(
        scorer,
        classes,
        test_X,
        test_y,
        np.empty((0, base_X.shape[1])),
        np.empty(0),
        np.empty((0, base_X.shape[1])),
    )
    initial_old_acc = float(initial["old_acc"])
    unseen_guard_rows = np.where(stream.probe_mask & stream.unseen_mask)[0]
    unseen_guard_X = (
        concat(stream.rep, unseen_guard_rows)
        if len(unseen_guard_rows)
        else np.empty((0, base_X.shape[1]))
    )

    class_buffers: dict[int, list[np.ndarray]] = {}
    introduced: set[str] = set()
    labelled_rows: list[int] = []
    labelled_labels: list[str] = []
    rollbacks = 0
    updates = 0
    false_accepts = 0
    ari_values: list[float] = []
    history: list[dict] = []
    labels_used = 0
    queried_rows = 0
    queried_novel = 0

    max_episode = int(stream.episodes.max()) if stream.n else -1
    for episode in range(max_episode + 1):
        rows = np.where(
            (stream.episodes == episode)
            & stream.update_mask
            & ~stream.unseen_mask
        )[0]
        if len(rows) == 0:
            continue
        X_rows = concat(stream.rep, rows)
        scores = scorer.scores(X_rows).max(axis=1)

        if policy == "frozen":
            query = np.empty(0, dtype=np.int64)
        elif policy == "oracle_full":
            query = rows
        else:
            query = select_query(
                stream.rep, rows, scores, budget, policy, rng, threshold
            )

        if len(query):
            updates += 1
            labels = stream.labels[query]
            if policy == "self_train":
                pred = scorer.classes_[scorer.scores(concat(stream.rep, query)).argmax(axis=1)]
                labels = np.asarray([classes[int(p)] for p in pred])
            labels_used += len(labels)
            queried_rows += len(query)
            queried_novel += int(np.sum(stream.is_novel[query]))
            introduced.update(str(x) for x in labels)
            labelled_rows.extend(int(r) for r in query)
            labelled_labels.extend(str(x) for x in labels)

            if policy != "self_train":
                novel_query = [
                    (int(r), str(l)) for r, l in zip(query, labels)
                    if stream.is_novel[r]
                ]
                if len(novel_query) >= 2 and len({l for _, l in novel_query}) >= 2:
                    clusters = cluster_features(concat(stream.rep, [r for r, _ in novel_query]))
                    ari_values.append(
                        float(
                            adjusted_rand_score(
                                [l for _, l in novel_query], clusters
                            )
                        )
                    )

            old_scorer = copy.deepcopy(scorer) if guard else None
            before = evaluate_probe(
                scorer,
                classes,
                test_X,
                test_y,
                np.empty((0, base_X.shape[1])),
                np.empty(0),
                unseen_guard_X,
            )

            if replay:
                X_all = concat(stream.rep, labelled_rows)
                y_all = np.asarray([cls_to_i[l] for l in labelled_labels], dtype=np.int64)
                scorer = MahalanobisScorer().fit(
                    np.vstack([base_X, X_all]), np.concatenate([base_y, y_all])
                )
            else:
                _apply_labels(scorer, concat(stream.rep, query), labels, cls_to_i, class_buffers)

            if guard and old_scorer is not None:
                after = evaluate_probe(
                    scorer,
                    classes,
                    test_X,
                    test_y,
                    np.empty((0, base_X.shape[1])),
                    np.empty(0),
                    unseen_guard_X,
                )
                degraded = (
                    after["old_acc"] < before["old_acc"] - 0.03
                    or after["ece"] > before["ece"] + 0.05
                    or after["unknown_auroc"] < before["unknown_auroc"] - 0.05
                )
                if degraded:
                    scorer = old_scorer
                    rollbacks += 1
            else:
                after = evaluate_probe(
                    scorer,
                    classes,
                    test_X,
                    test_y,
                    np.empty((0, base_X.shape[1])),
                    np.empty(0),
                    unseen_guard_X,
                )
                if after["old_acc"] < before["old_acc"] - 0.03:
                    false_accepts += 1

        # Probe after the episode: introduced novel labels only.
        novel_rows = np.where(
            stream.probe_mask
            & stream.is_novel
            & ~stream.unseen_mask
            & np.isin(stream.labels, list(introduced))
        )[0]
        unseen_rows = np.where(stream.probe_mask & stream.unseen_mask)[0]
        metrics = evaluate_probe(
            scorer,
            classes,
            test_X,
            test_y,
            concat(stream.rep, novel_rows) if len(novel_rows) else np.empty((0, base_X.shape[1])),
            stream.labels[novel_rows] if len(novel_rows) else np.empty(0, dtype=stream.labels.dtype),
            concat(stream.rep, unseen_rows) if len(unseen_rows) else np.empty((0, base_X.shape[1])),
        )
        metrics.update(
            {
                "episode": int(episode),
                "labels_used": int(labels_used),
                "introduced": len(introduced),
                "rollbacks": int(rollbacks),
            }
        )
        history.append(metrics)

    final = history[-1] if history else {}
    if final:
        final["old_forgetting"] = initial_old_acc - float(final.get("old_acc", float("nan")))
        if math.isnan(float(final.get("novel_acc", float("nan")))):
            final["novel_acc"] = 0.0
    all_novel_rows = np.where(
        stream.probe_mask
        & stream.is_novel
        & ~stream.unseen_mask
        & np.isin(stream.labels, list(introduced))
    )[0]
    if len(all_novel_rows):
        eval_X = np.vstack([test_X, concat(stream.rep, all_novel_rows)])
        eval_y = np.concatenate([test_y, stream.labels[all_novel_rows]])
    else:
        eval_X, eval_y = test_X, test_y
    pred = np.asarray(classes)[
        scorer.classes_[scorer.scores(eval_X).argmax(axis=1)]
    ]
    class_acc = {
        str(c): float(np.mean(pred[eval_y == c] == c))
        for c in np.unique(eval_y)
        if int(np.sum(eval_y == c)) > 0
    }
    return {
        "policy": policy,
        "budget": int(budget),
        "replay": bool(replay),
        "guard": bool(guard),
        "seed": int(seed),
        "final": final,
        "history": history,
        "labels_used": int(labels_used),
        "updates": int(updates),
        "rollback_rate": float(rollbacks / updates) if updates else 0.0,
        "update_false_accept": float(false_accepts / updates) if updates else 0.0,
        "cluster_ari": float(np.nanmean(ari_values)) if ari_values else float("nan"),
        "n_classes": len(classes),
        "n_novel_labels": int(len({str(x) for x in stream.labels[stream.is_novel]})),
        "initial_old_acc": initial_old_acc,
        "class_acc": class_acc,
        "novel_query_fraction": (
            float(queried_novel / queried_rows) if queried_rows else float("nan")
        ),
    }
