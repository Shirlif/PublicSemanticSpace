"""Open-category stream runner (O0–O6).

Executable form of ``docs/plans/PSS-开放类别流实验计划.md`` §8:

  o0_smoke     tiny tier-A stream + two arms (protocol check)
  o1_tier_a    held-out categories, frozen/oracle/random/active
  o2_tier_b    new instances of known categories, active-query arms
  o3_tier_c    new categories + lighting/tactile shift
  o4_update    no-replay vs replay vs replay+guard
  o5_cells     model-proposed class errors vs the human A-line audit
  final        aggregate and grade

The oracle labels are the generator's ground truth.  That models a *perfect*
human labeler, not a real annotation budget; ``label_noise`` is the robustness
knob, not a substitute for a human study.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.g_audit import mean_std
from pss.eval.open_stream import (
    StreamData,
    load_base,
    load_stream,
    simulate,
)
from pss.eval.open_stream_preregistration import check_open_stream_preregistration
from pss.sim.generator import build_open_stream_from_index, generate_open_stream
from pss.sim.scene import CATEGORY_SPECS, HELD_OUT_CATEGORIES


# --------------------------------------------------------------------------- #
# json / format helpers
# --------------------------------------------------------------------------- #
def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


# --------------------------------------------------------------------------- #
# stream setup
# --------------------------------------------------------------------------- #
def _instance_labels(base_index: str, novel_per_category: int, start: int = 100):
    index = json.loads(Path(base_index).read_text(encoding="utf-8"))
    train = [e for e in index["entries"] if e.get("split") == "train"]
    seen = sorted({str(e["instance"]) for e in train if e.get("instance")})
    categories = sorted({str(e["category"]) for e in train if e.get("category")})
    novel = [
        f"{category}_v{variation:03d}"
        for category in categories
        for variation in range(start, start + int(novel_per_category))
    ]
    return seen, novel, categories


def _stream_dir(tier: str, args) -> Path:
    return {
        "a": Path(args.tier_a_dir),
        "b": Path(args.tier_b_dir),
        "c": Path(args.tier_c_dir),
    }[tier]


def _stream_labels(tier: str, args):
    """Source index and the seen/novel label split for one tier."""

    if tier == "b":
        index = json.loads(Path(args.base_instance_index).read_text(encoding="utf-8"))
        entries = [e for e in index["entries"] if e.get("kind") == "category"]
        instances = sorted(
            {str(e["instance"]) for e in entries if e.get("split") != "test_category"}
        )
        rng = np.random.default_rng(int(args.seed))
        order = rng.permutation(len(instances))
        n_novel = max(1, int(round(len(instances) * float(args.tier_b_novel_fraction))))
        novel = [instances[int(i)] for i in order[:n_novel]]
        seen = [instances[int(i)] for i in order[n_novel:]]
        return args.base_instance_index, seen, novel

    index = json.loads(Path(args.base_category_index).read_text(encoding="utf-8"))
    entries = [e for e in index["entries"] if e.get("kind") == "category"]
    seen = sorted(
        {str(e["category"]) for e in entries if e.get("split") != "test_category"}
    )
    novel = sorted(
        {str(e["category"]) for e in entries if e.get("split") == "test_category"}
    )
    return args.base_category_index, seen, novel


def _ensure_stream(tier: str, args):
    out = _stream_dir(tier, args)
    source, seen, novel = _stream_labels(tier, args)
    if (out / "index.json").exists() and not args.force:
        return out, seen, novel
    cache = Path(args.out) / f"_cache_stream_{tier}" / "paired_features.npz"
    if cache.exists():
        cache.unlink()
    label_mode = "instance" if tier == "b" else "category"
    build_open_stream_from_index(
        out,
        source,
        seen_labels=seen,
        novel_labels=novel,
        tier=tier,
        label_mode=label_mode,
        episodes=int(args.episodes),
        per_episode=int(args.per_episode),
        novel_rate=float(args.novel_rate),
        novel_start=int(args.novel_start),
        drift_episode=int(args.drift_episode),
        drift_mode=args.drift_mode,
        seed=int(args.seed),
    )
    return out, seen, novel


def _filter_task(task: dict, keep_labels) -> dict:
    """Restrict the base split to the stream's seen labels (Tier B)."""

    keep = {str(c) for c in keep_labels}
    labels = task["labels"]
    out = dict(task)
    for key in ("train_idx", "val_idx", "test_idx", "held_idx"):
        idx = task[key]
        mask = np.asarray([str(labels[i]) in keep for i in idx], dtype=bool)
        out[key] = idx[mask]
    out["classes"] = [c for c in task["classes"] if str(c) in keep]
    cls_to_i = {c: i for i, c in enumerate(out["classes"])}
    out["cls_to_i"] = cls_to_i
    out["ytr"] = np.asarray(
        [cls_to_i[str(labels[i])] for i in out["train_idx"]], dtype=np.int64
    )
    out["yva"] = np.asarray(
        [cls_to_i[str(labels[i])] for i in out["val_idx"]], dtype=np.int64
    )
    out["yte"] = np.asarray(
        [cls_to_i[str(labels[i])] for i in out["test_idx"]], dtype=np.int64
    )
    return out


def _setup(tier: str, args):
    if tier == "b":
        base_data = args.base_instance
        experts = args.experts_instance
        projections = args.projection_instance
        regime = "view-instance"
        label_mode = "instance"
    else:
        base_data = args.base_category
        experts = args.experts_category
        projections = args.projection_category
        regime = "category"
        label_mode = "category"

    base_rep, base_task = load_base(
        base_data,
        experts,
        projections,
        regime=regime,
        device=args.device,
        cache_dir=Path(args.out) / f"_cache_base_{tier}",
    )
    stream_dir, seen_labels, novel_labels = _ensure_stream(tier, args)
    if tier == "b":
        base_task = _filter_task(base_task, seen_labels)
    stream = load_stream(
        stream_dir,
        experts,
        projections,
        label_mode=label_mode,
        device=args.device,
        cache_dir=Path(args.out) / f"_cache_stream_{tier}",
        split_seed=0,
    )
    return base_rep, base_task, stream, label_mode


# --------------------------------------------------------------------------- #
# arm aggregation
# --------------------------------------------------------------------------- #
ARM_SPECS = {
    "frozen": {"policy": "frozen", "replay": False, "guard": False},
    "oracle_full": {"policy": "oracle_full", "replay": True, "guard": False},
    "random_query": {"policy": "random", "replay": False, "guard": False},
    "novelty_query": {"policy": "novelty", "replay": False, "guard": False},
    "active_no_cluster": {"policy": "active_no_cluster", "replay": False, "guard": False},
    "active_cluster": {"policy": "active_cluster", "replay": False, "guard": False},
    "self_train": {"policy": "self_train", "replay": False, "guard": False},
    "active_no_replay": {"policy": "active_cluster", "replay": False, "guard": False},
    "active_replay": {"policy": "active_cluster", "replay": True, "guard": False},
    "active_replay_guard": {"policy": "active_cluster", "replay": True, "guard": True},
}


def _run_arm(base_rep, base_task, stream, arm: str, seeds, budget: int, acceptance: float):
    spec = ARM_SPECS[arm]
    runs = [
        simulate(
            base_rep,
            base_task,
            stream,
            budget=int(budget),
            seed=int(seed),
            acceptance=float(acceptance),
            **spec,
        )
        for seed in seeds
    ]
    keys = ("novel_acc", "old_acc", "unknown_auroc", "ece", "old_forgetting")
    out = {
        key: mean_std([r["final"].get(key, float("nan")) for r in runs])
        for key in keys
    }
    for key in (
        "cluster_ari",
        "rollback_rate",
        "update_false_accept",
        "labels_used",
        "novel_query_fraction",
    ):
        out[key] = mean_std([float(r.get(key, float("nan"))) for r in runs])
    out["n_novel_labels"] = int(runs[0]["n_novel_labels"])
    out["history"] = runs[0]["history"]
    out["class_acc"] = runs[0]["class_acc"]
    out["per_seed"] = [
        {
            "seed": int(r["seed"]),
            "final": r["final"],
            "labels_used": int(r["labels_used"]),
            "rollback_rate": float(r["rollback_rate"]),
        }
        for r in runs
    ]
    return out


def _observed(arms: dict) -> dict:
    observed = {}
    for arm, values in arms.items():
        observed[arm] = {
            key: values[key]["mean"]
            for key in (
                "novel_acc",
                "old_forgetting",
                "unknown_auroc",
                "cluster_ari",
                "ece",
                "rollback_rate",
                "update_false_accept",
            )
        }
    return observed


def _relative(arms: dict, kind: str) -> dict:
    def _mean(arm, key):
        if arm not in arms:
            return None
        value = float(arms[arm][key]["mean"])
        return None if math.isnan(value) else value

    if kind == "o2":
        out = {}
        active = _mean("active_cluster", "novel_acc")
        random = _mean("random_query", "novel_acc")
        if active is not None and random is not None:
            out["random_query_vs_active"] = random - active
        self_forget = _mean("self_train", "old_forgetting")
        if self_forget is not None:
            out["self_train_old_forgetting"] = self_forget
        self_novel = _mean("self_train", "novel_acc")
        if active is not None and self_novel is not None:
            out["self_train_novel_delta"] = self_novel - active
        return out
    if kind == "o4":
        no_replay = _mean("active_no_replay", "old_forgetting")
        replay = _mean("active_replay", "old_forgetting")
        if no_replay is not None and replay is not None:
            return {"no_replay_vs_replay": no_replay - replay}
        return {}
    return {}


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def _run_stage(tier: str, arms_to_run, args) -> dict:
    base_rep, base_task, stream, label_mode = _setup(tier, args)
    print(
        f"[{tier}] base classes={len(base_task['classes'])} "
        f"stream rows={stream.n} novel rows={int(stream.is_novel.sum())}",
        flush=True,
    )
    arms = {}
    for arm in arms_to_run:
        arms[arm] = _run_arm(
            base_rep,
            base_task,
            stream,
            arm,
            args.seeds,
            args.budget,
            args.acceptance,
        )
        print(
            f"[{tier}] {arm:<20} novel={_fmt(arms[arm]['novel_acc']['mean'])} "
            f"old={_fmt(arms[arm]['old_acc']['mean'])} "
            f"forget={_fmt(arms[arm]['old_forgetting']['mean'])} "
            f"unknown={_fmt(arms[arm]['unknown_auroc']['mean'])}",
            flush=True,
        )
    return {"tier": tier, "label_mode": label_mode, "arms": arms}


def _payload_header(stage: str, args) -> dict:
    return {
        "stage": stage,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config": {
            "device": args.device,
            "seeds": [int(s) for s in args.seeds],
            "budget": int(args.budget),
            "episodes": int(args.episodes),
            "per_episode": int(args.per_episode),
            "novel_rate": float(args.novel_rate),
            "novel_start": int(args.novel_start),
            "drift_episode": int(args.drift_episode),
            "acceptance": float(args.acceptance),
        },
    }


def stage_o0_smoke(args) -> dict:
    out = Path(args.out)
    smoke_dir = out / "_smoke_stream"
    if not (smoke_dir / "index.json").exists() or args.force:
        source, _, _ = _stream_labels("a", args)
        seen = [c for c in CATEGORY_SPECS if c not in HELD_OUT_CATEGORIES][:6]
        novel = list(HELD_OUT_CATEGORIES)[:2]
        build_open_stream_from_index(
            smoke_dir,
            source,
            seen_labels=seen,
            novel_labels=novel,
            tier="a",
            label_mode="category",
            episodes=6,
            per_episode=20,
            novel_rate=0.25,
            novel_start=2,
            seed=0,
        )
    smoke_cache = out / "_cache_stream_a" / "paired_features.npz"
    if smoke_cache.exists():
        smoke_cache.unlink()
    original = args.tier_a_dir
    original_force = args.force
    args.tier_a_dir = str(smoke_dir)
    args.force = False  # the 6x20 smoke stream is already built above
    try:
        base_rep, base_task, stream, label_mode = _setup("a", args)
        runs = {}
        for arm in ("frozen", "active_cluster"):
            runs[arm] = _run_arm(
                base_rep, base_task, stream, arm, [0], 5, args.acceptance
            )
    finally:
        args.tier_a_dir = original
        args.force = original_force
    payload = _payload_header("o0_smoke", args)
    payload["config"]["episodes"] = 6
    payload["config"]["per_episode"] = 20
    payload.update({"tier": "a", "label_mode": label_mode, "arms": runs})
    _write_json(out / "o0_smoke.json", payload)
    (out / "O0_SMOKE.md").write_text(_render_stage(payload, "O0 smoke"), encoding="utf-8")
    return payload


def stage_o1_tier_a(args) -> dict:
    result = _run_stage(
        "a",
        ("frozen", "oracle_full", "random_query", "novelty_query", "active_cluster"),
        args,
    )
    payload = _payload_header("o1_tier_a", args)
    payload.update(result)
    check = check_open_stream_preregistration(
        _observed(result["arms"]), _relative(result["arms"], "o2")
    )
    payload["preregistration"] = check
    _write_json(Path(args.out) / "o1_tier_a.json", payload)
    (Path(args.out) / "O1_TIER_A.md").write_text(
        _render_stage(payload, "O1 Tier A"), encoding="utf-8"
    )
    return payload


def stage_o1_budget(args) -> dict:
    """Label-efficiency sweep at B=5/10/20 for the Tier-A query arms."""

    out = Path(args.out)
    base_rep, base_task, stream, label_mode = _setup("a", args)
    budgets = [5, 10, 20]
    results = {}
    for budget in budgets:
        arms = {}
        for arm in ("random_query", "novelty_query", "active_cluster"):
            arms[arm] = _run_arm(
                base_rep, base_task, stream, arm, args.seeds, budget, args.acceptance
            )
            print(
                f"[a:b{budget}] {arm:<16} novel={_fmt(arms[arm]['novel_acc']['mean'])} "
                f"old={_fmt(arms[arm]['old_acc']['mean'])} "
                f"forget={_fmt(arms[arm]['old_forgetting']['mean'])}",
                flush=True,
            )
        results[str(budget)] = {
            "budget": budget,
            "arms": arms,
            "preregistration": check_open_stream_preregistration(
                _observed(arms), _relative(arms, "o2")
            ),
        }
    payload = _payload_header("o1_budget", args)
    payload["config"]["budgets"] = budgets
    payload.update({"tier": "a", "label_mode": label_mode, "budgets": results})
    _write_json(out / "o1_budget_sweep.json", payload)
    (out / "O1_BUDGET_SWEEP.md").write_text(_render_budget(payload), encoding="utf-8")
    return payload


def _render_budget(p: dict) -> str:
    lines = [
        "# O1 标签效率曲线（Tier A）",
        "",
        f"- generated: {p['generated']}",
        f"- seeds: {p['config']['seeds']} · budgets: {p['config']['budgets']}",
        "",
        "| budget | arm | novel acc | old acc | forgetting | unknown AUROC |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for budget, block in p["budgets"].items():
        for arm, values in block["arms"].items():
            lines.append(
                f"| {budget} | `{arm}` | {_fmt(values['novel_acc']['mean'])} | "
                f"{_fmt(values['old_acc']['mean'])} | "
                f"{_fmt(values['old_forgetting']['mean'])} | "
                f"{_fmt(values['unknown_auroc']['mean'])} |"
            )
        check = block["preregistration"]
        lines.append(
            f"| {budget} | _checker_ | ok={check['ok']} · "
            f"{'; '.join(check['mismatches']) if check['mismatches'] else 'all bars passed'} |"
            " | | |"
        )
    lines += [
        "",
        "## 判读",
        "",
        "- 看 `random_query` 与 `active_cluster` 的差距随 budget 缩小/扩大；"
        "B=20 时 random 可能已接近 ceiling，active 的优势应出现在低预算端。",
        "- `novelty_query` 是“只看分数、不聚类”的对照；若它≈active，聚类没有额外贡献。",
        "",
    ]
    return "\n".join(lines)


def stage_o2_tier_b(args) -> dict:
    result = _run_stage(
        "b",
        (
            "frozen",
            "oracle_full",
            "random_query",
            "novelty_query",
            "active_no_cluster",
            "active_cluster",
            "self_train",
        ),
        args,
    )
    payload = _payload_header("o2_tier_b", args)
    payload.update(result)
    check = check_open_stream_preregistration(
        _observed(result["arms"]), _relative(result["arms"], "o2")
    )
    payload["preregistration"] = check
    _write_json(Path(args.out) / "o2_tier_b.json", payload)
    (Path(args.out) / "O2_TIER_B.md").write_text(
        _render_stage(payload, "O2 Tier B"), encoding="utf-8"
    )
    return payload


def stage_o3_tier_c(args) -> dict:
    result = _run_stage(
        "c",
        ("frozen", "random_query", "novelty_query", "active_cluster"),
        args,
    )
    payload = _payload_header("o3_tier_c", args)
    payload.update(result)
    check = check_open_stream_preregistration(
        _observed(result["arms"]), _relative(result["arms"], "o2")
    )
    payload["preregistration"] = check
    _write_json(Path(args.out) / "o3_tier_c.json", payload)
    (Path(args.out) / "O3_TIER_C.md").write_text(
        _render_stage(payload, "O3 Tier C"), encoding="utf-8"
    )
    return payload


def stage_o4_update(args) -> dict:
    result = _run_stage(
        "b",
        ("active_no_replay", "active_replay", "active_replay_guard"),
        args,
    )
    payload = _payload_header("o4_update", args)
    payload.update(result)
    check = check_open_stream_preregistration(
        _observed(result["arms"]), _relative(result["arms"], "o4")
    )
    payload["preregistration"] = check
    _write_json(Path(args.out) / "o4_update.json", payload)
    (Path(args.out) / "O4_UPDATE.md").write_text(
        _render_stage(payload, "O4 update"), encoding="utf-8"
    )
    return payload


def _category_of(class_name: str) -> str:
    return str(class_name).split("_v", 1)[0]


def stage_o5_cells(args) -> dict:
    out = Path(args.out)
    # Prefer Tier B: its labels are instance ids, the same space as the A-line
    # audit's per_class rows.  Tier A is a different (category) task, so using
    # it here would compare two unrelated error rankings.
    o2 = _read_json(out / "o2_tier_b.json")
    o1 = _read_json(out / "o1_tier_a.json")
    source = o2 or o1
    source_name = "o2_tier_b" if o2 else "o1_tier_a"
    audit_path = Path(args.g_audit) / "a_coverage.json"
    if source is None:
        raise SystemExit("o1_tier_a.json / o2_tier_b.json missing")
    if not audit_path.exists():
        payload = {
            "stage": "o5_cells",
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "model_cells": None,
            "human_cells": None,
            "top1_match": None,
            "spearman": None,
            "note": f"{audit_path} missing; run the g audit A-line stage first",
        }
        _write_json(out / "o5_cells.json", payload)
        (out / "O5_CELLS.md").write_text(_render_o5(payload), encoding="utf-8")
        return payload

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if source_name == "o2_tier_b":
        human = {
            str(row["class"]): 1.0 - float(row["acc"])
            for row in audit.get("per_class", [])
        }
    else:
        human_by_cat = {}
        for row in audit.get("per_class", []):
            category = _category_of(row["class"])
            human_by_cat.setdefault(category, []).append(float(row["acc"]))
        human = {c: 1.0 - float(np.mean(v)) for c, v in human_by_cat.items()}

    class_acc = source["arms"]["active_cluster"].get("class_acc", {})
    model = {
        c: 1.0 - float(acc)
        for c, acc in class_acc.items()
        if not math.isnan(float(acc))
    }
    common = sorted(set(model) & set(human))
    if len(common) < 2:
        payload = {
            "stage": "o5_cells",
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "model_cells": model,
            "human_cells": human,
            "common": common,
            "top1_match": None,
            "spearman": None,
            "note": "fewer than 2 shared categories",
        }
        _write_json(out / "o5_cells.json", payload)
        (out / "O5_CELLS.md").write_text(_render_o5(payload), encoding="utf-8")
        return payload

    from scipy.stats import spearmanr

    model_rank = [model[c] for c in common]
    human_rank = [human[c] for c in common]
    spearman = float(spearmanr(model_rank, human_rank).statistic)
    model_worst = max(common, key=lambda c: model[c])
    human_worst = max(common, key=lambda c: human[c])
    payload = {
        "stage": "o5_cells",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model_cells": model,
        "human_cells": human,
        "common": common,
        "model_worst": model_worst,
        "human_worst": human_worst,
        "top1_match": float(model_worst == human_worst),
        "spearman": spearman,
        "source": source_name,
        "note": (
            f"source={source_name}; model cells = per-class error on the stream "
            "probe; human cells = per-class error from the A-line audit. This is "
            "a cross-check on the same class space, not the geometry cell audit."
        ),
    }
    _write_json(out / "o5_cells.json", payload)
    (out / "O5_CELLS.md").write_text(_render_o5(payload), encoding="utf-8")
    return payload


def stage_final(args) -> dict:
    out = Path(args.out)
    sources = {
        name: _read_json(out / f"{name}.json")
        for name in ("o1_tier_a", "o2_tier_b", "o3_tier_c", "o4_update", "o5_cells")
    }
    grading = {}
    for name, data in sources.items():
        if not data:
            continue
        if "arms" in data:
            grading[name] = {
                arm: {
                    "novel_acc": values["novel_acc"]["mean"],
                    "old_forgetting": values["old_forgetting"]["mean"],
                    "unknown_auroc": values["unknown_auroc"]["mean"],
                    "cluster_ari": values["cluster_ari"]["mean"],
                    "rollback_rate": values["rollback_rate"]["mean"],
                }
                for arm, values in data["arms"].items()
            }
        else:
            grading[name] = {k: data.get(k) for k in ("top1_match", "spearman")}
    payload = {
        "stage": "final",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "sources": {name: bool(data) for name, data in sources.items()},
        "grading": grading,
        "caveat": (
            "Oracle labels model a perfect human labeler; Tier C is bounded by "
            "MuJoCo/sensor primitives; the encoder is frozen. No claim of general "
            "autonomy or real-VLM transfer."
        ),
    }
    _write_json(out / "final_matrix.json", payload)
    (out / "OPEN_STREAM_FINAL.md").write_text(_render_final(payload), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------- #
# renderers
# --------------------------------------------------------------------------- #
def _render_stage(p: dict, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"- generated: {p['generated']}",
        f"- tier: `{p.get('tier')}` · label_mode: `{p.get('label_mode')}`",
        f"- seeds: {p['config']['seeds']} · budget: {p['config']['budget']} · "
        f"episodes: {p['config']['episodes']} × {p['config']['per_episode']}",
        "",
        "| arm | novel acc | old acc | forgetting | unknown AUROC | cluster ARI | "
        "labels used | rollbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in p["arms"].items():
        lines.append(
            f"| `{arm}` | {_fmt(values['novel_acc']['mean'])} ± "
            f"{_fmt(values['novel_acc']['std'])} | "
            f"{_fmt(values['old_acc']['mean'])} | "
            f"{_fmt(values['old_forgetting']['mean'])} | "
            f"{_fmt(values['unknown_auroc']['mean'])} | "
            f"{_fmt(values['cluster_ari']['mean'])} | "
            f"{_fmt(values['labels_used']['mean'], 0)} | "
            f"{_fmt(values['rollback_rate']['mean'])} |"
        )
    check = p.get("preregistration")
    if check:
        lines += [
            "",
            "## 预注册检查",
            "",
            f"- checked: {check['checked']} · ok: **{check['ok']}**",
        ]
        for mismatch in check["mismatches"]:
            lines.append(f"- `{mismatch}`")
    lines += [
        "",
        "## 判读",
        "",
        "- `novel acc` 只算已出现的新类 probe；`forgetting` 是 base probe 的下降。",
        "- `cluster ARI` 只在查询到 ≥2 个新类的 episode 上有定义。",
        "- `labels used` 是累计查询标签数；oracle_full 是上界参照。",
        "",
    ]
    return "\n".join(lines)


def _render_o5(p: dict) -> str:
    lines = [
        "# O5：模型 cell vs 人工 cell",
        "",
        f"- generated: {p.get('generated')}",
        f"- note: {p.get('note')}",
        "",
    ]
    if p.get("top1_match") is None:
        lines.append("- 无法比较（见 note）。")
        return "\n".join(lines)
    lines += [
        f"- model worst: `{p['model_worst']}`",
        f"- human worst: `{p['human_worst']}`",
        f"- top-1 match: **{p['top1_match']:.0f}**",
        f"- Spearman: **{p['spearman']:+.3f}**",
        "",
        "只展示 model/human 误差合计最高的 20 个类；完整表在 JSON。",
        "",
        "| category | model error | human error |",
        "|---|---:|---:|",
    ]
    top = sorted(
        p["common"],
        key=lambda c: -(p["model_cells"][c] + p["human_cells"][c]),
    )[:20]
    for category in top:
        lines.append(
            f"| `{category}` | {_fmt(p['model_cells'][category])} | "
            f"{_fmt(p['human_cells'][category])} |"
        )
    return "\n".join(lines)


def _render_final(p: dict) -> str:
    lines = [
        "# 开放类别流最终矩阵（O6）",
        "",
        f"- generated: {p['generated']}",
        f"- sources: {p['sources']}",
        "",
        "## 各阶段主结果",
        "",
        "| stage | arm | novel acc | forgetting | unknown AUROC | cluster ARI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for stage, arms in p["grading"].items():
        if not isinstance(arms, dict):
            continue
        for arm, values in arms.items():
            if not isinstance(values, dict) or "novel_acc" not in values:
                continue
            lines.append(
                f"| {stage} | `{arm}` | {_fmt(values.get('novel_acc'))} | "
                f"{_fmt(values.get('old_forgetting'))} | "
                f"{_fmt(values.get('unknown_auroc'))} | "
                f"{_fmt(values.get('cluster_ari'))} |"
            )
    lines += [
        "",
        "## 边界",
        "",
        f"- {p['caveat']}",
        "",
    ]
    return "\n".join(lines)


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_stage(args) -> dict:
    return {
        "o0_smoke": stage_o0_smoke,
        "o1_tier_a": stage_o1_tier_a,
        "o1_budget": stage_o1_budget,
        "o2_tier_b": stage_o2_tier_b,
        "o3_tier_c": stage_o3_tier_c,
        "o4_update": stage_o4_update,
        "o5_cells": stage_o5_cells,
        "final": stage_final,
    }[args.stage](args)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage",
        required=True,
        choices=(
            "o0_smoke",
            "o1_tier_a",
            "o1_budget",
            "o2_tier_b",
            "o3_tier_c",
            "o4_update",
            "o5_cells",
            "final",
        ),
    )
    ap.add_argument("--out", default="eval/results/open_stream")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--acceptance", type=float, default=0.95)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--per-episode", type=int, default=200)
    ap.add_argument("--novel-rate", type=float, default=0.2)
    ap.add_argument("--novel-start", type=int, default=5)
    ap.add_argument("--drift-episode", type=int, default=10)
    ap.add_argument("--drift-mode", choices=("lighting", "tactile", "both"), default="both")
    ap.add_argument("--novel-per-category", type=int, default=2)
    ap.add_argument("--tier-b-novel-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tier-a-dir", default="data/open_stream/tier_a")
    ap.add_argument("--tier-b-dir", default="data/open_stream/tier_b")
    ap.add_argument("--tier-c-dir", default="data/open_stream/tier_c")
    ap.add_argument("--base-category", default="data/category_pretrain")
    ap.add_argument("--base-category-index", default="data/category_pretrain/index.json")
    ap.add_argument("--base-instance", default="data/category_pretrain_instance8")
    ap.add_argument("--base-instance-index", default="data/category_pretrain_instance8/index.json")
    ap.add_argument("--experts-category", default="checkpoints/category_pretrain_experts")
    ap.add_argument(
        "--experts-instance", default="checkpoints/category_pretrain_instance8_experts"
    )
    ap.add_argument("--projection-category", default="checkpoints/phase45_projection")
    ap.add_argument(
        "--projection-instance",
        default="checkpoints/category_pretrain_instance8_projection",
    )
    ap.add_argument("--g-audit", default="eval/results/g_audit")
    args = ap.parse_args()
    payload = run_stage(args)
    print(f"output: {Path(args.out)}")
    print(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
