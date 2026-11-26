"""P0-P3 evaluation: derivation precision/recall on the spatial suite.

Three families are in scope, each testing a different operator:

- ``transitive``: the closure table over a two-edge path.  Two measurements:
  answering the suite's yes/no item, and recovering a sparse consecutive chain.
- ``frame``: reference-frame scoping.  The suite builds these items so the
  world and the named frame disagree, so a frame-blind evaluator must fail.
- ``perspective``: the same scoping test with ``viewer:cam1``.

The script is deterministic (seeded generator) and writes ``metrics.json`` and
``REPORT.md`` under ``reports/claim_gaps/operator_derivation``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.binding.frames import FrameOp, relation_in_frame
from pss.eval.spatial import predicates as P
from pss.eval.spatial.generator import generate_suite
from pss.reasoning.relations import derive

DIRECTIONAL = ("left_of", "right_of", "front_of", "behind")
ALL_PREDICATES = (
    "left_of",
    "right_of",
    "front_of",
    "behind",
    "above",
    "below",
    "contact",
    "near",
    "inside",
    "on",
)
DEFAULT_FAMILIES = ("transitive", "frame", "perspective")


def _binary_metrics(rows) -> dict:
    """``rows`` is a list of ``(gold, predicted)`` values in {"yes", "no"}."""
    tp = fp = tn = fn = 0
    for gold, predicted in rows:
        if gold == "yes" and predicted == "yes":
            tp += 1
        elif gold == "yes":
            fn += 1
        elif predicted == "yes":
            fp += 1
        else:
            tn += 1
    n = len(rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is not None and recall is not None and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else None,
        "precision": precision,
        "recall": recall,
        "f1_yes": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _frame_op(scene, frame: str, subject=None) -> FrameOp:
    right, front, up = P.frame_axes(scene, frame, subject)
    return FrameOp.from_axes(right, front, up, np.zeros(3), name=frame)


def _chain(scene, frame: str):
    """Consecutive strictly-left-of edges along ``frame``'s right axis."""
    right, _front, _up = P.frame_axes(scene, frame)
    ordered = sorted(scene.objects, key=lambda obj: float(np.dot(np.asarray(obj.pos), right)))
    observed = []
    for left, right_obj in zip(ordered, ordered[1:]):
        if P.relation(scene, left, right_obj, "left_of", frame):
            observed.append((left.id, "left_of", right_obj.id))
        else:
            break
    ids = [obj.id for obj in ordered[: len(observed) + 1]]
    return observed, ids


def evaluate_transitive(items) -> dict:
    rows = []
    direct_rows = []
    by_frame: dict[str, list] = {}
    for item in items:
        meta = item.query.meta
        predicate = meta["predicate"]
        (a, b), (b2, c) = meta["premises"]
        records = [(a, predicate, b), (b2, predicate, c)]
        result = derive(records, max_rounds=2, min_conf=0.0)
        query = (meta["subject"], predicate, meta["object"])
        predicted = "yes" if query in result.facts() else "no"
        rows.append((item.query.answer, predicted))
        # Baseline: answer "yes" only when the conclusion is directly observed.
        direct = "yes" if query in set(records) else "no"
        direct_rows.append((item.query.answer, direct))
        by_frame.setdefault(meta.get("frame", "world"), []).append((item.query.answer, predicted))
    return {
        "overall": _binary_metrics(rows),
        "direct_only": _binary_metrics(direct_rows),
        "by_frame": {frame: _binary_metrics(rs) for frame, rs in by_frame.items()},
    }


def evaluate_sparse_chain(items) -> dict:
    tp = fp = fn = 0
    derived_total = 0
    scenes = 0
    for item in items:
        scene = item.scene
        frame = item.query.meta.get("frame", "world")
        observed, ids = _chain(scene, frame)
        if len(observed) < 2:
            continue
        result = derive(observed, max_rounds=len(observed), min_conf=0.0)
        predicted = result.facts() | set(observed)
        gt = {
            (x, "left_of", y)
            for x in ids
            for y in ids
            if x != y and P.relation(scene, scene.obj(x), scene.obj(y), "left_of", frame)
        }
        tp += len(predicted & gt)
        fp += len(predicted - gt)
        fn += len(gt - predicted)
        derived_total += len(result.facts())
        scenes += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = None
    if precision is not None and recall is not None and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "scenes": scenes,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "derived_total": derived_total,
        "derived_mean": derived_total / scenes if scenes else None,
    }


def _gt_facts(scene, frame: str = "world") -> set[tuple[str, str, str]]:
    out = set()
    for a in scene.objects:
        for b in scene.objects:
            if a.id == b.id:
                continue
            for predicate in ALL_PREDICATES:
                if P.relation(scene, a, b, predicate, frame):
                    out.add((a.id, predicate, b.id))
    return out


def evaluate_holdout(items, holdout: float = 0.2, seed: int = 0, frame: str = "world") -> dict:
    """Hide a fraction of a full ground-truth fact set and try to re-derive it.

    Precision counts derived facts that are false against the full ground
    truth; recall counts hidden facts the table recovered.  This is where weak
    entries such as ``contact ∘ contact -> near`` show up as false positives.
    """
    tp = fp = fn = 0
    hidden_total = derived_total = 0
    scenes = 0
    seen: set[str] = set()
    for index, item in enumerate(items):
        scene = item.scene
        if scene.scene_id in seen:
            continue
        seen.add(scene.scene_id)
        gt = _gt_facts(scene, frame)
        if len(gt) < 4:
            continue
        ordered = sorted(gt)
        rng = np.random.default_rng(seed + index)
        k = max(1, int(round(holdout * len(ordered))))
        chosen = rng.choice(len(ordered), size=k, replace=False)
        hidden = {ordered[int(i)] for i in chosen}
        observed = gt - hidden
        result = derive(observed, max_rounds=2, min_conf=0.0)
        derived = result.facts()
        tp += len(derived & hidden)
        fp += len(derived - gt)
        fn += len(hidden - derived)
        hidden_total += len(hidden)
        derived_total += len(derived)
        scenes += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = None
    if precision is not None and recall is not None and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "scenes": scenes,
        "holdout": holdout,
        "hidden_total": hidden_total,
        "derived_total": derived_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_frame_scoping(items, family: str) -> dict:
    ours, world_only = [], []
    frame_sensitive = 0
    for item in items:
        scene = item.scene
        meta = item.query.meta
        a = scene.obj(meta["subject"])
        b = scene.obj(meta["object"])
        predicate = meta["predicate"]
        frame = meta["frame"]
        if predicate not in DIRECTIONAL:
            continue
        gold = P.relation(scene, a, b, predicate, frame)
        world = P.relation(scene, a, b, predicate, "world")
        subject = a if frame == "object" else None
        flags = relation_in_frame(a.pos, b.pos, _frame_op(scene, frame, subject))
        ours.append(("yes" if gold else "no", "yes" if predicate in flags else "no"))
        world_only.append(("yes" if gold else "no", "yes" if world else "no"))
        if gold != world:
            frame_sensitive += 1
    metrics = {
        "frame_op": _binary_metrics(ours),
        "world_only": _binary_metrics(world_only),
        "frame_sensitive_rate": frame_sensitive / len(ours) if ours else None,
    }
    metrics["family"] = family
    return metrics


def evaluate_cross_frame(items, target_frame: str) -> dict:
    """Derive in the world frame, then check the conclusion in ``target_frame``."""
    agree = contradict = 0
    scenes = set()
    for item in items:
        scene = item.scene
        if scene.scene_id in scenes:
            continue
        scenes.add(scene.scene_id)
        observed, _ids = _chain(scene, "world")
        if len(observed) < 2:
            continue
        result = derive(observed, max_rounds=len(observed), min_conf=0.0)
        for s, _p, o in result.facts():
            if P.relation(scene, scene.obj(s), scene.obj(o), "left_of", target_frame):
                agree += 1
            else:
                contradict += 1
    total = agree + contradict
    return {
        "scenes": len(scenes),
        "pairs_checked": total,
        "agree": agree,
        "contradict": contradict,
        "contradiction_rate": contradict / total if total else None,
    }


def run(n_per_family: int = 100, seed: int = 0) -> dict:
    items = generate_suite(
        n_per_family=n_per_family,
        seed=seed,
        split="test",
        ood="id",
        families=DEFAULT_FAMILIES,
    )
    transitive = [item for item in items if item.query.family == "transitive"]
    frame = [item for item in items if item.query.family == "frame"]
    perspective = [item for item in items if item.query.family == "perspective"]

    metrics = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "n_per_family": n_per_family,
        "seed": seed,
        "transitive_answer": evaluate_transitive(transitive),
        "transitive_sparse_chain": evaluate_sparse_chain(transitive),
        "holdout_derivation": evaluate_holdout(transitive),
        "frame_scoping": evaluate_frame_scoping(frame, "frame"),
        "perspective_scoping": evaluate_frame_scoping(perspective, "perspective"),
        "cross_frame": {
            "world_to_viewer": evaluate_cross_frame(frame, "viewer"),
            "world_to_cam1": evaluate_cross_frame(perspective, "viewer:cam1"),
        },
    }
    return metrics


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _report(metrics: dict) -> str:
    ta = metrics["transitive_answer"]
    sc = metrics["transitive_sparse_chain"]
    ho = metrics["holdout_derivation"]
    fs = metrics["frame_scoping"]
    ps = metrics["perspective_scoping"]
    cf = metrics["cross_frame"]
    lines = [
        "# Operator derivation on the spatial suite",
        "",
        f"> Generated: {metrics['generated_at']}",
        f"> Seed: {metrics['seed']}, items per family: {metrics['n_per_family']}",
        "",
        "## 1. Transitive items (closure table)",
        "",
        "| frame | n | accuracy | precision | recall | F1 (yes) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for frame, row in ta["by_frame"].items():
        lines.append(
            f"| {frame} | {row['n']} | {_fmt(row['accuracy'])} | {_fmt(row['precision'])} | "
            f"{_fmt(row['recall'])} | {_fmt(row['f1_yes'])} |"
        )
    overall = ta["overall"]
    lines.append(
        f"| **all** | {overall['n']} | {_fmt(overall['accuracy'])} | "
        f"{_fmt(overall['precision'])} | {_fmt(overall['recall'])} | {_fmt(overall['f1_yes'])} |"
    )
    direct = ta["direct_only"]
    lines += [
        "",
        f"Baseline `direct_only` (answer yes only when the conclusion is observed): "
        f"accuracy {_fmt(direct['accuracy'])}, F1 {_fmt(direct['f1_yes'])}.",
        "",
        "## 2. Sparse consecutive chain (derived left_of)",
        "",
        "Only consecutive edges are observed; all longer-range left_of facts are derived.",
        "",
        "| scenes | precision | recall | F1 | derived/scene |",
        "|---:|---:|---:|---:|---:|",
        f"| {sc['scenes']} | {_fmt(sc['precision'])} | {_fmt(sc['recall'])} | "
        f"{_fmt(sc['f1'])} | {_fmt(sc['derived_mean'])} |",
        "",
        "## 3. Holdout on the full ground-truth fact set",
        "",
        "A fraction of all true facts (all ten predicates) is hidden; the table must re-derive "
        "them without producing false facts.",
        "",
        "| scenes | hidden | derived | precision | recall | F1 | tp | fp | fn |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {ho['scenes']} | {ho['hidden_total']} | {ho['derived_total']} | "
        f"{_fmt(ho['precision'])} | {_fmt(ho['recall'])} | {_fmt(ho['f1'])} | "
        f"{ho['tp']} | {ho['fp']} | {ho['fn']} |",
        "",
        "## 4. Frame scoping",
        "",
        "`frame_op` builds the reference frame with `FrameOp.from_axes`; "
        "`world_only` ignores the named frame.",
        "",
        "| family | n | frame_op acc | world_only acc | frame-sensitive |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in (("frame", fs), ("perspective", ps)):
        lines.append(
            f"| {name} | {row['frame_op']['n']} | {_fmt(row['frame_op']['accuracy'])} | "
            f"{_fmt(row['world_only']['accuracy'])} | {_fmt(row['frame_sensitive_rate'])} |"
        )
    lines += [
        "",
        "## 5. Cross-frame transfer",
        "",
        "Derive a left_of chain in the world frame, then check the conclusion in another frame.",
        "",
        "| target frame | scenes | pairs | agree | contradict | contradiction rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in cf.items():
        lines.append(
            f"| {name} | {row['scenes']} | {row['pairs_checked']} | {row['agree']} | "
            f"{row['contradict']} | {_fmt(row['contradiction_rate'])} |"
        )
    lines += [
        "",
        "## 6. Decision",
        "",
        f"- Closure derivation is **sound**: holdout precision {_fmt(ho['precision'])} "
        f"with {ho['fp']} false facts.",
        f"- It is **low-coverage**: recall {_fmt(ho['recall'])} on the full "
        "ten-predicate fact set; most relations are not licensed by any composition entry.",
        f"- It **earns its place on transitive items**: accuracy {_fmt(overall['accuracy'])} vs "
        f"{_fmt(direct['accuracy'])} for a direct-only baseline.",
        f"- Frame scoping is exact with `FrameOp` ({_fmt(fs['frame_op']['accuracy'])}), while "
        f"ignoring the frame drops to {_fmt(fs['world_only']['accuracy'])}.",
        f"- Cross-frame transfer is unsafe: "
        f"{_fmt(cf['world_to_cam1']['contradiction_rate'])} of world-frame conclusions "
        "contradict cam1-frame truth.",
        "",
        "**Recommendation:** do not enable derivation in `PSSRuntime.ingest` by default. "
        "Expose it as an opt-in planner query once facts carry a frame tag; without that tag, "
        "derivations can mix reference frames.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(description="Operator derivation on the spatial suite")
    parser.add_argument("--out", default="reports/claim_gaps/operator_derivation")
    parser.add_argument("--n-per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    metrics = run(n_per_family=args.n_per_family, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "REPORT.md").write_text(_report(metrics), encoding="utf-8")
    print(_report(metrics))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
