"""Phase 8 evaluation: action head closed loop, text/value readouts, ablation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss.encoders.data import load_index
from pss.memory import AttentionScheduler, Track
from pss.reasoning.relations import relations_from_frame
from pss.sim.dataset import load_sample
from pss.sim.scene import CATEGORY_SPECS, SceneConfig, build_scene

from .action import (
    ActionHead,
    evaluate_head,
    load_learned_action_head,
    train_learned_action_head,
)
from .text import TextHead
from .value import (
    ValueHead,
    evaluate_rule_value,
    evaluate_value_scorer,
    load_learned_value_head,
    train_learned_value_head,
)

SAFETY = {"tool": 0.8, "cup": 0.5, "plate": 0.5, "bottle": 0.5, "toy_fruit": 0.4}


def action_closed_loop(n_trials: int = 10, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    head = ActionHead(gain=15.0, max_force=10.0, min_force=1.0, tolerance=0.03)
    successes = 0
    finals, pushes, starts = [], [], []
    for i in range(n_trials):
        start = np.array([rng.uniform(-0.30, -0.05), rng.uniform(-0.15, 0.15)])
        goal = np.array([0.25, rng.uniform(-0.12, 0.12)])
        cfg = SceneConfig(n_objects=1, seed=seed + 100 + i, width=64, height=64, bins=0)
        scene = build_scene(cfg, categories=["block"], instance_indices=[i % 3])
        obj = scene.objects[0]
        scene.set_object_pose(0, [start[0], start[1], obj.rest_height + 0.002])
        scene.step(10)
        # Lower friction so the proxy task is about control, not friction tuning.
        scene.model.geom_friction[scene.model.geom(obj.geom_name).id] = [0.2, 0.01, 0.001]

        used = 0
        for _ in range(20):
            pos, _ = scene.get_object_pose(0)
            if np.linalg.norm(pos[:2] - goal) < 0.06:
                break
            proposal = head.propose_push_to(pos[:2], goal, target=0)
            if proposal is None:
                break
            direction = np.asarray(proposal.params["direction"])
            force = float(proposal.params["force"])
            body_id = scene.model.body(obj.body_name).id
            scene.data.xfrc_applied[body_id, :3] = [direction[0] * force, direction[1] * force, 0.0]
            scene.step(8)
            scene.data.xfrc_applied[body_id, :3] = 0.0
            scene.step(12)
            used += 1

        pos, _ = scene.get_object_pose(0)
        final = float(np.linalg.norm(pos[:2] - goal))
        finals.append(final)
        pushes.append(used)
        starts.append({"start": start.tolist(), "goal": goal.tolist(), "final": final})
        successes += int(final < 0.06)

    return {
        "trials": n_trials,
        "success_rate": successes / max(1, n_trials),
        "mean_final_distance": float(np.mean(finals)) if finals else 0.0,
        "mean_pushes": float(np.mean(pushes)) if pushes else 0.0,
        "final_distances": finals,
        "per_trial": starts,
        "note": "Rule-based action head; physical push in MuJoCo with reduced friction.",
    }


def _episode_context(data_dir: Path):
    index = load_index(data_dir)
    entry = next(e for e in index["entries"] if e["kind"] == "episode")
    sample = load_sample(data_dir / entry["path"])
    frame = 0
    entities = []
    for i, instance_id in enumerate(sample["instance_ids"]):
        entities.append(
            {
                "id": str(instance_id),
                "category": str(sample["categories"][i]),
                "pos": np.asarray(sample["poses"][frame, i, :3], dtype=float),
                "quat": np.asarray(sample["poses"][frame, i, 3:7], dtype=float),
            }
        )
    sizes = {c: spec["size"] for c, spec in CATEGORY_SPECS.items()}
    relations = [r.as_tuple() for r in relations_from_frame(sample, frame, sizes)]
    return entry["id"], entities, relations


def text_demo(data_dir: str | Path) -> dict:
    sample_id, entities, relations = _episode_context(Path(data_dir))
    head = TextHead()
    text = head.render(entities, relations)
    return {
        "sample": sample_id,
        "entities": [
            {"id": e["id"], "category": e["category"], "pos": e["pos"].tolist()}
            for e in entities
        ],
        "relations": [list(r) for r in relations],
        "text": text,
        "parameters": 0,
    }


def value_demo(data_dir: str | Path) -> dict:
    sample_id, entities, _relations = _episode_context(Path(data_dir))
    goal_xy = np.array([0.20, 0.0])
    head = ValueHead(goal_xy=goal_xy)
    tracks = [
        Track(
            id=e["id"],
            pos=e["pos"],
            quat=e["quat"],
            category=e["category"],
        )
        for e in entities
    ]
    scores = head.score_tracks(tracks, SAFETY)
    ranking = sorted(
        (
            {"id": t.id, "category": t.category, "score": scores[t.id]}
            for t in tracks
        ),
        key=lambda x: -x["score"],
    )
    focus_ids, _ = AttentionScheduler(capacity=2).select(tracks, task_scores=scores)
    return {
        "sample": sample_id,
        "goal_xy": goal_xy.tolist(),
        "ranking": ranking,
        "focus_ids": focus_ids,
    }


def text_ablation(
    data_dir: str | Path,
    expert_dir: str | Path,
    projection_dir: str | Path,
    prototype_dir: str | Path,
) -> dict:
    from pss.prototypes import PrototypeBank, recognize
    from pss.prototypes.build import extract_projected_features

    feats, Z = extract_projected_features(data_dir, expert_dir, projection_dir, device="cpu")
    bank = PrototypeBank.load(Path(prototype_dir) / "prototypes.npz")
    test_idx = [
        i
        for i, split in enumerate(feats.splits)
        if split == "test" and feats.categories[i] in bank.categories
    ]
    weights = {"vision": 1.0, "tactile": 0.5, "geometry": 0.95}

    def accuracy() -> float:
        correct = 0
        for i in test_idx:
            query = {m: Z[m][i] for m in Z}
            result = recognize(query, bank, weights=weights, use_tree=False)
            correct += int(result.ranked[0][0] == feats.categories[i])
        return correct / max(1, len(test_idx))

    before = accuracy()
    proto_path = Path(prototype_dir) / "prototypes.npz"
    hash_before = hashlib.sha256(proto_path.read_bytes()).hexdigest()

    head = TextHead()
    for i in test_idx[:20]:
        head.render(
            [{"id": feats.sample_ids[i], "category": feats.categories[i], "pos": [0.0, 0.0, 0.0]}],
            relations=[],
        )

    after = accuracy()
    hash_after = hashlib.sha256(proto_path.read_bytes()).hexdigest()
    return {
        "test_examples": len(test_idx),
        "acc_before": before,
        "acc_after": after,
        "delta": after - before,
        "prototype_hash_unchanged": hash_before == hash_after,
        "note": "TextHead has no parameters and no gradient path; this checks accidental mutation.",
    }


def _write_report(path: Path, metrics: dict) -> None:
    action = metrics["action"]
    action_learned = metrics.get("action_learned", {})
    action_training = metrics.get("action_head_training", {})
    text = metrics["text"]
    value = metrics["value"]
    value_learned = metrics.get("value_learned", {})
    ablation = metrics["ablation"]
    lines = [
        "# Phase 8 report: multi-readout heads",
        "",
        "## Action head (closed loop)",
        "",
        f"- trials: {action['trials']}",
        f"- success rate: **{action['success_rate']:.3f}**",
        f"- mean final distance: {action['mean_final_distance']:.3f} m",
        f"- mean pushes: {action['mean_pushes']:.1f}",
        "",
        "| trial | start | goal | final distance |",
        "|---|---|---|---:|",
    ]
    for i, t in enumerate(action["per_trial"]):
        lines.append(
            f"| {i} | ({t['start'][0]:.2f}, {t['start'][1]:.2f}) | "
            f"({t['goal'][0]:.2f}, {t['goal'][1]:.2f}) | {t['final']:.3f} |"
        )
    lines += [
        "",
        "The rule head uses direction = goal - position and force proportional to distance.",
        "It is a readout-path test, not a learned policy.",
        "",
        "## Learned action head (behavior cloning)",
        "",
        f"- demos: {action_training.get('n_demos', 0)}, samples: {action_training.get('n_samples', 0)}",
        f"- epochs: {action_training.get('epochs', 0)}, final loss: {action_training.get('final_loss', 0.0):.5f}",
        "",
        "| head | success rate | mean final distance | mean pushes |",
        "|---|---:|---:|---:|",
        f"| rule | {action['success_rate']:.3f} | {action['mean_final_distance']:.3f} | {action['mean_pushes']:.1f} |",
        f"| learned BC | {action_learned.get('success_rate', 0.0):.3f} | "
        f"{action_learned.get('mean_final_distance', 0.0):.3f} | "
        f"{action_learned.get('mean_pushes', 0.0):.1f} |",
        "",
        "The learned head keeps the same `propose_push_to` interface and is evaluated on held-out starts/goals.",
        "",
        "## Text head",
        "",
        f"- parameters: {text['parameters']} (pure read-only projection)",
        "",
        "```",
        text["text"],
        "```",
        "",
        "## Value head",
        "",
        f"- goal: {value['goal_xy']}",
        f"- focus slots (capacity 2): {value.get('focus_ids', [])}",
        "",
        "| entity | category | value |",
        "|---|---|---:|",
    ]
    for r in value["ranking"]:
        lines.append(f"| {r['id']} | {r['category']} | {r['score']:.3f} |")
    lines += [
        "",
        "## Learned value head",
        "",
        f"- preference pairs: {value_learned.get('training', {}).get('n_pairs', 0)}",
        f"- final loss: {value_learned.get('training', {}).get('final_loss', 0.0):.5f}",
        "",
        "| scorer | pairwise accuracy | top-1 accuracy (5-way) |",
        "|---|---:|---:|",
        f"| rule | {value_learned.get('rule', {}).get('pairwise_accuracy', 0.0):.3f} | "
        f"{value_learned.get('rule', {}).get('top1_accuracy', 0.0):.3f} |",
        f"| learned | {value_learned.get('learned', {}).get('pairwise_accuracy', 0.0):.3f} | "
        f"{value_learned.get('learned', {}).get('top1_accuracy', 0.0):.3f} |",
        "",
        "Learned head is trained on synthetic nonlinear preferences, not human data.",
        "",
        "## Text-head ablation",
        "",
        "| item | value |",
        "|---|---:|",
        f"| test examples | {ablation['test_examples']} |",
        f"| accuracy before | {ablation['acc_before']:.4f} |",
        f"| accuracy after | {ablation['acc_after']:.4f} |",
        f"| delta | {ablation['delta']:+.4f} |",
        f"| prototype hash unchanged | {ablation['prototype_hash_unchanged']} |",
        "",
        "The text head is called between the two accuracy measurements. The delta is zero",
        "by construction because the head has no parameters and no gradient path; the test",
        "guards against accidental mutation of the core representation.",
        "",
        "## Caveats",
        "",
        "- Rule and learned action heads are both v1 proxies; the learned head is behavior cloning from the rule controller.",
        "- Text head is a template adapter; no LLM is installed. An external frozen LLM can be plugged in.",
        "- Value head has rule and learned versions; the learned one uses synthetic preferences, not human data.",
        "- Ablation is structural, not a training experiment; it verifies the architecture's separation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase8(
    data_dir: str | Path,
    expert_dir: str | Path,
    projection_dir: str | Path,
    prototype_dir: str | Path,
    out_dir: str | Path,
    n_trials: int = 10,
    seed: int = 0,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_seed = seed + 1000
    action_rule = action_closed_loop(n_trials, eval_seed)
    action_training = train_learned_action_head(
        out / "action_head", n_demos=60, epochs=200, seed=seed
    )
    learned_head = load_learned_action_head(
        out / "action_head" / "action_head.pt", max_force=10.0
    )
    action_learned = evaluate_head(learned_head, n_trials=n_trials, seed=eval_seed)
    value_training = train_learned_value_head(
        out / "value_head", n_pairs=4000, epochs=100, seed=seed
    )
    learned_value = load_learned_value_head(
        out / "value_head" / "value_head.pt", goal_xy=np.array([0.20, 0.0])
    )
    value_rule_eval = evaluate_rule_value(n_pairs=1000, seed=eval_seed)
    value_learned_eval = evaluate_value_scorer(
        learned_value.score_features, n_pairs=1000, seed=eval_seed
    )
    metrics = {
        "data_dir": str(data_dir),
        "action": action_rule,
        "action_learned": action_learned,
        "action_head_training": action_training,
        "value_learned": {
            "training": value_training,
            "rule": value_rule_eval,
            "learned": value_learned_eval,
        },
        "text": text_demo(data_dir),
        "value": value_demo(data_dir),
        "ablation": text_ablation(data_dir, expert_dir, projection_dir, prototype_dir),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "REPORT.md", metrics)
    return metrics
