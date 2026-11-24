"""P5b execution closed loop: relation goal -> push execution -> re-perceive.

Three objects are placed at tabletop regions. PSS plans a relational goal and
executes pushes; the transformer policy sees the same perceived region state
and outputs actions. After each push the scene is re-rendered and re-tracked.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from pss._console import ensure_utf8
from pss.eval.collision_planner import (
    plan_collision_aware,
    rows_collision_aware,
)
from pss.eval.relational_bench import (
    ACTIONS,
    RELATIONS,
    TEST_PAIRS,
    TRAIN_PAIRS,
    goal_ok,
    train_policy,
)
from pss.eval.relational_perception_bench import REGION_COORDS, _nearest_region
from pss.heads.action import ActionHead
from pss.memory import SceneGraph, Tracker, TrackerConfig
from pss.memory.perception import PoseNoise, extract_frame, load_perception_models
from pss.sim.generator import capture_scene_sample
from pss.sim.scene import SceneConfig, build_scene
from pss.sim.sensors import SensorRig

CATEGORIES = ["block", "brick", "box"]


def _make_world(seed):
    cfg = SceneConfig(
        n_objects=3,
        seed=seed,
        width=128,
        height=128,
        bins=0,
        min_separation=0.05,
        workspace_x=(-0.30, 0.30),
        workspace_y=(-0.25, 0.25),
    )
    scene = build_scene(cfg, categories=CATEGORIES, instance_indices=[0, 0, 0])
    rng = np.random.default_rng(seed)
    state = tuple(rng.choice([0, 1, 2, 3], size=3, replace=False))
    for i, region in enumerate(state):
        x, y = REGION_COORDS[region]
        scene.set_object_pose(i, [x, y, scene.objects[i].rest_height + 0.002])
    scene.forward()
    rig = SensorRig(scene, 128, 128)
    rig.set_camera(azimuth=0.0, elevation=-48.0, distance=1.05)
    tracker = Tracker(TrackerConfig(gate_radius=0.8, w_category=0.0), SceneGraph())
    return scene, rig, tracker, state


def _gt_state(scene):
    regions = [
        _nearest_region(scene.get_object_pose(i)[0][:2]) for i in range(len(scene.objects))
    ]
    if len(set(regions)) != len(regions):
        return None
    return tuple(regions)


def _perceive(scene, rig, tracker, models, t, seed, noise_sigma=0.01):
    sample = capture_scene_sample(scene, rig, seed=seed + t)
    obs = extract_frame(
        sample,
        0,
        models,
        PoseNoise(seed=seed + t, enabled=True, sigma_pos=noise_sigma),
    )
    events = tracker.step(obs, t)
    gt_to_track = {}
    for key in ("matched", "reid", "new"):
        for tid, gid in events.get(key, []):
            gt_to_track[str(gid)] = tid
    perceived = []
    for obj in scene.objects:
        tid = gt_to_track.get(obj.instance_id)
        track = tracker.graph.get(tid) if tid is not None else None
        if track is None:
            return None
        perceived.append(_nearest_region(track.pos))
    if len(set(perceived)) != 3:
        return None
    return tuple(perceived)


def _place_at(scene, obj_index, region):
    x, y = REGION_COORDS[region]
    z = scene.objects[obj_index].rest_height + 0.002
    scene.set_object_pose(obj_index, [x, y, z])
    scene.forward()
    return 1


def _push_to(scene, obj_index, region, head, max_pushes=40):
    goal = REGION_COORDS[region]
    pushes = 0
    for _ in range(max_pushes):
        pos = scene.get_object_pose(obj_index)[0]
        if float(np.linalg.norm(pos[:2] - goal)) < 0.03:
            break
        proposal = head.propose_push_to(pos[:2], goal, target=obj_index)
        if proposal is None:
            break
        direction = np.asarray(proposal.params["direction"], dtype=float)
        force = float(proposal.params["force"])
        body_id = scene.model.body(scene.objects[obj_index].body_name).id
        scene.data.xfrc_applied[body_id, :3] = [direction[0] * force, direction[1] * force, 0.0]
        scene.step(6)
        scene.data.xfrc_applied[body_id, :3] = 0.0
        scene.step(10)
        pushes += 1
    return pushes


def _policy_action(policy, perceived, rel1, rel2, device):
    state = torch.as_tensor([perceived], dtype=torch.long, device=device)
    r1 = torch.as_tensor([RELATIONS.index(rel1)], dtype=torch.long, device=device)
    r2 = torch.as_tensor([RELATIONS.index(rel2)], dtype=torch.long, device=device)
    with torch.no_grad():
        action = int(policy(state, r1, r2).argmax(dim=1).cpu())
    return ACTIONS[action]


def _episode(policy, models, rel1, rel2, seed, mode, device, max_steps=10, physical=False):
    scene, rig, tracker, _true_state = _make_world(seed)
    head = ActionHead(gain=20.0, max_force=6.0, min_force=2.0, tolerance=0.03)
    pushes, replans, invalid = 0, 0, 0
    perceived = None
    for t in range(max_steps):
        gt = _gt_state(scene)
        if gt is not None and goal_ok(gt, rel1, rel2):
            return True, pushes, replans, invalid
        perceived = _perceive(scene, rig, tracker, models, t, seed)
        if perceived is None:
            break
        if mode == "pss":
            solution = plan_collision_aware(perceived, rel1, rel2, clearance=0.12)
            replans += 1
            if not solution:
                break
            obj, region = solution[0]
        else:
            obj, region = _policy_action(policy, perceived, rel1, rel2, device)
            if obj == -1:
                break
        if perceived[obj] == region or region in perceived:
            invalid += 1
            break
        if physical:
            pushes += _push_to(scene, obj, region, head)
        else:
            pushes += _place_at(scene, obj, region)
    gt = _gt_state(scene)
    success = bool(gt is not None and goal_ok(gt, rel1, rel2))
    return success, pushes, replans, invalid


def _evaluate(policy, models, pairs, episodes, seed_base, mode, device, physical=False):
    rng = np.random.default_rng(seed_base)
    ok, pushes, replans, invalid = [], [], [], []
    for rel1, rel2 in pairs:
        for _ in range(episodes):
            seed = int(rng.integers(1 << 30))
            good, p, r, inv = _episode(
                policy, models, rel1, rel2, seed, mode, device, physical=physical
            )
            ok.append(float(good))
            pushes.append(p)
            replans.append(r)
            invalid.append(inv)
    return {
        "success": float(np.mean(ok)),
        "pushes": float(np.mean(pushes)),
        "replans": float(np.mean(replans)),
        "invalid": float(np.mean(invalid)),
        "n": len(ok),
    }


def run(
    out_dir="eval/results/pss_vs_transformer",
    seeds=3,
    episodes=1,
    epochs=300,
    device="cpu",
    physical=False,
):
    rng = np.random.default_rng(0)
    train_rows = rows_collision_aware(TRAIN_PAIRS, 80, rng, clearance=0.12)
    models = load_perception_models(
        "checkpoints/category_pretrain_experts",
        "checkpoints/phase45_projection",
        device=device,
    )
    runs = []
    for seed in range(seeds):
        policy = train_policy(train_rows, seed, epochs, device)
        policies = [policy]  # one seed; modes share the same scene seeds
        runs.append(
            {
                "seed": seed,
                "pss_id": _evaluate(policy, models, TRAIN_PAIRS, episodes, 100 + seed, "pss", device, physical),
                "pss_ood": _evaluate(policy, models, TEST_PAIRS, episodes, 200 + seed, "pss", device, physical),
                "tr_id": _evaluate(policy, models, TRAIN_PAIRS, episodes, 100 + seed, "tr", device, physical),
                "tr_ood": _evaluate(policy, models, TEST_PAIRS, episodes, 200 + seed, "tr", device, physical),
            }
        )

    def agg(key):
        vals = [r[key]["success"] for r in runs]
        return float(np.mean(vals)), float(np.std(vals))

    metrics = {
        "train_pairs": TRAIN_PAIRS,
        "test_pairs": TEST_PAIRS,
        "episodes_per_pair": episodes,
        "physical": bool(physical),
        "runs": runs,
        "pss": {"id_success": agg("pss_id"), "ood_success": agg("pss_ood")},
        "transformer": {"id_success": agg("tr_id"), "ood_success": agg("tr_ood")},
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p5b_closed_loop.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p5b_closed_loop.md", metrics)
    return metrics


def _report(path, metrics):
    p, t = metrics["pss"], metrics["transformer"]
    lines = [
        "# P5b execution closed loop",
        "",
        f"- episodes per pair: {metrics['episodes_per_pair']}",
        "",
        "| method | ID success | OOD success |",
        "|---|---:|---:|",
        f"| PSS explicit planner + execution | {p['id_success'][0]:.3f} ± {p['id_success'][1]:.3f} | "
        f"**{p['ood_success'][0]:.3f} ± {p['ood_success'][1]:.3f}** |",
        f"| Transformer BC + execution | {t['id_success'][0]:.3f} ± {t['id_success'][1]:.3f} | "
        f"{t['ood_success'][0]:.3f} ± {t['ood_success'][1]:.3f} |",
        "",
        "Execution mode: physical push." if metrics.get("physical") else "Execution mode: kinematic placement.",
        "After each action the scene is re-rendered, re-tracked, and re-planned.",
        "Success is checked on ground-truth region placement and the two goal relations.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--physical", action="store_true")
    args = ap.parse_args()
    m = run(args.out, args.seeds, args.episodes, args.epochs, physical=args.physical)
    p, t = m["pss"], m["transformer"]
    print(f"PSS ID/OOD {p['id_success'][0]:.3f}/{p['ood_success'][0]:.3f}")
    print(f"TR  ID/OOD {t['id_success'][0]:.3f}/{t['ood_success'][0]:.3f}")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
