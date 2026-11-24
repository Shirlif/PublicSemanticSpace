"""P5b perception version: rendered scene -> tracker -> relational planner.

This is a bridge between the symbolic relational benchmark and a full physical
closed loop. It renders three objects at five tabletop regions, runs the PSS
perception/tracker stack, maps tracked positions to regions, and compares the
explicit relational planner with the behavior-cloned transformer on the same
perceived state. No pushes are executed yet.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.relational_bench import (
    TEST_PAIRS,
    TRAIN_PAIRS,
    _rows,
    plan,
    rollout,
    train_policy,
)
from pss.memory import SceneGraph, Tracker, TrackerConfig
from pss.memory.perception import PoseNoise, extract_frame, load_perception_models
from pss.sim.generator import capture_scene_sample
from pss.sim.scene import SceneConfig, build_scene
from pss.sim.sensors import SensorRig

REGION_COORDS = np.array(
    [
        [-0.24, 0.00],
        [0.24, 0.00],
        [0.00, 0.18],
        [0.00, -0.18],
        [0.00, 0.00],
    ],
    dtype=float,
)
REGION_NAMES = ("A", "B", "C", "D", "G")


def _nearest_region(xy):
    d = np.linalg.norm(REGION_COORDS - np.asarray(xy, dtype=float)[:2], axis=1)
    return int(np.argmin(d))


def _perceive(models, seed, noise_sigma):
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
    scene = build_scene(
        cfg, categories=["cup", "block", "bottle"], instance_indices=[0, 0, 0]
    )
    rng = np.random.default_rng(seed)
    state = tuple(rng.choice(len(REGION_COORDS), size=3, replace=False))
    for i, region in enumerate(state):
        x, y = REGION_COORDS[region]
        scene.set_object_pose(i, [x, y, scene.objects[i].rest_height + 0.002])
    scene.forward()
    rig = SensorRig(scene, 128, 128)
    rig.set_camera(azimuth=0.0, elevation=-48.0, distance=1.05)
    sample = capture_scene_sample(scene, rig, seed=seed)
    obs = extract_frame(
        sample,
        0,
        models,
        PoseNoise(seed=seed, enabled=True, sigma_pos=noise_sigma),
    )
    tracker = Tracker(TrackerConfig(gate_radius=0.8, w_category=0.0), SceneGraph())
    events = tracker.step(obs, 0)
    gt_to_track = {}
    for key in ("matched", "reid", "new"):
        for tid, gid in events.get(key, []):
            gt_to_track[str(gid)] = tid
    perceived = []
    for obj in scene.objects:
        tid = gt_to_track.get(obj.instance_id)
        if tid is None:
            return state, None
        track = tracker.graph.get(tid)
        if track is None:
            return state, None
        perceived.append(_nearest_region(track.pos))
    if len(set(perceived)) != 3:
        return state, None
    return state, tuple(perceived)


def _evaluate(model, models, rel_pairs, n, seed, noise_sigma, device):
    rng = np.random.default_rng(seed)
    pss_ok, tr_ok, invalid, perception_fail = [], [], [], []
    for rel1, rel2 in rel_pairs:
        for _ in range(n):
            true_state, perceived = _perceive(
                models, int(rng.integers(1 << 30)), noise_sigma
            )
            if perceived is None:
                perception_fail.append(1.0)
                pss_ok.append(0.0)
                tr_ok.append(0.0)
                invalid.append(1.0)
                continue
            perception_fail.append(0.0)
            solution = plan(perceived, rel1, rel2)
            pss_ok.append(float(solution is not None))
            good, _steps, inv = rollout(model, perceived, rel1, rel2, device)
            tr_ok.append(float(good))
            invalid.append(float(inv > 0))
    return {
        "pss_success": float(np.mean(pss_ok)),
        "transformer_success": float(np.mean(tr_ok)),
        "transformer_invalid": float(np.mean(invalid)),
        "perception_failure": float(np.mean(perception_fail)),
        "n": len(pss_ok),
    }


def run(
    out_dir="eval/results/pss_vs_transformer",
    seeds=5,
    n_train_per_pair=80,
    n_test_per_pair=20,
    epochs=300,
    noise_sigma=0.01,
    device="cpu",
    expert_dir="checkpoints/category_pretrain_experts",
    projection_dir="checkpoints/phase45_projection",
    vision_expert=None,
    tactile_expert=None,
):
    rng = np.random.default_rng(0)
    train_rows = _rows(TRAIN_PAIRS, n_train_per_pair, rng)
    models = load_perception_models(
        expert_dir,
        projection_dir,
        device=device,
        vision_expert=vision_expert,
        tactile_expert=tactile_expert,
    )
    runs = []
    for seed in range(seeds):
        policy = train_policy(train_rows, seed, epochs, device)
        runs.append(
            {
                "seed": seed,
                "id": _evaluate(
                    policy,
                    models,
                    TRAIN_PAIRS,
                    n_test_per_pair,
                    100 + seed,
                    noise_sigma,
                    device,
                ),
                "ood": _evaluate(
                    policy,
                    models,
                    TEST_PAIRS,
                    n_test_per_pair,
                    200 + seed,
                    noise_sigma,
                    device,
                ),
            }
        )

    def agg(sub, key):
        vals = [r[sub][key] for r in runs]
        return float(np.mean(vals)), float(np.std(vals))

    metrics = {
        "train_pairs": TRAIN_PAIRS,
        "test_pairs": TEST_PAIRS,
        "noise_sigma": noise_sigma,
        "expert_dir": str(expert_dir),
        "projection_dir": str(projection_dir),
        "runs": runs,
        "pss": {
            "id_success": agg("id", "pss_success"),
            "ood_success": agg("ood", "pss_success"),
        },
        "transformer": {
            "id_success": agg("id", "transformer_success"),
            "ood_success": agg("ood", "transformer_success"),
            "ood_invalid": agg("ood", "transformer_invalid"),
            "ood_perception_failure": agg("ood", "perception_failure"),
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p5b_perception.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(out / "p5b_perception.md", metrics)
    return metrics


def _report(path, metrics):
    p, t = metrics["pss"], metrics["transformer"]
    lines = [
        "# P5b perception version: rendered scene -> tracker -> relational planner",
        "",
        f"- pose noise sigma: {metrics['noise_sigma']}",
        "",
        "| method | ID success | OOD success | OOD invalid | perception failure |",
        "|---|---:|---:|---:|---:|",
        f"| PSS explicit planner | {p['id_success'][0]:.3f} | "
        f"**{p['ood_success'][0]:.3f}** | 0.000 | "
        f"{t['ood_perception_failure'][0]:.3f} |",
        f"| Transformer BC | {t['id_success'][0]:.3f} ± {t['id_success'][1]:.3f} | "
        f"{t['ood_success'][0]:.3f} ± {t['ood_success'][1]:.3f} | "
        f"{t['ood_invalid'][0]:.3f} | {t['ood_perception_failure'][0]:.3f} |",
        "",
        "Three objects are rendered at five tabletop regions; the tracker supplies perceived positions.",
        "Planning is evaluated on the same perceived region state; no pushes are executed yet.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/pss_vs_transformer")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.01)
    args = ap.parse_args()
    m = run(args.out, args.seeds, epochs=args.epochs, noise_sigma=args.noise)
    print(
        f"PSS ID/OOD {m['pss']['id_success'][0]:.3f}/{m['pss']['ood_success'][0]:.3f}; "
        f"TR {m['transformer']['id_success'][0]:.3f}/{m['transformer']['ood_success'][0]:.3f}"
    )
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
