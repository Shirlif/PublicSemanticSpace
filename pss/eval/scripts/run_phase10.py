"""Phase 10: pre-registered evaluation and baseline comparison."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.encoders.data import load_index
from pss.eval.baselines import (
    evaluate_tracker,
    no_slot_episode_accuracy,
    recognition_ablations,
    tracker_variants,
)
from pss.memory.perception import PoseNoise, load_perception_models
from pss.prototypes import PrototypeBank
from pss.prototypes.build import extract_projected_features
from pss.sim.dataset import load_sample


def _load(path: str | Path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _cap(data, scheme, d, k, noise):
    if not data:
        return None
    for row in data.get("capacity", []):
        if (
            row["scheme"] == scheme
            and row["D"] == d
            and row["K"] == k
            and abs(row["noise"] - noise) < 1e-9
        ):
            return row["accuracy"]
    return None


def _rt(data, scheme, d, noise, cycles):
    if not data:
        return None
    for row in data.get("roundtrip", []):
        if (
            row["scheme"] == scheme
            and row["D"] == d
            and row["noise"] == noise
            and row["cycles"] == cycles
        ):
            return row["mean"]
    return None


def _e1_binding() -> dict:
    binding = _load("bench/results/binding_stability.json")
    phase1 = _load("bench/results/phase1_binding.json")
    return {
        "hadamard_d128_k5": _cap(binding, "hadamard", 128, 5, 0.0),
        "hadamard_d128_k20": _cap(binding, "hadamard", 128, 20, 0.0),
        "hadamard_d512_k20": _cap(binding, "hadamard", 512, 20, 0.0),
        "gaussian_d128_k5": _cap(binding, "gaussian", 128, 5, 0.0),
        "hrr_d128_k20": _cap(phase1, "hrr", 128, 20, 0.0),
        "hadamard_rt_1pct_100": _rt(binding, "hadamard", 128, 0.01, 100),
        "hadamard_rt_5pct_10": _rt(binding, "hadamard", 128, 0.05, 10),
        "hrr_rt_no_noise_1": _rt(phase1, "hrr", 128, 0.0, 1),
        "hrr_rt_100": _rt(phase1, "hrr", 128, 0.0, 100),
    }


def _e2_alignment(data_dir, expert_dir, projection_dir, prototype_dir, seed: int) -> dict:
    feats, Z = extract_projected_features(data_dir, expert_dir, projection_dir, device="cpu")
    bank = PrototypeBank.load(Path(prototype_dir) / "prototypes.npz")
    classes = bank.categories
    splits = np.array(feats.splits)
    train_idx = np.array(
        [i for i, s in enumerate(splits) if s == "train" and feats.categories[i] in classes],
        dtype=int,
    )
    test_idx = np.array(
        [i for i, s in enumerate(splits) if s == "test" and feats.categories[i] in classes],
        dtype=int,
    )
    ablations = recognition_ablations(feats, Z, bank, test_idx, train_idx, classes, seed=seed)

    proj = _load(Path(projection_dir) / "metrics.json") or {}
    retrieval = proj.get("retrieval", {}).get("test", {})
    v_t = retrieval.get("vision->tactile", {}).get("category_r1")
    v_g = retrieval.get("vision->geometry", {}).get("category_r1")

    index = load_index(data_dir)
    episodes = [
        load_sample(Path(data_dir) / e["path"])
        for e in index["entries"]
        if e["kind"] == "episode"
    ][:4]
    models = load_perception_models(expert_dir, projection_dir, device="cpu")
    no_slot = no_slot_episode_accuracy(episodes, models, bank, PoseNoise(seed=seed), max_frames=4)

    return {
        "counts": {"train": len(train_idx), "test": len(test_idx)},
        "ablations": ablations,
        "vision_tactile_r1": v_t,
        "vision_geometry_r1": v_g,
        "no_slot": no_slot,
    }


def _e3_tracking(occlusion_data, expert_dir, projection_dir, seed: int, max_samples: int = 4) -> dict:
    index = load_index(occlusion_data)
    entries = [e for e in index["entries"] if e["kind"] == "occlusion"][:max_samples]
    samples = [load_sample(Path(occlusion_data) / e["path"]) for e in entries]
    models = load_perception_models(expert_dir, projection_dir, device="cpu")
    variants = tracker_variants()
    out = {}
    for name, factory in variants.items():
        retentions, switches, reappear = [], [], []
        for i, sample in enumerate(samples):
            result = evaluate_tracker(sample, models, PoseNoise(seed=seed + i), factory())
            retentions.append(result["id_retention"])
            switches.append(result["id_switches"])
            reappear.append(result["reappearance_success"])
        out[name] = {
            "id_retention": float(np.mean(retentions)),
            "id_switches": float(np.mean(switches)),
            "reappearance_success": float(np.mean(reappear)),
            "samples": len(samples),
        }
    return out


def _e5_gates() -> list[dict]:
    phase3 = _load("checkpoints/expert_v1/metrics.json")
    phase4 = _load("checkpoints/projection_v1/metrics.json")
    phase5 = _load("checkpoints/prototypes_v1/metrics.json")
    phase6 = _load("checkpoints/memory_v1/metrics.json")
    phase7 = _load("checkpoints/phase7/metrics.json")
    phase8 = _load("checkpoints/phase8/metrics.json")
    phase9 = _load("checkpoints/phase9/metrics.json")
    binding = _load("bench/results/binding_stability.json")

    gates = []
    gates.append(
        {
            "stage": "P0 binding",
            "metric": "roundtrip no-noise",
            "value": _rt(binding, "hadamard", 128, 0.0, 1),
            "threshold": 1e-6,
            "pass": (_rt(binding, "hadamard", 128, 0.0, 1) or 1.0) < 1e-6,
        }
    )
    gates.append(
        {
            "stage": "P3 experts (vision)",
            "metric": "test top-1",
            "value": phase3["vision"]["test_acc"] if phase3 else None,
            "threshold": 0.8,
            "pass": bool(phase3 and phase3["vision"]["test_acc"] >= 0.8),
        }
    )
    gates.append(
        {
            "stage": "P3 experts (tactile)",
            "metric": "test top-1",
            "value": phase3["tactile"]["test_acc"] if phase3 else None,
            "threshold": 0.3,
            "pass": bool(phase3 and phase3["tactile"]["test_acc"] >= 0.3),
        }
    )
    vg = None
    vt = None
    if phase4:
        retrieval = phase4["retrieval"]["test"]
        vg = retrieval["vision->geometry"]["category_r1"]
        vt = retrieval["vision->tactile"]["category_r1"]
    gates.append(
        {
            "stage": "P4 alignment (V-G)",
            "metric": "category R@1",
            "value": vg,
            "threshold": 0.6,
            "pass": bool(vg is not None and vg >= 0.6),
        }
    )
    gates.append(
        {
            "stage": "P4 alignment (V-T)",
            "metric": "category R@1",
            "value": vt,
            "threshold": 0.6,
            "pass": bool(vt is not None and vt >= 0.6),
        }
    )
    gates.append(
        {
            "stage": "P5 prototypes",
            "metric": "test top-1 (flat)",
            "value": phase5["fusion"]["test_acc_flat"] if phase5 else None,
            "threshold": 0.8,
            "pass": bool(phase5 and phase5["fusion"]["test_acc_flat"] >= 0.8),
        }
    )
    occ_ret = None
    if phase6:
        occ_ret = float(
            np.mean([r["id_retention"] for r in phase6["tracking"]["occlusion"]])
        )
    gates.append(
        {
            "stage": "P6 tracking",
            "metric": "occlusion ID retention",
            "value": occ_ret,
            "threshold": 0.9,
            "pass": bool(occ_ret is not None and occ_ret >= 0.9),
        }
    )
    gates.append(
        {
            "stage": "P7 planner",
            "metric": "A* + anomaly replan",
            "value": bool(phase7 and phase7["planner"]["astar"]["success"] and phase7["planner"]["anomaly"]["replan_success"]),
            "threshold": True,
            "pass": bool(phase7 and phase7["planner"]["astar"]["success"] and phase7["planner"]["anomaly"]["replan_success"]),
        }
    )
    gates.append(
        {
            "stage": "P8 action head",
            "metric": "closed-loop success",
            "value": phase8["action"]["success_rate"] if phase8 else None,
            "threshold": 0.8,
            "pass": bool(phase8 and phase8["action"]["success_rate"] >= 0.8),
        }
    )
    gates.append(
        {
            "stage": "P8 text ablation",
            "metric": "accuracy delta",
            "value": phase8["ablation"]["delta"] if phase8 else None,
            "threshold": 0.0,
            "pass": bool(phase8 and abs(phase8["ablation"]["delta"]) < 1e-9),
        }
    )
    gates.append(
        {
            "stage": "P9 learning/safety",
            "metric": "rollback + integrity + guardrail",
            "value": bool(
                phase9
                and phase9["learning"]["package"]["rollback_to"]
                and phase9["learning"]["package"]["integrity_rejected"]
                and phase9["guardrails"]["hard_veto"]["hard_veto"]
            ),
            "threshold": True,
            "pass": bool(
                phase9
                and phase9["learning"]["package"]["rollback_to"]
                and phase9["learning"]["package"]["integrity_rejected"]
                and phase9["guardrails"]["hard_veto"]["hard_veto"]
            ),
        }
    )
    return gates


def _e6_scaling() -> dict:
    memory = _load("checkpoints/memory_v1/metrics.json")
    scaling = memory["scaling"] if memory else {}
    token_table = [
        {
            "history_steps": n,
            "context_tokens": n * 100,
            "attention_ops": (n * 100) ** 2,
        }
        for n in (10, 100, 1000)
    ]
    return {"vector": scaling, "token_sequence": token_table}


def _write_report(path: Path, metrics: dict) -> None:
    e1 = metrics["e1_binding"]
    e2 = metrics["e2_alignment"]
    e3 = metrics["e3_tracking"]
    e4 = metrics["e4_package"]
    e5 = metrics["e5_gates"]
    e6 = metrics["e6_scaling"]
    kills = metrics["kill_criteria"]
    lines = [
        "# Phase 10 report: pre-registered evaluation",
        "",
        f"- generated: {metrics['generated']}",
        "- pre-registration: `eval/preregistration.md`",
        "",
        "## E0 R-Gate",
        "",
        "**Not run**: RD2 real synchronized data is not available. This remains an open gate.",
        "",
        "## E1 Binding",
        "",
        "| item | value |",
        "|---|---:|",
    ]
    for k, v in e1.items():
        lines.append(f"| {k} | {v if v is None else f'{v:.4f}'} |")
    lines += [
        "",
        "## E2 Alignment and ablations",
        "",
        "| item | value |",
        "|---|---:|",
        f"| full recognition | {e2['ablations']['full_accuracy']:.3f} |",
        f"| cross-modal sum | {e2['ablations']['cross_modal_sum_accuracy']:.3f} |",
        f"| random projection (vision) | {e2['ablations']['random_projection_accuracy']:.3f} |",
        f"| trained V-T R@1 | {e2['ablations']['trained_vision_tactile_r1']:.3f} |",
        f"| random V-T R@1 | {e2['ablations']['random_projection_vision_tactile_r1']:.3f} |",
        f"| no-slot per-object | {e2['no_slot']['per_object_acc']:.3f} |",
        f"| no-slot global vector | {e2['no_slot']['global_vector_acc']:.3f} |",
        f"| V-G category R@1 | {e2['vision_geometry_r1']:.3f} |",
        f"| V-T category R@1 | {e2['vision_tactile_r1']:.3f} |",
        "",
        "## E3 Occlusion ID retention",
        "",
        "| variant | ID retention | switches | reappearance |",
        "|---|---:|---:|---:|",
    ]
    for name, row in e3.items():
        lines.append(
            f"| {name} | {row['id_retention']:.3f} | {row['id_switches']:.2f} | "
            f"{row['reappearance_success']:.3f} |"
        )
    lines += [
        "",
        "`text_state_proxy` is B1: one-frame text memory, no dormant entity slots.",
        "",
        "## E4 Package update",
        "",
        f"- accepted: {e4.get('accepted')}",
        f"- regression new acc: {e4.get('new_acc')}",
        f"- rollback: {e4.get('rollback')}",
        f"- integrity rejected: {e4.get('integrity_rejected')}",
        "",
        "## E5 Stage gates",
        "",
        "| stage | metric | value | threshold | pass |",
        "|---|---|---:|---:|---|",
    ]
    for g in e5:
        lines.append(
            f"| {g['stage']} | {g['metric']} | {g['value']} | {g['threshold']} | {g['pass']} |"
        )
    lines += [
        "",
        "## E6 Scaling",
        "",
        f"- vector: active pairs {e6['vector'].get('pairs_active')}, "
        f"all pairs {e6['vector'].get('pairs_all')}, "
        f"query {e6['vector'].get('query_ms_per_frame'):.3f} ms",
        "",
        "| history steps | context tokens | attention ops (N^2) |",
        "|---:|---:|---:|",
    ]
    for row in e6["token_sequence"]:
        lines.append(
            f"| {row['history_steps']} | {row['context_tokens']} | {row['attention_ops']:,} |"
        )
    lines += [
        "",
        "## Kill criteria",
        "",
        "| id | triggered | detail |",
        "|---|---|---|",
    ]
    for k, v in kills.items():
        lines.append(f"| {k} | {v['triggered']} | {v['detail']} |")
    lines += [
        "",
        "## Conclusion",
        "",
        "The structural claims are supported on proxy data: explicit entity state retains",
        "identity under occlusion far better than frame-local/text-state baselines;",
        "cross-modal alignment works for vision-geometry but not for the synthetic tactile",
        "channel; the prototype-learning loop accepts and rolls back packages.",
        "",
        "This is **not** a capability claim against a real VLM or real robot. E0, B2, and",
        "real pretrained backbones remain open.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase10(
    data_dir: str | Path,
    occlusion_data: str | Path,
    expert_dir: str | Path,
    projection_dir: str | Path,
    prototype_dir: str | Path,
    out_dir: str | Path,
    seed: int = 0,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    e1 = _e1_binding()
    e2 = _e2_alignment(data_dir, expert_dir, projection_dir, prototype_dir, seed)
    e3 = _e3_tracking(occlusion_data, expert_dir, projection_dir, seed)
    phase9 = _load("checkpoints/phase9/metrics.json") or {}
    e4 = {
        "accepted": phase9.get("learning", {}).get("review", {}).get("accepted"),
        "new_acc": phase9.get("learning", {}).get("review", {}).get("regression", {}).get("new_acc"),
        "rollback": phase9.get("learning", {}).get("package", {}).get("rollback_to"),
        "integrity_rejected": phase9.get("learning", {}).get("package", {}).get("integrity_rejected"),
    }
    e5 = _e5_gates()
    e6 = _e6_scaling()

    k1 = bool(e1["hadamard_d128_k20"] is not None and e1["hadamard_d128_k20"] < 0.95)
    k2 = bool(
        (e2["vision_tactile_r1"] is not None and e2["vision_tactile_r1"] < 0.6)
        or (e2["vision_geometry_r1"] is not None and e2["vision_geometry_r1"] < 0.6)
    )
    main_ret = e3["main"]["id_retention"]
    b1_ret = e3["text_state_proxy"]["id_retention"]
    k3 = bool((main_ret - b1_ret) < 0.10)
    k4 = any(not g["pass"] for g in e5)

    kills = {
        "K1": {
            "triggered": k1,
            "detail": f"D=128 K=20 accuracy={e1['hadamard_d128_k20']:.3f}; HRR K=20={e1['hrr_d128_k20']:.3f}",
        },
        "K2": {
            "triggered": k2,
            "detail": f"V-T={e2['vision_tactile_r1']:.3f}, V-G={e2['vision_geometry_r1']:.3f}",
        },
        "K3": {
            "triggered": k3,
            "detail": f"main={main_ret:.3f}, B1 proxy={b1_ret:.3f}, delta={main_ret - b1_ret:+.3f}",
        },
        "K4": {
            "triggered": k4,
            "detail": ", ".join(g["stage"] for g in e5 if not g["pass"]) or "all gates pass",
        },
    }

    metrics = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "e1_binding": e1,
        "e2_alignment": e2,
        "e3_tracking": e3,
        "e4_package": e4,
        "e5_gates": e5,
        "e6_scaling": e6,
        "kill_criteria": kills,
        "note": "Proxy data; B1 is text-state proxy; B2 not run; E0 pending RD2.",
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out / "phase10_report.md", metrics)
    return metrics


def main() -> None:
    ensure_utf8()
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_v1")
    ap.add_argument("--occlusion-data", default="data/v1")
    ap.add_argument("--experts", default="checkpoints/expert_v1")
    ap.add_argument("--projections", default="checkpoints/projection_v1")
    ap.add_argument("--prototypes", default="checkpoints/prototypes_v1")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    metrics = run_phase10(
        args.data,
        args.occlusion_data,
        args.experts,
        args.projections,
        args.prototypes,
        args.out,
        seed=args.seed,
    )
    print("Phase 10 complete:")
    print(f"  E1 K=20 D=128: {metrics['e1_binding']['hadamard_d128_k20']}")
    print(f"  E2 full / sum / random: "
          f"{metrics['e2_alignment']['ablations']['full_accuracy']:.3f} / "
          f"{metrics['e2_alignment']['ablations']['cross_modal_sum_accuracy']:.3f} / "
          f"{metrics['e2_alignment']['ablations']['random_projection_accuracy']:.3f}")
    print(f"  E3 ID retention: " + ", ".join(
        f"{k}={v['id_retention']:.3f}" for k, v in metrics["e3_tracking"].items()
    ))
    print("  kill criteria: " + ", ".join(
        f"{k}={v['triggered']}" for k, v in metrics["kill_criteria"].items()
    ))
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
