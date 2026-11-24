"""CLI: run Phase 6 tracking and micro-benchmarks.

Example:
    python -m pss.memory.scripts.run_tracking \
        --data data/tracking_v2 --experts checkpoints/category_pretrain_experts \
        --projections checkpoints/phase45_projection \
        --prototypes checkpoints/phase45_prototypes \
        --split test --out checkpoints/memory_v2_proto
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.memory.build import run_phase6


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--experts", default="checkpoints/expert_v1")
    ap.add_argument("--projections", default="checkpoints/projection_v1")
    ap.add_argument(
        "--prototypes",
        default=None,
        help="Phase 5 prototype package directory; omit to use GT categories",
    )
    ap.add_argument("--out", default="checkpoints/memory_v1")
    ap.add_argument("--occlusion", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default=None,
        help="select tracking sequences by split; omit for the first N entries",
    )
    ap.add_argument(
        "--recognition-weights",
        choices=("reliability", "tuned"),
        default="reliability",
        help="fusion weights from per-modality test accuracy or Phase 5 tuning",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    metrics = run_phase6(
        args.data,
        args.experts,
        args.projections,
        args.out,
        prototype_dir=args.prototypes,
        max_occlusion=args.occlusion,
        max_episodes=args.episodes,
        split=args.split,
        weights_mode=args.recognition_weights,
        seed=args.seed,
    )
    agg = metrics["tracking"]["aggregate"]
    wm = metrics["working_memory"]
    sc = metrics["scaling"]
    print("Tracking complete:")
    print(f"  ID retention: {agg['id_retention']:.3f}")
    print(f"  ID switches: {agg['id_switches']:.2f}")
    print(f"  reappearance success: {agg['reappearance_success']:.3f}")
    print(f"  wake latency: {agg['wake_latency_frames']:.1f} frames")
    print(f"  ID inflation: {agg['id_inflation']:.2f}")
    rec = metrics["recognition"]
    print(f"Recognition: top-1={rec['accuracy']:.3f} "
          f"known-only={rec['known_accuracy']:.3f} "
          f"unknown={rec['unknown_rate']:.3f} conflict={rec['conflict_rate']:.3f}")
    print(f"Working memory: focus recall={wm['focus_switch_recall']:.3f} "
          f"p50={wm['latency_ms_p50']:.3f} ms")
    print(f"Scaling: pairs active={sc['pairs_active']} all={sc['pairs_all']} "
          f"query={sc['query_ms_per_frame']:.3f} ms")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
