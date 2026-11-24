"""TAO official frames: IoU baseline vs PSS tracker (appearance + pose).

Requires frames extracted by ``prepare_tao_frames``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.suite.adapters.tao import TAOAdapter
from pss.eval.suite.predictors import predict_tao, predict_tao_pss
from pss.memory import TrackerConfig

OUT = "eval/results/tao_pss"


def _fmt(v):
    return "n/a" if v is None else f"{float(v):.3f}"


def run(device="cuda", videos=5) -> dict:
    adapter = TAOAdapter(root="data/public/tao")
    adapter.prepare()
    episodes = []
    for ep in adapter.episodes("test", max_videos=100, dataset="Charades"):
        frames = ep.meta.get("frames") or []
        if any((Path(ep.meta["frames_dir"]) / im["file_name"]).exists() for im in frames):
            episodes.append(ep)
        if len(episodes) >= videos:
            break
    if not episodes:
        return {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "videos": 0,
            "note": "no extracted TAO Charades frames; run prepare_tao_frames first",
        }
    configs = {
        "pss_g90": TrackerConfig(gate_radius=90, w_pose=0.4, w_app=0.6, w_category=0.0, max_cost=1.0),
        "pss_g180": TrackerConfig(gate_radius=180, w_pose=0.3, w_app=0.7, w_category=0.0, max_cost=1.1),
        "pss_g300": TrackerConfig(gate_radius=300, w_pose=0.2, w_app=0.8, w_category=0.0, max_cost=1.2),
        "pss_pose_only": TrackerConfig(gate_radius=300, w_pose=1.0, w_app=0.0, w_category=0.0, max_cost=0.9),
        "pss_app_only": TrackerConfig(gate_radius=400, w_pose=0.0, w_app=1.0, w_category=0.0, max_cost=0.9),
        "pss_g300_reid": TrackerConfig(
            gate_radius=300, w_pose=0.2, w_app=0.8, w_category=0.0, max_cost=1.2,
            reid_pose_gate=300.0, reid_app_min=0.2, reid_max_age=1000,
            dormancy_missed=1, missing_missed=3, historical_missed=10,
        ),
    }
    variants = {"iou": [predict_tao(ep) for ep in episodes]}
    for name, cfg in configs.items():
        variants[name] = [predict_tao_pss(ep, device=device, config=cfg) for ep in episodes]
    results = {}
    for name, preds in variants.items():
        metrics = adapter.evaluate(preds)
        results[name] = {m.metric: m.value for m in metrics}
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "videos": len(episodes),
        "video_names": [ep.meta.get("video") for ep in episodes],
        "results": results,
    }
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tao_pss.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# TAO official frames: IoU vs PSS tracker",
        "",
        f"> generated: {payload['generated']}",
        f"> videos: {payload['videos']}",
        "",
        "| method | IDF1 | ID retention | ID switches |",
        "|---|---:|---:|---:|",
    ]
    for name, row in results.items():
        lines.append(
            f"| {name} | {_fmt(row.get('idf1'))} | {_fmt(row.get('id_retention'))} | "
            f"{_fmt(row.get('id_switches'))} |"
        )
    lines += [
        "",
        "GT boxes are used as detections; IoU and PSS tracker differ only in the",
        "association rule. This is an entity-layer benchmark, not a detector.",
        "",
    ]
    (out / "tao_pss.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--videos", type=int, default=5)
    args = ap.parse_args()
    payload = run(args.device, args.videos)
    print(json.dumps(payload.get("results", {}), indent=1))
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
