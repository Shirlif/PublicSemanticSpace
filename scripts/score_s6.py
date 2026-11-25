"""S6: generate detections, replay the session, write the memory scorecard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pss.detect.oracle import detect_oracle, write_detections
from pss.detect.session_replay import replay_session
from pss.metrics.memory import compute_memory_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="bundles/dev/s6_memory")
    parser.add_argument("--out", default="scorecards")
    args = parser.parse_args()
    root = Path(args.session)
    seeds = sorted(root.glob("*/seed*"))
    if not seeds:
        raise FileNotFoundError(f"no S6 session under {root}")
    seed_dir = seeds[0]
    bundles = sorted(p for p in seed_dir.glob("ep*") if p.is_dir())
    if not bundles:
        raise FileNotFoundError(f"no episode bundles under {seed_dir}")
    for bundle in bundles:
        write_detections(bundle, detect_oracle(bundle))

    trace, report = replay_session(bundles)
    metrics = compute_memory_metrics(
        trace, rollback_exactness=report.get("rollback_exactness", 0.0)
    )
    metrics.update(
        {
            "session": str(seed_dir),
            "episodes": report.get("episodes"),
            "tracks": report.get("tracks"),
            "rollback_changed_state": report.get("rollback_changed_state"),
        }
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "S6.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# S6 long-horizon memory (dev)",
        "",
        f"- session: `{seed_dir}`",
        f"- episodes: {metrics['episodes']}  tracks: {metrics['tracks']}",
        f"- edited_accuracy: {metrics['edited_accuracy']:.4f}",
        f"- old_forgetting: {metrics['old_forgetting']:.4f}  (forgetting_rate {metrics['forgetting_rate']:.4f})",
        f"- id_retention: {metrics['id_retention']:.4f}",
        f"- **rollback_exactness: {metrics['rollback_exactness']:.4f}**",
        f"- rollback changed state before restore: {metrics['rollback_changed_state']}",
        "",
        "S6 uses oracle detections; PSS state persists across sub-episodes and objects are",
        "physically relocated between them.",
    ]
    (out / "S6.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
