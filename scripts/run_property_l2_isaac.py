#!/usr/bin/env python
"""Aggregate fresh Isaac S4-PSS episodes (P4).

Each scorecard comes from `scripts/run_s4_pss.py` in the Isaac 3.11 env.
This is a fresh simulator episode with PSS planning + skill execution.  It uses
oracle GT boxes; it does not exercise hot-swap or KB edits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pss.metrics.property import budget_adherence_rate, percentile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecards", type=Path, default=Path("scorecards/property"))
    parser.add_argument("--max-wall-s", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=Path("scorecards/property"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.scorecards.glob("l2_s4_pss_seed*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": int(data["seed"]),
                "success": bool(data.get("task_success")),
                "plan_actions": list(data.get("plan_actions", [])),
                "wall_s": float(data.get("wall_s", 0.0)),
                "steps": int(data.get("steps", 0)),
                "actions_ok": bool(data.get("actions_ok")),
            }
        )
    if not rows:
        raise SystemExit("no l2_s4_pss_seed*.json found")

    wall = [r["wall_s"] for r in rows]
    report = {
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "plan_actions_unique": sorted({tuple(r["plan_actions"]) for r in rows}),
        "mean_wall_s": float(np.mean(wall)),
        "p95_wall_s": percentile(wall, 95),
        "max_wall_s": float(np.max(wall)),
        "steps_mean": float(np.mean([r["steps"] for r in rows])),
        "BAR_wall": budget_adherence_rate([(w, args.max_wall_s) for w in wall]),
        "max_wall_bound_s": float(args.max_wall_s),
        "rows": rows,
        "note": "S4-PSS uses oracle GT boxes; does not exercise hot-swap or KB edits.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "l2_isaac.json"
    md_path = args.out / "l2_isaac.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# L2 fresh Isaac S4-PSS",
        "",
        f"- episodes: {report['episodes']}",
        f"- success rate: {report['success_rate']:.3f}",
        f"- plan actions: {report['plan_actions_unique']}",
        f"- wall mean/p95/max: {report['mean_wall_s']:.3f} / {report['p95_wall_s']:.3f} / {report['max_wall_s']:.3f} s",
        f"- steps mean: {report['steps_mean']:.1f}",
        f"- BAR (wall ≤ {args.max_wall_s}s): {report['BAR_wall']:.3f}",
        "",
        "注意：使用 oracle GT boxes；未测试 hot-swap 与 KB 更新。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"episodes={report['episodes']} success={report['success_rate']:.3f} BAR={report['BAR_wall']:.3f}")


if __name__ == "__main__":
    main()
