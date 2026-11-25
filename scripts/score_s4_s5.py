"""Aggregate S4/S5 scripted-baseline bundles into scorecards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAULTS = [
    "occlusion_insert",
    "object_relocate",
    "object_remove",
    "grasp_slip",
    "camera_dropout",
    "lighting_shock",
    "twin_swap",
    "joint_limit",
    "base_disturb",
    "path_blocked",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", default="bundles/dev/s4_g1_pick_place")
    parser.add_argument("--out", default="scorecards")
    args = parser.parse_args()
    root = Path(args.bundles)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    episodes = []
    for manifest_path in sorted(root.glob("*/*/seed*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fault = manifest_path.parent.parent.name
        episodes.append(
            {
                "fault": fault,
                "seed": manifest.get("seed"),
                "success": bool(manifest.get("outcome", {}).get("task_success")),
                "steps": manifest.get("outcome", {}).get("steps"),
                "wall_s": manifest.get("outcome", {}).get("wall_s"),
                "bundle": str(manifest_path.parent),
            }
        )

    baseline = [e for e in episodes if e["fault"] == "none"]
    s4_success = (
        sum(e["success"] for e in baseline) / len(baseline) if baseline else 0.0
    )
    s4 = {
        "suite": "S4",
        "scripted_baseline": True,
        "n_episodes": len(baseline),
        "execution_success": s4_success,
        "episodes": baseline,
        "note": "scripted skill sequence + suction proxy; PSS not in the loop",
    }
    (out / "S4.json").write_text(json.dumps(s4, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "S4.md").write_text(
        "# S4 execution baseline (dev)\n\n"
        f"- episodes: {len(baseline)}\n"
        f"- execution_success: {s4_success:.3f}\n"
        "- **scripted baseline** (home/reach/grasp/place + suction proxy); PSS is not in the loop.\n"
        "- bin is a solid pad; `in` is approximated by footprint containment.\n",
        encoding="utf-8",
    )

    per_fault = {}
    for fault in FAULTS:
        rows = [e for e in episodes if e["fault"] == fault]
        if not rows:
            continue
        per_fault[fault] = {
            "episodes": len(rows),
            "recovery_rate": sum(e["success"] for e in rows) / len(rows),
            "steps_mean": sum(e["steps"] or 0 for e in rows) / len(rows),
        }
    s5 = {
        "suite": "S5",
        "scripted_baseline": True,
        "recovery_rate_mean": (
            sum(v["recovery_rate"] for v in per_fault.values()) / len(per_fault)
            if per_fault
            else 0.0
        ),
        "per_fault": per_fault,
        "note": "scripted policy with suction proxy; not PSS recovery",
    }
    (out / "S5.json").write_text(json.dumps(s5, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# S5 fault recovery (dev, scripted baseline)",
        "",
        f"- faults: {len(per_fault)}",
        f"- mean recovery: {s5['recovery_rate_mean']:.3f}",
        "- **not PSS recovery**: scripted skill sequence with suction proxy.",
        "",
        "| fault | episodes | recovery | steps |",
        "|---|---|---|---|",
    ]
    for fault, stats in per_fault.items():
        lines.append(
            f"| {fault} | {stats['episodes']} | {stats['recovery_rate']:.3f} | {stats['steps_mean']:.0f} |"
        )
    (out / "S5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"S4 success={s4_success:.3f} faults={len(per_fault)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
