"""Aggregate PSS-in-loop S4 tasks across seeds."""

from __future__ import annotations

import json
from pathlib import Path

TASKS = {
    "basic": "S4_pss_seed*.json",
    "relation": "S4_pss_relation_seed*.json",
    "memory": "S4_pss_memory_seed*.json",
    "two_object": "S4_pss_two_seed*.json",
    "memory_no_memory": "S4_pss_memory_nomem_seed*.json",
}


def main() -> int:
    out = Path("scorecards")
    results = {}
    for name, pattern in TASKS.items():
        rows = []
        for path in sorted(out.glob(pattern)):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "seed": data.get("seed"),
                    "success": bool(data.get("task_success")),
                    "actions": [f"{a['skill']}:{a['status']}" for a in data.get("actions", [])],
                    "file": path.name,
                }
            )
        results[name] = {
            "runs": len(rows),
            "successes": sum(1 for r in rows if r["success"]),
            "success_rate": (sum(1 for r in rows if r["success"]) / len(rows)) if rows else 0.0,
            "episodes": rows,
        }
    payload = {"suite": "S4-PSS", "tasks": results}
    (out / "S4_pss_suite.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = ["# S4-PSS 干活能力（5 seeds）", "", "| task | runs | successes | success rate |", "|---|---|---|---|"]
    for name, stats in results.items():
        lines.append(
            f"| {name} | {stats['runs']} | {stats['successes']} | {stats['success_rate']:.2f} |"
        )
    lines.append("")
    lines.append(
        "Tasks: basic pick-place; relation (cup left of bowl); memory (2 s occlusion + "
        "relocation); two_object (two cups into the bin on their side, identity kept "
        "across two cycles); memory_no_memory (same memory task with the PSS memory "
        "layer ablated)."
    )
    (out / "S4_pss_suite.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v["success_rate"] for k, v in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
