"""CLI: evaluate Phase 7 relation records against simulation GT.

Example:
    python -m pss.reasoning.scripts.run_relation_eval \
        --data data/phase7_v2 --episodes 8 --out eval/results/relations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pss._console import ensure_utf8
from pss.reasoning.build import evaluate_relations


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phase7_v2")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", default="eval/results/relations")
    args = ap.parse_args()

    metrics = evaluate_relations(args.data, max_episodes=args.episodes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Phase 7 relation evaluation",
        "",
        f"- data: `{args.data}`",
        f"- episodes: {metrics['episodes']}",
        f"- frames: {metrics['frames']}",
        "",
        "| predicate | precision | recall | F1 | tp | fp | fn |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in [("overall", metrics["overall"]), *metrics["per_predicate"].items()]:
        lines.append(
            f"| {name} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['tp']} | {row['fp']} | {row['fn']} |"
        )
    lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    overall = metrics["overall"]
    print(
        f"relations: F1={overall['f1']:.3f} "
        f"P={overall['precision']:.3f} R={overall['recall']:.3f}"
    )
    for name, row in metrics["per_predicate"].items():
        print(f"  {name:8s} P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
