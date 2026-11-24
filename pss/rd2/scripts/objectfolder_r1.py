"""CLI: ObjectFolder 2.0 object-level cross-modal R@1.

Example:
    python -m pss.rd2.scripts.objectfolder_r1 \
        --root _external/ObjectFolder/ObjectFolder1-100 \
        --renderer _external/ObjectFolder \
        --out eval/results/objectfolder \
        --limit 100 --touches 4
"""

from __future__ import annotations

import argparse

from pss._console import ensure_utf8
from pss.rd2.objectfolder import run_objectfolder


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="_external/ObjectFolder/ObjectFolder1-100")
    parser.add_argument("--renderer", default="_external/ObjectFolder")
    parser.add_argument("--out", default="eval/results/objectfolder")
    parser.add_argument("--vision-query", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--touches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    results = run_objectfolder(
        args.root,
        args.renderer,
        args.out,
        vision_query=args.vision_query,
        limit=args.limit,
        touches=args.touches,
        seed=args.seed,
        render=not args.no_render,
        render_only=args.render_only,
    )
    if results.get("render_only"):
        rendered = sum(1 for report in results["render_reports"].values() if report.get("ok"))
        print(f"Render-only complete: {rendered}/{results['objects_requested']} objects ok")
        return
    print("ObjectFolder object-level R@1:")
    for name in ("zero_shot", "ridge_half_split"):
        print(f"  [{name}]")
        for direction, metric in results[name].items():
            print(
                f"    {direction}: R@1={metric['r1']:.3f} R@5={metric['r5']:.3f} "
                f"chance={metric['chance']:.3f}"
            )
    print(f"  report: {args.out}/OBJECTFOLDER_R1.md")


if __name__ == "__main__":
    main()
