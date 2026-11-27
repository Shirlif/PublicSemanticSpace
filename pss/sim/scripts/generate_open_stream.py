"""Generate an open-category stream (tier A/B/C).

Examples
--------
python -m pss.sim.scripts.generate_open_stream --tier a \
  --out data/open_stream/tier_a --episodes 20 --per-episode 200
python -m pss.sim.scripts.generate_open_stream --tier b \
  --out data/open_stream/tier_b --label-mode instance \
  --base-index data/category_pretrain_instance8/index.json \
  --novel-per-category 2 --episodes 20 --per-episode 200
python -m pss.sim.scripts.generate_open_stream --tier c \
  --out data/open_stream/tier_c --drift-mode both --episodes 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pss._console import ensure_utf8
from pss.sim.generator import build_open_stream_from_index, generate_open_stream
from pss.sim.scene import CATEGORY_SPECS, HELD_OUT_CATEGORIES


def _instance_labels(
    base_index: str,
    novel_per_category: int,
    novel_variation_start: int,
) -> tuple[list[str], list[str]]:
    index = json.loads(Path(base_index).read_text(encoding="utf-8"))
    train = [e for e in index["entries"] if e.get("split") == "train"]
    seen = sorted({str(e["instance"]) for e in train if e.get("instance")})
    categories = sorted({str(e["category"]) for e in train if e.get("category")})
    novel = [
        f"{category}_v{variation:03d}"
        for category in categories
        for variation in range(
            int(novel_variation_start),
            int(novel_variation_start) + int(novel_per_category),
        )
    ]
    return seen, novel


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier", choices=("a", "b", "c"), default="a")
    ap.add_argument("--label-mode", choices=("category", "instance"), default="category")
    ap.add_argument("--seen-categories", nargs="+", default=None)
    ap.add_argument("--novel-categories", nargs="+", default=None)
    ap.add_argument("--base-index", default="data/category_pretrain_instance8/index.json")
    ap.add_argument(
        "--source-index",
        default="",
        help="build by re-sampling this existing index instead of rendering MuJoCo scenes",
    )
    ap.add_argument("--novel-per-category", type=int, default=2)
    ap.add_argument("--novel-variation-start", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--per-episode", type=int, default=200)
    ap.add_argument("--novel-rate", type=float, default=0.2)
    ap.add_argument("--novel-start", type=int, default=5)
    ap.add_argument("--drift-episode", type=int, default=10)
    ap.add_argument("--drift-mode", choices=("lighting", "tactile", "both"), default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    args = ap.parse_args()

    seen_labels = args.seen_categories
    novel_labels = args.novel_categories
    if args.label_mode == "instance":
        seen_labels, novel_labels, _ = _instance_labels(
            args.base_index,
            args.novel_per_category,
            args.novel_variation_start,
        )
    elif seen_labels is None:
        seen_labels = [c for c in CATEGORY_SPECS if c not in HELD_OUT_CATEGORIES]
    if args.label_mode == "category" and novel_labels is None:
        novel_labels = list(HELD_OUT_CATEGORIES)

    if args.source_index:
        source = args.source_index
        if args.label_mode == "instance":
            seen_labels, novel_labels = _instance_labels(
                source, args.novel_per_category, args.novel_variation_start
            )[:2]
        elif seen_labels is None:
            import json

            index = json.loads(Path(source).read_text(encoding="utf-8"))
            entries = [e for e in index["entries"] if e.get("kind") == "category"]
            seen_labels = sorted(
                {str(e["category"]) for e in entries if e.get("split") != "test_category"}
            )
            novel_labels = sorted(
                {str(e["category"]) for e in entries if e.get("split") == "test_category"}
            )
        summary = build_open_stream_from_index(
            args.out,
            source,
            seen_labels=seen_labels,
            novel_labels=novel_labels,
            tier=args.tier,
            label_mode=args.label_mode,
            episodes=args.episodes,
            per_episode=args.per_episode,
            novel_rate=args.novel_rate,
            novel_start=args.novel_start,
            drift_episode=args.drift_episode,
            drift_mode=args.drift_mode,
            seed=args.seed,
        )
    else:
        summary = generate_open_stream(
            args.out,
            tier=args.tier,
            seen_labels=seen_labels,
            novel_labels=novel_labels,
            label_mode=args.label_mode,
            episodes=args.episodes,
            per_episode=args.per_episode,
            novel_rate=args.novel_rate,
            novel_start=args.novel_start,
            drift_episode=args.drift_episode,
            drift_mode=args.drift_mode,
            seed=args.seed,
            width=args.width,
            height=args.height,
        )
    print("Open stream generated:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
