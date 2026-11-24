"""Run S1 (T-text spatial perception).

Examples:

    python -m pss.eval.spatial.scripts.run_s1 --out eval/results/spatial_s1
    python -m pss.eval.spatial.scripts.run_s1 --llm-model Qwen/Qwen2.5-1.5B-Instruct

The exact and random baselines always run. The LLM baseline is opt-in and
local-only by default, so a missing model never triggers a download.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pss._console import ensure_utf8
from pss.eval.spatial.baselines import (
    ExactSolver,
    LLMPredictor,
    RandomBaseline,
    ToolLoopPredictor,
    WorldAxisOnlySolver,
    load_transformers_generator,
)
from pss.eval.spatial.generator import ALL_FAMILIES, generate_suite
from pss.eval.spatial.runner import evaluate_predictor, write_jsonl, write_report

DEFAULT_SPLITS = (
    ("test_id", "id"),
    ("test_layout", "layout"),
    ("test_count", "count"),
    ("test_object", "object"),
)


def _parse_splits(values) -> list[tuple[str, str]]:
    out = []
    for value in values:
        if ":" not in value:
            raise SystemExit(f"split must be name:ood, got {value!r}")
        name, ood = value.split(":", 1)
        out.append((name, ood))
    return out


def main(argv=None) -> int:
    ensure_utf8()
    parser = argparse.ArgumentParser(description="S1 T-text spatial suite")
    parser.add_argument("--out", default="eval/results/spatial_s1")
    parser.add_argument("--n-per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--families", nargs="+", default=list(ALL_FAMILIES))
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="name:ood, repeatable. Default: test_id:id test_layout:layout test_count:count test_object:object",
    )
    parser.add_argument("--train-items", type=int, default=18, help="few-shot pool size")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-device", default="auto")
    parser.add_argument("--llm-max-new-tokens", type=int, default=32)
    parser.add_argument("--llm-mode", choices=("zero_shot", "cot"), default="zero_shot")
    parser.add_argument(
        "--llm-tools",
        choices=("none", "low", "mid", "oracle"),
        default="none",
        help="wrap the LLM in a tool loop with this access tier",
    )
    parser.add_argument("--llm-max-steps", type=int, default=6)
    parser.add_argument(
        "--llm-allow-download",
        action="store_true",
        help="allow HF download; default is local_files_only=True",
    )
    parser.add_argument("--llm-limit", type=int, default=0, help="cap LLM items (0 = all)")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--skip-naive", action="store_true")
    args = parser.parse_args(argv)

    splits = _parse_splits(args.split) if args.split else list(DEFAULT_SPLITS)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_items = generate_suite(
        n_per_family=max(2, args.train_items // max(1, len(args.families))),
        seed=args.seed + 10_000,
        split="train",
        ood="id",
        families=args.families,
    )

    items = []
    suite_summary = {"splits": [], "n_per_family": args.n_per_family, "seed": args.seed}
    for name, ood in splits:
        split_items = generate_suite(
            n_per_family=args.n_per_family,
            seed=args.seed,
            split=name,
            ood=ood,
            families=args.families,
        )
        write_jsonl(split_items, out_dir / "suite" / f"{name}.jsonl")
        items.extend(split_items)
        suite_summary["splits"].append({"split": name, "ood": ood, "items": len(split_items)})

    write_jsonl(train_items, out_dir / "suite" / "train.jsonl")

    baselines = {"pss_exact": ExactSolver()}
    if not args.skip_naive:
        baselines["naive_world"] = WorldAxisOnlySolver()
    if not args.skip_random:
        baselines["random"] = RandomBaseline(seed=args.seed)

    if args.llm_model:
        try:
            generate = load_transformers_generator(
                args.llm_model,
                device=args.llm_device,
                max_new_tokens=args.llm_max_new_tokens,
                local_files_only=not args.llm_allow_download,
            )
        except Exception as exc:
            print(f"[llm] skipped: {type(exc).__name__}: {exc}")
            suite_summary["llm_error"] = f"{type(exc).__name__}: {exc}"
        else:
            few_shot = _few_shot(train_items, per_family=1)
            if args.llm_tools != "none":
                name = f"llm_tools:{args.llm_model}:{args.llm_tools}"
                if args.llm_mode != "zero_shot":
                    print("[llm] --llm-mode is ignored when --llm-tools is set")
                baselines[name] = ToolLoopPredictor(
                    generate,
                    profile=args.llm_tools,
                    max_steps=args.llm_max_steps,
                    name=name,
                )
            else:
                baselines[f"llm:{args.llm_model}"] = LLMPredictor(
                    generate,
                    name=f"llm:{args.llm_model}",
                    few_shot=few_shot,
                    mode=args.llm_mode,
                )

    results = {}
    for name, predictor in baselines.items():
        subset = items
        if name.startswith("llm") and args.llm_limit > 0:
            subset = items[: args.llm_limit]
        print(f"[eval] {name} on {len(subset)} items")
        result = evaluate_predictor(predictor, subset, name)
        rows = result.pop("rows")
        results[name] = result
        with (out_dir / f"rows_{name.replace(':', '_').replace('/', '_')}.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[eval] {name}: acc={result['overall']['accuracy']:.3f} "
            f"invalid={result['overall']['invalid_rate']:.3f}"
        )

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "suite": suite_summary,
        "results": results,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = write_report(results, out_dir, suite_summary)
    print(f"[done] {report}")
    return 0


def _few_shot(items, per_family=1):
    picked = []
    seen = {}
    for item in items:
        count = seen.get(item.query.family, 0)
        if count < per_family:
            picked.append(item)
            seen[item.query.family] = count + 1
    return picked


if __name__ == "__main__":
    raise SystemExit(main())
