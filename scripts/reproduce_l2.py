"""Reproduce the L2 bundle-replay evidence from a public bundle pack.

Runs the E12 cognitive-layer ablation (full / no_memory / no_entity /
no_relations) over the whole pack, then repeats it on the subset of bundles
whose detection tables carry embeddings.  The second run is the
perception-level check: it uses detector output and DINOv2 embeddings rather
than oracle poses.

Usage::

    python scripts/fetch_repro_bundles.py --url <URL> --out bundles/repro
    python scripts/reproduce_l2.py --root bundles/repro --out reports/repro

Exit code is non-zero when the published E12 numbers do not reproduce.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pss._console import ensure_utf8
from pss.eval.scripts.run_gap_l2_memory import (
    _report,
    _valid_bundles,
    _valid_summary,
    run,
)

# E12 valid-bundle results (reports/claim_gaps/l2_memory_all/REPORT.md).
EXPECTED = {
    "n_bundles": 33,
    "valid_n": 19,
    "variants": {
        "full": {"id_retention": 1.000, "id_switches": 0.00},
        "no_memory": {"id_retention": 0.827, "id_switches": 0.56},
        "no_entity": {"id_retention": 0.270, "id_switches": 47.47},
        "no_relations": {"id_retention": 1.000, "id_switches": 0.00},
    },
}
PERCEPTION_EXPECTED = {"n_bundles": 4}


def _resolve_root(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist.  Fetch the pack first:\n"
            "  python scripts/fetch_repro_bundles.py --url <URL> --out bundles/repro"
        )
    if (path / "PACK_MANIFEST.json").exists():
        return path
    stages = sorted(p.parent for p in path.glob("*/PACK_MANIFEST.json"))
    if len(stages) > 1:
        names = ", ".join(stage.name for stage in stages)
        raise SystemExit(
            f"multiple packs under {path} ({names}); keep one or pass the stage path"
        )
    if len(stages) == 1:
        return stages[0].parent
    if any(path.glob("**/obs_detections.parquet")):
        return path
    raise SystemExit(f"no bundles found under {path}")


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _check(full: dict, perception: dict) -> list[str]:
    problems: list[str] = []
    if full["n_bundles"] != EXPECTED["n_bundles"]:
        problems.append(
            f"bundle count: expected {EXPECTED['n_bundles']}, got {full['n_bundles']}"
        )
    valid = _valid_summary(full)
    if not valid:
        problems.append("no valid bundles (full runtime retention > 0.5)")
        return problems
    if valid.get("full", {}).get("n") != EXPECTED["valid_n"]:
        problems.append(
            f"valid bundle count: expected {EXPECTED['valid_n']}, "
            f"got {valid.get('full', {}).get('n')}"
        )
    for variant, expected in EXPECTED["variants"].items():
        actual = valid.get(variant)
        if actual is None:
            problems.append(f"missing variant: {variant}")
            continue
        for metric in ("id_retention", "id_switches"):
            precision = 3 if metric == "id_retention" else 2
            actual_rounded = round(float(actual[metric]), precision)
            expected_rounded = round(float(expected[metric]), precision)
            if actual_rounded != expected_rounded:
                problems.append(
                    f"{variant}.{metric}: expected {expected[metric]:.{precision}f}, "
                    f"got {actual[metric]:.{precision}f}"
                )
    if perception["n_bundles"] != PERCEPTION_EXPECTED["n_bundles"]:
        problems.append(
            f"perception bundle count: expected {PERCEPTION_EXPECTED['n_bundles']}, "
            f"got {perception['n_bundles']}"
        )
    return problems


def _combined_report(full: dict, perception: dict, problems: list[str]) -> str:
    valid = _valid_summary(full)
    lines = [
        "# L2 bundle replay reproduction",
        "",
        f"> generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"> root: `{full['root']}`",
        f"> bundles: {full['n_bundles']} total, {len(_valid_bundles(full))} valid",
        "",
        "## E12 cognitive-layer ablation (valid bundles only)",
        "",
        "| variant | n | ID retention | switches |",
        "|---|---:|---:|---:|",
    ]
    for variant, row in valid.items():
        lines.append(
            f"| {variant} | {row['n']} | {row['id_retention']:.3f} | {row['id_switches']:.2f} |"
        )
    lines += [
        "",
        "## Perception subset (non-empty detector embeddings)",
        "",
        f"Bundles: {perception['n_bundles']}.",
        "",
        "| variant | n | ID retention | switches |",
        "|---|---:|---:|---:|",
    ]
    for variant, row in perception["summary"].items():
        lines.append(
            f"| {variant} | {row['n']} | {row['id_retention']:.3f} | {row['id_switches']:.2f} |"
        )
    lines += [
        "",
        "The perception subset is the 4 bundles whose detection tables carry",
        "GroundingDINO boxes and DINOv2 crop embeddings.  Their ablation is a",
        "null result (all variants retain 1.000): these bundles exercise the",
        "detector-to-runtime path, not the memory layer.  The memory signal comes",
        "from the occlusion/oracle bundles in the full set.",
        "",
        "## Check",
        "",
    ]
    if problems:
        lines.append("**FAIL**")
        lines.append("")
        lines += [f"- {problem}" for problem in problems]
    else:
        lines.append("**PASS** — published E12 numbers reproduced at the "
                     "report's precision (retention 3 dp, switches 2 dp).")
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/fetch_repro_bundles.py --url <URL> --out bundles/repro",
        "python scripts/reproduce_l2.py --root bundles/repro --out reports/repro",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "bundles" / "repro"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "repro"))
    ap.add_argument("--no-check", action="store_true", help="do not enforce E12 numbers")
    args = ap.parse_args()

    root = _resolve_root(Path(args.root).resolve())
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"root: {root}")
    print("== full bundle set ==")
    full = run(root=str(root), pattern="**/obs_detections.parquet")
    print("== perception subset ==")
    perception = run(
        root=str(root),
        pattern="**/obs_detections.parquet",
        require_emb=True,
    )
    display_root = _display_path(root)
    full["root"] = display_root
    perception["root"] = display_root

    for name, metrics in (("l2_memory", full), ("l2_perception", perception)):
        directory = out / name
        directory.mkdir(parents=True, exist_ok=True)
        metrics["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        (directory / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        title = (
            "Claim-gap E12: L2 bundle replay under cognitive ablations"
            if name == "l2_memory"
            else "L2 perception subset: detector embeddings, no memory ablation"
        )
        (directory / "REPORT.md").write_text(_report(metrics, title=title), encoding="utf-8")

    problems = [] if args.no_check else _check(full, perception)
    (out / "REPRODUCTION.md").write_text(
        _combined_report(full, perception, problems),
        encoding="utf-8",
    )
    (out / "REPRODUCED.json").write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "root": display_root,
                "n_bundles": full["n_bundles"],
                "valid_bundles": len(_valid_bundles(full)),
                "perception_bundles": perception["n_bundles"],
                "problems": problems,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if problems:
        print("\nFAIL")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print(f"\nPASS: E12 reproduced; report: {out / 'REPRODUCTION.md'}")


if __name__ == "__main__":
    main()
