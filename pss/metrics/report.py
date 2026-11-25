"""A9/A10: bundle index, metric recomputation and report assembly."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .scorecard import score_suite, write_scorecard


def bundles_index(bundles_root: str | Path = "bundles") -> pd.DataFrame:
    root = Path(bundles_root)
    rows = []
    for manifest_path in sorted(root.glob("**/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle = manifest_path.parent
        relative = bundle.relative_to(root).parts
        device = relative[0] if relative else ""
        suite = relative[1] if len(relative) > 1 else ""
        rows.append(
            {
                "episode_id": manifest.get("episode_id"),
                "bundle": str(bundle),
                "suite": suite,
                "device_class": manifest.get("device_class"),
                "spec_id": manifest.get("spec", {}).get("id"),
                "spec_hash": manifest.get("spec", {}).get("hash"),
                "seed": manifest.get("seed"),
                "detector_tier": manifest.get("detector_tier"),
                "task_success": manifest.get("outcome", {}).get("task_success"),
                "steps": manifest.get("outcome", {}).get("steps"),
                "wall_s": manifest.get("outcome", {}).get("wall_s"),
            }
        )
    return pd.DataFrame(rows)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def verify_recompute(
    tag: str,
    *,
    scorecards_dir: str | Path = "scorecards",
    bundles_root: str | Path = "bundles",
    out_dir: str | Path = "verify",
) -> dict[str, Any]:
    """Recompute entity scorecards from bundles and compare to stored JSON."""
    stored_dir = Path(scorecards_dir)
    suites: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(stored_dir.glob("*.json")):
        if path.name.startswith(("S7", "S8")) or path.name.endswith(
            ("_relations.json", "_ablation.json", "_vs_baselines.json")
        ):
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        suite = report.get("suite")
        device = report.get("device_class")
        if not suite or not device:
            continue
        suites[(str(suite), str(device))] = {"path": str(path), "stored": report}

    results = []
    ok_all = True
    for (suite, device), item in suites.items():
        try:
            recomputed = score_suite(bundles_root, suite, device=device)
        except Exception as exc:  # noqa: BLE001
            results.append({"suite": suite, "device_class": device, "ok": False, "error": str(exc)})
            ok_all = False
            continue
        stored_metrics = item["stored"].get("metrics", {})
        mismatches = []
        for name, stats in recomputed.get("metrics", {}).items():
            if name not in stored_metrics:
                continue
            if not _close(stats["mean"], stored_metrics[name]["mean"]):
                mismatches.append(
                    {"metric": name, "stored": stored_metrics[name]["mean"], "recomputed": stats["mean"]}
                )
        ok = not mismatches
        ok_all = ok_all and ok
        results.append(
            {
                "suite": suite,
                "device_class": device,
                "ok": ok,
                "episodes": recomputed.get("n_episodes"),
                "mismatches": mismatches,
            }
        )

    payload = {"tag": tag, "ok": ok_all, "suites": results}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def build_report(
    tag: str,
    *,
    out_root: str | Path = "reports",
    bundles_root: str | Path = "bundles",
    scorecards_dir: str | Path = "scorecards",
    prereg: str | Path = "eval/prereg_isaac_v1.md",
    freeze: str | Path = "freeze/p7-w11.json",
    verify: str | Path = "verify",
    notes: str | Path = "notes/w13_status.md",
) -> Path:
    out = Path(out_root) / tag
    out.mkdir(parents=True, exist_ok=True)

    freeze_path = Path(freeze)
    if freeze_path.exists():
        shutil.copyfile(freeze_path, out / "freeze.json")

    specs_out = out / "specs"
    specs_out.mkdir(exist_ok=True)
    for spec in Path("specs").glob("*.yaml"):
        shutil.copyfile(spec, specs_out / spec.name)
    registry = Path("specs/registry.jsonl")
    if registry.exists():
        shutil.copyfile(registry, specs_out / "registry.jsonl")

    if Path(prereg).exists():
        shutil.copyfile(prereg, out / "prereg.md")

    score_out = out / "scorecards"
    score_out.mkdir(exist_ok=True)
    for path in Path(scorecards_dir).glob("*"):
        if path.is_file() and path.suffix in (".json", ".md"):
            shutil.copyfile(path, score_out / path.name)

    index = bundles_index(bundles_root)
    index.to_parquet(out / "bundles.index.parquet", index=False)

    verify_path = Path(verify) / f"{tag}.json"
    if not verify_path.exists():
        verify_recompute(tag, scorecards_dir=scorecards_dir, bundles_root=bundles_root, out_dir=verify)
    if verify_path.exists():
        shutil.copyfile(verify_path, out / "verify.json")

    overrides = out / "overrides.jsonl"
    if not overrides.exists():
        overrides.write_text("", encoding="utf-8")

    notes_path = Path(notes)
    if notes_path.exists():
        shutil.copyfile(notes_path, out / "notes.md")
    else:
        (out / "notes.md").write_text("", encoding="utf-8")
    return out
