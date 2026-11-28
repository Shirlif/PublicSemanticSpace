"""Minimal seed set for modality onboarding (W7).

The compatibility matrix is synthetic; a small **real** seed set is the
hardware-backed regression that catches sensor, calibration and timing drift.
This module owns the on-disk contract, a validator, and a synthetic builder
used by CI.  It deliberately does not pretend synthetic data is real:

    <root>/
      manifest.json
      data_card.md
      objects/<object_id>/<modality>/session-<n>.npz
          features       (N, D) float32
          timestamps_ns  (N,)   int64, strictly increasing
          sample_ids     (N,)   unicode

The physical 30-object x 3-modality x 2-session collection is hardware work;
``build-synthetic`` gives CI the same shape so the validator and downstream
loaders can be tested without pretending it is a capability result.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0"
DEFAULT_MODALITIES = ("vision", "tactile", "geometry")
DEFAULT_OBJECTS = 30
DEFAULT_SESSIONS = 2
DEFAULT_SAMPLES = 6
DEFAULT_DIMS = {"vision": 128, "tactile": 64, "geometry": 16}
REAL_MIN_OBJECTS = 30
REAL_MIN_SESSIONS = 2


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cell_path(root: str | Path, object_id: str, modality: str, session: int) -> Path:
    return Path(root) / "objects" / object_id / modality / f"session-{int(session)}.npz"


def _unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def build_synthetic(
    out: str | Path,
    *,
    n_objects: int = DEFAULT_OBJECTS,
    sessions: int = DEFAULT_SESSIONS,
    samples_per_cell: int = DEFAULT_SAMPLES,
    modalities=DEFAULT_MODALITIES,
    dims: dict[str, int] | None = None,
    seed: int = 0,
    kind: str = "synthetic",
) -> dict:
    """Write a seed set with the production layout and deterministic vectors.

    ``kind`` stays ``synthetic`` by default.  Only a hardware collection may
    set ``kind="real"``; the validator enforces the larger real-set minimum.
    """
    out = Path(out)
    dims = {**DEFAULT_DIMS, **(dims or {})}
    rng = np.random.default_rng(seed)
    object_ids = [f"obj_{i:03d}" for i in range(int(n_objects))]
    cells = 0
    samples = 0
    for object_id in object_ids:
        centers = {
            modality: _unit(rng.normal(size=dims[modality])) for modality in modalities
        }
        for session in range(1, int(sessions) + 1):
            for modality in modalities:
                path = cell_path(out, object_id, modality, session)
                path.parent.mkdir(parents=True, exist_ok=True)
                features = centers[modality] + rng.normal(
                    scale=0.25, size=(int(samples_per_cell), dims[modality])
                )
                if modality in ("vision", "geometry"):
                    features = _unit(features)
                features = features.astype(np.float32)
                base_ns = 1_700_000_000_000_000_000 + session * 10_000_000_000
                timestamps = base_ns + np.arange(int(samples_per_cell), dtype=np.int64)
                sample_ids = np.array(
                    [
                        f"{object_id}-s{session}-{i:03d}"
                        for i in range(int(samples_per_cell))
                    ],
                    dtype=np.str_,
                )
                np.savez_compressed(
                    path,
                    features=features,
                    timestamps_ns=timestamps,
                    sample_ids=sample_ids,
                )
                cells += 1
                samples += int(samples_per_cell)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": str(kind),
        "created_at": _utcnow(),
        "objects": object_ids,
        "sessions": [f"session-{i}" for i in range(1, int(sessions) + 1)],
        "modalities": list(modalities),
        "samples_per_cell": int(samples_per_cell),
        "dims": {m: int(dims[m]) for m in modalities},
        "coverage": {"cells": cells, "samples": samples},
        "sensor": {
            "vision": "unspecified",
            "tactile": "unspecified",
            "geometry": "unspecified",
        },
        "calibration": {"version": "unset", "date": ""},
        "collection": {"operator": "", "location": "", "date": ""},
        "license": "unset",
        "notes": (
            "Synthetic seed set: validates layout only, not a capability claim."
            if kind != "real"
            else "Hardware seed set."
        ),
    }
    _write_manifest(out, manifest)
    write_data_card(out, manifest)
    return manifest


def _write_manifest(root: Path, manifest: dict) -> Path:
    path = Path(root) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_manifest(root: str | Path) -> dict:
    return json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))


def write_data_card(root: str | Path, manifest: dict | None = None) -> Path:
    manifest = manifest or load_manifest(root)
    coverage = manifest.get("coverage", {})
    lines = [
        "# PSS 模态最小种子集 · 数据卡",
        "",
        f"- 类型：**{manifest.get('kind', 'unknown')}**",
        f"- schema：`{manifest.get('schema_version')}`",
        f"- 生成/采集时间：{manifest.get('created_at', '')}",
        f"- 物体数：{len(manifest.get('objects', []))}",
        f"- 会话数：{len(manifest.get('sessions', []))}",
        f"- 模态：{', '.join(manifest.get('modalities', []))}",
        f"- 每个 (物体, 模态, 会话) 样本数：{manifest.get('samples_per_cell')}",
        f"- 覆盖：{coverage.get('cells', 0)} 个文件 / {coverage.get('samples', 0)} 个样本",
        "",
        "## 传感器与标定",
        "",
        "| 项 | 值 |",
        "|---|---|",
    ]
    for modality, sensor in (manifest.get("sensor") or {}).items():
        lines.append(f"| {modality} 传感器 | {sensor} |")
    calibration = manifest.get("calibration") or {}
    lines.append(f"| 标定版本 | {calibration.get('version', 'unset')} |")
    lines.append(f"| 标定日期 | {calibration.get('date', '')} |")
    collection = manifest.get("collection") or {}
    lines.append(f"| 采集人员 | {collection.get('operator', '')} |")
    lines.append(f"| 采集地点 | {collection.get('location', '')} |")
    lines.append(f"| 采集日期 | {collection.get('date', '')} |")
    lines.extend(
        [
            f"| 许可 | {manifest.get('license', 'unset')} |",
            "",
            "## 用途与边界",
            "",
            "- 只用于回归：验证新模态/新专家/新标定没有破坏既有契约。",
            "- 不参与训练，不作为能力声明。",
            "- 真实种子集不得提交到公开仓库；记录许可并单独保管。",
            "",
        ]
    )
    if manifest.get("notes"):
        lines.extend([f"> {manifest['notes']}", ""])
    path = Path(root) / "data_card.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def validate_seedset(
    root: str | Path,
    *,
    min_objects: int | None = None,
    min_sessions: int | None = None,
    min_samples: int = 1,
) -> dict:
    """Validate the on-disk contract and return a machine-readable report."""
    root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    if not (root / "manifest.json").exists():
        return {
            "ok": False,
            "errors": [f"missing manifest: {root / 'manifest.json'}"],
            "warnings": [],
            "coverage": {},
        }
    try:
        manifest = load_manifest(root)
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        return {
            "ok": False,
            "errors": [f"unreadable manifest: {exc}"],
            "warnings": [],
            "coverage": {},
        }

    kind = str(manifest.get("kind", "unknown"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"manifest schema_version={manifest.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION!r}"
        )
    if kind not in ("real", "synthetic"):
        errors.append(f"manifest kind={kind!r}; expected 'real' or 'synthetic'")

    objects = [str(o) for o in manifest.get("objects", [])]
    sessions = list(range(1, len(manifest.get("sessions", [])) + 1))
    modalities = [str(m) for m in manifest.get("modalities", [])]
    if not objects:
        errors.append("manifest lists no objects")
    if not sessions:
        errors.append("manifest lists no sessions")
    if not modalities:
        errors.append("manifest lists no modalities")
    if len(set(modalities)) != len(modalities):
        errors.append("duplicate modality in manifest")

    required_objects = min_objects
    required_sessions = min_sessions
    if kind == "real":
        required_objects = REAL_MIN_OBJECTS if required_objects is None else required_objects
        required_sessions = (
            REAL_MIN_SESSIONS if required_sessions is None else required_sessions
        )
    elif kind == "synthetic":
        required_objects = 1 if required_objects is None else required_objects
        required_sessions = 1 if required_sessions is None else required_sessions

    if required_objects is not None and len(objects) < required_objects:
        errors.append(
            f"{kind} seed set has {len(objects)} objects; need >= {required_objects}"
        )
    if required_sessions is not None and len(sessions) < required_sessions:
        errors.append(
            f"{kind} seed set has {len(sessions)} sessions; need >= {required_sessions}"
        )

    files = 0
    samples = 0
    for object_id in objects:
        for modality in modalities:
            for session in sessions:
                path = cell_path(root, object_id, modality, session)
                if not path.exists():
                    errors.append(f"missing cell: {path.relative_to(root)}")
                    continue
                try:
                    with np.load(path, allow_pickle=False) as data:
                        features = np.asarray(data["features"])
                        timestamps = np.asarray(data["timestamps_ns"])
                        sample_ids = np.asarray(data["sample_ids"])
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"unreadable cell {path.relative_to(root)}: {exc}")
                    continue
                files += 1
                samples += int(features.shape[0]) if features.ndim >= 1 else 0
                if features.ndim != 2:
                    errors.append(
                        f"{path.relative_to(root)}: features must be 2-D, got {features.shape}"
                    )
                elif features.shape[0] < int(min_samples):
                    errors.append(
                        f"{path.relative_to(root)}: {features.shape[0]} samples "
                        f"< {min_samples}"
                    )
                if not np.all(np.isfinite(features)):
                    errors.append(f"{path.relative_to(root)}: non-finite features")
                if features.size and float(np.abs(features).max()) == 0.0:
                    warnings.append(f"{path.relative_to(root)}: all-zero features")
                if timestamps.ndim != 1 or timestamps.shape[0] != features.shape[0]:
                    errors.append(
                        f"{path.relative_to(root)}: timestamps shape {timestamps.shape}"
                    )
                elif timestamps.size and not np.all(np.diff(timestamps) > 0):
                    errors.append(
                        f"{path.relative_to(root)}: timestamps not strictly increasing"
                    )
                if sample_ids.ndim != 1 or sample_ids.shape[0] != features.shape[0]:
                    errors.append(
                        f"{path.relative_to(root)}: sample_ids shape {sample_ids.shape}"
                    )
                elif len(set(sample_ids.tolist())) != len(sample_ids):
                    errors.append(f"{path.relative_to(root)}: duplicate sample_ids")

    if not (root / "data_card.md").exists():
        warnings.append("data_card.md missing; run `card` to regenerate")
    if manifest.get("license", "unset") in ("", "unset"):
        warnings.append("license is unset")

    coverage = {
        "kind": kind,
        "objects": len(objects),
        "modalities": len(modalities),
        "sessions": len(sessions),
        "files": files,
        "samples": samples,
        "expected_files": len(objects) * len(modalities) * len(sessions),
    }
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "coverage": coverage,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-synthetic", help="write a synthetic seed set")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--objects", type=int, default=DEFAULT_OBJECTS)
    build.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    build.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    build.add_argument("--seed", type=int, default=0)

    validate = sub.add_parser("validate", help="validate an existing seed set")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--min-objects", type=int, default=None)
    validate.add_argument("--min-sessions", type=int, default=None)
    validate.add_argument("--min-samples", type=int, default=1)

    card = sub.add_parser("card", help="regenerate data_card.md")
    card.add_argument("--root", type=Path, required=True)

    ns = parser.parse_args(args)
    if ns.command == "build-synthetic":
        manifest = build_synthetic(
            ns.out,
            n_objects=ns.objects,
            sessions=ns.sessions,
            samples_per_cell=ns.samples,
            seed=ns.seed,
        )
        print(json.dumps(manifest["coverage"], ensure_ascii=False))
        return 0
    if ns.command == "card":
        print(write_data_card(ns.root))
        return 0
    report = validate_seedset(
        ns.root,
        min_objects=ns.min_objects,
        min_sessions=ns.min_sessions,
        min_samples=ns.min_samples,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
