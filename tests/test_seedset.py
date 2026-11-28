"""Seed-set layout, validation and data-card tests (W7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pss.eval.seedset import (
    build_synthetic,
    cell_path,
    load_manifest,
    validate_seedset,
    write_data_card,
)


def _tiny(tmp_path: Path, **kwargs) -> Path:
    root = tmp_path / "seedset"
    build_synthetic(
        root, n_objects=kwargs.pop("n_objects", 3), sessions=2, samples_per_cell=4,
        **kwargs,
    )
    return root


def test_build_and_validate_synthetic(tmp_path):
    root = _tiny(tmp_path)
    report = validate_seedset(root)
    assert report["ok"], report["errors"]
    assert report["coverage"]["objects"] == 3
    assert report["coverage"]["files"] == 3 * 3 * 2
    assert report["coverage"]["samples"] == 3 * 3 * 2 * 4
    assert (root / "data_card.md").exists()
    assert load_manifest(root)["kind"] == "synthetic"


def test_real_seed_set_requires_thirty_objects(tmp_path):
    root = _tiny(tmp_path, n_objects=3)
    manifest = load_manifest(root)
    manifest["kind"] = "real"
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    report = validate_seedset(root)
    assert not report["ok"]
    assert any("need >= 30" in error for error in report["errors"])


def test_validator_reports_missing_cell(tmp_path):
    root = _tiny(tmp_path)
    cell_path(root, "obj_000", "tactile", 1).unlink()
    report = validate_seedset(root)
    assert not report["ok"]
    assert any("missing cell" in error for error in report["errors"])


def test_validator_rejects_non_finite_and_bad_timestamps(tmp_path):
    root = _tiny(tmp_path)
    path = cell_path(root, "obj_000", "vision", 1)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload["features"] = payload["features"].copy()
    payload["features"][0, 0] = np.nan
    payload["timestamps_ns"] = payload["timestamps_ns"][::-1].copy()
    np.savez_compressed(path, **payload)
    report = validate_seedset(root)
    assert not report["ok"]
    assert any("non-finite" in error for error in report["errors"])
    assert any("strictly increasing" in error for error in report["errors"])


def test_validator_rejects_duplicate_sample_ids(tmp_path):
    root = _tiny(tmp_path)
    path = cell_path(root, "obj_001", "geometry", 2)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    ids = payload["sample_ids"].copy()
    ids[1] = ids[0]
    payload["sample_ids"] = ids
    np.savez_compressed(path, **payload)
    report = validate_seedset(root)
    assert any("duplicate sample_ids" in error for error in report["errors"])


def test_data_card_records_license_and_kind(tmp_path):
    root = _tiny(tmp_path)
    card = write_data_card(root)
    text = Path(card).read_text(encoding="utf-8")
    assert "synthetic" in text
    assert "许可" in text
