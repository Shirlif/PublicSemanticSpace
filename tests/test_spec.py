from __future__ import annotations

import json
from pathlib import Path

import pytest

from pss.spec import hash_spec, lint_file, load_spec, register, validate
from pss.spec.loader import load_yaml
from pss.spec.schema import json_schema

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "specs" / "s1_occlusion.yaml"


def test_s1_occlusion_spec_is_valid():
    spec = load_spec(SPEC_PATH)
    assert spec.id == "s1_occlusion_v1"
    assert len(spec.assets) == 4
    assert spec.sha256.startswith("sha256:")


def test_hash_is_independent_of_key_order():
    raw = load_yaml(SPEC_PATH)
    reordered = {k: raw[k] for k in reversed(list(raw))}
    assert hash_spec(raw) == hash_spec(reordered)


def test_lint_rejects_missing_required_sections():
    result = validate({"spec_version": 1, "id": "x"})
    assert not result.ok
    assert any("task_family" in e for e in result.errors)
    assert any("assets" in e for e in result.errors)


def test_lint_rejects_unknown_keys():
    raw = load_yaml(SPEC_PATH)
    raw["surprise"] = 1
    result = validate(raw)
    assert not result.ok
    assert any("surprise" in e for e in result.errors)


def test_lint_warns_when_noisy_without_profile():
    raw = load_yaml(SPEC_PATH)
    raw["perception"] = {"mode": "offline", "detector_tier": "noisy"}
    result = validate(raw)
    assert not result.ok
    assert any("noise_profile" in e for e in result.errors)


def test_registry_is_idempotent(tmp_path):
    result = lint_file(SPEC_PATH)
    assert result.ok
    registry = tmp_path / "registry.jsonl"
    assert register(result, SPEC_PATH, registry=registry) is True
    assert register(result, SPEC_PATH, registry=registry) is False
    lines = [line for line in registry.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    assert json.loads(lines[0])["sha256"] == result.sha256


def test_json_schema_is_serializable():
    schema = json_schema()
    json.dumps(schema)
    assert schema["properties"]["spec_version"]["const"] == 1


def test_missing_file_returns_error_result(tmp_path):
    result = lint_file(tmp_path / "nope.yaml")
    assert not result.ok
    assert result.errors
