"""Protocol-level tests: no PSS imports, canonical digests, capabilities."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pss_eval.protocol import (
    Observation,
    Prediction,
    capabilities_of,
    predictions_equal,
    state_digest,
    state_size_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "pss_eval" / "protocol.py"


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_protocol_source_has_no_pss_imports():
    modules = _imported_modules(PROTOCOL_PATH)
    forbidden = [
        module
        for module in modules
        if module == "pss" or module.startswith("pss.")
    ]
    assert forbidden == []


def test_protocol_imports_without_loading_pss():
    code = (
        "import sys\n"
        "import pss_eval.protocol\n"
        "loaded = sorted(m for m in sys.modules if m == 'pss' or m.startswith('pss.'))\n"
        "assert not loaded, loaded\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_observation_copies_and_validates_features():
    source = np.array([1.0, 2.0])
    observation = Observation("s0", "vision", source)
    source[0] = 99.0
    assert observation.features[0] == 1.0
    assert observation.features.dtype == np.float64
    with pytest.raises(ValueError):
        Observation("s0", "vision", np.zeros((2, 2)))


def test_prediction_validates_embedding_and_coerces_scores():
    prediction = Prediction(
        "s0", np.array([1, 2, 3]), label="cup", scores={"cup": np.float32(0.7)}
    )
    assert prediction.embedding.shape == (3,)
    assert prediction.scores == {"cup": pytest.approx(0.7)}
    with pytest.raises(ValueError):
        Prediction("s0", np.zeros((2, 2)))


def test_state_digest_is_key_order_independent():
    left = {"b": np.array([1.0, 2.0]), "a": 1}
    right = {"a": 1, "b": np.array([1.0, 2.0])}
    assert state_digest(left) == state_digest(right)
    assert state_size_bytes(left) > 0
    assert len(state_digest(left)) == 64


def test_state_digest_rejects_non_finite_values():
    with pytest.raises(ValueError):
        state_digest({"x": np.array([np.nan])})
    with pytest.raises(ValueError):
        state_digest({"x": float("inf")})


def test_predictions_equal_is_exact():
    left = Prediction("s0", np.array([1.0, 2.0]), "cup")
    same = Prediction("s0", np.array([1.0, 2.0]), "cup")
    almost = Prediction("s0", np.array([1.0, 2.0 + 1e-12]), "cup")
    other_label = Prediction("s0", np.array([1.0, 2.0]), "block")
    assert predictions_equal(left, same)
    assert not predictions_equal(left, almost)
    assert not predictions_equal(left, other_label)


class _SnapshotOnly:
    def snapshot(self) -> str:
        return "token"

    def rollback(self, token: str) -> bool:
        return True


class _AllCapabilities:
    def snapshot(self) -> str:
        return "token"

    def rollback(self, token: str) -> bool:
        return True

    def swap(self, module) -> bool:
        return True

    def audit(self) -> list[dict]:
        return [{"event": "probe"}]


def test_capabilities_of_detects_optional_protocols():
    assert capabilities_of(object()) == frozenset()
    assert capabilities_of(_SnapshotOnly()) == frozenset({"snapshot"})
    assert capabilities_of(_AllCapabilities()) == frozenset(
        {"snapshot", "swap", "audit"}
    )
