import json

import pytest

from pss.inference import resolve_weights
from pss.projection.calibration import (
    ModalityCalibration,
    load_calibration,
    save_calibration,
)


def _records():
    return {
        "vision": ModalityCalibration(
            modality="vision",
            temperature=0.24,
            reliability=0.968,
            val_accuracy=0.97,
            val_nll=0.1,
            n_val=36,
        ),
        "tactile": ModalityCalibration(
            modality="tactile",
            temperature=1.11,
            reliability=0.188,
            val_accuracy=0.24,
            val_nll=1.4,
            n_val=36,
        ),
    }


def test_calibration_roundtrip(tmp_path):
    path = tmp_path / "calibration.json"
    save_calibration(path, _records())
    loaded = load_calibration(path)
    assert set(loaded) == {"vision", "tactile"}
    assert loaded["vision"].reliability == pytest.approx(0.968)
    assert loaded["vision"].temperature == pytest.approx(0.24)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == "calibration_v1"
    assert payload["space_version"]


def test_resolve_weights_prefers_calibration():
    weights = resolve_weights(("vision", "tactile"), _records())
    assert weights["vision"] == pytest.approx(0.968)
    assert weights["tactile"] == pytest.approx(0.188)


def test_resolve_weights_falls_back_to_registry():
    weights = resolve_weights(("vision", "geometry"))
    assert 0.0 <= weights["vision"] <= 1.0
    assert 0.0 <= weights["geometry"] <= 1.0


def test_flat_calibration_payload_is_accepted(tmp_path):
    path = tmp_path / "flat.json"
    path.write_text(
        json.dumps(
            {
                "vision": {
                    "modality": "vision",
                    "temperature": 0.3,
                    "reliability": 0.8,
                    "val_accuracy": 0.9,
                    "val_nll": 0.2,
                    "n_val": 10,
                }
            }
        ),
        encoding="utf-8",
    )
    loaded = load_calibration(path)
    assert loaded["vision"].reliability == pytest.approx(0.8)
