import pytest

from pss.inference import DEFAULT_WEIGHTS, load_available_heads
from pss.contracts import get_spec


def test_projection_weights_come_from_registry():
    assert DEFAULT_WEIGHTS["vision"] == pytest.approx(get_spec("vision").reliability_prior)
    assert DEFAULT_WEIGHTS["tactile"] == pytest.approx(get_spec("tactile").reliability_prior)
    assert DEFAULT_WEIGHTS["geometry"] == pytest.approx(get_spec("geometry").reliability_prior)


def test_missing_projection_is_inactive_not_an_error(tmp_path, monkeypatch):
    (tmp_path / "vision.pt").write_text("placeholder", encoding="utf-8")
    (tmp_path / "geometry.pt").write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        "pss.inference.load_projection",
        lambda path, device="cpu": {"path": str(path)},
    )
    heads, inactive = load_available_heads(
        tmp_path, modalities=("vision", "tactile", "geometry")
    )
    assert list(heads) == ["vision", "geometry"]
    assert inactive == ["tactile"]


def test_all_present_heads_have_no_inactive(tmp_path, monkeypatch):
    for modality in ("vision", "tactile", "geometry"):
        (tmp_path / f"{modality}.pt").write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        "pss.inference.load_projection",
        lambda path, device="cpu": {"path": str(path)},
    )
    heads, inactive = load_available_heads(tmp_path)
    assert set(heads) == {"vision", "tactile", "geometry"}
    assert inactive == []
