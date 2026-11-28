import textwrap

import pytest

from pss.contracts import (
    MODALITIES,
    ModalityContractError,
    get_modalities,
    get_spec,
    load_modalities,
    resolve_modalities,
)


def test_default_registry_has_three_active_incumbents():
    assert get_modalities() == ("vision", "tactile", "geometry")
    assert MODALITIES == get_modalities()


def test_experimental_audio_is_registered_but_not_default():
    specs = load_modalities()
    assert "audio" in specs
    assert specs["audio"].status == "experimental"
    assert "audio" not in get_modalities()
    assert "audio" in get_modalities(include_experimental=True)


def test_specs_carry_required_contract_fields():
    vision = get_spec("vision")
    assert vision.native_dim == 256
    assert vision.normalize == "l2"
    assert 0.0 <= vision.reliability_prior <= 1.0
    assert vision.max_latency_ms >= 0
    assert vision.version


def test_resolve_modalities_is_canonical_order():
    assert resolve_modalities(("geometry", "vision")) == ("vision", "geometry")
    with pytest.raises(ModalityContractError):
        resolve_modalities(("vision", "sonar"))


def _write(tmp_path, body: str):
    path = tmp_path / "modalities.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_missing_field_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        """
        modalities:
          vision:
            kind: dense_rgb
            preprocess: x
            reliability_prior: 0.9
        """,
    )
    with pytest.raises(ModalityContractError):
        load_modalities(str(path))


def test_invalid_enum_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        """
        modalities:
          vision:
            kind: dense_rgb
            preprocess: x
            reliability_prior: 0.9
            version: "0.1.0"
            normalize: minmax
        """,
    )
    with pytest.raises(ModalityContractError):
        load_modalities(str(path))


def test_name_key_mismatch_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        """
        modalities:
          vision:
            name: tactile
            kind: dense_rgb
            preprocess: x
            reliability_prior: 0.9
            version: "0.1.0"
        """,
    )
    with pytest.raises(ModalityContractError):
        load_modalities(str(path))


def test_unknown_spec_field_warns(tmp_path, recwarn):
    path = _write(
        tmp_path,
        """
        modalities:
          vision:
            kind: dense_rgb
            preprocess: x
            reliability_prior: 0.9
            version: "0.1.0"
            made_up_field: 1
        """,
    )
    specs = load_modalities(str(path))
    assert "vision" in specs
    assert any("unknown fields" in str(w.message) for w in recwarn)
