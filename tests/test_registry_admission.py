"""Production admission: the registry must enforce the expert contract.

These tests exercise the path the property suite does **not** cover: a module
built through :mod:`pss.encoders.registry` is validated before it is returned.
The property suite validates 128-d contract objects; here the encoders are raw
native-width feature extractors.
"""

from __future__ import annotations

import numpy as np
import pytest

from pss.encoders.registry import build_tactile_encoder, build_vision_encoder
from pss.experts.adapter import EncoderExpert, admit_encoder, resolve_out_dim
from pss.experts.contract import ExpertValidationError, validate_expert
from pss.experts.stubs import NonNormalizedExpert


class _FnEncoder:
    def __init__(self, out_dim=6, fn=None):
        self.out_dim = int(out_dim)
        self._fn = fn or (lambda x: np.ones((len(x), self.out_dim)))

    def encode(self, x):
        return self._fn(x)


class _NestedEncoder:
    class _Inner:
        out_dim = 7

    encoder = _Inner()

    def encode(self, x):
        return np.ones((len(x), 7))


class _NoEncode:
    out_dim = 6


class _NoDim:
    def encode(self, x):
        return np.ones((len(x), 6))


def _probe(rows: int = 2, dim: int = 3) -> np.ndarray:
    return np.zeros((rows, dim), dtype=np.float32)


# ── width resolution ────────────────────────────────────────────────────────


def test_resolve_out_dim_flat_and_nested():
    assert resolve_out_dim(_FnEncoder(out_dim=6)) == 6
    assert resolve_out_dim(_NestedEncoder()) == 7


def test_resolve_out_dim_rejects_missing_width():
    with pytest.raises(ExpertValidationError, match="declares no output width"):
        resolve_out_dim(_NoDim())


def test_resolve_out_dim_rejects_nonpositive():
    with pytest.raises(ExpertValidationError, match="must be positive"):
        resolve_out_dim(_FnEncoder(out_dim=0))


# ── adapter / admit_encoder ─────────────────────────────────────────────────


def test_adapter_rejects_missing_encode_and_bad_metadata():
    with pytest.raises(ExpertValidationError, match="no callable encode"):
        EncoderExpert(_NoEncode(), modality="vision", version="v1")
    with pytest.raises(ExpertValidationError, match="modality"):
        EncoderExpert(_FnEncoder(), modality="", version="v1")
    with pytest.raises(ExpertValidationError, match="version"):
        EncoderExpert(_FnEncoder(), modality="vision", version="")
    with pytest.raises(ExpertValidationError, match="does not match"):
        EncoderExpert(_FnEncoder(), modality="vision", version="v1", dim=99)


def test_admit_encoder_probes_shape_and_finiteness():
    good = _FnEncoder(fn=lambda x: np.full((len(x), 6), 1.0 / np.sqrt(6)))
    expert = admit_encoder(good, modality="vision", version="v1", probe=_probe())
    assert expert.dim == 6
    assert validate_expert(expert, public_dim=6, probe=_probe()) is True

    with pytest.raises(ExpertValidationError, match="returned dim 5"):
        admit_encoder(
            _FnEncoder(fn=lambda x: np.zeros((len(x), 5))),
            modality="vision",
            version="v1",
            probe=_probe(),
        )
    with pytest.raises(ExpertValidationError, match="NaN/Inf"):
        admit_encoder(
            _FnEncoder(fn=lambda x: np.full((len(x), 6), np.nan)),
            modality="vision",
            version="v1",
            probe=_probe(),
        )


def test_adapter_normalize_is_opt_in():
    raw = _FnEncoder(fn=lambda x: np.full((len(x), 6), 3.0))

    # Default: raw numerics are preserved, unit-norm is not required.
    plain = admit_encoder(raw, modality="vision", version="v1", probe=_probe())
    assert not plain.normalize
    assert np.allclose(plain.encode(_probe()), 3.0)

    # Opt-in: the adapter L2-normalizes and then satisfies the public contract.
    normalized = admit_encoder(
        raw, modality="vision", version="v1", probe=_probe(), normalize=True
    )
    out = normalized.encode(_probe())
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_validate_expert_keeps_unit_norm_default():
    """Backward compatibility: the property-suite default is unchanged."""
    from pss.experts.stubs import CategoryCodebook

    expert = NonNormalizedExpert(CategoryCodebook(["a", "b"], dim=8), dim=8)
    with pytest.raises(ExpertValidationError, match="non-unit"):
        validate_expert(expert, public_dim=8, probe={"category": "a"})
    assert (
        validate_expert(
            expert, public_dim=8, probe={"category": "a"}, require_unit_norm=False
        )
        is True
    )


# ── registry wiring ─────────────────────────────────────────────────────────


def test_registry_admits_handcrafted_encoders_with_probe():
    torch = pytest.importorskip("torch")
    vision = build_vision_encoder(
        "handcrafted", image_size=32, probe=torch.zeros(2, 3, 32, 32)
    )
    assert vision.params() == 0

    tactile = build_tactile_encoder(
        "handcrafted", out_shape=(8, 8), probe=torch.zeros(2, 8, 8)
    )
    assert tactile.params() == 0


def test_registry_rejects_malformed_module(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import pss.encoders.vision as vision_pkg

    (tmp_path / "vision.pt").write_bytes(b"not-a-checkpoint")
    monkeypatch.setattr(
        vision_pkg,
        "load_vision_expert",
        lambda path, device="cpu": _FnEncoder(fn=lambda x: np.zeros((len(x), 5))),
    )

    with pytest.raises(ExpertValidationError, match="returned dim 5"):
        build_vision_encoder("smallcnn", expert_dir=tmp_path, probe=_probe())

    # Static-only admission still catches a module with no declared width.
    monkeypatch.setattr(
        vision_pkg, "load_vision_expert", lambda path, device="cpu": _NoDim()
    )
    with pytest.raises(ExpertValidationError, match="declares no output width"):
        build_vision_encoder("smallcnn", expert_dir=tmp_path)


def test_registry_auto_probes_by_default(tmp_path, monkeypatch):
    """Production builders must run the dynamic check without a caller probe."""
    pytest.importorskip("torch")
    import pss.encoders.vision as vision_pkg

    (tmp_path / "vision.pt").write_bytes(b"not-a-checkpoint")
    bad = _FnEncoder(out_dim=6, fn=lambda x: np.zeros((len(x), 5)))
    monkeypatch.setattr(
        vision_pkg, "load_vision_expert", lambda path, device="cpu": bad
    )

    with pytest.raises(ExpertValidationError, match="returned dim 5"):
        build_vision_encoder("smallcnn", expert_dir=tmp_path)

    # Static-only mode remains available for latency-sensitive callers.
    returned = build_vision_encoder(
        "smallcnn", expert_dir=tmp_path, auto_probe=False
    )
    assert returned is bad


def test_registry_validate_false_bypasses_admission(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import pss.encoders.vision as vision_pkg

    (tmp_path / "vision.pt").write_bytes(b"not-a-checkpoint")
    bad = _FnEncoder(fn=lambda x: np.zeros((len(x), 5)))
    monkeypatch.setattr(vision_pkg, "load_vision_expert", lambda path, device="cpu": bad)

    returned = build_vision_encoder(
        "smallcnn", expert_dir=tmp_path, probe=_probe(), validate=False
    )
    assert returned is bad


def test_registry_still_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown vision encoder"):
        build_vision_encoder("not-a-backbone")
    with pytest.raises(ValueError, match="unknown tactile encoder"):
        build_tactile_encoder("not-a-backbone")
