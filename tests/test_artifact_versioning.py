import json

import numpy as np
import pytest

from pss.contracts.schema import SPACE_VERSION
from pss.contracts.versioning import (
    ArtifactManifest,
    IncompatibleArtifactError,
    assert_compatible,
    is_compatible,
    parse_version,
    sha256_file,
)
from pss.prototypes.bank import PrototypeBank
from pss.prototypes.joint_readout import JointReadout
from pss.runtime.artifacts import ArtifactRegistry
from pss.runtime.migrate import migrate_bank, migrate_readout


def test_version_parsing_and_compatibility():
    assert parse_version("v0.2.1") == (0, 2, 1)
    assert is_compatible("0.1.1", "0.2.0")
    assert is_compatible("0.2.0", "0.2.0")
    assert not is_compatible("0.3.0", "0.2.0")
    assert not is_compatible("1.0.0", "0.2.0")


def test_assert_compatible_has_migration_hint():
    with pytest.raises(IncompatibleArtifactError, match="rebuild"):
        assert_compatible("1.0.0", "0.2.0", artifact_name="bank")


def test_manifest_checksum_roundtrip(tmp_path):
    file = tmp_path / "bank.npz"
    file.write_bytes(b"payload")
    manifest = ArtifactManifest.build(
        artifact="bank",
        space_version="0.2.0",
        public_dim=128,
        modalities=["vision"],
        files={"bank": file},
    )
    assert manifest.checksums["bank"] == sha256_file(file)
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = ArtifactManifest.load(path)
    assert loaded.to_dict() == manifest.to_dict()


def _legacy_bank(path):
    payload = {
        "categories_json": np.array(json.dumps(["a", "b"])),
        "stats_json": np.array(json.dumps({"a": {}, "b": {}})),
        "version": np.array("legacy"),
        "proto_vision": np.eye(2, 4, dtype=np.float32),
    }
    np.savez_compressed(path, **payload)


def test_bank_version_defaults_and_migrates(tmp_path):
    path = tmp_path / "bank.npz"
    _legacy_bank(path)
    bank = PrototypeBank.load(path)
    assert bank.space_version == "0.1.1"
    manifest = migrate_bank(path)
    assert manifest.space_version == SPACE_VERSION
    assert PrototypeBank.load(path).space_version == SPACE_VERSION


def test_incompatible_bank_migration_is_rejected(tmp_path):
    path = tmp_path / "bank.npz"
    _legacy_bank(path)
    bank = PrototypeBank.load(path)
    bank.space_version = "99.0.0"
    bank.save(path)
    with pytest.raises(IncompatibleArtifactError):
        migrate_bank(path)


def _readout():
    rng = np.random.default_rng(0)
    Z = {"a": rng.normal(size=(20, 4)), "b": rng.normal(size=(20, 4))}
    y = np.array([0] * 10 + [1] * 10)
    return JointReadout.fit(Z, y, ["x", "y"], modalities=("a", "b"))


def test_readout_version_migration(tmp_path):
    path = tmp_path / "readout.npz"
    readout = _readout()
    readout.save(path)
    assert JointReadout.load(path).space_version == SPACE_VERSION

    # Simulate an artifact written before the field existed.
    with np.load(path, allow_pickle=False) as f:
        arrays = {name: f[name] for name in f.files}
    meta = json.loads(str(arrays["meta_json"]))
    meta.pop("space_version", None)
    arrays["meta_json"] = np.array(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(path, **arrays)
    assert JointReadout.load(path).space_version == "0.1.1"
    migrate_readout(path)
    assert JointReadout.load(path).space_version == SPACE_VERSION


def test_artifact_registry_rollback(tmp_path):
    registry = ArtifactRegistry(tmp_path / "registry")
    first = tmp_path / "bank_v1.npz"
    first.write_bytes(b"v1")
    second = tmp_path / "bank_v2.npz"
    second.write_bytes(b"v2")
    registry.register("bank", first, version="v1")
    registry.register("bank", second, version="v2")
    assert len(registry.list("bank")) == 2
    assert registry.resolve("bank").read_bytes() == b"v2"
    restored = registry.rollback("bank", "v1", target=tmp_path / "restored.npz")
    assert restored.read_bytes() == b"v1"


def test_bank_load_enforces_version_with_escape_hatch(tmp_path):
    path = tmp_path / "bank.npz"
    _legacy_bank(path)
    bank = PrototypeBank.load(path)
    bank.space_version = "99.0.0"
    bank.save(path)
    with pytest.raises(IncompatibleArtifactError):
        PrototypeBank.load(path)
    assert PrototypeBank.load(path, strict=False).space_version == "99.0.0"


def test_readout_load_enforces_version_with_escape_hatch(tmp_path):
    path = tmp_path / "readout.npz"
    readout = _readout()
    readout.space_version = "99.0.0"
    readout.save(path)
    with pytest.raises(IncompatibleArtifactError):
        JointReadout.load(path)
    assert JointReadout.load(path, strict=False).space_version == "99.0.0"


def test_projection_load_enforces_version_with_escape_hatch(tmp_path):
    pytest.importorskip("torch")
    from pss.projection.heads import ProjectionHead, load_projection, save_projection

    path = tmp_path / "vision.pt"
    head = ProjectionHead(4, 8)
    head.space_version = "99.0.0"
    save_projection(path, head)
    with pytest.raises(IncompatibleArtifactError):
        load_projection(path)
    assert load_projection(path, strict=False).space_version == "99.0.0"
