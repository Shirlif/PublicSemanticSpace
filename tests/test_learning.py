import numpy as np
import pytest

from pss.learning import (
    AdapterRegistry,
    L1Record,
    L1Store,
    PackageManager,
    PrototypePackage,
    RuleRegistry,
    adapter_package,
    rule_package,
)
from pss.learning.signing import generate_keypair


def _unit(seed, d=8):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


def test_l1_proposal_is_scrubbed_and_health_gated(tmp_path):
    store = L1Store(tmp_path / "l1.jsonl")
    for i in range(30):
        store.append(
            L1Record(
                kind="observation",
                category="cloth",
                entity_id=f"e{i}",
                vectors={"vision": _unit(i).tolist()},
                confidence=1.0,
                environment={"lighting": "normal", "secret": "drop_me"},
            )
        )
    store.save()
    loaded = L1Store.load(tmp_path / "l1.jsonl")
    proposal = loaded.make_proposal("cloth", health="normal")
    assert proposal is not None
    assert proposal["sample_count"] == 30
    assert proposal["scrubbed"] is True
    assert "entity_id" not in proposal
    assert "secret" not in proposal["environment"]
    assert loaded.make_proposal("cloth", health="degraded") is None


def test_package_append_verify_and_rollback(tmp_path):
    base = PrototypePackage(
        version="v1",
        categories=["a"],
        protos={"vision": np.eye(2)[:1]},
        stats={"a": {"vision": {"sample_count": 1, "variance": 0.0}}},
    )
    new = base.append_category(
        "b",
        {"vision": np.array([0.0, 1.0])},
        {"vision": {"sample_count": 1, "variance": 0.0}},
        version="v2",
    )
    manager = PackageManager(tmp_path / "packages")
    manager.install(base)
    manager.install(new)
    manager.activate("v1")
    manager.activate("v2")
    assert "b" in manager.active_package().categories
    assert manager.rollback() == "v1"
    assert "b" not in manager.active_package().categories

    corrupt = manager.copy_package("v2", tmp_path / "corrupt")
    (corrupt / "prototypes.npz").write_bytes(b"corrupt")
    with pytest.raises(Exception):
        PrototypePackage.load(corrupt)


def test_registries_version_and_rollback():
    adapters = AdapterRegistry()
    adapters.install(adapter_package("a1", "projection", {"vision.pt": "h1"}))
    adapters.install(adapter_package("a2", "projection", {"vision.pt": "h2"}, parent="a1"))
    adapters.activate("a1")
    adapters.activate("a2")
    assert adapters.active == "a2"
    assert adapters.rollback() == "a1"

    rules = RuleRegistry()
    rules.install(rule_package("r1", {"task_fidelity": 0.5}, ["human_in_zone"]))
    rules.activate("r1")
    assert rules.active == "r1"
    assert rules.active_package().payload["blacklist"] == ["human_in_zone"]


def test_ed25519_signed_package_verification(tmp_path):
    private_key, public_key = generate_keypair()
    package = PrototypePackage(
        version="v1",
        categories=["a"],
        protos={"vision": np.eye(2)[:1]},
        stats={"a": {"vision": {"sample_count": 1, "variance": 0.0}}},
    )
    manager = PackageManager(
        tmp_path / "packages",
        trusted_public_key=public_key,
        private_key=private_key,
    )
    manager.install(package)
    manager.activate("v1")
    assert manager.verify("v1")
    assert manager.active_package().version == "v1"

    other_private, other_public = generate_keypair()
    with pytest.raises(Exception):
        PrototypePackage.load(
            manager.packages["v1"], trusted_public_key=other_public
        )

    unsigned = PrototypePackage(
        version="v2",
        categories=["a"],
        protos={"vision": np.eye(2)[:1]},
        stats={"a": {"vision": {"sample_count": 1, "variance": 0.0}}},
    )
    unsigned.save(tmp_path / "unsigned")
    with pytest.raises(Exception):
        PrototypePackage.load(
            tmp_path / "unsigned", trusted_public_key=public_key
        )

    signed_path = manager.packages["v1"]
    npz = signed_path / "prototypes.npz"
    npz.write_bytes(npz.read_bytes() + b"tamper")
    with pytest.raises(Exception):
        PrototypePackage.load(signed_path, trusted_public_key=public_key)


def test_registry_payload_files_and_tamper(tmp_path):
    registry = AdapterRegistry(tmp_path / "l3")
    payload = b'{"adapter":"vision"}'
    import hashlib

    registry.install(
        adapter_package(
            "a1",
            "projection",
            {"vision.pt": hashlib.sha256(payload).hexdigest()},
            artifacts={"vision.pt": payload},
        )
    )
    assert registry.verify("a1")
    registry.activate("a1")
    target = tmp_path / "l3" / "a1" / "files" / "vision.pt"
    target.write_bytes(b"tampered")
    assert not registry.verify("a1")
    with pytest.raises(Exception):
        registry.activate("a1")
