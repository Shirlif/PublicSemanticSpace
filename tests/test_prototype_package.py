import numpy as np
import pytest

pytest.importorskip("torch")

from pss.prototypes.bank import PrototypeBank
from pss.prototypes.package import init_package, load_manager, rollback


def _bank() -> PrototypeBank:
    categories = ["a", "b"]
    protos = {
        "vision": np.eye(2, 4, dtype=np.float32),
        "tactile": np.eye(2, 4, dtype=np.float32),
    }
    stats = {
        category: {
            modality: {"sample_count": 5, "variance": 0.1}
            for modality in protos
        }
        for category in categories
    }
    return PrototypeBank(categories, protos, stats)


def test_init_add_rollback(tmp_path):
    bank = _bank()
    package, created = init_package(bank, tmp_path, version="v1")
    assert created
    assert package.version == "v1"

    new_package = package.append_category(
        "c",
        {
            "vision": np.ones(4, dtype=np.float32) / 2,
            "tactile": np.ones(4, dtype=np.float32) / 2,
        },
        {
            "vision": {"sample_count": 3, "variance": 0.1},
            "tactile": {"sample_count": 3, "variance": 0.1},
        },
        version="v2",
    )
    manager = load_manager(tmp_path)
    manager.install(new_package)
    manager.activate("v2")
    assert manager.active == "v2"

    result = rollback(tmp_path, to_version="v1")
    assert result["to"] == "v1"
    assert result["integrity"] is True
    assert "c" not in result["categories"]
