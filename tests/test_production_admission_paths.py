"""Every production loader that touches a raw checkpoint admits it first.

These calls used to bypass the registry (and therefore the contract).  Each
test monkeypatches the loader with a wrong-width module and expects admission
to fail before any data/projection work happens.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

import pss.projection.anchor as anchor_mod
import pss.projection.train as train_mod
import pss.prototypes.build as build_mod
import pss.prototypes.package as package_mod
from pss.experts.contract import ExpertValidationError


class _BadVision:
    out_dim = 6
    image_size = 8

    def encode(self, images):
        return np.zeros((len(images), 5))


def test_prototype_feature_extraction_admits(monkeypatch, tmp_path):
    monkeypatch.setattr(build_mod, "load_vision_expert", lambda *a, **k: _BadVision())
    with pytest.raises(ExpertValidationError, match="returned dim 5"):
        build_mod.extract_projected_features(
            tmp_path / "data", tmp_path / "experts", tmp_path / "projections"
        )


def test_projection_training_admits(monkeypatch, tmp_path):
    monkeypatch.setattr(train_mod, "load_vision_expert", lambda *a, **k: _BadVision())
    with pytest.raises(ExpertValidationError, match="returned dim 5"):
        train_mod.train_projections(
            tmp_path / "data", tmp_path / "ckpt", tmp_path / "out"
        )


def test_anchor_expert_loading_admits(monkeypatch, tmp_path):
    monkeypatch.setattr(anchor_mod, "load_vision_expert", lambda *a, **k: _BadVision())
    with pytest.raises(ExpertValidationError, match="returned dim 5"):
        anchor_mod._load_experts(tmp_path, "vision", None, "cpu")


def test_package_add_category_admits(monkeypatch, tmp_path):
    active = SimpleNamespace(categories=[], protos={}, stats={}, version="v0")
    monkeypatch.setattr(
        package_mod,
        "load_manager",
        lambda root: SimpleNamespace(active="v0", active_package=lambda: active),
    )
    monkeypatch.setattr(
        package_mod, "load_vision_expert", lambda *a, **k: _BadVision()
    )
    with pytest.raises(ExpertValidationError, match="returned dim 5"):
        package_mod.add_category(
            tmp_path / "data",
            tmp_path / "experts",
            tmp_path / "projections",
            tmp_path / "prototypes",
            tmp_path / "root",
            "newcat",
        )
