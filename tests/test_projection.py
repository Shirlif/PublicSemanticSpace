import pytest

pytest.importorskip("torch")

import torch

from pss.projection import (
    ProjectionHead,
    load_projection,
    save_projection,
    supcon_cross,
)
from pss.projection.losses import material_within_category


def test_projection_head_shape_and_norm():
    head = ProjectionHead(16, out_dim=8)
    x = torch.randn(5, 16)
    z = head(x)
    assert z.shape == (5, 8)
    assert torch.allclose(z.norm(dim=1), torch.ones(5), atol=1e-5)


def test_supcon_cross_finite():
    z1 = torch.randn(6, 8)
    z2 = torch.randn(6, 8)
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = supcon_cross(z1, z2, y, temp=0.07)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_material_within_category_finite():
    z = torch.randn(8, 8)
    categories = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    materials = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loss = material_within_category(z, categories, materials, temp=0.07, margin=0.5)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_save_load_roundtrip(tmp_path):
    head = ProjectionHead(16, out_dim=8)
    path = tmp_path / "projection.pt"
    save_projection(path, head)
    loaded = load_projection(path)
    x = torch.randn(3, 16)
    assert torch.allclose(head(x), loaded(x), atol=1e-6)
