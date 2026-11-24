import pytest

torch = pytest.importorskip("torch")

from pss.rd2.objectfolder_symmetric import VisionProjector


def test_vision_projector_shape_and_norm():
    model = VisionProjector(in_dim=384, out_dim=64, hidden=128)
    out = model(torch.zeros(3, 384))
    assert tuple(out.shape) == (3, 64)
    assert torch.allclose(out.norm(dim=1), torch.ones(3), atol=1e-5)
