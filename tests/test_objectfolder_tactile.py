import pytest

torch = pytest.importorskip("torch")

from pss.rd2.objectfolder_tactile import TactileEncoder


def test_tactile_encoder_shape_and_norm():
    model = TactileEncoder(out_dim=64)
    out = model(torch.zeros(2, 3, 96, 128))
    assert tuple(out.shape) == (2, 64)
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)
