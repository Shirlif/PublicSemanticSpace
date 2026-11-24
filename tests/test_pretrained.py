import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch

from pss.encoders.vision.pretrained import (
    ResNet18Encoder,
    load_resnet_expert,
    save_resnet_expert,
)


def test_resnet_encoder_forward_and_checkpoint(tmp_path):
    encoder = ResNet18Encoder(out_dim=16, pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    assert encoder(x).shape == (2, 16)
    assert encoder.backbone.training is False

    path = tmp_path / "resnet.pt"
    save_resnet_expert(path, encoder, ["a", "b"], 64)
    loaded = load_resnet_expert(path)
    assert loaded(x).shape == (2, 16)
