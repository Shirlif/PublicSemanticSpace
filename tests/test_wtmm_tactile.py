import numpy as np
import pytest
import torch

from pss.encoders.tactile.wtmm import TactileCNNEncoder, WTMMTactileExpert


def test_tactile_encoder_shapes():
    model = TactileCNNEncoder(out_dim=32)
    assert model(torch.rand(2, 12, 32)).shape == (2, 32)
    assert model(torch.rand(2, 8, 8)).shape == (2, 32)
    assert model(torch.rand(2, 1, 12, 32)).shape == (2, 32)


def test_expert_adapter_accepts_flat_8x8():
    model = TactileCNNEncoder(out_dim=16)
    expert = WTMMTactileExpert(encoder=model, device="cpu", input_dim=64)
    out = expert.encode(np.random.rand(4, 64).astype(np.float32))
    assert out.shape == (4, 16)


def test_expert_adapter_accepts_12x32():
    model = TactileCNNEncoder(out_dim=16)
    expert = WTMMTactileExpert(encoder=model, device="cpu")
    out = expert.encode(np.random.rand(3, 12, 32).astype(np.float32))
    assert out.shape == (3, 16)
