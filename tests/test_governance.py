import numpy as np
import torch

from pss.eval.scripts.run_p16_reliability import ReliabilityGate
from pss.eval.scripts.run_p17_conformal import _conformal_quantile
from pss.eval.scripts.run_p18_prototype_governance import _prototype


def test_conformal_quantile_lower_tail():
    values = np.arange(100, dtype=float)
    q = _conformal_quantile(values, 0.05)
    assert 4.0 <= q <= 5.0


def test_reliability_gate_forward():
    model = ReliabilityGate(n_classes=5, hidden=16)
    scores = {
        "vision": torch.randn(4, 5),
        "tactile": torch.randn(4, 5),
        "geometry": torch.randn(4, 5),
    }
    logits, weights = model(scores)
    assert logits.shape == (4, 5)
    assert weights.shape == (4, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)


def test_prototype_modes():
    Z = {
        "vision": np.random.default_rng(0).normal(size=(9, 8)).astype(np.float32),
        "tactile": np.random.default_rng(1).normal(size=(9, 8)).astype(np.float32),
        "geometry": np.random.default_rng(2).normal(size=(9, 8)).astype(np.float32),
    }
    idx = list(range(9))
    for mode in ("mean", "trimmed", "medoid", "median"):
        proto = _prototype(Z, idx, mode)
        assert set(proto) == set(Z)
        for v in proto.values():
            assert v.shape == (8,)
