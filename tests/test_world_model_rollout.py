import torch

from pss.world_model.multi import build_multi_transitions
from pss.world_model.set_model import FEATURE_DIM, SetDeltaMLP, _rollout_loss


def test_rollout_loss_is_differentiable():
    rows = build_multi_transitions("data/phase7_v2_more")[:8]
    model = SetDeltaMLP(FEATURE_DIM, hidden=16)
    loss = _rollout_loss(model, rows, "cpu")
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
