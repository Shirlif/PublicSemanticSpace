import torch

from pss.memory.learned_metric import TACTILE_TOKEN_DIM
from pss.memory.object_attention import ObjectAttentionMatcher


def test_object_attention_shapes():
    model = ObjectAttentionMatcher(TACTILE_TOKEN_DIM, d_model=64, heads=4, layers=1)
    pre = torch.randn(2, 3, TACTILE_TOKEN_DIM)
    post = torch.randn(2, 3, TACTILE_TOKEN_DIM)
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    scores = model(pre, post, mask, mask)
    assert scores.shape == (2, 3, 3)
