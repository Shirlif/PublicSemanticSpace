import torch

from pss.eval.baselines.entity_transformer import EntityTransformer
from pss.memory.learned_metric import TACTILE_TOKEN_DIM


def test_sparse_attention_forward():
    model = EntityTransformer(
        token_dim=TACTILE_TOKEN_DIM, d_model=32, layers=1, sparse_topk=2
    )
    pre = torch.randn(2, 3, TACTILE_TOKEN_DIM)
    post = torch.randn(2, 3, TACTILE_TOKEN_DIM)
    mask = torch.ones(2, 3)
    assert model(pre, post, mask, mask).shape == (2, 3, 3)
