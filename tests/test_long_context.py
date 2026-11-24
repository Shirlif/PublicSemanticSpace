import torch

from pss.eval.long_context_bench import N_KEYS, N_VALUES, ContextTransformer


def test_context_transformer_forward():
    model = ContextTransformer(d_model=32, layers=1, heads=2)
    keys = torch.zeros((2, 10), dtype=torch.long)
    values = torch.zeros((2, 10), dtype=torch.long)
    query = torch.zeros((2,), dtype=torch.long)
    logits = model(keys, values, query)
    assert logits.shape == (2, N_VALUES)
