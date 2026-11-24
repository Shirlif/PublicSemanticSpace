import torch

from pss.eval.scripts.run_rag_baseline import RAGContextTransformer
from pss.eval.long_context_bench import N_VALUES


def test_rag_forward():
    model = RAGContextTransformer(d_model=32, layers=1, heads=2)
    keys = torch.zeros((2, 10), dtype=torch.long)
    values = torch.zeros((2, 10), dtype=torch.long)
    query = torch.zeros((2,), dtype=torch.long)
    assert model(keys, values, query).shape == (2, N_VALUES)
