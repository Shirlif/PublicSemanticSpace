import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pss.eval.baselines.entity_transformer import (
    TOKEN_DIM,
    EntityTransformer,
    TokenPair,
    evaluate_transformer,
    train_transformer,
)


def _pair(seed: int) -> TokenPair:
    rng = np.random.default_rng(seed)
    return TokenPair(
        pre=rng.normal(size=(3, TOKEN_DIM)).astype(np.float32),
        post=rng.normal(size=(3, TOKEN_DIM)).astype(np.float32),
        pre_mask=np.ones(3, dtype=np.float32),
        post_mask=np.ones(3, dtype=np.float32),
        labels=np.eye(3, dtype=np.float32),
        subset="mixed",
        sample_id=str(seed),
    )


def test_entity_transformer_forward_shape():
    model = EntityTransformer(d_model=32, nhead=4, layers=1)
    pre = torch.zeros(2, 3, TOKEN_DIM)
    post = torch.zeros(2, 3, TOKEN_DIM)
    mask = torch.ones(2, 3)
    scores = model(pre, post, mask, mask)
    assert scores.shape == (2, 3, 3)


def test_entity_transformer_training_smoke():
    pairs = [_pair(i) for i in range(8)]
    model, info = train_transformer(
        pairs, pairs, epochs=2, batch_size=4, seed=0, device="cpu"
    )
    assert "best_val_reid" in info
    result = evaluate_transformer(model, pairs, device="cpu")
    assert 0.0 <= result["reid_accuracy"] <= 1.0
    assert "mixed" in result["per_subset"]
