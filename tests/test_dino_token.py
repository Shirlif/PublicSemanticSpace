import numpy as np
import torch

from pss.eval.scripts.run_p0_dino import _select, zero_modality
from pss.memory.graph_matcher import GraphMatcher
from pss.memory.learned_metric import (
    DINO_DIM,
    DINO_TOKEN_DIM,
    FULL_TOKEN_DIM,
    TACTILE_DIM,
    TOKEN_DIM,
    entity_token,
    token_flags,
    token_layout,
)


def _raw():
    return {
        "vision": np.ones(128, dtype=np.float32),
        "geometry": np.ones(128, dtype=np.float32) * 2,
        "tactile": np.ones(TACTILE_DIM, dtype=np.float32) * 3,
        "dino": np.ones(DINO_DIM, dtype=np.float32) * 4,
    }


def test_token_dims_and_layout():
    base = entity_token([0, 0, 0], _raw(), "block")
    dino = entity_token([0, 0, 0], _raw(), "block", include_dino=True)
    both = entity_token(
        [0, 0, 0], _raw(), "block", include_tactile=True, include_dino=True
    )
    assert base.shape == (TOKEN_DIM,)
    assert dino.shape == (DINO_TOKEN_DIM,)
    assert both.shape == (FULL_TOKEN_DIM,)
    layout = token_layout(True, True)
    assert layout["tactile"] == (TOKEN_DIM, TOKEN_DIM + TACTILE_DIM)
    assert layout["dino"] == (TOKEN_DIM + TACTILE_DIM, FULL_TOKEN_DIM)
    assert token_flags(TOKEN_DIM) == (False, False)
    assert token_flags(DINO_TOKEN_DIM) == (False, True)
    assert token_flags(FULL_TOKEN_DIM) == (True, True)


def test_select_and_zero():
    full = np.arange(FULL_TOKEN_DIM, dtype=np.float32)[None]
    dino = _select(full, False, True)
    assert dino.shape == (1, DINO_TOKEN_DIM)
    assert np.allclose(dino[0, :TOKEN_DIM], full[0, :TOKEN_DIM])
    assert np.allclose(dino[0, TOKEN_DIM:], full[0, TOKEN_DIM + TACTILE_DIM :])

    from pss.eval.baselines.entity_transformer import TokenPair

    pair = TokenPair(
        pre=full.copy(),
        post=full.copy(),
        pre_mask=np.ones(3, dtype=np.float32),
        post_mask=np.ones(3, dtype=np.float32),
        labels=np.zeros((3, 3), dtype=np.float32),
        subset="mixed",
        sample_id="0",
    )
    zeroed = zero_modality([pair], "dino")[0]
    start, end = token_layout(True, True)["dino"]
    assert np.all(zeroed.pre[:, start:end] == 0)
    assert np.all(zeroed.pre[:, :start] == pair.pre[:, :start])


def test_graph_matcher_accepts_dino_token():
    model = GraphMatcher(token_dim=DINO_TOKEN_DIM, d_model=32, layers=1)
    pre = torch.randn(1, 3, DINO_TOKEN_DIM)
    post = torch.randn(1, 3, DINO_TOKEN_DIM)
    mask = torch.ones(1, 3)
    out = model(pre, post, mask, mask)
    assert out.shape == (1, 3, 3)
