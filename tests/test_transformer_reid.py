from types import SimpleNamespace

import numpy as np

from pss.eval.baselines.entity_transformer import EntityTransformer
from pss.memory.learned_metric import TACTILE_TOKEN_DIM
from pss.memory.transformer_reid import TransformerReID


def _entity(pos):
    v = np.ones(128, dtype=np.float32) / np.sqrt(128)
    return SimpleNamespace(
        pos=np.asarray(pos, dtype=float),
        category="block",
        appearance=v,
        raw={"vision": v, "geometry": v, "tactile": v},
    )


def test_transformer_reid_accepts_tactile_tokens():
    model = EntityTransformer(token_dim=TACTILE_TOKEN_DIM, d_model=32, layers=1)
    adapter = TransformerReID(model, include_tactile=True)
    tracks = [_entity([0, 0, 0]), _entity([1, 0, 0])]
    obs = [_entity([0.01, 0, 0]), _entity([1.01, 0, 0])]
    cost = adapter.cost_matrix(tracks, obs)
    assert cost.shape == (2, 2)
