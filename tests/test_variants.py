from __future__ import annotations

import numpy as np

from pss.memory.perception import Observation
from pss.runtime.variants import VARIANTS, FlatEntityLayer, build_runtime


def _obs(gt_id="a"):
    return Observation(
        gt_id=gt_id,
        pos=np.array([0.0, 0.0, 0.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        appearance=np.array([1.0, 0.0]),
        raw={"vision": np.zeros(2)},
        confidence=1.0,
        category="cube",
    )


def test_nine_variants_are_registered():
    assert len(VARIANTS) == 9
    assert "full" in VARIANTS and "no_entity" in VARIANTS and "flat_context" in VARIANTS


def test_build_runtime_accepts_every_variant():
    for variant in VARIANTS:
        runtime = build_runtime(variant, tier="oracle")
        assert runtime is not None


def test_flat_entity_layer_does_not_persist_identity():
    layer = FlatEntityLayer()
    layer.ingest([_obs()], 0)
    first = set(layer.active_ids())
    layer.ingest([_obs()], 1)
    second = set(layer.active_ids())
    assert first.isdisjoint(second)
