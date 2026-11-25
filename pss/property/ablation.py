"""Additional H-series experiments and KB trace data for the §15 report.

These are L0 simulations: they use the real registry/KB classes but structured
observations and stubs, so they isolate the contract rather than perception.
"""

from __future__ import annotations

import numpy as np

from pss.experts import make_quality_expert
from pss.experts.stubs import CategoryCodebook
from pss.metrics.property import rollback_exactness, zero_shot_compatibility
from pss.observations import make_scene
from pss.property.kb import VersionedKnowledgeBase
from pss.property.swap import ModuleRegistry


def _predictions(expert, codebook: CategoryCodebook, obs) -> list[str]:
    return [codebook.predict(v) for v in expert.encode(obs)]


def run_h3_add_remove(seed: int = 0) -> dict:
    """Add an audio module, then remove it; old modalities must not change."""

    scene = make_scene(50, dim=128, seed=seed)
    obs = scene.observations()
    codebook = scene.codebook
    registry = ModuleRegistry(public_dim=128, probe=obs[:8])
    registry.register(make_quality_expert("Q0", codebook, modality="vision", seed=seed), probe=obs[:8])
    registry.register(make_quality_expert("Q0", codebook, modality="geometry", seed=seed), probe=obs[:8])
    geo_before = registry.behavior_hash("geometry", probe=obs[:8])

    audio = make_quality_expert("Q0", codebook, modality="audio", seed=seed)
    registry.register(audio, probe=obs[:8])
    geo_after = registry.behavior_hash("geometry", probe=obs[:8])
    zcr_new = zero_shot_compatibility(
        _predictions(audio, codebook, obs), [o.gt_category for o in obs]
    )

    removed = registry.remove("audio")
    cleanup = 1.0 if removed and registry.get("audio") is None else 0.0
    return {
        "INT": 0.0 if geo_before == geo_after else 1.0,
        "ZCR_new": float(zcr_new),
        "cleanup": float(cleanup),
        "RBE": float(rollback_exactness(geo_before, geo_after)),
    }


def run_h4_version_skew(seed: int = 0) -> dict:
    """Register two vision versions, route to the new one, detect disagreement."""

    scene = make_scene(50, dim=128, seed=seed)
    obs = scene.observations()
    codebook = scene.codebook
    registry = ModuleRegistry(public_dim=128, probe=obs[:8])
    v1 = make_quality_expert("Q0", codebook, modality="vision", seed=seed)
    v2 = make_quality_expert("Q1", codebook, modality="vision", seed=seed)
    registry.register(v1, probe=obs[:8])
    before = registry.behavior_hash("vision", probe=obs[:8])

    swap = registry.swap(v2, probe=obs[:8])
    routing = 1.0 if registry.get("vision").version == v2.version else 0.0
    p1 = _predictions(v1, codebook, obs)
    p2 = _predictions(v2, codebook, obs)
    disagreement = float(np.mean([a != b for a, b in zip(p1, p2)]))
    detected = 1.0 if disagreement > 0.0 else 0.0

    registry.rollback()
    rbe = rollback_exactness(before, registry.behavior_hash("vision", probe=obs[:8]))
    return {
        "RCR": routing,
        "CDR": detected,
        "RBE": float(rbe),
        "SML": float(swap.latency_ms),
        "disagreement": disagreement,
    }


def run_h5_hot_swap_under_load(
    seed: int = 0,
    n_requests: int = 200,
    swap_at: int = 100,
) -> dict:
    """Swap mid-stream and count dropped/failed requests."""

    scene = make_scene(50, dim=128, seed=seed)
    obs = scene.observations()
    codebook = scene.codebook
    registry = ModuleRegistry(public_dim=128, probe=obs[:8])
    registry.register(make_quality_expert("Q0", codebook, modality="vision", seed=seed), probe=obs[:8])
    hash_probe = obs[:8]
    before = registry.behavior_hash("vision", probe=hash_probe)

    failures = 0
    swaps = 0
    request_probe = obs[:1]
    for i in range(int(n_requests)):
        if i == int(swap_at):
            result = registry.swap(
                make_quality_expert("Q2", codebook, modality="vision", seed=seed),
                probe=request_probe,
            )
            swaps += 1
            failures += int(not result.success)
        try:
            registry.get("vision").encode(request_probe)
        except Exception:
            failures += 1

    registry.rollback()
    rbe = rollback_exactness(before, registry.behavior_hash("vision", probe=hash_probe))
    return {
        "DT": float(failures / max(1, n_requests)),
        "SSR": float(1.0 if swaps and failures == 0 else 0.0),
        "RBE": float(rbe),
        "requests": int(n_requests),
        "failures": int(failures),
    }


def kb_state_trace() -> dict:
    """A short KB operation trace for Figure 2."""

    kb = VersionedKnowledgeBase()
    steps = [{"step": 0, "label": "init", "hash": kb.state_hash(), "facts": 0}]

    def facts_n() -> int:
        return sum(1 for values in kb.store.assertions.values() if values)

    kb.assert_fact("cup", "material", "glass", source="s1")
    steps.append({"step": 1, "label": "assert cup.glass", "hash": kb.state_hash(), "facts": facts_n()})
    target_index = kb.snapshot()["index"]
    target_hash = kb.state_hash()

    kb.assert_fact("block", "color", "red", source="s1")
    steps.append({"step": 2, "label": "assert block.red", "hash": kb.state_hash(), "facts": facts_n()})
    kb.update("cup", "material", "metal", source="s1")
    steps.append({"step": 3, "label": "update cup.metal", "hash": kb.state_hash(), "facts": facts_n()})
    kb.retract("block", "color", "s1")
    steps.append({"step": 4, "label": "retract block.red", "hash": kb.state_hash(), "facts": facts_n()})
    kb.rollback(target_index)
    steps.append({"step": 5, "label": "rollback v1", "hash": kb.state_hash(), "facts": facts_n()})

    return {
        "steps": steps,
        "target_index": int(target_index),
        "rollback_exact": float(rollback_exactness(target_hash, kb.state_hash())),
    }
