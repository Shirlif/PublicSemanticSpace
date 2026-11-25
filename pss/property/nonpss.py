"""Contract-free baseline for the property suite (claim-gap E5).

The property metrics are only meaningful if an architecture *without* the
contract fails them.  This module builds the simplest contract-free baseline:

- ``FlatRegistry``: one module dict per modality, no admission validation, no
  rollback, no per-modality isolation, no versioned routing;
- ``FlatKB``: one fact dict, no versions, no audit, no rollback, no
  justification propagation;
- no budget controller.

It is not a competitive model; it is the negative architecture the suite must
separate from PSS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pss.experts import (
    CategoryCodebook,
    NaNExpert,
    NonNormalizedExpert,
    WrongDimExpert,
    make_quality_expert,
)
from pss.observations import make_scene
from pss.reasoning.planner import AStarPlanner
from pss.world_model.actions import initial_laundry_facts, laundry_library


@dataclass
class FlatSwap:
    success: bool
    modality: str
    version: str


class FlatRegistry:
    def __init__(self, probe=None):
        self.probe = probe
        self.modules: dict[str, object] = {}
        self.swaps = 0

    def register(self, expert, probe=None) -> None:
        self.modules[expert.modality] = expert

    def get(self, modality: str):
        return self.modules.get(modality)

    def swap(self, expert, probe=None) -> FlatSwap:
        # No contract validation, no stack, no rollback.
        self.modules[expert.modality] = expert
        self.swaps += 1
        return FlatSwap(True, expert.modality, getattr(expert, "version", "?"))

    def rollback(self) -> FlatSwap:
        return FlatSwap(False, "", "no-rollback")


class FlatKB:
    def __init__(self):
        self.facts: dict[tuple[str, str], str] = {}
        self.changes = 0

    def assert_fact(self, entity, attribute, value, source="flat", priority=0, reason=""):
        # Last writer wins; no history, no audit, no conflict set.
        self.facts[(entity, attribute)] = value
        self.changes += 1

    def update(self, entity, attribute, value, source="flat", reason=""):
        self.assert_fact(entity, attribute, value, source=source, reason=reason)

    def retract(self, entity, attribute, source="flat", reason=""):
        self.facts.pop((entity, attribute), None)
        self.changes += 1

    def resolve(self, entity, attribute):
        return self.facts.get((entity, attribute))

    def conflicts(self):
        return {}

    def rollback(self, version):
        return False

    def audit_count(self) -> int:
        return 0

    def change_count(self) -> int:
        return int(self.changes)


def _predict(codebook: CategoryCodebook, vectors: np.ndarray) -> list[str]:
    return [codebook.predict(v) for v in np.asarray(vectors)]


def _class_means(vectors: np.ndarray, observations, classes: list[str]) -> np.ndarray:
    """Per-class mean vectors, the shape ``CategoryCodebook.from_vectors`` wants."""
    vectors = np.asarray(vectors, dtype=np.float64)
    rows = []
    for category in classes:
        idx = [i for i, obs in enumerate(observations) if obs.gt_category == category]
        if not idx:
            idx = list(range(len(vectors)))
        rows.append(vectors[idx].mean(axis=0))
    return np.stack(rows)


def run_nonpss_baseline(seed: int = 0, dim: int = 128) -> dict:
    scene = make_scene(200, dim=dim, seed=seed)
    obs = scene.observations()
    classes = scene.codebook.categories

    registry = FlatRegistry(probe=obs[:8])
    vision = make_quality_expert("Q0", scene.codebook, modality="vision", seed=seed)
    geometry = make_quality_expert("Q0", scene.codebook, modality="geometry", seed=seed)
    registry.register(vision)
    registry.register(geometry)

    # Safety: invalid modules are accepted (no admission contract).
    invalid = [WrongDimExpert(scene.codebook), NonNormalizedExpert(scene.codebook), NaNExpert(scene.codebook)]
    invalid_accepted = sum(1 for bad in invalid if registry.swap(bad).success)
    srr = 0.0 if invalid_accepted == len(invalid) else 1.0

    # Swap all quality levels; a flat registry never rejects.
    swap_success = []
    for level in ("Q0", "Q1", "Q2", "Q3", "Q4"):
        candidate = make_quality_expert(level, scene.codebook, modality="vision", seed=seed)
        swap_success.append(registry.swap(candidate).success)

    # Interference: a single shared codebook is rebuilt from the active vision
    # expert, so swapping vision changes geometry predictions.
    geo_preds_before = _predict(
        CategoryCodebook.from_vectors(
            classes, _class_means(vision.encode(obs), obs, classes)
        ),
        geometry.encode(obs),
    )
    swapped = make_quality_expert("Q4", scene.codebook, modality="vision", seed=seed)
    geo_preds_after = _predict(
        CategoryCodebook.from_vectors(
            classes, _class_means(swapped.encode(obs), obs, classes)
        ),
        geometry.encode(obs),
    )
    int_value = 0.0 if geo_preds_before == geo_preds_after else 1.0

    # No rollback, no versioned routing.
    rbe = 0.0 if not registry.rollback().success else 1.0
    rcr = 0.0

    # Flat KB: updates are local, but there is no propagation/audit/rollback.
    kb = FlatKB()
    kb.assert_fact("cup", "material", "glass", source="vision")
    kb.assert_fact("block", "color", "red", source="vision")
    unrelated_before = kb.resolve("block", "color")
    kb.update("cup", "material", "metal", source="runtime")
    loc = 0.0 if kb.resolve("block", "color") == unrelated_before else 1.0
    kb_rollback = 0.0 if not kb.rollback(0) else 1.0
    aud = 0.0
    pcr = 0.0  # no derived-justification propagation
    usr = 1.0  # flat updates always "succeed"

    # No budget controller: an unbounded search exceeds the PSS bound.
    unreachable = frozenset({("never", "happens", "yes")})
    result = AStarPlanner(laundry_library(), max_expansions=10_000).plan(
        initial_laundry_facts(), unreachable
    )
    bar = 0.0 if int(result.nodes) > 20 else 1.0
    mon = 0.0  # no anytime curve

    return {
        "seed": int(seed),
        "dim": int(dim),
        "metrics": {
            "SSR": float(np.mean(swap_success)),
            "SRR": srr,
            "INT": int_value,
            "RBE": rbe,
            "RCR": rcr,
            "USR": usr,
            "LOC": loc,
            "PCR": pcr,
            "AUD": aud,
            "RBE-KB": kb_rollback,
            "BAR": bar,
            "MON": mon,
        },
        "details": {
            "invalid_injected": len(invalid),
            "invalid_accepted": int(invalid_accepted),
            "planner_nodes": int(result.nodes),
            "kb_changes": kb.change_count(),
        },
    }
