"""Adapter for the E5 contract-free flat runtime.

This wraps :mod:`pss.property.nonpss` -- the negative architecture used by
claim-gap E5 -- behind the standalone :class:`~pss_eval.protocol.SystemUnderTest`
interface.  The adapter deliberately preserves the flat baseline's defining
properties:

* every module swap is accepted, including wrong-dimension, non-normalised and
  NaN modules;
* there is no rollback and no versioned routing;
* there is no audit log;
* knowledge is one flat last-writer-wins dictionary.

It is a *negative architecture*, not a competitive model.  Its purpose is to
show that the property suite separates a contract-free system from PSS.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from pss.experts import (
    NaNExpert,
    NonNormalizedExpert,
    WrongDimExpert,
    make_quality_expert,
)
from pss.observations import DEFAULT_CATEGORIES, make_scene
from pss.property.nonpss import FlatKB, FlatRegistry

from ..protocol import Observation, Prediction, StateSnapshot, state_digest

__all__ = ["FlatNonPSSAdapter"]


class FlatNonPSSAdapter:
    """``SystemUnderTest`` view of the E5 flat baseline."""

    name = "e5-flat"

    def __init__(self, dim: int = 32, seed: int = 0) -> None:
        self.dim = int(dim)
        self.reset(seed=seed)

    # -- lifecycle ---------------------------------------------------------

    def reset(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self._scene = make_scene(
            len(DEFAULT_CATEGORIES), dim=self.dim, seed=self.seed
        )
        self._codebook = self._scene.codebook
        self.registry = FlatRegistry(probe=self._scene.observations()[:8])
        self._vision = make_quality_expert(
            "Q0", self._codebook, modality="vision", seed=self.seed
        )
        self._geometry = make_quality_expert(
            "Q0", self._codebook, modality="geometry", seed=self.seed
        )
        self.registry.register(self._vision)
        self.registry.register(self._geometry)
        self.kb = FlatKB()
        self._last: Observation | None = None
        self._n_ingested = 0

    # -- core protocol -----------------------------------------------------

    def ingest(self, observation: Observation) -> None:
        if observation.features.shape[0] != self.dim:
            raise ValueError(
                f"{self.name} expects dim {self.dim}, got {observation.features.shape[0]}"
            )
        self._last = observation
        self._n_ingested += 1
        # The flat KB has no per-observation semantics; recording a fact is the
        # closest faithful analogue and makes state growth visible.
        self.kb.assert_fact(observation.observation_id, "seen", "1", source="stream")

    def step(self) -> Prediction:
        if self._last is None:
            raise RuntimeError("step() called before ingest()")
        # The E5 stubs encode from a category, not from raw features.  Derive a
        # deterministic pseudo-category from the feature vector so the active
        # vision expert is still the component that produces the embedding.
        pseudo_category = self._codebook.predict(self._last.features)
        encoded = self._vision.encode([{"gt_category": pseudo_category}])[0]
        label = self._codebook.predict(encoded)
        return Prediction(
            observation_id=self._last.observation_id,
            embedding=np.asarray(encoded, dtype=np.float64),
            label=label,
        )

    def state(self) -> StateSnapshot:
        modules = {
            modality: {
                "version": str(getattr(expert, "version", "?")),
                "dim": int(getattr(expert, "dim", 0)),
            }
            for modality, expert in sorted(self.registry.modules.items())
        }
        return {
            "system": self.name,
            "seed": self.seed,
            "dim": self.dim,
            "n_ingested": self._n_ingested,
            "last_id": self._last.observation_id if self._last is not None else None,
            "modules": modules,
            "swaps": int(self.registry.swaps),
            "kb_facts": {
                f"{entity}\x1f{attribute}": str(value)
                for (entity, attribute), value in self.kb.facts.items()
            },
            "kb_changes": int(self.kb.change_count()),
        }

    # -- optional capabilities (implemented with E5's negative behaviour) --

    def snapshot(self) -> str:
        # No real snapshots; the token is only an identifier.
        return state_digest(self.state())

    def rollback(self, token: str) -> bool:
        # The flat registry/KB has no rollback.
        return False

    def swap(self, module: Mapping[str, Any]) -> bool:
        expert = self._make_module(module)
        result = self.registry.swap(expert)
        # E5 accepts every module; the ``valid`` flag is intentionally ignored.
        return bool(result.success)

    def audit(self) -> list[dict[str, Any]]:
        # No audit trail.
        return []

    # -- helpers -----------------------------------------------------------

    def _make_module(self, module: Mapping[str, Any]):
        modality = str(module.get("modality", "vision"))
        kind = str(module.get("kind", "quality"))
        if kind == "invalid_dim":
            return WrongDimExpert(self._codebook)
        if kind == "invalid_norm":
            return NonNormalizedExpert(self._codebook)
        if kind == "invalid_nan":
            return NaNExpert(self._codebook)
        quality = str(module.get("quality", "Q2"))
        return make_quality_expert(
            quality, self._codebook, modality=modality, seed=self.seed
        )
