"""Identity and decomposition rules (plan 7.4).

Two branches:

1. **Witnessed event**: the tracker saw the parent break. The rule calls
   ``SceneGraph.fragment`` and records a deterministic hypothesis.
2. **Result-only**: only the shards are observed. The rule fuses geometric
   fit and prototype similarity into a confidence-scored hypothesis. It is
   applied only above ``accept_threshold`` and can be retracted later.

The rule never claims certainty it does not have. A weak hypothesis stays in
``hypotheses`` without mutating the scene graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from pss.prototypes import PrototypeBank, recognize

from .scene_graph import SceneGraph, Track


@dataclass
class FragmentHypothesis:
    id: str
    parent_id: str
    shard_ids: list[str]
    confidence: float
    status: str  # deterministic | probabilistic | candidate | uncertain | retracted
    evidence: dict = field(default_factory=dict)
    frame: int = 0
    applied: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class DecompositionRule:
    def __init__(
        self,
        graph: SceneGraph,
        bank: PrototypeBank | None = None,
        accept_threshold: float = 0.72,
        uncertain_threshold: float = 0.45,
        max_parent_distance: float = 0.25,
        expected_extent: float = 0.20,
    ):
        self.graph = graph
        self.bank = bank
        self.accept_threshold = float(accept_threshold)
        self.uncertain_threshold = float(uncertain_threshold)
        self.max_parent_distance = float(max_parent_distance)
        self.expected_extent = float(expected_extent)
        self.hypotheses: dict[str, FragmentHypothesis] = {}
        self._next = 1

    # ------------------------------------------------------------------
    # Branch 1: the event was observed.
    # ------------------------------------------------------------------
    def witness(self, parent_id: str, children: list[Track], t: int) -> FragmentHypothesis:
        parent = self.graph.get(parent_id)
        if parent is None:
            raise KeyError(f"unknown parent: {parent_id}")
        self.graph.fragment(parent_id, children, t)
        parent.provenance.update(
            {
                "decomposition_status": "deterministic",
                "fragment_frame": int(t),
            }
        )
        for child in children:
            child.provenance.update(
                {
                    "relation": "part-of-fracture",
                    "decomposition_status": "deterministic",
                }
            )
        hypothesis = FragmentHypothesis(
            id=self._new_id(),
            parent_id=parent_id,
            shard_ids=[c.id for c in children],
            confidence=1.0,
            status="deterministic",
            evidence={"branch": "witnessed_event", "frame": int(t)},
            frame=int(t),
            applied=True,
        )
        parent.provenance["fragment_hypothesis"] = hypothesis.id
        self.hypotheses[hypothesis.id] = hypothesis
        return hypothesis

    # ------------------------------------------------------------------
    # Branch 2: only the result is observed.
    # ------------------------------------------------------------------
    def infer_result_only(
        self,
        parent: Track,
        shards: list[Track],
        t: int,
        expected_extent: float | None = None,
    ) -> FragmentHypothesis:
        evidence = self._evidence(parent, shards, expected_extent)
        confidence = 0.55 * evidence["geometry"] + 0.45 * evidence["prototype"]
        if confidence >= self.accept_threshold:
            status = "probabilistic"
        elif confidence >= self.uncertain_threshold:
            status = "candidate"
        else:
            status = "uncertain"

        hypothesis = FragmentHypothesis(
            id=self._new_id(),
            parent_id=parent.id,
            shard_ids=[s.id for s in shards],
            confidence=float(confidence),
            status=status,
            evidence={
                "branch": "result_only",
                "frame": int(t),
                **evidence,
            },
            frame=int(t),
        )
        self.hypotheses[hypothesis.id] = hypothesis

        if status == "probabilistic":
            self._apply(hypothesis, parent, shards, t)
        return hypothesis

    def retract(
        self,
        hypothesis_id: str,
        reason: str,
        t: int | None = None,
    ) -> FragmentHypothesis:
        """Overturn a hypothesis. Restores the parent if it was applied."""
        hypothesis = self.hypotheses.get(hypothesis_id)
        if hypothesis is None:
            raise KeyError(f"unknown hypothesis: {hypothesis_id}")
        parent = self.graph.get(hypothesis.parent_id)
        if hypothesis.applied and parent is not None:
            parent.status = parent.provenance.pop("previous_status", "active")
            parent.provenance.pop("decomposition_status", None)
            parent.provenance.pop("fragment_hypothesis", None)
            parent.provenance["decomposition_retracted"] = hypothesis_id
            for shard_id in hypothesis.shard_ids:
                shard = self.graph.get(shard_id)
                if shard is None:
                    continue
                shard.parent = None
                shard.provenance["decomposition_status"] = "retracted"
                shard.provenance["decomposition_retracted"] = hypothesis_id
                shard.provenance.pop("fractured_from", None)
                shard.provenance.pop("relation", None)
                shard.provenance.pop("decomposition_hypothesis", None)
        hypothesis.status = "retracted"
        hypothesis.confidence = 0.0
        hypothesis.applied = False
        hypothesis.evidence["retracted_reason"] = reason
        if t is not None:
            hypothesis.evidence["retracted_frame"] = int(t)
        return hypothesis

    def revise(
        self,
        hypothesis_id: str,
        parent: Track,
        shards: list[Track],
        t: int,
        expected_extent: float | None = None,
    ) -> FragmentHypothesis:
        """Re-score a hypothesis with new evidence, promoting or retracting it."""
        old = self.hypotheses.get(hypothesis_id)
        if old is None:
            raise KeyError(f"unknown hypothesis: {hypothesis_id}")
        if old.applied:
            self.retract(hypothesis_id, reason="superseded_by_revision", t=t)
        revised = self.infer_result_only(parent, shards, t, expected_extent)
        revised.evidence["revised_from"] = old.id
        return revised

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------
    def _evidence(
        self,
        parent: Track,
        shards: list[Track],
        expected_extent: float | None,
    ) -> dict:
        if not shards:
            return {"geometry": 0.0, "prototype": 0.0, "count": 0, "shard_count": 0}
        parent_pos = np.asarray(parent.pos, dtype=float)
        distances = np.array(
            [float(np.linalg.norm(s.pos - parent_pos)) for s in shards], dtype=float
        )
        mean_distance = float(np.mean(distances))
        within = float(np.exp(-mean_distance / max(self.max_parent_distance, 1e-9)))

        if len(shards) >= 2:
            positions = np.stack([np.asarray(s.pos, dtype=float) for s in shards])
            spread = float(
                np.max(
                    np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
                )
            )
        else:
            spread = 0.0
        extent = float(expected_extent or self.expected_extent)
        compactness = float(np.exp(-spread / max(extent, 1e-9)))
        count_score = min(1.0, len(shards) / 2.0)
        geometry = 0.45 * within + 0.35 * compactness + 0.20 * count_score

        prototype = self._prototype_similarity(parent, shards)
        return {
            "geometry": float(geometry),
            "prototype": float(prototype),
            "within": within,
            "compactness": compactness,
            "spread": spread,
            "mean_distance": mean_distance,
            "count": count_score,
            "shard_count": len(shards),
        }

    def _prototype_similarity(self, parent: Track, shards: list[Track]) -> float:
        cosines = []
        for shard in shards:
            modalities = [
                m
                for m in parent.raw
                if m in shard.raw
                and np.linalg.norm(parent.raw[m]) > 0
                and np.linalg.norm(shard.raw[m]) > 0
            ]
            if modalities:
                cosines.append(
                    float(
                        np.mean(
                            [
                                np.dot(_unit(parent.raw[m]), _unit(shard.raw[m]))
                                for m in modalities
                            ]
                        )
                    )
                )
        cosine = float(np.mean(cosines)) if cosines else 0.0
        cosine = float(np.clip(cosine, 0.0, 1.0))

        category_match = None
        if self.bank is not None and parent.category is not None:
            matches = []
            for shard in shards:
                if not shard.raw:
                    continue
                result = recognize(shard.raw, self.bank, use_tree=False)
                if result.ranked:
                    matches.append(result.ranked[0][0] == parent.category)
            if matches:
                category_match = float(np.mean(matches))
        if category_match is None:
            return cosine
        return 0.6 * cosine + 0.4 * category_match

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def _apply(
        self,
        hypothesis: FragmentHypothesis,
        parent: Track,
        shards: list[Track],
        t: int,
    ) -> None:
        previous_status = parent.status
        parent.status = "fragmented"
        parent.provenance.update(
            {
                "decomposition_status": "probabilistic",
                "fragment_frame": int(t),
                "fragment_hypothesis": hypothesis.id,
                "previous_status": previous_status,
            }
        )
        for shard in shards:
            shard.parent = parent.id
            shard.provenance.update(
                {
                    "fractured_from": parent.id,
                    "frame": int(t),
                    "relation": "part-of-fracture",
                    "decomposition_status": "probabilistic",
                    "decomposition_hypothesis": hypothesis.id,
                }
            )
            if self.graph.get(shard.id) is None:
                self.graph.add(shard)
        hypothesis.applied = True

    def _new_id(self) -> str:
        hypothesis_id = f"frag_{self._next:04d}"
        self._next += 1
        return hypothesis_id
