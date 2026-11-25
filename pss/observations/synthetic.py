"""Structured synthetic observations for the L0 runtime sandbox.

L0 removes perception from the loop: observations are generated with known
ground truth and fed directly to the runtime.  Expert stubs then encode them,
so any change in a property metric is attributable to the architecture, not to
a backbone.

The returned objects are the same ``pss.memory.perception.Observation`` the
tracker consumes, which keeps the sandbox honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pss.experts.stubs import CategoryCodebook
from pss.memory.perception import Observation

DEFAULT_CATEGORIES = ("cup", "block", "bottle", "plate", "bowl", "mug", "box", "can")
DEFAULT_MODALITIES = ("vision", "tactile", "geometry")


@dataclass
class SyntheticEntity:
    id: str
    category: str
    pos: np.ndarray
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    attributes: dict = field(default_factory=dict)
    visible: bool = True


@dataclass
class SyntheticScene:
    entities: list[SyntheticEntity]
    codebook: CategoryCodebook
    relations: list[tuple[str, str, str]] = field(default_factory=list)
    facts: dict[tuple[str, str], str] = field(default_factory=dict)
    dim: int = 128
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def _embedding(self, entity: SyntheticEntity, modality: str) -> np.ndarray:
        base = self.codebook.vector(entity.category).astype(np.float64)
        noise = self._rng.normal(scale=0.02, size=self.dim)
        vec = base + noise
        return (vec / np.linalg.norm(vec)).astype(np.float32)

    def observations(
        self,
        frame: int = 0,
        visible_ids: set[str] | None = None,
        modalities=DEFAULT_MODALITIES,
    ) -> list[Observation]:
        out: list[Observation] = []
        for entity in self.entities:
            if visible_ids is not None and entity.id not in visible_ids:
                continue
            if not entity.visible:
                continue
            raw = {m: self._embedding(entity, m) for m in modalities}
            out.append(
                Observation(
                    gt_id=entity.id,
                    pos=np.asarray(entity.pos, dtype=float).copy(),
                    quat=np.asarray(entity.quat, dtype=float).copy(),
                    appearance=raw.get("vision", self._embedding(entity, "vision")),
                    raw=raw,
                    confidence=0.95,
                    category=entity.category,
                    frame=int(frame),
                    gt_category=entity.category,
                    category_score=1.0,
                    unknown=False,
                    conflict=False,
                    modality_reliability={m: 1.0 for m in modalities},
                )
            )
        return out

    def entity_ids(self) -> list[str]:
        return [e.id for e in self.entities]


def make_scene(
    n_entities: int = 5,
    categories=None,
    dim: int = 128,
    seed: int = 0,
    spacing: float = 0.15,
) -> SyntheticScene:
    categories = tuple(categories or DEFAULT_CATEGORIES)
    codebook = CategoryCodebook(categories, dim=dim, seed=seed)
    rng = np.random.default_rng(seed + 17)
    entities: list[SyntheticEntity] = []
    side = max(1, int(np.ceil(np.sqrt(n_entities))))
    for i in range(int(n_entities)):
        row, col = divmod(i, side)
        pos = np.array(
            [col * spacing, row * spacing, 0.5 + 0.01 * (i % 3)], dtype=float
        )
        category = categories[i % len(categories)]
        entities.append(
            SyntheticEntity(
                id=f"e{i:03d}",
                category=category,
                pos=pos,
                attributes={"index": i, "seed": int(rng.integers(0, 10_000))},
            )
        )
    return SyntheticScene(
        entities=entities,
        codebook=codebook,
        dim=int(dim),
        seed=int(seed),
    )


def make_episode(
    n_entities: int = 5,
    n_frames: int = 10,
    occlusion_prob: float = 0.0,
    dim: int = 128,
    seed: int = 0,
) -> tuple[SyntheticScene, list[list[Observation]]]:
    """Return a scene and a list of per-frame observation lists.

    Occlusion hides a random subset of entities per frame.  At least one
    entity is always visible so a frame is never empty.
    """

    scene = make_scene(n_entities=n_entities, dim=dim, seed=seed)
    rng = np.random.default_rng(seed + 101)
    frames: list[list[Observation]] = []
    for frame in range(int(n_frames)):
        ids = [e.id for e in scene.entities]
        if occlusion_prob > 0.0 and len(ids) > 1:
            keep = [eid for eid in ids if rng.random() >= occlusion_prob]
            if not keep:
                keep = [ids[int(rng.integers(0, len(ids)))]]
        else:
            keep = ids
        # Small motion so the tracker has a non-trivial association problem.
        for entity in scene.entities:
            entity.pos = entity.pos + rng.normal(scale=0.002, size=3)
        frames.append(scene.observations(frame=frame, visible_ids=set(keep)))
    return scene, frames


def make_large_scene(
    n_entities: int = 200,
    n_relations: int = 1000,
    dim: int = 128,
    seed: int = 0,
) -> SyntheticScene:
    """A large scene for active-set and relation-cost pressure tests."""

    scene = make_scene(n_entities=n_entities, dim=dim, seed=seed)
    rng = np.random.default_rng(seed + 211)
    ids = [e.id for e in scene.entities]
    predicates = ("left_of", "right_of", "near", "on_top_of", "inside")
    relations: list[tuple[str, str, str]] = []
    if len(ids) >= 2:
        for _ in range(int(n_relations)):
            a, b = rng.choice(len(ids), size=2, replace=False)
            relations.append((ids[int(a)], str(rng.choice(predicates)), ids[int(b)]))
    scene.relations = relations
    return scene
