"""CLEVR-CoGenT adapter (official data).

Expected layout after extracting ``CLEVR_CoGenT_v1.0.zip``:

```
data/public/clevr_cogent/
  scenes/CLEVR_CoGenT_TrainA_R1_scenes.json
  scenes/CLEVR_CoGenT_TrainB_R1_scenes.json
  scenes/CLEVR_CoGenT_ValA_scenes.json
  scenes/CLEVR_CoGenT_ValB_scenes.json
  images/trainA/*.png
  images/trainB/*.png
  images/valA/*.png
  images/valB/*.png
```

The adapter is tolerant of file-name variants. It yields relation episodes
(left/right/front/behind) from the 3D coordinates in each scene. The smoke path
uses tiny synthetic scenes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from pss.eval.suite.adapter import BenchmarkAdapter, BenchmarkUnavailable
from pss.eval.suite.metrics import accuracy
from pss.eval.suite.schema import EntitySpec, EvalEpisode, EvalResult, Prediction


class CLEVRCoGenTAdapter(BenchmarkAdapter):
    name = "clevr_cogent"
    layer = "relation"
    license = "CC-BY-4.0"
    homepage = "https://cs.stanford.edu/people/jcjohns/clevr/"

    def _scene_files(self) -> list[Path]:
        if self.root is None:
            raise BenchmarkUnavailable(
                "CLEVR-CoGenT not found. Extract CLEVR_CoGenT_v1.0.zip under "
                "data/public/clevr_cogent/."
            )
        root = Path(self.root)
        candidates = sorted(
            set(root.glob("scenes/*.json")) | set(root.glob("**/scenes/*.json"))
        )
        if not candidates:
            raise BenchmarkUnavailable(f"no scenes/*.json under {root}")
        return candidates

    def _split_name(self, path: Path) -> str:
        name = path.stem.lower()
        for split in ("traina", "trainb", "vala", "valb"):
            if split in name:
                return split
        return "unknown"

    def _image_path(self, split: str, image_index: int, filename: str | None = None) -> Path | None:
        if self.root is None:
            return None
        root = Path(self.root)
        if getattr(self, "_image_dirs", None) is None:
            dirs: dict[str, Path] = {}
            for images_root in root.glob("**/images"):
                if not images_root.is_dir():
                    continue
                for child in images_root.iterdir():
                    if child.is_dir() and child.name not in dirs:
                        dirs[child.name.lower()] = child
            self._image_dirs = dirs
        names = [filename] if filename else []
        names += [
            f"CLEVR_{split}_{image_index:06d}.png",
            f"CLEVR_{split}_{image_index:06d}.jpg",
            f"{image_index:06d}.png",
            f"{image_index:06d}.jpg",
        ]
        directory = self._image_dirs.get(split.lower())
        for name in names:
            if not name:
                continue
            path = (directory / name) if directory is not None else None
            if path is not None and path.exists():
                return path
        return None

    def prepare(self) -> None:
        self._scene_files()

    def episodes(self, split: str = "test", max_scenes: int | None = None) -> Iterator[EvalEpisode]:
        count = 0
        if split == "test":
            allowed = {"vala", "valb"}
        elif split == "all":
            allowed = None
        else:
            allowed = {split.lower()}
        for path in self._scene_files():
            split_name = self._split_name(path)
            if allowed is not None and split_name not in allowed:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            scenes = payload.get("scenes", payload if isinstance(payload, list) else [])
            for scene in scenes:
                if max_scenes is not None and count >= max_scenes:
                    return
                objects = scene.get("objects", [])
                if len(objects) < 2:
                    continue
                image_index = int(scene.get("image_index", count))
                image = self._image_path(
                    split_name, image_index, scene.get("image_filename")
                )
                entity_specs = [
                    EntitySpec(
                        id=f"obj_{i}",
                        category=obj.get("shape"),
                        pos=np.asarray(obj.get("3d_coords", [0, 0, 0]), dtype=float),
                        attributes={
                            "size": obj.get("size"),
                            "material": obj.get("material"),
                            "color": obj.get("color"),
                            "shape": obj.get("shape"),
                            "pixel_coords": list(obj.get("pixel_coords", [0, 0, 0])),
                        },
                    )
                    for i, obj in enumerate(objects)
                ]
                predicate, pair = _true_pair(scene)
                if predicate is None:
                    continue
                i, j = pair
                true_rel = predicate
                false_rel = _false_relation(true_rel)
                for answer, relation in ((True, true_rel), (False, false_rel)):
                    yield EvalEpisode(
                        id=f"clevr_{scene.get('image_index', count)}_{relation}_{int(answer)}",
                        benchmark=self.name,
                        layer=self.layer,
                        split=split_name,
                        capabilities={"entities", "relations"},
                        entities=entity_specs,
                        expected={"answer": answer, "relation": relation},
                        query=f"Object {i} is {relation} object {j}?",
                        meta={
                            "objects": [
                                {
                                    "pos": list(map(float, o.get("3d_coords", [0, 0, 0]))),
                                    "pixel": list(map(float, o.get("pixel_coords", [0, 0, 0]))),
                                }
                                for o in objects
                            ],
                            "pair": [i, j],
                            "image": None if image is None else str(image),
                            "scene": scene.get("image_index", count),
                        },
                    )
                count += 1

    def evaluate(self, predictions: list[Prediction]) -> list[EvalResult]:
        overall = [bool(p.output.get("answer")) == bool(p.meta["expected"]["answer"]) for p in predictions]
        results = [
            EvalResult(self.name, self.layer, "all", "relation_accuracy", accuracy(overall, [True] * len(overall)), len(overall))
        ]
        for split in ("traina", "trainb", "vala", "valb"):
            vals = [
                bool(p.output.get("answer")) == bool(p.meta["expected"]["answer"])
                for p in predictions
                if p.meta.get("split") == split
            ]
            if vals:
                results.append(
                    EvalResult(self.name, self.layer, split, "relation_accuracy", accuracy(vals, [True] * len(vals)), len(vals))
                )
        return results

    def sample_episodes(self) -> list[EvalEpisode]:
        episodes = []
        for k, (split, xs) in enumerate(
            (("vala", (0.0, 2.0)), ("valb", (2.0, 0.0)))
        ):
            objects = [
                {"pos": [xs[0], 0.0, 0.0], "pixel": [xs[0] * 100, 100.0, 5.0]},
                {"pos": [xs[1], 0.0, 0.0], "pixel": [xs[1] * 100, 100.0, 5.0]},
            ]
            entities = [
                EntitySpec(f"obj_{i}", "cube", np.asarray(o["pos"], dtype=float))
                for i, o in enumerate(objects)
            ]
            true_rel = _relation(objects[0]["pos"], objects[1]["pos"])
            false_rel = _false_relation(true_rel)
            for answer, relation in ((True, true_rel), (False, false_rel)):
                episodes.append(
                    EvalEpisode(
                        id=f"clevr_sample_{k}_{relation}_{int(answer)}",
                        benchmark=self.name,
                        layer=self.layer,
                        split=split,
                        capabilities={"entities", "relations"},
                        entities=entities,
                        expected={"answer": answer, "relation": relation},
                        query=f"Object 0 is {relation} object 1?",
                        meta={"objects": objects, "pair": [0, 1], "split": split},
                    )
                )
        return episodes


def _true_pair(scene) -> tuple[str | None, tuple[int, int]]:
    relationships = scene.get("relationships") or {}
    objects = scene.get("objects", [])
    for i in range(len(objects)):
        for predicate in ("left", "right", "front", "behind"):
            rows = relationships.get(predicate)
            if rows is None or i >= len(rows):
                continue
            for j in rows[i]:
                j = int(j)
                if j != i and j < len(objects):
                    # relationships[pred][i] lists objects that are *pred of i*,
                    # so the subject is j and the object is i.
                    return predicate, (j, i)
    if len(objects) >= 2:
        return _relation(objects[0]["3d_coords"], objects[1]["3d_coords"]), (0, 1)
    return None, (0, 0)


def _relation(a, b, eps: float = 1e-3) -> str:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    if ax < bx - eps:
        return "left"
    if ax > bx + eps:
        return "right"
    if ay < by - eps:
        return "front"
    return "behind"


def _false_relation(relation: str) -> str:
    return {
        "left": "right",
        "right": "left",
        "front": "behind",
        "behind": "front",
    }[relation]
