"""F10 spike: replay ObjectFolder 2.0 object-level data through ``pss_eval``.

What this module is
-------------------
ObjectFolder 2.0 is a public, CC-BY-4.0 dataset of implicit object
representations.  Its rendered vision/tactile images are *not* raw robot
captures: there are no scenes, no 6-D poses, no depth, no segmentation, and no
synchronized multi-frame interaction stream.  This module therefore does **not**
claim L2 identity/tracking or sim2real transfer.

What it does is the honest subset:

* extract DINOv2 features from the rendered vision and tactile images with the
  existing :mod:`pss.rd2.objectfolder` helpers;
* measure object-level cross-modal binding (R@1/R@5, Ridge half-split, a
  permutation test);
* put the real features on the same :class:`pss_eval.fixtures.Stream` and
  property path used by the synthetic bundles, and run the PSS runtime adapter
  plus a passthrough adapter through it;
* add a ``cross_modal_binding`` property that computes object-level R@1 from a
  system's own embeddings, so the number is produced *inside* the eval path.

The standard ``identity_retention`` property is deliberately not used here:
it expects a temporal track stream and harness-side instance labels.  With no
poses every detection collapses to one track (we verified ``1`` entity for
``8`` detections), so the property would report a degenerate 1.0.  That is not
evidence of binding.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss.rd2.objectfolder import (
    _encode_paths,
    _l2,
    _load_encoder,
    _retrieval,
    _ridge_retrieval,
)
from pss_eval.adapters.pss_runtime import PSSRuntimeAdapter
from pss_eval.fixtures import LabeledSample, Stream
from pss_eval.properties import FAIL, PASS, SKIP, PropertyResult, extended_properties
from pss_eval.protocol import Observation, Prediction
from pss_eval.runner import SystemSpec, run_suite

__all__ = [
    "CrossModalBindingAdapter",
    "CrossModalBindingProperty",
    "PassthroughAdapter",
    "RenderObject",
    "binding_metrics",
    "build_stream",
    "discover_renders",
    "extract_features",
    "json_default",
    "load_object_metadata",
    "run_objectfolder_replay",
]


@dataclass(frozen=True)
class RenderObject:
    """One object's pre-rendered vision and tactile image paths."""

    object_id: str
    vision: tuple[Path, ...]
    touch: tuple[Path, ...]
    name: str = ""
    material: str = ""
    size_m: float | None = None
    source_url: str = ""

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "name": self.name,
            "material": self.material,
            "size_m": self.size_m,
            "source_url": self.source_url,
            "vision_images": len(self.vision),
            "touch_images": len(self.touch),
        }


def _object_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (0, f"{int(path.name):09d}")
    except ValueError:
        return (1, path.name)


def load_object_metadata(path: str | Path) -> dict[str, dict]:
    """Read ``objects.csv`` (id,name,size,material,url) into a dict."""

    path = Path(path)
    if not path.exists():
        return {}
    metadata: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 5:
                continue
            object_id = row[0].strip()
            if not object_id:
                continue
            try:
                size = float(row[2])
            except ValueError:
                size = None
            metadata[object_id] = {
                "name": row[1].strip(),
                "size_m": size,
                "material": row[3].strip(),
                "source_url": row[4].strip(),
            }
    return metadata


def discover_renders(
    renders_root: str | Path,
    *,
    limit: int | None = None,
    object_ids: list[str] | None = None,
    metadata: dict[str, dict] | None = None,
) -> list[RenderObject]:
    """Return render directories that contain both vision and touch images.

    The render tree is the one produced by
    ``python -m pss.rd2.scripts.objectfolder_r1 --render-only``::

        renders/<object_id>/vision/*.png
        renders/<object_id>/touch/*.png
    """

    root = Path(renders_root)
    if not root.exists():
        raise FileNotFoundError(f"render root does not exist: {root}")
    metadata = metadata or {}
    selected = set(str(value) for value in object_ids) if object_ids else None
    objects: list[RenderObject] = []
    for path in sorted(root.iterdir(), key=_object_sort_key):
        if not path.is_dir():
            continue
        if selected is not None and path.name not in selected:
            continue
        vision = tuple(sorted((path / "vision").glob("*.png")))
        touch = tuple(sorted((path / "touch").glob("*.png")))
        if not vision or not touch:
            continue
        meta = metadata.get(path.name, {})
        objects.append(
            RenderObject(
                object_id=path.name,
                vision=vision,
                touch=touch,
                name=str(meta.get("name", "")),
                material=str(meta.get("material", "")),
                size_m=meta.get("size_m"),
                source_url=str(meta.get("source_url", "")),
            )
        )
        if limit is not None and len(objects) >= int(limit):
            break
    return objects


def extract_features(
    objects: list[RenderObject],
    *,
    device: str | None = None,
    batch_size: int = 8,
    cache_path: str | Path | None = None,
) -> dict:
    """Encode the mean DINOv2 vision/tactile feature of every object.

    The encoder and image preprocessing are the ones already used by
    :func:`pss.rd2.objectfolder.run_objectfolder`; this function only adds a
    reusable feature cache and per-object image counts.
    """

    if len(objects) < 2:
        raise ValueError("need at least two rendered objects")

    cache = Path(cache_path) if cache_path is not None else None
    names = [obj.object_id for obj in objects]
    vision_counts = [len(obj.vision) for obj in objects]
    touch_counts = [len(obj.touch) for obj in objects]
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as payload:
            if (
                list(payload["names"].astype(str)) == names
                and list(payload["vision_counts"]) == vision_counts
                and list(payload["touch_counts"]) == touch_counts
            ):
                return {
                    "names": names,
                    "vision_features": payload["vision_features"],
                    "touch_features": payload["touch_features"],
                    "vision_images": vision_counts,
                    "touch_images": touch_counts,
                    "device": str(payload["device"]),
                    "cached": True,
                }

    resolved = device or ("cuda" if _torch_cuda_available() else "cpu")
    model = _load_encoder().to(resolved)
    vision_rows: list[np.ndarray] = []
    touch_rows: list[np.ndarray] = []
    for obj in objects:
        vision_rows.append(_encode_paths(model, list(obj.vision), resolved, batch_size))
        touch_rows.append(_encode_paths(model, list(obj.touch), resolved, batch_size))
    # Mean over the images of each object; L2-normalise so cosine retrieval is
    # comparable with the historical ObjectFolder R@1 runs.
    vision = _l2(np.stack([row.mean(axis=0) for row in vision_rows]))
    touch = _l2(np.stack([row.mean(axis=0) for row in touch_rows]))

    result = {
        "names": names,
        "vision_features": vision,
        "touch_features": touch,
        "vision_images": vision_counts,
        "touch_images": touch_counts,
        "device": resolved,
        "cached": False,
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            names=np.asarray(names),
            vision_features=vision,
            touch_features=touch,
            vision_counts=np.asarray(vision_counts, dtype=np.int64),
            touch_counts=np.asarray(touch_counts, dtype=np.int64),
            device=np.asarray(resolved),
        )
    return result


def _torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dep for this path
        return False
    return bool(torch.cuda.is_available())


def binding_metrics(
    vision: np.ndarray,
    touch: np.ndarray,
    names: list[str],
    *,
    seed: int = 0,
    permutations: int = 1000,
) -> dict:
    """Object-level cross-modal binding metrics with a chance baseline.

    ``vision_to_touch`` R@1 is the binding accuracy: a tactile embedding of
    object X must retrieve the vision embedding of the same X.  The permutation
    test asks how often a shuffled pairing reaches the observed R@1.
    """

    vision = _l2(np.asarray(vision, dtype=np.float32))
    touch = _l2(np.asarray(touch, dtype=np.float32))
    if vision.shape != touch.shape:
        raise ValueError(f"vision/touch shape mismatch: {vision.shape} vs {touch.shape}")
    n = int(vision.shape[0])
    if n < 2:
        raise ValueError("need at least two objects")

    zero_shot = {
        "vision_to_touch": _retrieval(vision, touch),
        "touch_to_vision": _retrieval(touch, vision),
    }

    order = np.random.default_rng(seed).permutation(n)
    cut = n // 2
    train_idx, test_idx = order[:cut], order[cut:]
    if len(train_idx) >= 2 and len(test_idx) >= 1:
        ridge = {
            "vision_to_touch": _ridge_retrieval(
                vision[train_idx], touch[train_idx], vision[test_idx], touch[test_idx]
            ),
            "touch_to_vision": _ridge_retrieval(
                touch[train_idx], vision[train_idx], touch[test_idx], vision[test_idx]
            ),
        }
    else:  # pragma: no cover - only for tiny smoke streams
        ridge = {}

    similarity = vision @ touch.T
    same = np.diag(similarity)
    off_mask = ~np.eye(n, dtype=bool)
    same_mean = float(same.mean())
    cross_mean = float(similarity[off_mask].mean())

    observed = float(zero_shot["vision_to_touch"]["r1"])
    rng = np.random.default_rng(seed + 104729)
    null = []
    for _ in range(int(permutations)):
        perm = rng.permutation(n)
        null.append(float(_retrieval(vision, touch[perm])["r1"]))
    null_array = np.asarray(null, dtype=np.float64)
    p_value = float((1 + np.sum(null_array >= observed)) / (1 + len(null_array)))

    return {
        "n": n,
        "chance_r1": float(1.0 / n),
        "chance_r5": float(min(5, n) / n),
        "zero_shot": zero_shot,
        "ridge_half_split": ridge,
        "cosine": {
            "same_object_mean": same_mean,
            "cross_object_mean": cross_mean,
            "gap": same_mean - cross_mean,
        },
        "permutation": {
            "observed_r1": observed,
            "null_mean": float(null_array.mean()),
            "null_p95": float(np.percentile(null_array, 95)),
            "p_value": p_value,
            "permutations": int(permutations),
        },
    }


def build_stream(
    vision: np.ndarray,
    touch: np.ndarray,
    names: list[str],
    *,
    objects: list[RenderObject] | None = None,
    seed: int = 0,
) -> Stream:
    """Build a ``pss_eval`` stream: one vision and one touch item per object.

    Vision items come first so an online binder can use them as a gallery; the
    PSS runtime does not rely on this ordering.  ``identity_labels`` is
    harness-side only and maps every item to its object identity.
    """

    vision = _l2(np.asarray(vision, dtype=np.float32)).astype(np.float64)
    touch = _l2(np.asarray(touch, dtype=np.float32)).astype(np.float64)
    if vision.shape != touch.shape or len(names) != vision.shape[0]:
        raise ValueError("vision/touch/names must describe the same objects")
    material_by_id = {
        obj.object_id: obj.material for obj in (objects or []) if obj.material
    }

    samples: list[LabeledSample] = []
    labels: dict[str, str] = {}
    for modality, matrix in (("vision", vision), ("touch", touch)):
        for index, name in enumerate(names):
            observation_id = f"{name}:{modality}"
            observation = Observation(
                observation_id=observation_id,
                modality=modality,
                features=matrix[index],
                metadata={"modality": modality},
            )
            samples.append(LabeledSample(observation=observation, label=None))
            labels[observation_id] = f"object:{name}"

    categories = tuple(sorted({value for value in material_by_id.values() if value}))
    return Stream(
        samples=tuple(samples),
        categories=categories,
        dim=int(vision.shape[1]),
        seed=int(seed),
        identity_labels=labels,
    )


# ── adapters used inside the eval path ────────────────────────────────────


class PassthroughAdapter:
    """Emits the observation's own features; the control system for binding."""

    name = "objectfolder-passthrough"

    def __init__(self) -> None:
        self.reset()

    def reset(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self._last: Prediction | None = None
        self._count = 0

    def ingest(self, observation: Observation) -> None:
        self._count += 1
        self._last = Prediction(
            observation_id=observation.observation_id,
            embedding=np.asarray(observation.features, dtype=np.float64),
        )

    def step(self) -> Prediction:
        if self._last is None:
            raise RuntimeError("step() called before ingest()")
        return self._last

    def state(self) -> dict:
        dim = int(self._last.embedding.shape[0]) if self._last is not None else 0
        return {"system": self.name, "count": int(self._count), "dim": dim}


class CrossModalBindingAdapter:
    """Nearest-vision online binder for the standard identity capability.

    Vision items create gallery tracks; a tactile item is assigned to the
    nearest gallery track by cosine similarity.  This is a *replay adapter*, not
    the PSS runtime.  Its ``identity_assignments`` let the harness see the
    cross-modal consistency of the binding.
    """

    name = "objectfolder-nn-binding"

    def __init__(self) -> None:
        self.reset()

    def reset(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self._gallery: list[np.ndarray] = []
        self._assignments: dict[str, str] = {}
        self._last: Prediction | None = None
        self._count = 0

    def ingest(self, observation: Observation) -> None:
        vector = np.asarray(observation.features, dtype=np.float64)
        self._count += 1
        if observation.modality == "vision":
            index = len(self._gallery)
            self._gallery.append(vector)
            track = f"track{index:04d}"
        elif self._gallery:
            gallery = np.stack(self._gallery)
            gallery = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-9)
            query = vector / (np.linalg.norm(vector) + 1e-9)
            track = f"track{int(np.argmax(gallery @ query)):04d}"
        else:
            track = "unassigned"
        self._assignments[observation.observation_id] = track
        self._last = Prediction(
            observation_id=observation.observation_id, embedding=vector
        )

    def step(self) -> Prediction:
        if self._last is None:
            raise RuntimeError("step() called before ingest()")
        return self._last

    def state(self) -> dict:
        return {
            "system": self.name,
            "count": int(self._count),
            "gallery": int(len(self._gallery)),
            "assignments": dict(sorted(self._assignments.items())),
        }

    def identity_assignments(self) -> dict[str, str]:
        return dict(self._assignments)


# ── custom property: object-level binding inside the eval path ────────────


class CrossModalBindingProperty:
    """Object-level R@1 computed from a system's own predictions.

    The harness groups predictions by ``Stream.identity_labels`` and by the
    observation modality, then retrieves touch from vision.  This is the
    static-object analogue of identity retention; it needs no poses and no
    temporal track.  A permutation test over object pairings guards against the
    "slightly above chance" artefact: status is ``pass`` only when R@1 is above
    chance *and* the pairing is significant at p < 0.05.
    """

    name = "cross_modal_binding"
    smoke = False

    def __init__(self, permutations: int = 200, alpha: float = 0.05) -> None:
        self.permutations = int(permutations)
        self.alpha = float(alpha)

    def evaluate(self, factory, stream: Stream, *, seed: int) -> PropertyResult:
        labels = dict(getattr(stream, "identity_labels", {}) or {})
        if not labels:
            return PropertyResult(
                self.name, SKIP, {}, "stream has no object identity labels", smoke=False
            )

        system = factory()
        system.reset(seed=seed)
        predictions: dict[str, tuple[str, np.ndarray]] = {}
        for sample in stream.samples:
            observation = sample.observation
            system.ingest(observation)
            prediction = system.step()
            predictions[observation.observation_id] = (
                observation.modality,
                np.asarray(prediction.embedding, dtype=np.float64).reshape(-1),
            )

        by_object: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
            lambda: {"vision": [], "touch": []}
        )
        for observation_id, identity in labels.items():
            entry = predictions.get(observation_id)
            if entry is None:
                continue
            modality, embedding = entry
            if modality in ("vision", "touch"):
                by_object[str(identity)][modality].append(embedding)

        names = sorted(
            identity
            for identity, entry in by_object.items()
            if entry["vision"] and entry["touch"]
        )
        if len(names) < 2:
            return PropertyResult(
                self.name, SKIP, {}, "fewer than two objects have both modalities"
            )

        vision = _l2(
            np.stack([np.mean(by_object[name]["vision"], axis=0) for name in names])
        )
        touch = _l2(
            np.stack([np.mean(by_object[name]["touch"], axis=0) for name in names])
        )
        forward = _retrieval(vision, touch)
        backward = _retrieval(touch, vision)
        chance = float(forward["chance"])
        observed = float(forward["r1"])
        above_chance = observed > chance + 1e-12

        rng = np.random.default_rng(seed + 7919)
        null = np.asarray(
            [
                float(_retrieval(vision, touch[rng.permutation(len(names))])["r1"])
                for _ in range(self.permutations)
            ],
            dtype=np.float64,
        )
        p_value = float((1 + np.sum(null >= observed)) / (1 + len(null)))
        significant = bool(p_value < self.alpha)

        metrics = {
            "r1": observed,
            "r5": float(forward["r5"]),
            "chance": chance,
            "chance_r5": float(min(5, len(names)) / len(names)),
            "n": float(forward["n"]),
            "touch_to_vision_r1": float(backward["r1"]),
            "touch_to_vision_r5": float(backward["r5"]),
            "permutation_p": p_value,
            "permutations": float(self.permutations),
        }
        return PropertyResult(
            self.name,
            PASS if (above_chance and significant) else FAIL,
            metrics,
            "object-level cross-modal retrieval; pass means R@1 > chance and "
            f"permutation p < {self.alpha}",
            smoke=False,
        )


# ── orchestration ─────────────────────────────────────────────────────────


def _render_markdown(results: dict) -> str:
    binding = results["binding"]
    dataset = results["dataset"]
    features = results["features"]
    lines = [
        "# ObjectFolder 2.0 real-data replay (F10 spike)",
        "",
        "## Data inventory",
        "",
        f"- render root: `{dataset['renders_root']}`",
        f"- objects: {dataset['objects']}",
        f"- license: {dataset['license']}",
        f"- citation: {dataset['citation']}",
        f"- vision images/object: {features['vision_images_per_object']}",
        f"- touch images/object: {features['touch_images_per_object']}",
        "",
        "## Object-level cross-modal binding",
        "",
        "| protocol | direction | R@1 | R@5 | chance |",
        "|---|---|---:|---:|---:|",
    ]
    for protocol in ("zero_shot", "ridge_half_split"):
        for direction, metric in binding.get(protocol, {}).items():
            lines.append(
                f"| {protocol} | {direction} | {metric['r1']:.3f} "
                f"| {metric['r5']:.3f} | {metric['chance']:.3f} |"
            )
    cosine = binding["cosine"]
    permutation = binding["permutation"]
    lines += [
        "",
        f"- R@1 chance = 1/n = {binding['chance_r1']:.3f}",
        f"- R@5 chance = min(5, n)/n = {binding['chance_r5']:.3f}",
        f"- same-object cosine: {cosine['same_object_mean']:.4f}",
        f"- cross-object cosine: {cosine['cross_object_mean']:.4f}",
        f"- cosine gap: {cosine['gap']:.4f}",
        f"- permutation test: observed R@1 {permutation['observed_r1']:.3f}, "
        f"null mean {permutation['null_mean']:.3f}, "
        f"p={permutation['p_value']:.4f} ({permutation['permutations']} perms)",
        "",
        "## pss_eval path",
        "",
        "| system | property | status | metrics |",
        "|---|---|---|---|",
    ]
    for run in results["pss_eval"]["runs"]:
        for row in run["results"]:
            metrics = ", ".join(
                f"{key}={value:.3f}" for key, value in sorted(row["metrics"].items())
            )
            lines.append(
                f"| {row['system']} | {row['property']} | {row['status']} | {metrics} |"
            )
    lines += [
        "",
        "## Caveats",
        "",
        *[f"- {item}" for item in results["caveats"]],
        "",
    ]
    return "\n".join(lines)


def run_objectfolder_replay(
    *,
    renders_root: str | Path,
    out_dir: str | Path,
    objects_csv: str | Path | None = None,
    limit: int = 20,
    seed: int = 0,
    permutations: int = 1000,
    device: str | None = None,
    public_dim: int = 128,
    budget_ms: float | None = 50.0,
    cache_path: str | Path | None = None,
) -> dict:
    """Run the F10 ObjectFolder replay spike and write JSON/Markdown outputs."""

    metadata = load_object_metadata(objects_csv) if objects_csv else {}
    objects = discover_renders(
        renders_root, limit=limit, metadata=metadata
    )
    if len(objects) < 2:
        raise SystemExit(f"need at least two rendered objects under {renders_root}")

    features = extract_features(objects, device=device, cache_path=cache_path)
    binding = binding_metrics(
        features["vision_features"],
        features["touch_features"],
        features["names"],
        seed=seed,
        permutations=permutations,
    )
    stream = build_stream(
        features["vision_features"],
        features["touch_features"],
        features["names"],
        objects=objects,
        seed=seed,
    )

    systems = [
        SystemSpec(
            "pss-runtime",
            lambda: PSSRuntimeAdapter(
                input_dim=int(stream.dim),
                public_dim=int(public_dim),
                budget_ms=budget_ms,
                tier="frozen_real",
            ),
        ),
        SystemSpec("objectfolder-passthrough", lambda: PassthroughAdapter()),
        SystemSpec("objectfolder-nn-binding", lambda: CrossModalBindingAdapter()),
    ]
    properties = [
        prop for prop in extended_properties() if prop.name != "identity_retention"
    ] + [CrossModalBindingProperty()]
    report = run_suite(
        systems,
        properties,
        stream,
        seeds=(seed,),
        preregistration_expectations={},
    )

    results = {
        "spike": "F10 ObjectFolder real-data replay",
        "dataset": {
            "renders_root": str(Path(renders_root).resolve()),
            "objects_csv": str(Path(objects_csv).resolve()) if objects_csv else None,
            "objects": len(objects),
            "object_ids": [obj.object_id for obj in objects],
            "materials": sorted({obj.material for obj in objects if obj.material}),
            "license": "CC-BY-4.0",
            "citation": "Gao et al., ObjectFolder 2.0, CVPR 2022",
            "object_metadata": [obj.to_dict() for obj in objects],
        },
        "features": {
            "encoder": "facebook/dinov2-small (AutoModel, pooler/mean)",
            "dim": int(features["vision_features"].shape[1]),
            "device": features["device"],
            "cached": bool(features["cached"]),
            "vision_images_per_object": sorted(set(features["vision_images"])),
            "touch_images_per_object": sorted(set(features["touch_images"])),
        },
        "binding": binding,
        "pss_eval": report,
        "caveats": [
            "ObjectFolder is a public dataset of implicit object representations; "
            "rendered vision/tactile images are not raw robot sensor captures.",
            "No poses, depth, segmentation, scenes, or synchronized multi-frame "
            "stream: L2 identity/tracking and sim2real claims are out of scope.",
            "The standard identity_retention property is not applicable: with no "
            "poses every detection collapses to one track (degenerate pass). It "
            "was removed; cross_modal_binding is used instead.",
            "The PSS runtime's projection here is a random adapter projection, not "
            "a projection head trained on real ObjectFolder pairs; its binding R@1 "
            "is a path-connectivity measurement, not a PSS capability result.",
            "Historical 100-object ObjectFolder results (zero-shot R@1 ~chance; "
            "trained tactile encoder held-out R@1 ~0.22) are the stronger evidence "
            "and should be cited instead of this 20-object spike for capability.",
        ],
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    (out / "real_replay.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    (out / "REPLAY.md").write_text(_render_markdown(results), encoding="utf-8")
    return results


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")
