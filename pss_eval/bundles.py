"""Real bundle streams for ``pss_eval``.

This module turns the detection tables of the four development bundles that
carry detector embeddings into :class:`~pss_eval.fixtures.Stream` objects, so
the same property runner can drive the PSS runtime and the external baselines
over real data instead of synthetic Gaussians.

Boundary
--------
The *system* only ever receives :class:`~pss_eval.protocol.Observation` objects.
Those carry per-detection pose/score/category/embedding in ``metadata`` but no
ground truth.  ``gt.parquet`` is opened **only** here, and only when the caller
asks for harness-side identity labels (``with_gt_identities=True``); the labels
land in ``Stream.identity_labels`` and never on an observation.  The PSS
adapter has no reference to ``gt.parquet``.

Bundles
-------
``DEFAULT_BUNDLE_NAMES`` are the four bundles the F2 perception-swap report
used: they are the only development bundles whose ``obs_detections.parquet``
has a populated ``emb`` column.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fixtures import LabeledSample, Stream
from .protocol import Observation

__all__ = [
    "DEFAULT_BUNDLE_NAMES",
    "DEFAULT_BUNDLE_ROOT",
    "BundleFrame",
    "Detection",
    "find_bundle_dirs",
    "frames_to_stream",
    "load_bundle_frames",
    "load_bundle_stream",
    "load_bundle_streams",
]

DEFAULT_BUNDLE_ROOT = Path("D:/PublicSemanticSpace/bundles/dev")

# The four bundles with a populated ``emb`` column (see F2 REPORT.md).
DEFAULT_BUNDLE_NAMES: tuple[str, ...] = (
    "s1_hard",
    "s1_occlusion_real",
    "s2_twins",
    "s1_camera_real",
)


@dataclass(frozen=True)
class Detection:
    """One detector output.  ``ref`` is a per-detection id, not an identity."""

    ref: str
    pos: np.ndarray
    quat: np.ndarray
    score: float
    category: str | None
    embedding: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", np.asarray(self.pos, dtype=np.float64).reshape(3))
        object.__setattr__(
            self, "quat", np.asarray(self.quat, dtype=np.float64).reshape(4)
        )
        object.__setattr__(
            self, "embedding", np.asarray(self.embedding, dtype=np.float64).reshape(-1)
        )


@dataclass(frozen=True)
class BundleFrame:
    """One camera frame's detections, ordered by the bundle's row order."""

    frame_index: int
    step: int
    t_s: float
    detections: tuple[Detection, ...]

    @property
    def dim(self) -> int:
        return int(self.detections[0].embedding.shape[0]) if self.detections else 0

    def mean_embedding(self) -> np.ndarray:
        """Unit-norm mean of the detection embeddings (the frame's features)."""

        if not self.detections:
            return np.zeros(0, dtype=np.float64)
        matrix = np.stack([det.embedding for det in self.detections])
        vector = matrix.mean(axis=0)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return vector.astype(np.float64)
        return (vector / norm).astype(np.float64)


# ── discovery / loading ────────────────────────────────────────────────────


def _has_embeddings(path: Path) -> bool:
    import pandas as pd

    try:
        frame = pd.read_parquet(path, columns=["emb"])
    except Exception:
        return False
    if frame.empty or "emb" not in frame.columns:
        return False
    return bool(frame["emb"].notna().any())


def find_bundle_dirs(
    root: str | Path = DEFAULT_BUNDLE_ROOT,
    names: Sequence[str] | None = None,
) -> list[Path]:
    """Return one seed directory per requested bundle, preferring ``seed0``."""

    root = Path(root)
    if not root.exists():
        return []
    selected = tuple(names) if names is not None else DEFAULT_BUNDLE_NAMES
    found: list[Path] = []
    for name in selected:
        candidates = sorted(
            path.parent
            for path in (root / name).glob("*/*/obs_detections.parquet")
            if path.parent.name.startswith("seed")
        )
        # Prefer seed0; otherwise the first seed with embeddings.
        candidates.sort(key=lambda path: (path.name != "seed0", str(path)))
        for candidate in candidates:
            if _has_embeddings(candidate / "obs_detections.parquet"):
                found.append(candidate)
                break
    return found


def _bundle_tag(root: Path, bundle_dir: Path) -> str:
    try:
        return str(bundle_dir.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return bundle_dir.name


def load_bundle_frames(
    bundle_dir: str | Path,
    *,
    max_frames: int | None = None,
) -> list[BundleFrame]:
    """Read ``obs_detections.parquet`` and group rows by frame step."""

    import pandas as pd

    bundle_dir = Path(bundle_dir)
    detections = pd.read_parquet(bundle_dir / "obs_detections.parquet")
    if detections.empty:
        return []
    if "emb" not in detections.columns or not detections["emb"].notna().any():
        raise ValueError(f"{bundle_dir} has no populated 'emb' column")

    frames: list[BundleFrame] = []
    for frame_index, (step, group) in enumerate(
        detections.groupby("step", sort=True)
    ):
        if max_frames is not None and frame_index >= int(max_frames):
            break
        dets: list[Detection] = []
        for _, row in group.iterrows():
            embedding = row.get("emb")
            if embedding is None or len(embedding) == 0:
                continue
            dets.append(
                Detection(
                    ref=str(row["emb_ref"]),
                    pos=np.asarray(row["pose_xyz"], dtype=np.float64),
                    quat=np.asarray(row["pose_quat"], dtype=np.float64),
                    score=float(row["score"]),
                    category=str(row["class_pred"]),
                    embedding=np.asarray(embedding, dtype=np.float64),
                )
            )
        if not dets:
            continue
        frames.append(
            BundleFrame(
                frame_index=len(frames),
                step=int(step),
                t_s=float(group.iloc[0]["t_s"]),
                detections=tuple(dets),
            )
        )
    return frames


def _match_gt_identities(
    bundle_dir: Path,
    frames: Sequence[BundleFrame],
    *,
    max_distance_m: float = 0.5,
) -> dict[str, str]:
    """Harness-only: map each detection ref to the nearest GT instance.

    GT rows and detection frames share ``physics_step == step`` (the detection
    table subsamples the same physics steps).  A detection farther than
    ``max_distance_m`` from every GT pose is left unlabelled.
    """

    import pandas as pd

    gt_path = bundle_dir / "gt.parquet"
    if not gt_path.exists():
        return {}
    gt = pd.read_parquet(gt_path, columns=["physics_step", "instance_id", "pose_xyz"])
    by_step: dict[int, tuple[np.ndarray, list[str]]] = {}
    for step, group in gt.groupby("physics_step", sort=False):
        positions = np.asarray(list(group["pose_xyz"]), dtype=np.float64).reshape(
            len(group), 3
        )
        by_step[int(step)] = (positions, [str(v) for v in group["instance_id"]])

    labels: dict[str, str] = {}
    for frame in frames:
        entry = by_step.get(int(frame.step))
        if entry is None:
            continue
        positions, instance_ids = entry
        for det in frame.detections:
            distances = np.linalg.norm(positions - det.pos[None, :], axis=1)
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) <= float(max_distance_m):
                labels[det.ref] = instance_ids[nearest]
    return labels


def _observation_metadata(frame: BundleFrame) -> dict[str, Any]:
    """Adapter-facing frame metadata.  Contains no ground truth."""

    return {
        "frame_index": int(frame.frame_index),
        "step": int(frame.step),
        "t_s": float(frame.t_s),
        "detections": tuple(
            {
                "ref": det.ref,
                "pos": det.pos.copy(),
                "quat": det.quat.copy(),
                "score": float(det.score),
                "category": det.category,
                "embedding": det.embedding.copy(),
            }
            for det in frame.detections
        ),
    }


def frames_to_stream(
    frames: Sequence[BundleFrame],
    *,
    tag: str = "bundle",
    identity_labels: Mapping[str, str] | None = None,
) -> Stream:
    """Build a :class:`Stream` of one observation per frame."""

    samples: list[LabeledSample] = []
    categories: set[str] = set()
    dim = 0
    for frame in frames:
        features = frame.mean_embedding()
        dim = int(features.shape[0])
        observation = Observation(
            observation_id=f"{tag}:f{frame.frame_index:04d}",
            modality="vision",
            features=features,
            metadata=_observation_metadata(frame),
        )
        samples.append(LabeledSample(observation=observation, label=None))
        categories.update(
            det.category for det in frame.detections if det.category is not None
        )
    return Stream(
        samples=tuple(samples),
        categories=tuple(sorted(categories)),
        dim=dim,
        seed=0,
        identity_labels=dict(identity_labels or {}),
    )


def load_bundle_stream(
    bundle_dir: str | Path,
    *,
    root: str | Path = DEFAULT_BUNDLE_ROOT,
    with_gt_identities: bool = True,
    max_frames: int | None = None,
) -> Stream:
    """Load one bundle as a stream.

    ``with_gt_identities`` opens ``gt.parquet`` to build harness-side identity
    labels only.  The observations passed to a system never contain them.
    """

    bundle_dir = Path(bundle_dir)
    frames = load_bundle_frames(bundle_dir, max_frames=max_frames)
    labels = (
        _match_gt_identities(bundle_dir, frames) if with_gt_identities else {}
    )
    return frames_to_stream(
        frames, tag=_bundle_tag(Path(root), bundle_dir), identity_labels=labels
    )


def load_bundle_streams(
    root: str | Path = DEFAULT_BUNDLE_ROOT,
    names: Iterable[str] | None = None,
    *,
    with_gt_identities: bool = True,
    max_frames: int | None = None,
) -> dict[str, Stream]:
    """Load the requested bundles, keyed by bundle tag."""

    root = Path(root)
    dirs = find_bundle_dirs(root, tuple(names) if names is not None else None)
    streams: dict[str, Stream] = {}
    for bundle_dir in dirs:
        tag = _bundle_tag(root, bundle_dir)
        streams[tag] = load_bundle_stream(
            bundle_dir,
            root=root,
            with_gt_identities=with_gt_identities,
            max_frames=max_frames,
        )
    return streams
