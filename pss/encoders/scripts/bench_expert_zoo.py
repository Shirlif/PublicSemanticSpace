"""CLI: bake off sensory experts against the same PSS evaluation protocol.

The script reads ``contracts/experts.yaml``, loads every runnable ExpertSpec,
extracts frozen features on the category dataset, and reports:

- closed-set top-1 on val/test;
- open-set AUROC and unseen reject rate;
- attribute linear probes (hardness/roughness/thermal/size);
- same-instance cross-modal R@1/R@5;
- single-sample latency, peak memory, parameter count, license gate.

Example:
    python -m pss.encoders.scripts.bench_expert_zoo \
        --data data/category_pretrain \
        --experts contracts/experts.yaml \
        --out eval/results/expert_zoo

The report is written to ``<out>/expert_zoo.md`` and ``<out>/expert_zoo.json``.
This is an evaluation harness, not a trainer: only the probe is fit.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pss._console import ensure_utf8
from pss.encoders.data import crop_object, load_index, normalize_image
from pss.encoders.geometry import geometry_features
from pss.encoders.tactile import load_tactile_expert
from pss.encoders.vision import load_vision_expert
from pss.encoders.vision.pretrained import load_pretrained_expert
from pss.sim.dataset import load_sample

ATTR_NAMES = ("hardness", "roughness", "thermal", "size_x", "size_y")
# Same scaling as the projection stage: sizes are metres, so keep them visible.
ATTR_SCALE = np.array([1.0, 1.0, 1.0, 10.0, 10.0], dtype=np.float32)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
DEFAULT_ALLOWED_LICENSES = ("project-internal", "Apache-2.0", "MIT", "BSD-3-Clause")


class SkipExpert(RuntimeError):
    """Raised when an ExpertSpec cannot be loaded in this environment."""


@dataclass
class RawView:
    rgb: np.ndarray              # (H, W, 3) uint8
    depth: np.ndarray            # (H, W) float32
    seg: np.ndarray              # (H, W, 2) int32
    geom_id: int
    tactile: np.ndarray          # (64,) float32
    geometry: np.ndarray         # (16,) float32
    attrs: np.ndarray            # (5,) float32
    category: str
    split: str
    instance: str
    sample_id: str
    view: int


# ── data ────────────────────────────────────────────────────────────────────


def load_raw_views(data_dir: str | Path, max_samples: int | None = None) -> tuple[list[RawView], bool]:
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    preserve_aspect = bool(index.get("config", {}).get("crop_preserve_aspect", False))
    entries = [e for e in index["entries"] if e.get("kind") == "category"]
    if max_samples is not None:
        entries = entries[: int(max_samples)]
    views: list[RawView] = []
    for entry in entries:
        sample = load_sample(data_dir / entry["path"])
        geom_id = int(sample["geom_id"])
        material = sample.get("material", {})
        size = np.asarray(sample.get("size", [0.0, 0.0]), dtype=np.float32).reshape(-1)
        size_x = float(size[0]) if size.size else 0.0
        size_y = float(size[1]) if size.size > 1 else size_x
        attrs = np.array(
            [
                float(material.get("hardness", 0.0)),
                float(material.get("roughness", 0.0)),
                float(material.get("thermal", 0.0)),
                size_x,
                size_y,
            ],
            dtype=np.float32,
        ) * ATTR_SCALE
        tactile_all = sample["tactile"]
        for v in range(int(sample["rgb"].shape[0])):
            views.append(
                RawView(
                    rgb=sample["rgb"][v],
                    depth=sample["depth"][v],
                    seg=sample["segmentation"][v],
                    geom_id=geom_id,
                    tactile=np.asarray(tactile_all[v], dtype=np.float32).reshape(-1),
                    geometry=geometry_features(
                        sample["depth"][v], sample["segmentation"][v], geom_id
                    ),
                    attrs=attrs.copy(),
                    category=str(sample["category"]),
                    split=str(sample["split"]),
                    instance=str(sample["instance_id"]),
                    sample_id=str(entry["id"]),
                    view=v,
                )
            )
    return views, preserve_aspect


# ── experts ─────────────────────────────────────────────────────────────────


class ExpertAdapter:
    modality = "unknown"

    def __init__(self, name: str) -> None:
        self.name = name
        self.model: torch.nn.Module | None = None

    def encode(
        self, views: list[RawView], preserve_aspect: bool, batch_size: int = 32
    ) -> np.ndarray:
        raise NotImplementedError

    def encode_one(self, view: RawView, preserve_aspect: bool) -> np.ndarray:
        return self.encode([view], preserve_aspect, batch_size=1)[0]

    def params(self) -> int:
        if self.model is None:
            return 0
        return int(sum(p.numel() for p in self.model.parameters()))


class VisionTorchAdapter(ExpertAdapter):
    modality = "vision"

    def __init__(self, name: str, model: torch.nn.Module, image_size: int, device: str) -> None:
        super().__init__(name)
        self.model = model.to(device).eval()
        self.image_size = int(image_size)
        self.device = device

    def _batch(self, views: list[RawView], preserve_aspect: bool) -> np.ndarray:
        xs = []
        for view in views:
            crop = crop_object(
                view.rgb,
                view.seg,
                view.geom_id,
                self.image_size,
                preserve_aspect=preserve_aspect,
            )
            xs.append(normalize_image(crop))
        return np.stack(xs).astype(np.float32)

    def _forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self.model(xb)  # type: ignore[misc]

    def encode(
        self, views: list[RawView], preserve_aspect: bool, batch_size: int = 32
    ) -> np.ndarray:
        out = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(views), batch_size):
                batch = views[start : start + batch_size]
                xb = torch.as_tensor(self._batch(batch, preserve_aspect), device=self.device)
                out.append(self._forward(xb).detach().cpu().numpy())
        return np.concatenate(out, axis=0)


class Dinov2Adapter(VisionTorchAdapter):
    def __init__(
        self,
        name: str,
        model_id: str,
        image_size: int,
        device: str,
        allow_download: bool,
    ) -> None:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_id, local_files_only=not allow_download
        )
        super().__init__(name, model, image_size, device)

    def _forward(self, xb: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=xb)  # type: ignore[union-attr]
        feat = getattr(out, "pooler_output", None)
        if feat is None:
            feat = out.last_hidden_state.mean(dim=1)
        return feat


class TactileTorchAdapter(ExpertAdapter):
    modality = "tactile"

    def __init__(self, name: str, model: torch.nn.Module, device: str) -> None:
        super().__init__(name)
        self.model = model.to(device).eval()
        self.device = device

    def encode(
        self, views: list[RawView], preserve_aspect: bool, batch_size: int = 256
    ) -> np.ndarray:
        X = np.stack([v.tactile for v in views]).astype(np.float32)
        out = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(X), batch_size):
                xb = torch.as_tensor(X[start : start + batch_size], device=self.device)
                out.append(self.model(xb).detach().cpu().numpy())
        return np.concatenate(out, axis=0)


class OctopiTactileAdapter(ExpertAdapter):
    """Octopi 7B tactile encoder: CLIP ViT-L/14 + per-layer VPT prompts.

    The checkpoint is ``7b/encoder.pt`` from Octopi's Google Drive. The frozen
    encoder expects ``(B, L, 3, 224, 224)`` tactile frames and returns the mean
    pooled feature over frames. Synthetic 8x8 contact grids are upsampled to
    224x224 and repeated to 3 channels / L frames; this domain bridge is a
    deliberate compromise so the C group can run on the same category dataset.
    """

    modality = "tactile"

    def __init__(
        self,
        name: str,
        checkpoint: str | Path,
        device: str,
        frames: int = 3,
        image_size: int = 224,
    ) -> None:
        super().__init__(name)
        from transformers import CLIPVisionConfig, CLIPVisionModel

        config = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14")
        model = CLIPVisionModel(config)
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        state: dict[str, torch.Tensor] = {}
        vpt: torch.Tensor | None = None
        vpt_shallow: dict[int, torch.Tensor] = {}
        for key, value in payload.items():
            clean = key[len("model.") :] if key.startswith("model.") else key
            if clean == "vision_model.VPT":
                vpt = value
            elif clean.endswith(".VPT_shallow"):
                layer_idx = int(clean.split(".")[3])
                vpt_shallow[layer_idx] = value
            else:
                state[clean] = value
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SkipExpert(
                "Octopi checkpoint mismatch: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        if vpt is None:
            raise SkipExpert("Octopi checkpoint has no VPT tensor")
        self.model = model.to(device).eval()
        self.vpt = vpt.to(device).float()
        self.vpt_shallow = {k: v.to(device).float() for k, v in vpt_shallow.items()}
        self.device = device
        self.frames = int(frames)
        self.image_size = int(image_size)

    def _batch(self, views: list[RawView]) -> torch.Tensor:
        mean = torch.as_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.as_tensor(CLIP_STD).view(1, 3, 1, 1)
        xs = []
        for view in views:
            grid = torch.as_tensor(view.tactile, dtype=torch.float32).reshape(1, 1, 8, 8)
            grid = F.interpolate(
                grid,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            grid = grid.repeat(1, 3, 1, 1)
            grid = (grid - mean) / std
            xs.append(grid.repeat(self.frames, 1, 1, 1))
        return torch.stack(xs, dim=0)  # (B, L, 3, H, W)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.model.vision_model.embeddings(x)
        batch = emb.shape[0]
        vpt = self.vpt.unsqueeze(0).expand(batch, -1, -1)
        hidden = torch.cat([emb, vpt], dim=1)
        hidden = self.model.vision_model.pre_layrnorm(hidden)
        for idx, layer in enumerate(self.model.vision_model.encoder.layers):
            if idx in self.vpt_shallow:
                ctx = self.vpt_shallow[idx].unsqueeze(0).expand(batch, -1, -1)
                hidden = torch.cat(
                    [hidden[:, : hidden.shape[1] - ctx.shape[1], :], ctx], dim=1
                )
            hidden = layer(hidden, attention_mask=None, causal_attention_mask=None)[0]
        return self.model.vision_model.post_layernorm(hidden[:, 0, :])

    def encode(
        self, views: list[RawView], preserve_aspect: bool, batch_size: int = 8
    ) -> np.ndarray:
        out = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(views), batch_size):
                x = self._batch(views[start : start + batch_size]).to(self.device)
                b, l, c, h, w = x.shape
                feats = self._forward(x.reshape(b * l, c, h, w))
                feats = feats.reshape(b, l, -1).mean(dim=1)
                out.append(feats.detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    def params(self) -> int:
        total = sum(p.numel() for p in self.model.parameters())
        total += int(self.vpt.numel())
        total += sum(int(v.numel()) for v in self.vpt_shallow.values())
        return int(total)


class RealTactileAdapter(ExpertAdapter):
    """R2 PhysiCLeAR encoder: trained on real GelSight frames, probed on 8x8 grids.

    The encoder was trained on 64x64 real tactile RGB frames. Synthetic 8x8
    contact grids are upsampled to the checkpoint's input size and replicated
    to 3 channels. This is the same deliberate domain bridge as the Octopi
    adapter; report it as such.
    """

    modality = "tactile"

    def __init__(self, name: str, checkpoint: str | Path, device: str) -> None:
        super().__init__(name)
        from pss.encoders.tactile.physiclear import RealTactileEncoder

        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        self.model = RealTactileEncoder(int(payload["out_dim"])).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.size = int(payload.get("input_shape", [3, 64, 64])[-1])
        self.device = device

    def _batch(self, views: list[RawView]) -> torch.Tensor:
        xs = []
        for view in views:
            grid = torch.as_tensor(view.tactile, dtype=torch.float32).reshape(1, 1, 8, 8)
            grid = F.interpolate(
                grid, size=(self.size, self.size), mode="bilinear", align_corners=False
            )
            xs.append(grid.repeat(1, 3, 1, 1))
        return torch.cat(xs, dim=0)  # (B, 3, S, S)

    def encode(
        self, views: list[RawView], preserve_aspect: bool, batch_size: int = 128
    ) -> np.ndarray:
        X = self._batch(views)
        out = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(X), batch_size):
                xb = torch.as_tensor(X[start : start + batch_size], device=self.device)
                out.append(self.model(xb).detach().cpu().numpy())
        return np.concatenate(out, axis=0)


class GeometryAdapter(ExpertAdapter):
    modality = "geometry"

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def encode(
        self, views: list[RawView], preserve_aspect: bool, batch_size: int = 1024
    ) -> np.ndarray:
        return np.stack([v.geometry for v in views]).astype(np.float32)

    def encode_one(self, view: RawView, preserve_aspect: bool) -> np.ndarray:
        return geometry_features(view.depth, view.seg, view.geom_id)


def resolve_file(source: str, default_name: str | None = None) -> Path:
    path = Path(str(source))
    if path.is_dir() and default_name:
        path = path / default_name
    if not path.exists():
        raise SkipExpert(f"source not found: {path}")
    return path


def hf_repo_id(source: str) -> str:
    value = str(source).rstrip("/")
    prefix = "https://huggingface.co/"
    return value[len(prefix) :] if value.startswith(prefix) else value


def spec_image_size(spec: dict, default: int = 64) -> int:
    rgb = spec.get("input_schema", {}).get("rgb")
    if isinstance(rgb, (list, tuple)) and rgb:
        return int(rgb[-1])
    return int(default)


def build_adapter(spec: dict, device: str, allow_download: bool) -> ExpertAdapter:
    name = str(spec.get("name", ""))
    if name == "vision_smallcnn_v0":
        expert = load_vision_expert(resolve_file(spec["source"], "vision.pt"), device=device)
        return VisionTorchAdapter(name, expert.encoder, spec_image_size(spec), device)
    if name == "vision_squeezenet_expert_v0":
        model = load_pretrained_expert(resolve_file(spec["source"]), device=device)
        return VisionTorchAdapter(name, model, spec_image_size(spec), device)
    if name == "vision_dinov2_vit_s14":
        try:
            return Dinov2Adapter(
                name,
                hf_repo_id(spec["source"]),
                spec_image_size(spec, 224),
                device,
                allow_download,
            )
        except Exception as exc:  # missing package, offline cache, bad weights
            raise SkipExpert(f"DINOv2 unavailable: {type(exc).__name__}: {exc}") from exc
    if name == "tactile_mlp_v0":
        expert = load_tactile_expert(resolve_file(spec["source"], "tactile.pt"), device=device)
        return TactileTorchAdapter(name, expert.encoder, device)
    if name == "tactile_octopi_v0":
        schema = spec.get("input_schema", {}).get("tactile_frames", [3, 224, 224])
        frames = int(schema[0]) if isinstance(schema, (list, tuple)) and schema else 3
        size = int(schema[-1]) if isinstance(schema, (list, tuple)) and schema else 224
        return OctopiTactileAdapter(
            name, resolve_file(spec["source"]), device, frames=frames, image_size=size
        )
    if name == "tactile_real_smallcnn_v0":
        return RealTactileAdapter(name, resolve_file(spec["source"]), device)
    if name == "geometry_handcrafted_v0":
        return GeometryAdapter(name)
    raise SkipExpert("no loader registered for this ExpertSpec")


# ── metrics ─────────────────────────────────────────────────────────────────


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def _fit_probe(X: np.ndarray, y: np.ndarray):
    if len(X) == 0:
        raise ValueError("empty probe training set")
    if len(np.unique(y)) < 2:
        return DummyClassifier(strategy="most_frequent").fit(X, y)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )
    clf.fit(X, y)
    return clf


def _top1(clf, X: np.ndarray, y: np.ndarray) -> float:
    if len(X) == 0:
        return float("nan")
    return float((clf.predict(X) == y).mean())


def _max_proba(clf, X: np.ndarray) -> np.ndarray:
    return np.asarray(clf.predict_proba(X)).max(axis=1)


def _open_set(clf, X_val, X_seen, X_unseen) -> dict:
    if len(X_val) == 0 or len(X_seen) == 0:
        return {
            "auroc": float("nan"),
            "unseen_reject": float("nan"),
            "seen_false_reject": float("nan"),
            "threshold": float("nan"),
        }
    threshold = float(np.percentile(_max_proba(clf, X_val), 5))
    seen_scores = _max_proba(clf, X_seen)
    unseen_scores = _max_proba(clf, X_unseen) if len(X_unseen) else np.array([])
    auroc = float("nan")
    if len(unseen_scores):
        # Match pss/prototypes/build.py: seen=1, unseen=0.
        y_true = np.r_[np.ones(len(seen_scores)), np.zeros(len(unseen_scores))]
        y_score = np.r_[seen_scores, unseen_scores]
        auroc = float(roc_auc_score(y_true, y_score))
    return {
        "auroc": auroc,
        "unseen_reject": float((unseen_scores < threshold).mean())
        if len(unseen_scores)
        else float("nan"),
        "seen_false_reject": float((seen_scores < threshold).mean()),
        "threshold": threshold,
    }


def _attribute_probe(X_train, A_train, X_test, A_test) -> dict:
    out = {}
    for i, name in enumerate(ATTR_NAMES):
        if len(X_train) < 2 or len(X_test) < 2:
            out[name] = {"r2": float("nan"), "spearman": float("nan")}
            continue
        reg = Ridge(alpha=1.0)
        reg.fit(X_train, A_train[:, i])
        pred = reg.predict(X_test)
        out[name] = {
            "r2": float(r2_score(A_test[:, i], pred)),
            "spearman": _spearman(A_test[:, i], pred),
        }
    return out


def _stratified_subset(y: np.ndarray, seed: int, frac: float = 0.8) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    take: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * frac)))
        take.extend(idx[:k].tolist())
    if len(take) < 2:
        return np.arange(len(y), dtype=np.int64)
    return np.array(sorted(take), dtype=np.int64)


def _run_seed(feats, attrs, train_idx, val_idx, test_idx, unseen_idx, y_all, seed):
    if seed is None:
        sub = train_idx
    else:
        sub = train_idx[_stratified_subset(y_all[train_idx], seed)]
    clf = _fit_probe(feats[sub], y_all[sub])
    return {
        "val_top1": _top1(clf, feats[val_idx], y_all[val_idx]),
        "test_top1": _top1(clf, feats[test_idx], y_all[test_idx]),
        "open": _open_set(clf, feats[val_idx], feats[test_idx], feats[unseen_idx]),
        "attr": _attribute_probe(feats[sub], attrs[sub], feats[test_idx], attrs[test_idx]),
    }


def _agg(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "per_seed": [float("nan")] * len(values),
        }
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "per_seed": [float(v) for v in arr],
    }


def evaluate_expert(feats: np.ndarray, meta: dict, seeds: list[int]) -> dict:
    cats = np.asarray(meta["categories"])
    splits = np.asarray(meta["splits"])
    attrs = np.asarray(meta["attrs"], dtype=np.float32)
    seen_classes = sorted(set(cats[splits == "train"].tolist()))
    class_to_idx = {c: i for i, c in enumerate(seen_classes)}
    y_all = np.array([class_to_idx.get(c, -1) for c in cats], dtype=np.int64)
    seen_mask = y_all >= 0
    train_idx = np.where((splits == "train") & seen_mask)[0]
    val_idx = np.where((splits == "val") & seen_mask)[0]
    test_idx = np.where((splits == "test") & seen_mask)[0]
    unseen_idx = np.where(splits == "test_category")[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise SkipExpert("empty train/test split for seen classes")

    runs = [
        _run_seed(feats, attrs, train_idx, val_idx, test_idx, unseen_idx, y_all, int(seed))
        for seed in seeds
    ]
    full = _run_seed(feats, attrs, train_idx, val_idx, test_idx, unseen_idx, y_all, None)

    attribute = {
        name: {
            "r2": _agg([r["attr"][name]["r2"] for r in runs]),
            "spearman": _agg([r["attr"][name]["spearman"] for r in runs]),
        }
        for name in ATTR_NAMES
    }
    return {
        "seen_classes": seen_classes,
        "counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "test_category": int(len(unseen_idx)),
        },
        "closed_set": {
            "val_top1": _agg([r["val_top1"] for r in runs]),
            "test_top1": _agg([r["test_top1"] for r in runs]),
            "full_train": {
                "val_top1": full["val_top1"],
                "test_top1": full["test_top1"],
            },
        },
        "open_set": {
            "auroc": _agg([r["open"]["auroc"] for r in runs]),
            "unseen_reject": _agg([r["open"]["unseen_reject"] for r in runs]),
            "seen_false_reject": _agg([r["open"]["seen_false_reject"] for r in runs]),
            "full_train": full["open"],
        },
        "attribute_probe": attribute,
        "full_train_attribute": full["attr"],
    }


def cross_modal(feats_a, feats_b, meta: dict, split: str = "test") -> dict:
    """Same-instance retrieval after a linear Ridge map A -> B.

    The map is fit on the train split, so this is an expert-level probe, not the
    Phase 4 InfoNCE projection. It answers: can A linearly predict B's frozen
    features well enough to retrieve the matching view?
    """
    splits = np.asarray(meta["splits"])
    train_idx = np.where(splits == "train")[0]
    idx = np.where(splits == split)[0]
    n = len(idx)
    if n < 2 or len(train_idx) < 2:
        return {
            "r1": float("nan"),
            "r5": float("nan"),
            "positives": 0,
            "n": n,
            "method": "ridge_linear_probe",
        }
    scaler_a = StandardScaler().fit(feats_a[train_idx])
    scaler_b = StandardScaler().fit(feats_b[train_idx])
    reg = Ridge(alpha=1.0)
    reg.fit(scaler_a.transform(feats_a[train_idx]), scaler_b.transform(feats_b[train_idx]))
    A = _l2(reg.predict(scaler_a.transform(feats_a[idx])))
    B = _l2(scaler_b.transform(feats_b[idx]))
    sim = A @ B.T
    np.fill_diagonal(sim, -np.inf)
    sample_ids = np.asarray(meta["sample_ids"])[idx]
    pos = sample_ids[:, None] == sample_ids[None, :]
    np.fill_diagonal(pos, False)
    if not pos.any():
        return {
            "r1": float("nan"),
            "r5": float("nan"),
            "positives": 0,
            "n": n,
            "method": "ridge_linear_probe",
        }
    order = np.argsort(-sim, axis=1)
    hits1 = pos[np.arange(n), order[:, 0]]
    k = min(5, n)
    hits5 = np.array([pos[i, order[i, :k]].any() for i in range(n)])
    return {
        "r1": float(hits1.mean()),
        "r5": float(hits5.mean()),
        "positives": int(pos.sum()),
        "n": n,
        "method": "ridge_linear_probe",
    }


# ── cost ────────────────────────────────────────────────────────────────────


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_cost(adapter, views, preserve_aspect, device, repeats: int) -> dict:
    if not views:
        return {
            "latency_ms_p50": float("nan"),
            "latency_ms_p95": float("nan"),
            "peak_memory_mb": float("nan"),
            "params": 0,
            "memory_note": "no views",
        }
    sample = views[0]
    for _ in range(min(3, repeats)):
        adapter.encode_one(sample, preserve_aspect)
    times = []
    for _ in range(max(1, repeats)):
        _sync(device)
        t0 = time.perf_counter()
        adapter.encode_one(sample, preserve_aspect)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    cost_views = views[: min(len(views), 64)]
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        adapter.encode(cost_views, preserve_aspect)
        _sync(device)
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        memory_note = "cuda peak allocated; first 64 views"
    else:
        tracemalloc.start()
        adapter.encode(cost_views, preserve_aspect)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1e6
        memory_note = "tracemalloc peak; torch CPU allocator not included; first 64 views"

    return {
        "latency_ms_p50": float(np.percentile(times, 50)),
        "latency_ms_p95": float(np.percentile(times, 95)),
        "peak_memory_mb": float(peak_mb),
        "params": int(adapter.params()),
        "memory_note": memory_note,
    }


# ── report ──────────────────────────────────────────────────────────────────


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(val) or math.isinf(val):
        return "n/a"
    return f"{val:.{digits}f}"


def _mean_std(agg: dict | None) -> str:
    if not agg:
        return "n/a"
    return f"{_fmt(agg.get('mean'))} ± {_fmt(agg.get('std'))}"


def _cell(value) -> str:
    """Keep one markdown table cell on one line."""
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if math.isnan(val) or math.isinf(val) else val
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj


def _split_counts(splits) -> dict:
    return dict(sorted(Counter(splits).items()))


def render_markdown(payload: dict) -> str:
    lines = [
        "# Phase 3 专家选型结果（自动生成）",
        "",
        f"> 生成时间：{payload['generated']}",
        f"> 数据：`{payload['data']}`",
        f"> 契约：`{payload['experts_config']}`",
        f"> 设备：`{payload['device']}`；随机种子：{payload['seeds']}",
        f"> 视图数：{payload['n_views']}",
        "",
        "> 本文件由 `pss.encoders.scripts.bench_expert_zoo` 生成。"
        "结论请人工写入 `eval/expert_zoo.md`。",
        "",
        "## 1. 数据划分",
        "",
        "| split | views |",
        "|---|---:|",
    ]
    for split, count in payload["counts"].items():
        lines.append(f"| {split} | {count} |")

    results = payload["results"]
    lines += [
        "",
        "## 2. 候选状态",
        "",
        "| ExpertSpec | 模态 | 路线 | 状态 | 说明 |",
        "|---|---|---|---|---|",
    ]
    for name, result in results.items():
        spec = result.get("spec", {})
        detail = result.get("reason", "")
        if result["status"] == "ran":
            detail = f"features {result.get('features_shape')}"
        lines.append(
            f"| `{name}` | {_cell(spec.get('modality', ''))} "
            f"| {_cell(spec.get('route', ''))} | {result['status']} "
            f"| {_cell(detail)} |"
        )

    ran = [(name, result) for name, result in results.items() if result["status"] == "ran"]

    lines += [
        "",
        "## 3. 闭集 top-1",
        "",
        "| ExpertSpec | val | test (mean ± std) | full-train test |",
        "|---|---:|---:|---:|",
    ]
    for name, result in ran:
        closed = result["evaluation"]["closed_set"]
        lines.append(
            f"| `{name}` | {_fmt(closed['val_top1']['mean'])} "
            f"| {_mean_std(closed['test_top1'])} "
            f"| {_fmt(closed['full_train']['test_top1'])} |"
        )

    lines += [
        "",
        "## 4. 开放集（单专家 max-prob 探针，AUROC 以 seen=1）",
        "",
        "| ExpertSpec | AUROC | 未见拒绝率 | 已见误拒率 |",
        "|---|---:|---:|---:|",
    ]
    for name, result in ran:
        open_set = result["evaluation"]["open_set"]
        lines.append(
            f"| `{name}` | {_mean_std(open_set['auroc'])} "
            f"| {_mean_std(open_set['unseen_reject'])} "
            f"| {_mean_std(open_set['seen_false_reject'])} |"
        )

    lines += [
        "",
        "## 5. 属性线性探针（R²，test）",
        "",
        "| ExpertSpec | hardness | roughness | thermal | size_x | size_y | 平均 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in ran:
        probe = result["evaluation"]["attribute_probe"]
        values = [probe.get(attr, {}).get("r2", {}).get("mean") for attr in ATTR_NAMES]
        finite = [float(v) for v in values if v is not None and not math.isnan(float(v))]
        mean = float(np.mean(finite)) if finite else float("nan")
        cells = " | ".join(_fmt(v) for v in values)
        lines.append(f"| `{name}` | {cells} | {_fmt(mean)} |")

    lines += [
        "",
        "## 6. 同实例跨模态检索（test，线性 Ridge 映射）",
        "",
        "| 方向 | R@1 | R@5 | positives |",
        "|---|---:|---:|---:|",
    ]
    if payload["cross_modal"]:
        for pair, metric in payload["cross_modal"].items():
            lines.append(
                f"| `{pair}` | {_fmt(metric.get('r1'))} | {_fmt(metric.get('r5'))} "
                f"| {metric.get('positives', 0)} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | 0 |")

    lines += [
        "",
        "## 7. 成本与许可",
        "",
        "| ExpertSpec | 延迟 p50 (ms) | 延迟 p95 (ms) | 峰值显存 (MB) "
        "| 参数量 | 许可 | 许可通过 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for name, result in ran:
        cost = result["cost"]
        spec = result["spec"]
        lines.append(
            f"| `{name}` | {_fmt(cost['latency_ms_p50'])} "
            f"| {_fmt(cost['latency_ms_p95'])} | {_fmt(cost['peak_memory_mb'])} "
            f"| {cost['params']:,} | {spec.get('license', '')} "
            f"| {'yes' if result.get('license_ok') else 'no'} |"
        )

    lines += [
        "",
        "## 8. 说明",
        "",
        "- 闭集/开放集/属性探针使用 `StandardScaler + LogisticRegression`"
        "（属性用 `Ridge`），只拟合探针，不改专家权重。",
        "- `mean ± std` 来自 YAML 的 seeds；每个 seed 对 train 做 80% 分层子采样，"
        "测试集固定。",
        "- `full-train test` 用完整 train 拟合，作为主参考。",
        "- 延迟是 batch=1 的单样本耗时；峰值显存为近似值。",
        "- 跨模态检索先用 train split 拟合线性 `Ridge` 映射 A→B，再在 test split 上检索；"
        "这是专家级探针，不是 Phase 4 的 InfoNCE 投影。",
        "- 开放集是单专家 max-prob 探针，不是 Phase 5 的原型融合；"
        "不要与 `CATEGORY_PRETRAIN_REPORT.md` 的融合 AUROC 0.960 直接比较。",
        "- `skipped` 不等于失败：多为 `status=planned`、缺少权重或环境不可用。",
        "- 本报告不自动下结论；按 `eval/expert_zoo.md` 的 G1–G4 与 K-E1–K-E5 判据人工决策。",
        "",
    ]
    return "\n".join(lines)


# ── orchestration ───────────────────────────────────────────────────────────


def load_config(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def run_bakeoff(args) -> dict:
    ensure_utf8()
    config = load_config(args.experts)
    views, preserve_aspect = load_raw_views(args.data, args.max_samples)
    if not views:
        raise SystemExit(f"no category views found in {args.data}")

    meta = {
        "categories": [v.category for v in views],
        "splits": [v.split for v in views],
        "instances": [v.instance for v in views],
        "sample_ids": [v.sample_id for v in views],
        "views": [v.view for v in views],
        "attrs": np.stack([v.attrs for v in views]).astype(np.float32),
    }
    device = resolve_device(args.device)
    seeds = (
        [int(seed) for seed in args.seeds]
        if args.seeds is not None
        else [int(seed) for seed in config.get("bakeoff", {}).get("seeds", [0, 1, 2])]
    )
    allowed = set(args.allow_licenses or DEFAULT_ALLOWED_LICENSES)

    results: dict = {}
    features: dict[str, np.ndarray] = {}
    for spec in config.get("experts", []):
        name = str(spec.get("name", ""))
        if args.only and name not in args.only:
            continue
        try:
            adapter = build_adapter(spec, device, args.allow_download)
            feats = adapter.encode(views, preserve_aspect, batch_size=args.batch_size)
            if feats.shape[0] != len(views):
                raise SkipExpert(
                    f"feature rows {feats.shape[0]} != views {len(views)}"
                )
            results[name] = {
                "status": "ran",
                "spec": spec,
                "features_shape": list(feats.shape),
                "evaluation": evaluate_expert(feats, meta, seeds),
                "cost": measure_cost(
                    adapter, views, preserve_aspect, device, args.latency_repeats
                ),
                "license_ok": spec.get("license") in allowed,
            }
            features[name] = feats
            print(f"[ran] {name}: {tuple(feats.shape)}")
        except SkipExpert as exc:
            results[name] = {
                "status": "skipped",
                "spec": spec,
                "reason": str(exc),
            }
            print(f"[skip] {name}: {exc}")
        except Exception as exc:  # keep the bake-off going
            results[name] = {
                "status": "error",
                "spec": spec,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            print(f"[error] {name}: {type(exc).__name__}: {exc}")

    ran_names = [name for name, result in results.items() if result["status"] == "ran"]
    cross: dict = {}
    for i, first in enumerate(ran_names):
        for second in ran_names[i + 1 :]:
            if results[first]["spec"].get("modality") == results[second]["spec"].get("modality"):
                continue
            cross[f"{first}->{second}"] = cross_modal(
                features[first], features[second], meta
            )
            cross[f"{second}->{first}"] = cross_modal(
                features[second], features[first], meta
            )

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data": str(args.data),
        "experts_config": str(args.experts),
        "device": device,
        "seeds": seeds,
        "n_views": len(views),
        "counts": _split_counts(meta["splits"]),
        "results": results,
        "cross_modal": cross,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "expert_zoo.json").write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "expert_zoo.md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {out / 'expert_zoo.md'}")
    print(f"Wrote {out / 'expert_zoo.json'}")
    return payload


def main() -> None:
    ensure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/category_pretrain")
    parser.add_argument("--experts", default="contracts/experts.yaml")
    parser.add_argument("--out", default="eval/results/expert_zoo")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--latency-repeats", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--allow-licenses", nargs="+", default=list(DEFAULT_ALLOWED_LICENSES)
    )
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    run_bakeoff(args)


if __name__ == "__main__":
    main()
