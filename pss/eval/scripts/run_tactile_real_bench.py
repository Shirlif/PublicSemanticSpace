"""Real-tactile bake-off: handcrafted contact statistics vs learned encoders.

Two real datasets are supported:

``physiclear`` (primary)
    Real GelSight clips shipped as 5 RGB frames per clip at 64x64.  Each clip
    carries ordinal hardness / roughness / texture ranks (0-2).  The official
    split is object-level, so clips of one object never cross splits.
    Experts: handcrafted, synthetic 8x8 tactile MLP, PhysiCLeAR CNN
    (real GelSight pretraining), and the WT-MM CNN via a 12x32 bridge.

``wtmm`` (secondary)
    Real WT-MM ``12x32`` contact grids from a SO-100 tactile arm.  There are no
    object labels, so the task is a weak semantic target: classify the gripper
    position (open/closed) from a 5-frame tactile window.  Treat this as a
    signal sanity check, not object recognition.  Absolute gripper position was
    also probed and is not linearly predictable from tactile (R² < 0 for every
    expert), so only the classification is reported.

Only a linear probe is fit; expert weights are frozen.

Example:
    python -m pss.eval.scripts.run_tactile_real_bench \
        --dataset physiclear --out eval/results/tactile_real_bench
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pss._console import ensure_utf8
from pss.encoders.tactile import load_tactile_expert
from pss.encoders.tactile.handcrafted import handcrafted_tactile_features
from pss.encoders.tactile.physiclear import RealTactileEncoder
from pss.encoders.tactile.wtmm import load_wtmm_tactile

PHYS_ROOT = Path("data/physiclear")
WTMM_ROOT = Path("data/public/wt_mm")
PROPERTIES = ("hardness", "roughness", "texture")
DEFAULT_CHECKS = {
    "physiclear_cnn": "checkpoints/tactile_physiclear/tactile_real.pt",
    "synthetic_mlp": "checkpoints/category_pretrain_experts/tactile.pt",
    "wtmm_cnn": "checkpoints/tactile_wtmm/tactile_wtmm.pt",
}


# ── data ────────────────────────────────────────────────────────────────────


def load_physiclear(root: Path = PHYS_ROOT) -> dict[str, list[tuple[np.ndarray, dict]]]:
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    out: dict[str, list[tuple[np.ndarray, dict]]] = {"train": [], "val": [], "test": []}
    for entry in index["entries"]:
        frames = np.stack(
            [
                np.asarray(
                    Image.open(root / Path(str(frame).replace("\\", "/"))).convert("RGB"),
                    dtype=np.uint8,
                )
                for frame in entry["frames"]
            ]
        )
        labels = {name: str(entry[name]) for name in PROPERTIES}
        out[entry["split"]].append((frames, labels))
    return out


def load_wtmm(
    root: Path = WTMM_ROOT, window: int = 5, seed: int = 0
) -> dict[str, list[tuple[np.ndarray, dict]]]:
    from pss.rd2.public.lerobot import LeRobotDataset

    dataset = LeRobotDataset(root, repo_id="WT-MM/lerobot_tactile")
    tactile, frame_df = dataset.tactile_matrix()
    states, _ = dataset.state_matrix()
    if states is None:
        raise SystemExit("WT-MM has no observation.state")
    episodes = frame_df["episode_index"].to_numpy()
    frame_ids = frame_df["frame_index"].to_numpy()

    unique_episodes = sorted(set(int(e) for e in episodes))

    # Only a few WT-MM episodes are cached locally (3 at the time of writing),
    # so an episode-level 70/15/15 split would leave val empty.  Split each
    # episode chronologically instead; the test set is the later part of every
    # episode, which is a temporal generalisation check.
    windows: dict[str, list[tuple[np.ndarray, dict]]] = {"train": [], "val": [], "test": []}
    for episode in unique_episodes:
        mask = episodes == episode
        ep_tactile = tactile[mask]
        ep_states = states[mask]
        ep_frames = frame_ids[mask]
        order = np.argsort(ep_frames, kind="stable")
        ep_tactile, ep_states = ep_tactile[order], ep_states[order]
        if len(ep_tactile) < window:
            continue
        span = max(len(ep_tactile) - 1, 1)
        for start in range(0, len(ep_tactile) - window + 1):
            center = start + window // 2
            fraction = center / span
            split = "train" if fraction < 0.7 else ("val" if fraction < 0.85 else "test")
            windows[split].append(
                (
                    ep_tactile[start : start + window],
                    {"gripper": float(ep_states[center, -1])},
                )
            )
    return windows


# ── experts ─────────────────────────────────────────────────────────────────


class Expert:
    name = "expert"
    kind = "unknown"
    params = 0

    def encode(self, clips: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def encode_one(self, clip: np.ndarray) -> np.ndarray:
        return self.encode([clip])[0]


class HandcraftedTactileExpert(Expert):
    name = "handcrafted_v0"
    kind = "handcrafted_contact_statistics"
    params = 0

    def encode(self, clips: list[np.ndarray]) -> np.ndarray:
        return np.stack([handcrafted_tactile_features(clip) for clip in clips]).astype(
            np.float32
        )

    def encode_one(self, clip: np.ndarray) -> np.ndarray:
        return handcrafted_tactile_features(clip)


def _to_8x8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    if float(arr.max(initial=0.0)) > 1.5:
        arr = arr / 255.0
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    img = img.resize((8, 8), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0


class SyntheticMLPExpert(Expert):
    name = "synthetic_mlp"
    kind = "synthetic_8x8_mlp"
    params = 0

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        self.expert = load_tactile_expert(checkpoint, device=device)
        self.params = int(sum(p.numel() for p in self.expert.encoder.parameters()))

    def encode(self, clips: list[np.ndarray]) -> np.ndarray:
        import torch

        out = []
        for clip in clips:
            x = np.stack([_to_8x8(frame) for frame in clip])
            feats = self.expert.encode(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
            out.append(np.asarray(feats).mean(axis=0))
        return np.stack(out).astype(np.float32)

    def encode_one(self, clip: np.ndarray) -> np.ndarray:
        return self.encode([clip])[0]


class PhysiClearCNNExpert(Expert):
    name = "physiclear_cnn"
    kind = "real_gelsight_cnn"
    params = 0

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        import torch

        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        self.model = RealTactileEncoder(int(payload["out_dim"]))
        self.model.load_state_dict(payload["state_dict"])
        self.model = self.model.to(device).eval()
        self.size = int(payload.get("input_shape", [3, 64, 64])[-1])
        self.device = device
        self.params = int(sum(p.numel() for p in self.model.parameters()))

    def _batch(self, clip: np.ndarray):
        import torch

        arr = np.asarray(clip, dtype=np.float32)
        if float(arr.max(initial=0.0)) > 1.5:
            arr = arr / 255.0
        if arr.ndim == 4:
            if arr.shape[-1] in (3, 4):
                arr = np.transpose(arr[..., :3], (0, 3, 1, 2))
            elif arr.shape[1] == 1:
                arr = np.repeat(arr, 3, axis=1)
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            arr = np.transpose(arr, (0, 3, 1, 2))
        elif arr.ndim == 3:
            arr = np.repeat(arr[..., None], 3, axis=-1)
            arr = np.transpose(arr, (0, 3, 1, 2))
        if arr.shape[-1] != self.size or arr.shape[-2] != self.size:
            import torch.nn.functional as F

            arr = F.interpolate(
                torch.as_tensor(arr), size=(self.size, self.size), mode="bilinear"
            ).numpy()
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def encode(self, clips: list[np.ndarray]) -> np.ndarray:
        import torch

        out = []
        with torch.inference_mode():
            for clip in clips:
                feats = self.model(self._batch(clip)).detach().cpu().numpy()
                out.append(feats.mean(axis=0))
        return np.stack(out).astype(np.float32)

    def encode_one(self, clip: np.ndarray) -> np.ndarray:
        return self.encode([clip])[0]


class WTMMCNNExpert(Expert):
    name = "wtmm_cnn"
    kind = "wtmm_12x32_cnn"
    params = 0

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        self.encoder = load_wtmm_tactile(checkpoint, device=device)
        self.device = device
        self.params = int(sum(p.numel() for p in self.encoder.parameters()))

    def _batch(self, clip: np.ndarray):
        import torch

        arr = np.asarray(clip, dtype=np.float32)
        if float(arr.max(initial=0.0)) > 1.5:
            arr = arr / 255.0
        if arr.ndim == 4:
            arr = arr[..., :3].mean(axis=-1)
        x = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x

    def encode(self, clips: list[np.ndarray]) -> np.ndarray:
        import torch

        out = []
        with torch.inference_mode():
            for clip in clips:
                feats = self.encoder(self._batch(clip)).detach().cpu().numpy()
                out.append(feats.mean(axis=0))
        return np.stack(out).astype(np.float32)

    def encode_one(self, clip: np.ndarray) -> np.ndarray:
        return self.encode([clip])[0]


# ── evaluation ──────────────────────────────────────────────────────────────


def _fit_classifier(train_x, train_y):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )
    clf.fit(train_x, train_y)
    return clf


def _classification_report(
    train_x, train_y, val_x, val_y, test_x, test_y
) -> dict:
    clf = _fit_classifier(train_x, train_y)
    pred_val = clf.predict(val_x)
    pred_test = clf.predict(test_x)
    classes = sorted(set(train_y))
    majority = Counter(train_y).most_common(1)[0][1] / len(train_y)
    return {
        "n_train": len(train_y),
        "n_val": len(val_y),
        "n_test": len(test_y),
        "classes": classes,
        "train_class_counts": dict(Counter(train_y)),
        "majority_baseline": float(majority),
        "val_top1": float(accuracy_score(val_y, pred_val)),
        "test_top1": float(accuracy_score(test_y, pred_test)),
        "test_macro_f1": float(
            f1_score(test_y, pred_test, average="macro", zero_division=0)
        ),
    }


def _measure_latency(expert: Expert, clips: list[np.ndarray], repeats: int = 10) -> dict:
    for _ in range(min(3, repeats)):
        expert.encode_one(clips[0])
    times = []
    for i in range(max(1, repeats)):
        clip = clips[i % len(clips)]
        t0 = time.perf_counter()
        expert.encode_one(clip)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "latency_ms_p50": float(np.percentile(times, 50)),
        "latency_ms_p95": float(np.percentile(times, 95)),
        "params": int(expert.params),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _fmt(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def render_markdown(payload: dict) -> str:
    lines = [
        f"# Real-tactile bake-off: {payload['dataset_label']}",
        "",
        f"> Generated: {payload['generated']}",
        f"> Dataset: {payload['input_note']}",
        f"> Splits: {payload['splits_note']}",
        f"> Samples: {payload['n_train']} / {payload['n_val']} / {payload['n_test']} "
        "(train / val / test).",
        f"> Device: `{payload['device']}`; seed: {payload['seed']}.",
        "",
        "## Experts",
        "",
        "| expert | kind | params |",
        "|---|---|---:|",
    ]
    for name, meta in payload["experts"].items():
        lines.append(f"| `{name}` | {meta['kind']} | {meta['params']:,} |")
    lines += [
        "",
        "## Accuracy by target",
        "",
        "| task | expert | val top-1 | test top-1 | test macro-F1 | majority |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task, report in payload["tasks"].items():
        for name, row in report["experts"].items():
            lines.append(
                f"| `{task}` | `{name}` | {_fmt(row['val_top1'])} "
                f"| {_fmt(row['test_top1'])} | {_fmt(row['test_macro_f1'])} "
                f"| {_fmt(report['majority_baseline'])} |"
            )
    if payload.get("regression"):
        lines += [
            "",
            "## Continuous target (Ridge)",
            "",
            "| target | expert | val R² | test R² |",
            "|---|---|---:|---:|",
        ]
        for target, report in payload["regression"].items():
            for name, row in report["experts"].items():
                lines.append(
                    f"| `{target}` | `{name}` | {_fmt(row['val_r2'])} "
                    f"| {_fmt(row['test_r2'])} |"
                )
    lines += [
        "",
        "## Cost (one clip / window, batch=1)",
        "",
        "| expert | p50 (ms) | p95 (ms) | params |",
        "|---|---:|---:|---:|",
    ]
    for name, cost in payload["cost"].items():
        lines.append(
            f"| `{name}` | {_fmt(cost['latency_ms_p50'])} "
            f"| {_fmt(cost['latency_ms_p95'])} | {cost['params']:,} |"
        )
    lines += ["", "## Automatic reading", ""]
    for task, report in payload["tasks"].items():
        ranked = sorted(
            report["experts"].items(), key=lambda kv: kv[1]["test_top1"], reverse=True
        )
        best = ranked[0]
        hand = report["experts"].get("handcrafted_v0")
        text = f"- `{task}`: best = `{best[0]}` ({_fmt(best[1]['test_top1'])})."
        if hand is not None:
            text += (
                f" handcrafted {_fmt(hand['test_top1'])} "
                f"(macro-F1 {_fmt(hand['test_macro_f1'])})."
            )
        lines.append(text)
    lines += [
        "",
        "## Class distribution (train)",
        "",
    ]
    for task, report in payload["tasks"].items():
        first = next(iter(report["experts"].values()), {})
        lines.append(f"- `{task}`: {first.get('train_class_counts', {})}")
    lines += [
        "",
        "## Boundaries",
        "",
        "- PhysiCLeAR and WT-MM are real tactile datasets, but the tasks are",
        "  property ranks and gripper state, not object recognition.",
        "- PhysiCLeAR tests 39 clips (object-level split); macro-F1 is noisy.",
        "- WT-MM has no object labels; the gripper target is a weak semantic",
        "  proxy and tactile may encode contact directly.",
        "- Absolute gripper position is not linearly predictable from tactile",
        "  (R² < 0 for every expert), so only open/closed is reported.",
        "- The WT-MM slice here is ~4k frames, not the full 103k-frame dataset.",
        "- PhysiCLeAR/Octopi weights and data have no LICENSE; research only.",
        "",
    ]
    return "\n".join(lines)


def run(args) -> dict:
    if args.dataset == "physiclear":
        splits = load_physiclear(Path(args.root) if args.root else PHYS_ROOT)
        dataset_label = "PhysiCLeAR real GelSight"
        input_note = "Real GelSight clips, 5 RGB frames each at 64x64."
        splits_note = "Official object-level train/val/test split."
    else:
        splits = load_wtmm(Path(args.root) if args.root else WTMM_ROOT, seed=args.seed)
        dataset_label = "WT-MM real 12x32 contact grids"
        input_note = "Real WT-MM tactile grids from a SO-100 tactile arm."
        splits_note = (
            "Within-episode chronological 70/15/15 split (only 3 cached episodes); "
            "windows of 5 frames."
        )

    experts: list[Expert] = [HandcraftedTactileExpert()]
    for name in args.experts:
        if name == "synthetic_mlp":
            experts.append(SyntheticMLPExpert(DEFAULT_CHECKS[name], device=args.device))
        elif name == "physiclear_cnn" and args.dataset == "physiclear":
            experts.append(PhysiClearCNNExpert(DEFAULT_CHECKS[name], device=args.device))
        elif name == "wtmm_cnn":
            experts.append(WTMMCNNExpert(DEFAULT_CHECKS[name], device=args.device))

    labels = {
        split: {
            task: [label[task] for _, label in items]
            for task in (
                PROPERTIES if args.dataset == "physiclear" else ("gripper",)
            )
        }
        for split, items in splits.items()
    }
    clips = {split: [clip for clip, _ in items] for split, items in splits.items()}
    if args.dataset == "wtmm":
        values = [label["gripper"] for _, label in splits["train"]]
        threshold = float(np.median(values))
        labels = {
            split: {
                "gripper": [
                    "open" if label["gripper"] >= threshold else "closed"
                    for _, label in items
                ]
            }
            for split, items in splits.items()
        }

    features: dict[str, dict[str, np.ndarray]] = {}
    for expert in experts:
        print(f"[extract] {expert.name}")
        features[expert.name] = {
            split: expert.encode(items) for split, items in clips.items()
        }
        print(f"  {expert.name}: {features[expert.name]['train'].shape}")

    task_names = PROPERTIES if args.dataset == "physiclear" else ("gripper",)
    tasks: dict[str, dict] = OrderedDict()
    for task in task_names:
        tasks[task] = {
            "experts": {
                expert.name: _classification_report(
                    features[expert.name]["train"],
                    labels["train"][task],
                    features[expert.name]["val"],
                    labels["val"][task],
                    features[expert.name]["test"],
                    labels["test"][task],
                )
                for expert in experts
            },
            "majority_baseline": 0.0,
        }
        counts = Counter(labels["train"][task])
        tasks[task]["majority_baseline"] = counts.most_common(1)[0][1] / len(
            labels["train"][task]
        )

    cost = {
        expert.name: _measure_latency(expert, clips["val"][:8], repeats=args.repeats)
        for expert in experts
    }

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "dataset": args.dataset,
        "dataset_label": dataset_label,
        "input_note": input_note,
        "splits_note": splits_note,
        "device": args.device,
        "seed": int(args.seed),
        "n_train": len(clips["train"]),
        "n_val": len(clips["val"]),
        "n_test": len(clips["test"]),
        "experts": {
            expert.name: {"kind": expert.kind, "params": int(expert.params)}
            for expert in experts
        },
        "tasks": tasks,
        "cost": cost,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tactile_real_bench.json").write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "TACTILE_REAL_BENCH.md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {out / 'TACTILE_REAL_BENCH.md'}")
    return payload


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="physiclear", choices=["physiclear", "wtmm"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="eval/results/tactile_real_bench")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument(
        "--experts",
        nargs="+",
        default=["synthetic_mlp", "physiclear_cnn", "wtmm_cnn"],
    )
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
