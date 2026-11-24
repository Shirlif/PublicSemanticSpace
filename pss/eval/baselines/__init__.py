"""Baselines and ablations for Phase 10.

B1 is a **text-state proxy**: no persistent entity slots, one-frame text memory.
It is not a real VLM. B2 is not run (no goal-conditioned policy data).
"""

from __future__ import annotations

import numpy as np

from pss.eval.metrics import id_retention
from pss.memory import Tracker, TrackerConfig
from pss.memory.perception import extract_frame
from pss.prototypes import PrototypeBank, recognize


class TextStateTracker:
    """One-frame text memory: category + position, no dormant tracks."""

    def __init__(self, gate: float = 0.15):
        self.gate = float(gate)
        self.memory: list[dict] = []
        self.next_id = 1

    def step(self, observations, t: int) -> dict:
        events = {"matched": [], "new": [], "dormant": [], "reid": []}
        used: set[int] = set()
        new_memory: list[dict] = []
        for obs in observations:
            best, best_d = None, self.gate
            for i, entry in enumerate(self.memory):
                if i in used or entry["category"] != obs.category:
                    continue
                d = float(np.linalg.norm(entry["pos"] - obs.pos))
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                used.add(best)
                track_id = self.memory[best]["id"]
                events["matched"].append((track_id, obs.gt_id))
            else:
                track_id = f"txt_{self.next_id:05d}"
                self.next_id += 1
                events["new"].append((track_id, obs.gt_id))
            new_memory.append({"id": track_id, "category": obs.category, "pos": obs.pos})
        self.memory = new_memory
        return events


class NoSceneGraphTracker:
    """Frame-local detection: every observation gets a fresh ID."""

    def __init__(self):
        self.next_id = 1

    def step(self, observations, t: int) -> dict:
        events = {"matched": [], "new": [], "dormant": [], "reid": []}
        for obs in observations:
            track_id = f"nsg_{self.next_id:05d}"
            self.next_id += 1
            events["new"].append((track_id, obs.gt_id))
        return events


def evaluate_tracker(sample: dict, models, noise, tracker) -> dict:
    gt_tracks: dict[str, list[tuple[int, str]]] = {}
    for t in range(sample["rgb"].shape[0]):
        observations = extract_frame(sample, t, models, noise)
        events = tracker.step(observations, t)
        for key in ("matched", "reid", "new"):
            for track_id, gt_id in events.get(key, []):
                gt_tracks.setdefault(gt_id, []).append((t, track_id))
    return id_retention(gt_tracks)


def tracker_variants() -> dict:
    return {
        "main": lambda: Tracker(TrackerConfig(), None),
        "pose_only": lambda: Tracker(
            TrackerConfig(
                w_app=0.0,
                w_category=0.0,
                reid_app_min=0.0,
                reid_w_pose=1.0,
                reid_w_app=0.0,
            ),
            None,
        ),
        "appearance_only": lambda: Tracker(
            TrackerConfig(
                w_pose=0.0,
                w_category=0.0,
                gate_radius=10.0,
                reid_pose_gate=10.0,
                reid_w_pose=0.0,
                reid_w_app=1.0,
            ),
            None,
        ),
        "text_state_proxy": lambda: TextStateTracker(),
        "no_scene_graph": lambda: NoSceneGraphTracker(),
    }


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


def _build_bank(features: dict[str, np.ndarray], feats, train_idx, classes: list[str]) -> PrototypeBank:
    protos, stats = {}, {c: {} for c in classes}
    for modality, arr in features.items():
        rows = []
        for c in classes:
            idx = [i for i in train_idx if feats.categories[i] == c]
            v = np.asarray(arr[idx], dtype=float)
            mean = _unit(v.mean(0))
            rows.append(mean)
            stats[c][modality] = {
                "sample_count": len(idx),
                "variance": float(np.mean(1.0 - v @ mean)),
            }
        protos[modality] = np.stack(rows).astype(np.float32)
    return PrototypeBank(categories=list(classes), protos=protos, stats=stats, version="ablation")


def _accuracy(features, bank, feats, test_idx, weights) -> float:
    if len(test_idx) == 0:
        return 0.0
    correct = 0
    for i in test_idx:
        query = {m: features[m][i] for m in features}
        result = recognize(query, bank, weights=weights, use_tree=False)
        correct += int(bool(result.ranked) and result.ranked[0][0] == feats.categories[i])
    return correct / len(test_idx)


def _cross_modal_r1(features, feats, test_idx) -> float:
    if len(test_idx) == 0:
        return 0.0
    correct = 0
    for i in test_idx:
        query = _unit(features["vision"][i])
        sims = [
            float(np.dot(query, _unit(features["tactile"][j])))
            for j in test_idx
        ]
        best = test_idx[int(np.argmax(sims))]
        correct += int(feats.categories[best] == feats.categories[i])
    return correct / len(test_idx)


def recognition_ablations(feats, Z, bank, test_idx, train_idx, classes, seed: int = 0) -> dict:
    weights = {"vision": 1.0, "tactile": 0.5, "geometry": 0.95}
    full = _accuracy(Z, bank, feats, test_idx, weights)
    trained_r1 = _cross_modal_r1(Z, feats, test_idx)

    summed = np.stack([_unit(sum(Z[m][i] for m in Z)) for i in range(len(feats.categories))])
    summed_features = {"sum": summed}
    summed_bank = _build_bank(summed_features, feats, train_idx, classes)
    summed_acc = _accuracy(summed_features, summed_bank, feats, test_idx, {"sum": 1.0})

    rng = np.random.default_rng(seed)
    rotations = {m: _random_orthogonal(Z[m].shape[1], rng) for m in Z}
    random_Z = {
        m: np.stack([_unit(rotations[m] @ Z[m][i]) for i in range(len(Z[m]))])
        for m in Z
    }
    random_bank = _build_bank(random_Z, feats, train_idx, classes)
    random_acc = _accuracy(random_Z, random_bank, feats, test_idx, {"vision": 1.0})
    random_r1 = _cross_modal_r1(random_Z, feats, test_idx)

    return {
        "full_accuracy": full,
        "trained_vision_tactile_r1": trained_r1,
        "cross_modal_sum_accuracy": summed_acc,
        "random_projection_accuracy": random_acc,
        "random_projection_vision_tactile_r1": random_r1,
    }


def no_slot_episode_accuracy(samples, models, bank, noise, max_frames: int = 4) -> dict:
    per_correct = global_correct = n = 0
    for sample in samples:
        for t in range(min(sample["rgb"].shape[0], max_frames)):
            observations = extract_frame(sample, t, models, noise)
            if not observations:
                continue
            for obs in observations:
                result = recognize(
                    {"vision": obs.raw["vision"]},
                    bank,
                    weights={"vision": 1.0},
                    use_tree=False,
                )
                per_correct += int(bool(result.ranked) and result.ranked[0][0] == obs.category)
                n += 1
            global_vec = _unit(np.mean([o.raw["vision"] for o in observations], axis=0))
            result = recognize(
                {"vision": global_vec}, bank, weights={"vision": 1.0}, use_tree=False
            )
            if result.ranked:
                global_correct += sum(
                    int(result.ranked[0][0] == o.category) for o in observations
                )
    return {
        "per_object_acc": per_correct / max(1, n),
        "global_vector_acc": global_correct / max(1, n),
        "n_observations": n,
    }
