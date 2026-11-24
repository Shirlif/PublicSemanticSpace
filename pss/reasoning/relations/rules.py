"""Learnable, versioned geometric relation rules.

The hand-written relation layer used fixed thresholds (``near_dist``,
``contact_pad``, ``bin_half``, ``bin_z``). This module learns those thresholds
from labelled frames and keeps them versioned/auditable. The generated records
use the same ``RelationRecord`` type as ``geometric_relations``, so the runtime
and planners do not change.

Fitting is intentionally small: a grid search per predicate against simulator
ground truth. It is a rule calibration, not a new network.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from pss.sim.scene import CATEGORY_SPECS

from . import (
    BIN_HALF,
    BIN_Z,
    DEFAULT_BINS,
    RelationRecord,
    precision_recall_f1,
)

HAND_RULES = {
    "near_dist": 0.12,
    "contact_pad": 0.01,
    "bin_half": BIN_HALF,
    "bin_z": BIN_Z,
}


@dataclass
class RelationRuleSet:
    near_dist: float = 0.12
    contact_pad: float = 0.01
    bin_half: float = BIN_HALF
    bin_z: float = BIN_Z
    version: int = 1
    source: str = "hand"
    history: list[dict] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "near_dist": float(self.near_dist),
            "contact_pad": float(self.contact_pad),
            "bin_half": float(self.bin_half),
            "bin_z": float(self.bin_z),
            "version": int(self.version),
            "source": self.source,
        }

    def update(self, reason: str = "", **kwargs) -> int:
        before = self.snapshot()
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise KeyError(f"unknown rule {key!r}")
            setattr(self, key, float(value))
        self.version += 1
        self.source = "learned"
        self.history.append(
            {"version": self.version, "before": before, "after": self.snapshot(), "reason": reason}
        )
        return self.version

    def rollback(self, snapshot: dict) -> int:
        for key in ("near_dist", "contact_pad", "bin_half", "bin_z", "source"):
            setattr(self, key, snapshot[key])
        self.version = int(snapshot["version"])
        self.history.append(
            {"version": self.version, "before": None, "after": self.snapshot(), "reason": "rollback"}
        )
        return self.version

    def relations(self, instances: dict[str, dict], bins: dict | None = None) -> list[RelationRecord]:
        bins = DEFAULT_BINS if bins is None else bins
        out: list[RelationRecord] = []
        ids = list(instances)
        for sid, inst in instances.items():
            p = np.asarray(inst["pos"], dtype=float)
            for bname, bpos in bins.items():
                if (
                    abs(p[0] - bpos[0]) < self.bin_half - 0.01
                    and abs(p[1] - bpos[1]) < self.bin_half - 0.01
                    and p[2] < self.bin_z
                ):
                    out.append(RelationRecord(sid, "in", bname, 1.0, "learned_rule"))
        for i, a in enumerate(ids):
            pa = np.asarray(instances[a]["pos"], dtype=float)
            ra = float(instances[a].get("radius", 0.05))
            for b in ids[i + 1 :]:
                pb = np.asarray(instances[b]["pos"], dtype=float)
                rb = float(instances[b].get("radius", 0.05))
                d = float(np.linalg.norm(pa - pb))
                if d <= ra + rb + self.contact_pad:
                    out.append(RelationRecord(a, "contact", b, 1.0, "learned_rule"))
                    out.append(RelationRecord(b, "contact", a, 1.0, "learned_rule"))
                if d < self.near_dist:
                    conf = float(np.clip(1.0 - d / self.near_dist, 0.05, 1.0))
                    out.append(RelationRecord(a, "near", b, conf, "learned_rule"))
                    out.append(RelationRecord(b, "near", a, conf, "learned_rule"))
        return out

    # ── fitting ─────────────────────────────────────────────────────────────

    @classmethod
    def fit(cls, episodes, min_f1: float = 0.0) -> "RelationRuleSet":
        """Fit thresholds on ``[(sample, frame), ...]`` with GT relations."""
        pair_rows, in_rows = [], []
        for sample, frame in episodes:
            instances = _instances(sample, frame)
            truth = _truth(sample["relations"][frame])
            ids = list(instances)
            for i, a in enumerate(ids):
                pa = np.asarray(instances[a]["pos"], dtype=float)
                for bname, bpos in DEFAULT_BINS.items():
                    in_rows.append(
                        (
                            abs(pa[0] - bpos[0]),
                            abs(pa[1] - bpos[1]),
                            pa[2],
                            float((a, "in", bname) in truth),
                        )
                    )
                for b in ids[i + 1 :]:
                    pb = np.asarray(instances[b]["pos"], dtype=float)
                    d = float(np.linalg.norm(pa - pb))
                    rsum = float(instances[a].get("radius", 0.05) + instances[b].get("radius", 0.05))
                    pair_rows.append(
                        (
                            d,
                            d - rsum,
                            float((a, "near", b) in truth or (b, "near", a) in truth),
                            float((a, "contact", b) in truth or (b, "contact", a) in truth),
                        )
                    )
        rules = cls()
        if pair_rows and any(r[2] > 0 for r in pair_rows):
            rules.near_dist = _best_threshold(
                [r[0] for r in pair_rows], [r[2] for r in pair_rows], lower_is_positive=True
            )
        if pair_rows and any(r[3] > 0 for r in pair_rows):
            rules.contact_pad = _best_threshold(
                [r[1] for r in pair_rows], [r[3] for r in pair_rows], lower_is_positive=True
            )
        if in_rows and any(r[3] > 0 for r in in_rows):
            best = (-1.0, rules.bin_half, rules.bin_z)
            for half in np.arange(0.03, 0.121, 0.005):
                for z in np.arange(0.08, 0.201, 0.01):
                    pred = [
                        float(abs(dx) < half - 0.01 and abs(dy) < half - 0.01 and pz < z)
                        for dx, dy, pz, _ in in_rows
                    ]
                    truth = [row[3] for row in in_rows]
                    f1 = _f1(pred, truth)
                    if f1 > best[0]:
                        best = (f1, float(half), float(z))
            rules.bin_half, rules.bin_z = best[1], best[2]
        rules.source = "learned"
        return rules


def _instances(sample, frame: int) -> dict[str, dict]:
    out = {}
    for i, instance_id in enumerate(sample["instance_ids"]):
        category = str(sample["categories"][i])
        size = CATEGORY_SPECS.get(category, {}).get("size", (0.05, 0.05))
        out[str(instance_id)] = {
            "pos": np.asarray(sample["poses"][frame, i, :3], dtype=float),
            "radius": max(float(x) for x in size) / 2.0,
        }
    return out


def _truth(rows) -> set[tuple]:
    out = set()
    for row in rows:
        fact = tuple(row)
        out.add(fact)
        if fact[1] in ("contact", "near"):
            out.add((fact[2], fact[1], fact[0]))
    return out


def _f1(pred, truth) -> float:
    tp = sum(1 for p, t in zip(pred, truth) if p and t)
    fp = sum(1 for p, t in zip(pred, truth) if p and not t)
    fn = sum(1 for p, t in zip(pred, truth) if not p and t)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _best_threshold(values, labels, lower_is_positive: bool = True) -> float:
    order = np.argsort(values)
    values = np.asarray(values)[order]
    labels = np.asarray(labels)[order]
    best = (-1.0, float(np.median(values)) if len(values) else 0.0)
    for i in range(len(values) + 1):
        threshold = float(values[i]) if i < len(values) else float(values[-1]) + 1e-6
        pred = [float(v < threshold) for v in values] if lower_is_positive else [
            float(v > threshold) for v in values
        ]
        f1 = _f1(pred, labels)
        if f1 > best[0]:
            best = (f1, threshold)
    return best[1]
