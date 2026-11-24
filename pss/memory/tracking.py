"""Multi-object tracker with pose + appearance + time association (Phase 6).

Association never uses GT ids. GT ids ride along in observations for
evaluation only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .perception import Observation
from .scene_graph import SceneGraph, Track


@dataclass
class TrackerConfig:
    gate_radius: float = 0.25
    max_cost: float = 1.0
    w_pose: float = 0.6
    w_app: float = 0.3
    w_time: float = 0.1
    w_category: float = 0.05
    appearance_alpha: float = 0.2
    velocity_alpha: float = 0.3
    dormancy_missed: int = 2
    missing_missed: int = 6
    historical_missed: int = 12
    reid_max_age: int = 150
    reid_pose_gate: float = 0.5
    reid_app_min: float = 0.5
    reid_w_pose: float = 0.0
    reid_w_app: float = 1.0
    metric: object | None = None
    metric_max_cost: float = 0.9


def _normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _category_distance(track_category: str | None, obs_category: str | None) -> float:
    """0 when compatible (unknown on either side, or equal), 1 on conflict."""
    if track_category is None or obs_category is None:
        return 0.0
    return 0.0 if track_category == obs_category else 1.0


class Tracker:
    def __init__(self, config: TrackerConfig | None = None, graph: SceneGraph | None = None):
        self.config = config or TrackerConfig()
        self.graph = graph or SceneGraph()

    def step(self, observations: list[Observation], t: int) -> dict:
        cfg = self.config
        active = [tr for tr in self.graph.tracks.values() if tr.status == "active"]
        predicted = {tr.id: tr.pos + tr.velocity for tr in active}

        cost = np.full((len(active), len(observations)), 1e6, dtype=float)
        for i, track in enumerate(active):
            for j, obs in enumerate(observations):
                d_pos = float(np.linalg.norm(predicted[track.id] - obs.pos))
                if d_pos > cfg.gate_radius:
                    continue
                d_app = (
                    1.0 - float(np.dot(track.appearance, obs.appearance))
                    if track.appearance is not None
                    else 0.5
                )
                c = (
                    cfg.w_pose * (d_pos / cfg.gate_radius)
                    + cfg.w_app * max(0.0, d_app)
                    + cfg.w_time * min(1.0, track.missed / 5.0)
                    + cfg.w_category
                    * _category_distance(track.category, obs.category)
                )
                if c <= cfg.max_cost:
                    cost[i, j] = c

        events = {"matched": [], "new": [], "dormant": [], "reid": []}
        matched_tracks: set[str] = set()
        matched_obs: set[int] = set()

        if cost.size:
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= cfg.max_cost:
                    track, obs = active[int(r)], observations[int(c)]
                    self._update(track, obs, t)
                    matched_tracks.add(track.id)
                    matched_obs.add(id(obs))
                    events["matched"].append((track.id, obs.gt_id))

        for track in active:
            if track.id in matched_tracks:
                continue
            track.missed += 1
            if track.missed >= cfg.historical_missed:
                track.status = "historical"
            elif track.missed >= cfg.missing_missed:
                track.status = "missing"
            elif track.missed >= cfg.dormancy_missed:
                track.status = "dormant"
            if track.status != "active":
                events["dormant"].append((track.id, track.status))

        unmatched = [obs for obs in observations if id(obs) not in matched_obs]
        if (
            cfg.metric is not None
            and hasattr(cfg.metric, "cost_matrix")
            and unmatched
        ):
            candidates = self.graph.reid_candidates(t, cfg.reid_max_age)
            if candidates:
                cost = cfg.metric.cost_matrix(candidates, unmatched)
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    if cost[r, c] <= cfg.metric_max_cost:
                        self._update(candidates[r], unmatched[c], t)
                        events["reid"].append((candidates[r].id, unmatched[c].gt_id))
                        matched_obs.add(id(unmatched[c]))
                for obs in unmatched:
                    if id(obs) not in matched_obs:
                        track = self._new_track(obs, t)
                        events["new"].append((track.id, obs.gt_id))
                return events

        for obs in observations:
            if id(obs) in matched_obs:
                continue
            best: Track | None = None
            best_cost = 1e9
            for track in self.graph.reid_candidates(t, cfg.reid_max_age):
                d_pos = float(np.linalg.norm(track.pos - obs.pos))
                if cfg.metric is not None:
                    c = float(cfg.metric.cost(track, obs))
                else:
                    if d_pos > cfg.reid_pose_gate:
                        continue
                    d_app = (
                        1.0 - float(np.dot(track.appearance, obs.appearance))
                        if track.appearance is not None
                        else 1.0
                    )
                    if d_app > (1.0 - cfg.reid_app_min):
                        continue
                    c = (
                        cfg.reid_w_pose * (d_pos / cfg.reid_pose_gate)
                        + cfg.reid_w_app * d_app
                        + 0.1 * _category_distance(track.category, obs.category)
                    )
                if c < best_cost:
                    best, best_cost = track, c
            if best is not None:
                self._update(best, obs, t)
                events["reid"].append((best.id, obs.gt_id))
            else:
                track = self._new_track(obs, t)
                events["new"].append((track.id, obs.gt_id))

        return events

    def _update(self, track: Track, obs: Observation, t: int) -> None:
        cfg = self.config
        old_pos = track.pos.copy()
        if track.appearance is not None:
            track.appearance = _normalize(
                (1.0 - cfg.appearance_alpha) * track.appearance
                + cfg.appearance_alpha * obs.appearance
            )
        track.velocity = (
            (1.0 - cfg.velocity_alpha) * track.velocity
            + cfg.velocity_alpha * (obs.pos - track.pos)
        )
        track.pos = obs.pos.copy()
        track.quat = obs.quat.copy()
        if obs.category is not None:
            track.category = obs.category
        track.raw = {k: np.asarray(v, dtype=float).copy() for k, v in obs.raw.items()}
        track.confidence = obs.confidence
        track.last_seen = t
        track.missed = 0
        track.status = "active"
        track.observations += 1
        self.graph.index.update(track.id, old_pos, track.pos)

    def _new_track(self, obs: Observation, t: int) -> Track:
        track = Track(
            id=self.graph.new_id(),
            pos=obs.pos.copy(),
            quat=obs.quat.copy(),
            appearance=_normalize(obs.appearance),
            category=obs.category,
            confidence=obs.confidence,
            first_seen=t,
            last_seen=t,
            raw={k: np.asarray(v, dtype=float).copy() for k, v in obs.raw.items()},
        )
        self.graph.add(track)
        return track
