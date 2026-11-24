from pathlib import Path

import numpy as np
import pytest

from pss.memory import (
    AttentionScheduler,
    DecompositionRule,
    SceneGraph,
    Track,
    Tracker,
    WorkingMemory,
)
from pss.memory.perception import Observation, PoseNoise

DATA = Path("data/v1")
EXPERTS = Path("checkpoints/expert_v1")
PROJECTIONS = Path("checkpoints/projection_v1")


def _obs(gt_id, pos, appearance=None, frame=0):
    app = np.array([1.0, 0.0, 0.0, 0.0]) if appearance is None else np.asarray(appearance, float)
    return Observation(
        gt_id=gt_id,
        pos=np.asarray(pos, dtype=float),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        appearance=app,
        raw={"vision": app},
        confidence=1.0,
        category="x",
        frame=frame,
    )


def test_tracking_keeps_id_across_frames():
    tracker = Tracker()
    ev0 = tracker.step([_obs("a", [0.0, 0.0, 0.0])], 0)
    track_id = ev0["new"][0][0]
    ev1 = tracker.step([_obs("a", [0.01, 0.0, 0.0])], 1)
    assert ev1["matched"][0][0] == track_id


def test_reid_after_occlusion():
    tracker = Tracker()
    ev0 = tracker.step([_obs("a", [0.0, 0.0, 0.0])], 0)
    track_id = ev0["new"][0][0]
    for t in (1, 2, 3):
        tracker.step([], t)
    ev = tracker.step([_obs("a", [0.01, 0.0, 0.0])], 4)
    assert ev["reid"]
    assert ev["reid"][0][0] == track_id


def test_new_object_creates_track():
    tracker = Tracker()
    ev = tracker.step([_obs("a", [0.0, 0.0, 0.0]), _obs("b", [1.0, 1.0, 1.0])], 0)
    assert len(ev["new"]) == 2


def test_spatial_index_query():
    graph = SceneGraph(cell_size=0.1)
    graph.add(Track(id="a", pos=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0])))
    graph.add(Track(id="b", pos=np.array([1.0, 1.0, 1.0]), quat=np.array([1.0, 0.0, 0.0, 0.0])))
    near = graph.query_near([0.0, 0.0, 0.0], 0.2)
    assert [t.id for t in near] == ["a"]


def test_scene_graph_save_load(tmp_path):
    graph = SceneGraph()
    graph.add(
        Track(
            id="a",
            pos=np.array([0.1, 0.2, 0.3]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            raw={"vision": np.ones(4)},
        )
    )
    path = tmp_path / "graph.json"
    graph.save(path)
    loaded = SceneGraph.load(path)
    assert "a" in loaded.tracks
    assert np.allclose(loaded.get("a").pos, [0.1, 0.2, 0.3])
    assert np.allclose(loaded.get("a").raw["vision"], np.ones(4))


def _unit(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=128)
    return v / np.linalg.norm(v)


def _track(i):
    raw = {
        "vision": _unit(10 * i + 1),
        "tactile": _unit(10 * i + 2),
        "geometry": _unit(10 * i + 3),
    }
    return Track(
        id=f"e{i}",
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        appearance=raw["vision"],
        raw=raw,
        confidence=1.0,
    )


def test_working_memory_capacity_and_recall():
    tracks = [_track(i) for i in range(6)]
    wm = WorkingMemory(AttentionScheduler(capacity=4))
    selected = wm.set_scene(
        tracks, task_scores={t.id: (1.0 if t.id == "e5" else 0.0) for t in tracks}
    )
    assert "e5" in selected
    assert len(wm.active_ids()) == 4
    assert wm.recall("e5") > 0.999

    selected = wm.set_scene(
        tracks, task_scores={t.id: (1.0 if t.id == "e0" else 0.0) for t in tracks}
    )
    assert "e0" in selected
    assert wm.recall("e0") > 0.999


def test_pose_noise_deterministic_and_disable():
    n1, n2 = PoseNoise(seed=0), PoseNoise(seed=0)
    p1, q1 = n1.apply("a", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    p2, q2 = n2.apply("a", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(p1, p2) and np.allclose(q1, q2)

    off = PoseNoise(seed=0, enabled=False)
    p, q = off.apply("a", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(p, [0.0, 0.0, 0.0]) and np.allclose(q, [1.0, 0.0, 0.0, 0.0])


def test_reid_candidates_include_historical_with_age_bound():
    graph = SceneGraph()
    graph.add(
        Track(
            id="old",
            pos=np.zeros(3),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            status="historical",
            last_seen=0,
        )
    )
    assert [t.id for t in graph.reid_candidates(t=5, max_age=10)] == ["old"]
    assert graph.reid_candidates(t=20, max_age=10) == []


def _unit_vec(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_witnessed_fragmentation_is_deterministic():
    graph = SceneGraph()
    parent = Track(
        id="cup",
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        raw={"vision": _unit_vec([1.0, 0.0, 0.0])},
        category="cup",
    )
    graph.add(parent)
    children = [
        Track(
            id="shard0",
            pos=np.array([0.02, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            raw={"vision": _unit_vec([1.0, 0.0, 0.0])},
        ),
        Track(
            id="shard1",
            pos=np.array([0.03, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            raw={"vision": _unit_vec([1.0, 0.0, 0.0])},
        ),
    ]
    rule = DecompositionRule(graph)
    hypothesis = rule.witness("cup", children, t=4)
    assert hypothesis.status == "deterministic"
    assert hypothesis.confidence == 1.0
    assert hypothesis.applied
    assert graph.get("cup").status == "fragmented"
    assert graph.get("shard0").parent == "cup"
    assert graph.get("shard0").provenance["relation"] == "part-of-fracture"


def test_result_only_fragmentation_accepts_consistent_shards():
    graph = SceneGraph()
    parent = Track(
        id="cup",
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        raw={"vision": _unit_vec([1.0, 0.0, 0.0])},
        category="cup",
    )
    graph.add(parent)
    shards = [
        Track(
            id="shard0",
            pos=np.array([0.03, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            raw={"vision": _unit_vec([1.0, 0.0, 0.0])},
        ),
        Track(
            id="shard1",
            pos=np.array([0.04, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            raw={"vision": _unit_vec([1.0, 0.0, 0.0])},
        ),
    ]
    rule = DecompositionRule(graph)
    hypothesis = rule.infer_result_only(parent, shards, t=5)
    assert hypothesis.status == "probabilistic"
    assert hypothesis.confidence > 0.9
    assert hypothesis.applied
    assert graph.get("cup").status == "fragmented"
    assert graph.get("shard1").provenance["decomposition_status"] == "probabilistic"


def test_result_only_fragmentation_stays_uncertain_when_evidence_is_weak():
    graph = SceneGraph()
    parent = Track(
        id="cup",
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        raw={"vision": _unit_vec([1.0, 0.0, 0.0])},
        category="cup",
    )
    graph.add(parent)
    shards = [
        Track(
            id="shard0",
            pos=np.array([0.5, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            raw={"vision": _unit_vec([0.0, 1.0, 0.0])},
        ),
        Track(
            id="shard1",
            pos=np.array([0.51, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            raw={"vision": _unit_vec([0.0, 0.0, 1.0])},
        ),
    ]
    rule = DecompositionRule(graph)
    hypothesis = rule.infer_result_only(parent, shards, t=5)
    assert hypothesis.status == "uncertain"
    assert not hypothesis.applied
    assert graph.get("cup").status == "active"
    assert graph.get("shard0") is None


def test_decomposition_hypothesis_can_be_retracted():
    graph = SceneGraph()
    parent = Track(
        id="cup",
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        raw={"vision": _unit_vec([1.0, 0.0, 0.0])},
        category="cup",
    )
    graph.add(parent)
    shards = [
        Track(
            id="shard0",
            pos=np.array([0.03, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            raw={"vision": _unit_vec([1.0, 0.0, 0.0])},
        ),
        Track(
            id="shard1",
            pos=np.array([0.04, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            raw={"vision": _unit_vec([1.0, 0.0, 0.0])},
        ),
    ]
    rule = DecompositionRule(graph)
    hypothesis = rule.infer_result_only(parent, shards, t=5)
    assert hypothesis.applied
    rule.retract(hypothesis.id, reason="parent observed intact", t=7)
    assert hypothesis.status == "retracted"
    assert graph.get("cup").status == "active"
    assert graph.get("shard0").parent is None
    assert graph.get("shard0").provenance["decomposition_status"] == "retracted"


def test_decomposition_hypothesis_revision_can_promote():
    graph = SceneGraph()
    parent = Track(
        id="cup",
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        raw={"vision": _unit_vec([1.0, 0.0, 0.0])},
        category="cup",
    )
    graph.add(parent)
    weak_shards = [
        Track(
            id="shard0",
            pos=np.array([0.5, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            raw={"vision": _unit_vec([0.0, 1.0, 0.0])},
        ),
        Track(
            id="shard1",
            pos=np.array([0.51, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            raw={"vision": _unit_vec([0.0, 0.0, 1.0])},
        ),
    ]
    rule = DecompositionRule(graph)
    weak = rule.infer_result_only(parent, weak_shards, t=5)
    assert weak.status == "uncertain"

    strong_shards = [
        Track(
            id="shard0",
            pos=np.array([0.03, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            raw={"vision": _unit_vec([1.0, 0.0, 0.0])},
        ),
        Track(
            id="shard1",
            pos=np.array([0.04, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            raw={"vision": _unit_vec([1.0, 0.0, 0.0])},
        ),
    ]
    revised = rule.revise(weak.id, parent, strong_shards, t=6)
    assert revised.status == "probabilistic"
    assert revised.applied
    assert graph.get("cup").status == "fragmented"


def test_scene_graph_fragment_marks_parent_and_children():
    graph = SceneGraph()
    parent = Track(
        id="cup",
        pos=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        status="active",
    )
    graph.add(parent)
    children = [
        Track(id="shard0", pos=np.array([0.0, 0.0, 0.0]), quat=np.array([1.0, 0.0, 0.0, 0.0])),
        Track(id="shard1", pos=np.array([0.1, 0.0, 0.0]), quat=np.array([1.0, 0.0, 0.0, 0.0])),
    ]
    out = graph.fragment("cup", children, t=3)
    assert len(out) == 2
    assert graph.get("cup").status == "fragmented"
    assert graph.get("shard0").parent == "cup"
    assert graph.get("shard0").provenance["fractured_from"] == "cup"


@pytest.mark.skipif(
    not (DATA.exists() and EXPERTS.exists() and PROJECTIONS.exists()),
    reason="run dataset generation, expert training, and projection training first",
)
def test_extract_frame_real_data():
    pytest.importorskip("mujoco")
    pytest.importorskip("torch")

    from pss.memory.perception import extract_frame, load_perception_models
    from pss.sim.dataset import load_sample

    sample = load_sample(DATA / "occlusion" / "occ_0000.npz")
    models = load_perception_models(EXPERTS, PROJECTIONS, device="cpu")
    obs = extract_frame(sample, 0, models, PoseNoise(seed=0))
    assert len(obs) >= 1
    assert obs[0].appearance.shape == (128,)
    assert "vision" in obs[0].raw
    assert set(obs[0].raw) <= {"vision", "tactile", "geometry"}
    # Zero-signal channels are treated as absent sensors.
    tac_std = float(np.std(sample["tactile"][0]))
    assert ("tactile" in obs[0].raw) == (tac_std > 1e-6)
