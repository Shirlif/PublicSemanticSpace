import numpy as np
import pytest

from pss.heads import ActionHead, CallableAdapter, TextHead, ValueHead
from pss.memory import Track


def test_action_head_push_direction_and_force():
    head = ActionHead(tolerance=0.03)
    proposal = head.propose_push_to([0.0, 0.0], [0.2, 0.0])
    assert proposal is not None
    assert proposal.kind == "push"
    assert proposal.target == 0
    assert proposal.params["direction"] == [1.0, 0.0]
    assert proposal.params["force"] > 0
    assert head.propose_push_to([0.0, 0.0], [0.01, 0.0]) is None


def test_action_head_place():
    head = ActionHead()
    proposal = head.propose_place_at(2, [0.1, -0.1])
    assert proposal.kind == "place"
    assert proposal.target == 2
    assert proposal.params["target_xy"] == [0.1, -0.1]


def test_text_head_renders_and_does_not_mutate_state():
    head = TextHead()
    entities = [
        {"id": "cup_00", "category": "cup", "pos": [0.1, 0.2, 0.05]},
        {"id": "plate_00", "category": "plate", "pos": [0.0, 0.0, 0.01]},
    ]
    relations = [("cup_00", "near", "plate_00")]
    snapshot = [dict(e) for e in entities]
    text = head.render(entities, relations)
    assert "cup_00" in text and "靠近" in text
    assert entities == snapshot


def test_text_head_adapter():
    seen = {}

    def fake_llm(prompt: str) -> str:
        seen["prompt"] = prompt
        return "LLM:" + prompt

    head = TextHead(adapter=CallableAdapter(fake_llm))
    text = head.render([{"id": "a", "category": "box", "pos": [0, 0, 0]}], [])
    assert text.startswith("LLM:")
    assert "a" in seen["prompt"]


def test_value_head_prefers_nearer_track():
    goal = np.array([0.2, 0.0])
    head = ValueHead(goal_xy=goal)
    near = Track(id="near", pos=np.array([0.2, 0.0, 0.05]), quat=np.array([1.0, 0, 0, 0]))
    far = Track(id="far", pos=np.array([-0.3, 0.2, 0.05]), quat=np.array([1.0, 0, 0, 0]))
    scores = head.score_tracks([near, far], safety={"near": 0.0, "far": 0.0})
    assert scores["near"] > scores["far"]


def test_value_head_drives_attention_scheduler():
    from pss.memory import AttentionScheduler

    head = ValueHead(goal_xy=np.array([0.2, 0.0]))
    near = Track(id="near", pos=np.array([0.2, 0.0, 0.05]), quat=np.array([1.0, 0, 0, 0]))
    far = Track(id="far", pos=np.array([-0.3, 0.2, 0.05]), quat=np.array([1.0, 0, 0, 0]))
    scores = head.score_tracks([near, far])
    selected, _ = AttentionScheduler(capacity=1).select([near, far], task_scores=scores)
    assert selected == ["near"]


def test_learned_value_head_training_smoke(tmp_path):
    pytest.importorskip("torch")
    from pss.heads.value import (
        evaluate_value_scorer,
        load_learned_value_head,
        train_learned_value_head,
    )

    metrics = train_learned_value_head(
        tmp_path, n_pairs=800, epochs=20, hidden=32, seed=0
    )
    assert metrics["n_pairs"] == 800
    head = load_learned_value_head(tmp_path / "value_head.pt")
    result = evaluate_value_scorer(head.score_features, n_pairs=200, seed=1)
    assert result["pairwise_accuracy"] > 0.7
    assert 0.0 <= result["top1_accuracy"] <= 1.0


def test_learned_action_head_training_smoke(tmp_path):
    pytest.importorskip("mujoco")
    pytest.importorskip("torch")
    from pss.heads.action import load_learned_action_head, train_learned_action_head

    metrics = train_learned_action_head(
        tmp_path, n_demos=4, epochs=2, hidden=16, seed=0
    )
    assert metrics["n_samples"] > 0
    head = load_learned_action_head(tmp_path / "action_head.pt", max_force=10.0)
    proposal = head.propose_push_to([0.0, 0.0], [0.2, 0.0])
    assert proposal is not None
    assert proposal.kind == "push"
    assert proposal.params["force"] > 0
