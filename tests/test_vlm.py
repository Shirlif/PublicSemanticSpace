import numpy as np
import pytest

pytest.importorskip("torch")

from pss.eval.baselines.vlm import (
    MockVLMClient,
    evaluate_vlm_text_state,
    parse_vlm_text,
)


def test_parse_vlm_text():
    text = "obj0: category=cup, x=0.12, y=-0.03\nnoise line"
    parsed = parse_vlm_text(text)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "obj0"
    assert parsed[0]["category"] == "cup"
    assert parsed[0]["pos"] == [0.12, -0.03]


def test_mock_vlm_text_state_retention():
    sample = {
        "rgb": np.zeros((4, 8, 8, 3), dtype=np.uint8),
        "instance_ids": ["cup_00"],
        "categories": ["cup"],
        "poses": np.zeros((4, 1, 7)),
    }
    sample["poses"][:, 0, :3] = [0.1, 0.0, 0.05]
    client = MockVLMClient(sample, dropout=0.0, pos_noise=0.0, seed=0)
    result = evaluate_vlm_text_state(sample, client)
    assert result["id_retention"] == 1.0
