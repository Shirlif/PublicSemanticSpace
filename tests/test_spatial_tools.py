"""Tests for the B3 tool layer and the ReAct loop."""

from __future__ import annotations

from pss.eval.spatial import predicates as P
from pss.eval.spatial.baselines import ToolLoopPredictor
from pss.eval.spatial.generator import generate_suite
from pss.eval.spatial.schema import Camera, Item, ObjectSpec, Query, Scene
from pss.eval.spatial.tools import (
    TOOL_PROFILES,
    execute_tool,
    parse_tool_call,
    tool_manifest,
)


def _item() -> Item:
    cam = Camera("cam0", (0.0, -1.0, 0.08), (0.0, 0.0, 0.0))
    a = ObjectSpec("obj_0", "cup", (0.1, 0.0, 0.05), (0.04, 0.04, 0.05))
    b = ObjectSpec("obj_1", "cup", (-0.1, 0.0, 0.05), (0.04, 0.04, 0.05))
    scene = Scene("s", (a, b), (cam,), (-0.3, 0.3, -0.2, 0.2))
    query = Query(
        "q_1",
        "relation",
        "bool",
        "Is obj_0 to the left of obj_1 in the world frame?",
        "no",
        "Answer with exactly one word: yes or no.",
        {"op": "relation", "subject": "obj_0", "object": "obj_1", "predicate": "left_of", "frame": "world"},
    )
    return Item(scene, query)


def test_tool_profiles_are_tiered():
    assert "distance" in TOOL_PROFILES["low"]
    assert "relation" not in TOOL_PROFILES["low"]
    assert "relation" in TOOL_PROFILES["oracle"]
    assert "occluded" not in TOOL_PROFILES["mid"]
    assert "occluded" in TOOL_PROFILES["oracle"]
    low_manifest = tool_manifest("low")
    assert "relation(" not in low_manifest


def test_execute_distance_and_project():
    item = _item()
    out = execute_tool(item, "distance", {"a": "obj_0", "b": "obj_1"}, "low")
    assert out["ok"] and abs(out["result"]["distance"] - 0.2) < 1e-6

    out = execute_tool(
        item,
        "project",
        {"a": "obj_0", "b": "obj_1", "axis": "left", "frame": "world"},
        "low",
    )
    assert out["ok"]
    # obj_0 is on the right, so its projection on the left axis is negative.
    assert out["result"]["value"] < 0
    assert P.left_of(item.scene, item.scene.obj("obj_0"), item.scene.obj("obj_1"), "world") is False


def test_execute_get_object_has_no_answer_leak():
    item = _item()
    out = execute_tool(item, "get_object", {"object_id": "obj_0"}, "low")
    assert out["ok"]
    payload = out["result"]
    assert payload["id"] == "obj_0"
    assert "answer" not in payload
    assert "query" not in payload


def test_execute_unknown_object_and_profile_errors():
    item = _item()
    out = execute_tool(item, "distance", {"a": "obj_99", "b": "obj_1"}, "low")
    assert out["ok"] is False and "error" in out

    out = execute_tool(
        item,
        "relation",
        {"a": "obj_0", "b": "obj_1", "predicate": "left_of"},
        "low",
    )
    assert out["ok"] is False and "not available" in out["error"]


def test_execute_count_and_ordinal_oracle():
    item = _item()
    out = execute_tool(
        item,
        "count",
        {"predicate": "right_of", "reference": "obj_1", "frame": "world"},
        "oracle",
    )
    assert out["ok"] and out["result"]["count"] == 1

    out = execute_tool(item, "ordinal", {"category": "cup", "k": 1, "frame": "world"}, "oracle")
    assert out["ok"] and out["result"]["object_id"] == "obj_1"


def test_execute_relation_matches_predicates():
    item = _item()
    out = execute_tool(
        item,
        "relation",
        {"a": "obj_0", "b": "obj_1", "predicate": "left_of", "frame": "world"},
        "oracle",
    )
    assert out["ok"] and out["result"]["value"] is False


def test_parse_tool_call_tolerates_fences_and_prose():
    text = (
        "I will measure first.\n"
        "```\n"
        'TOOL: {"name": "distance", "args": {"a": "obj_0", "b": "obj_1"}}\n'
        "```"
    )
    call = parse_tool_call(text)
    assert call is not None
    assert call.name == "distance"
    assert call.args == {"a": "obj_0", "b": "obj_1"}
    assert parse_tool_call("no tool here") is None


def test_tool_loop_scripted_model():
    item = _item()
    outputs = [
        'TOOL: {"name": "distance", "args": {"a": "obj_0", "b": "obj_1"}}',
        "Answer: no",
    ]

    class Scripted:
        def __init__(self):
            self.outputs = list(outputs)

        def __call__(self, prompt):
            return self.outputs.pop(0)

    predictor = ToolLoopPredictor(
        Scripted(), profile="low", max_steps=3, supports_messages=False
    )
    assert predictor(item) == "no"
    assert len(predictor.last_trace) == 1
    assert predictor.last_trace[0]["name"] == "distance"
    assert predictor.last_trace[0]["ok"] is True


def test_tool_loop_bad_tool_then_answer():
    item = _item()
    outputs = [
        'TOOL: {"name": "relation", "args": {"a": "obj_0", "b": "obj_1"}}',
        "Answer: yes",
    ]

    class Scripted:
        def __init__(self):
            self.outputs = list(outputs)

        def __call__(self, prompt):
            return self.outputs.pop(0)

    predictor = ToolLoopPredictor(Scripted(), profile="low", max_steps=3)
    assert predictor(item) == "yes"
    assert predictor.last_trace[0]["ok"] is False


def test_tool_loop_stops_at_max_steps():
    item = _item()

    class AlwaysTool:
        def __call__(self, prompt):
            return 'TOOL: {"name": "distance", "args": {"a": "obj_0", "b": "obj_1"}}'

    predictor = ToolLoopPredictor(AlwaysTool(), profile="low", max_steps=2)
    assert predictor(item) is None
    assert len(predictor.last_trace) == 2


def test_tool_loop_passes_chat_messages():
    item = _item()
    seen = []

    def generate(messages):
        seen.append(messages)
        if len(seen) == 1:
            return 'TOOL: {"name": "distance", "args": {"a": "obj_0", "b": "obj_1"}}'
        return "Answer: no"

    generate.supports_messages = True
    predictor = ToolLoopPredictor(generate, profile="low", max_steps=3)
    assert predictor(item) == "no"
    assert isinstance(seen[0], list)
    assert any(
        message["role"] == "user" and message["content"].startswith("TOOL_RESULT:")
        for message in seen[1]
    )


def test_tool_loop_on_generated_item():
    items = generate_suite(n_per_family=1, seed=5, families=["relation"])
    item = items[0]
    answer = item.query.answer

    class Scripted:
        def __init__(self):
            self.outputs = [
                'TOOL: {"name": "distance", "args": {"a": "obj_0", "b": "obj_1"}}',
                f"Answer: {answer}",
            ]

        def __call__(self, prompt):
            return self.outputs.pop(0)

    predictor = ToolLoopPredictor(Scripted(), profile="low", max_steps=3)
    assert predictor(item) == answer
    assert predictor.last_trace
