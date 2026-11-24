"""Tests for structured state, tools, and the LLM language adapter."""

from __future__ import annotations

import json

import numpy as np

from pss.heads.text import (
    LLMLanguageHead,
    PSSState,
    ToolAgent,
    ToolRegistry,
    parse_tool_call,
    verify_answer_entities,
    verify_citations,
)
from pss.knowledge import FactStore
from pss.memory.scene_graph import SceneGraph, Track


def _graph():
    graph = SceneGraph()
    for i, (x, category) in enumerate(((0.0, "cup"), (0.2, "bowl"))):
        graph.add(
            Track(
                id=f"obj_{i:04d}",
                pos=np.array([x, 0.0, 0.0]),
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
                category=category,
                confidence=0.9,
            )
        )
    graph.set_relation(
        {
            "source": "obj_0000",
            "target": "obj_0001",
            "type": "spatial",
            "predicate": "near",
            "probability": 0.8,
        }
    )
    return graph


def _store():
    facts = FactStore(
        {
            ("obj_0000", "category"): "cup",
            ("obj_0000", "material"): "ceramic",
            ("obj_0001", "category"): "bowl",
        }
    )
    facts.update("obj_0000", "material", "metal", reason="correction")
    return facts


def test_state_from_scene_graph():
    state = PSSState.from_scene_graph(_graph(), facts=_store(), active_ids=["obj_0000"])
    assert len(state.objects) == 2
    assert state.objects[0]["active"] is True
    assert len(state.relations) == 1
    assert any(f["entity"] == "obj_0000" and f["attribute"] == "material" for f in state.facts)
    assert len(state.audit) >= 1
    assert state.token_estimate() > 0
    assert "STATE" in state.to_context()


def test_tool_registry_read_and_write_permissions():
    facts = _store()
    state = PSSState.from_scene_graph(_graph(), facts=facts)
    read_only = ToolRegistry(state, facts=facts)
    assert read_only.call("get_objects")["ok"] is True
    denied = read_only.call(
        "update_fact", {"entity": "obj_0000", "attribute": "material", "value": "wood"}
    )
    assert denied["ok"] is False
    assert facts.get("obj_0000", "material") == "metal"

    writer = ToolRegistry(state, facts=facts, allow_write=True)
    result = writer.call(
        "update_fact", {"entity": "obj_0000", "attribute": "material", "value": "wood"}
    )
    assert result["ok"] is True
    assert facts.get("obj_0000", "material") == "wood"
    rolled = writer.call("rollback_fact", {"version": 0})
    assert rolled["ok"] is True
    assert facts.get("obj_0000", "material") == "ceramic"


def test_parse_tool_call_nested_args():
    text = 'prefix {"tool": "get_facts", "args": {"entity": "obj_0000"}} suffix'
    call = parse_tool_call(text)
    assert call == {"tool": "get_facts", "args": {"entity": "obj_0000"}}
    assert parse_tool_call("no call here") is None


def test_llm_adapter_tool_loop_and_citations():
    facts = _store()
    state = PSSState.from_scene_graph(_graph(), facts=facts)

    def decoder(prompt: str) -> str:
        if "TOOL_RESULTS:" not in prompt:
            return json.dumps({"tool": "get_facts", "args": {"entity": "obj_0000"}})
        return "obj_0000 material is metal."

    head = LLMLanguageHead(state, facts=facts, decoder=decoder)
    result = head.answer("obj_0000 的材质是什么？")
    assert result["mode"] == "decoder"
    assert len(result["tool_trace"]) == 1
    assert result["tool_trace"][0]["call"]["tool"] == "get_facts"
    citation = verify_citations(result, state)
    assert citation["n_citations"] >= 1
    assert citation["coverage"] == 1.0


def test_hallucination_guard():
    state = PSSState.from_scene_graph(_graph(), facts=_store())
    check = verify_answer_entities("obj_9999 位于 bin7。", state)
    assert check["unsupported"] == ["obj_9999"]
    assert check["supported_rate"] == 0.0


def test_tool_agent_update_and_rollback():
    facts = _store()
    state = PSSState.from_scene_graph(_graph(), facts=facts)
    agent = ToolAgent(state, facts=facts, allow_write=True)
    answer = agent.answer("把 obj_0000 的材质改成 wood，并告诉我新版本号")
    assert facts.get("obj_0000", "material") == "wood"
    assert "version" in answer["answer"].lower()
    rollback = agent.answer("回滚到 v0 并列出审计")
    assert facts.version == 0
    assert rollback["tool_trace"][0]["call"]["tool"] == "rollback_fact"


def test_llm_language_head_converges_on_repeated_tool_call():
    facts = _store()
    state = PSSState.from_scene_graph(_graph(), facts=facts)

    def decoder(prompt: str) -> str:
        return json.dumps({"tool": "get_objects", "args": {}})

    head = LLMLanguageHead(state, facts=facts, decoder=decoder, max_tool_rounds=3)
    result = head.answer("列出所有对象")
    assert result["mode"] == "converged_on_repeated_call"
    assert len(result["tool_trace"]) == 1
