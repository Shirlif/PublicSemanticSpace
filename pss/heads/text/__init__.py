"""Text readout head (Phase 8).

The head is a pure read-only projection of entity/relation state into text.
It has no parameters and no gradient path to the core representation. An
external frozen LLM can be plugged in via an adapter; the template adapter is
the default because no LLM is installed in this environment.
"""

from __future__ import annotations

from typing import Callable

_PREDICATE_ZH = {
    "in": "在",
    "contact": "接触",
    "near": "靠近",
    "open": "打开",
    "closed": "关闭",
    "held": "拿着",
    "running": "运行中",
    "stopped": "已停止",
}


class TemplateAdapter:
    def render(self, prompt: str) -> str:
        return prompt


class CallableAdapter:
    """Wrap an external (frozen) LLM callable. It must not be trained here."""

    def __init__(self, fn: Callable[[str], str]):
        self.fn = fn

    def render(self, prompt: str) -> str:
        return self.fn(prompt)


class TextHead:
    def __init__(self, adapter=None):
        self.adapter = adapter or TemplateAdapter()

    def render_entities(self, entities: list[dict]) -> str:
        lines = []
        for e in entities:
            pos = e.get("pos", [0.0, 0.0, 0.0])
            category = e.get("category") or "物体"
            lines.append(
                f"{e['id']} 是 {category}，位于 "
                f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})。"
            )
        return "\n".join(lines)

    def render_relations(self, relations) -> str:
        lines = []
        for fact in relations:
            subject, predicate, obj = fact[0], fact[1], (fact[2] if len(fact) > 2 else None)
            verb = _PREDICATE_ZH.get(predicate, predicate)
            if obj is None:
                lines.append(f"{subject} {verb}。")
            else:
                lines.append(f"{subject} {verb} {obj}。")
        return "\n".join(lines)

    def render_plan(self, actions: list[str]) -> str:
        return "计划：" + " → ".join(actions)

    def render(self, entities: list[dict], relations=(), plan: list[str] | None = None) -> str:
        parts = [self.render_entities(entities)]
        if relations:
            parts.append(self.render_relations(relations))
        if plan:
            parts.append(self.render_plan(plan))
        return self.adapter.render("\n".join(p for p in parts if p))


from .structured import (  # noqa: E402  (kept after TextHead to keep the file readable)
    LLMLanguageHead,
    PSSState,
    ToolAgent,
    ToolRegistry,
    build_prompt,
    load_local_decoder,
    parse_tool_call,
    verify_answer_entities,
    verify_citations,
)

__all__ = [
    "CallableAdapter",
    "LLMLanguageHead",
    "PSSState",
    "TemplateAdapter",
    "TextHead",
    "load_local_decoder",
    "ToolAgent",
    "ToolRegistry",
    "build_prompt",
    "parse_tool_call",
    "verify_answer_entities",
    "verify_citations",
]
