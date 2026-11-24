"""Structured state, tools, and an LLM adapter for the language output head.

The fourth priority is not "train an MLP to write sentences". PSS exposes:

- a compact, auditable structured state (objects, typed relations, facts,
  memory status, uncertainty);
- a small tool registry the decoder can call (query, update, rollback, audit);
- provenance on every tool result, so the language output can cite facts and
  versions.

``LLMLanguageHead`` is the adapter for a real frozen decoder. The default
``TemplateAdapter`` remains the offline path. ``ToolAgent`` is a deterministic
stand-in used by the evaluation harness; it is *not* a language model and the
experiment does not claim language quality from it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pss.knowledge import FactStore
from pss.memory.scene_graph import SceneGraph


# ── structured state ────────────────────────────────────────────────────────


def _round(value, digits: int = 3):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


@dataclass
class PSSState:
    objects: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)
    audit: list[dict] = field(default_factory=list)
    memory: dict = field(default_factory=dict)
    uncertainty: dict = field(default_factory=dict)

    @classmethod
    def from_scene_graph(
        cls,
        graph: SceneGraph,
        facts: FactStore | None = None,
        active_ids: list[str] | None = None,
        uncertainty: dict | None = None,
    ) -> "PSSState":
        objects = []
        for track in graph.tracks.values():
            objects.append(
                {
                    "id": track.id,
                    "category": track.category,
                    "pos": [_round(x) for x in np.asarray(track.pos)[:3]],
                    "confidence": _round(track.confidence),
                    "status": track.status,
                    "last_seen": int(track.last_seen),
                    "missed": int(track.missed),
                    "active": track.id in set(active_ids or []),
                }
            )
        relations = [
            {
                "source": r["source"],
                "target": r["target"],
                "type": r.get("type"),
                "predicate": r["predicate"],
                "probability": _round(r.get("probability", 0.0)),
            }
            for r in graph.get_relations()
        ]
        fact_rows = []
        audit_rows = []
        if facts is not None:
            for (entity, attribute), value in facts.all_facts().items():
                fact_rows.append(
                    {
                        "entity": entity,
                        "attribute": attribute,
                        "value": value,
                        "version": int(facts.version),
                    }
                )
            for entry in facts.audit[-20:]:
                audit_rows.append(
                    {
                        "version": int(entry.version),
                        "action": entry.action,
                        "key": list(entry.key),
                        "old": entry.old,
                        "new": entry.new,
                        "reason": entry.reason,
                    }
                )
        memory = {
            "active_ids": list(active_ids or []),
            "n_objects": len(objects),
        }
        return cls(
            objects=objects,
            relations=relations,
            facts=fact_rows,
            audit=audit_rows,
            memory=memory,
            uncertainty=dict(uncertainty or {}),
        )

    def to_dict(self, max_objects=24, max_relations=48, max_facts=48, max_audit=12) -> dict:
        return {
            "objects": self.objects[:max_objects],
            "relations": self.relations[:max_relations],
            "facts": self.facts[:max_facts],
            "audit": self.audit[-max_audit:],
            "memory": self.memory,
            "uncertainty": self.uncertainty,
            "counts": {
                "objects": len(self.objects),
                "relations": len(self.relations),
                "facts": len(self.facts),
                "audit": len(self.audit),
            },
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(**kwargs), ensure_ascii=False, sort_keys=True)

    def to_context(self, **kwargs) -> str:
        payload = self.to_dict(**kwargs)
        lines = [
            "STATE (structured; only entities and facts listed here are real):",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        ]
        if self.uncertainty:
            lines.append(
                "UNCERTAINTY: "
                + json.dumps(self.uncertainty, ensure_ascii=False, sort_keys=True)
            )
        return "\n".join(lines)

    def token_estimate(self, **kwargs) -> int:
        # Rough whitespace/4 estimate; good enough to compare prompt sizes.
        return max(1, len(self.to_context(**kwargs)) // 4)


# ── tools ───────────────────────────────────────────────────────────────────


TOOL_SPECS = [
    {
        "name": "get_objects",
        "description": "List tracked objects, optionally filtered by category or status.",
        "parameters": {"category": "str?", "status": "str?"},
    },
    {
        "name": "get_relations",
        "description": "List typed relations, optionally filtered by source/target/predicate.",
        "parameters": {"source": "str?", "target": "str?", "predicate": "str?", "min_probability": "float?"},
    },
    {
        "name": "get_facts",
        "description": "List direct facts, optionally filtered by entity/attribute.",
        "parameters": {"entity": "str?", "attribute": "str?"},
    },
    {
        "name": "list_audit",
        "description": "List the most recent audit entries with versions.",
        "parameters": {"limit": "int?"},
    },
    {
        "name": "get_uncertainty",
        "description": "Return low-confidence objects/modalities, if any.",
        "parameters": {},
    },
    {
        "name": "update_fact",
        "description": "Update a fact; requires write permission. Returns the new version.",
        "parameters": {"entity": "str", "attribute": "str", "value": "str", "reason": "str?"},
    },
    {
        "name": "rollback_fact",
        "description": "Roll back the fact store to a version; requires write permission.",
        "parameters": {"version": "int"},
    },
]


class ToolRegistry:
    def __init__(
        self,
        state: PSSState,
        facts: FactStore | None = None,
        allow_write: bool = False,
    ):
        self.state = state
        self.facts = facts
        self.allow_write = bool(allow_write)

    def specs(self) -> list[dict]:
        specs = [s for s in TOOL_SPECS if self.allow_write or not self._is_write(s["name"])]
        return specs

    @staticmethod
    def _is_write(name: str) -> bool:
        return name in ("update_fact", "rollback_fact")

    def call(self, name: str, args: dict | None = None) -> dict:
        args = args or {}
        if self._is_write(name) and not self.allow_write:
            return {"ok": False, "error": "write permission denied", "tool": name}
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown tool {name!r}", "tool": name}
        try:
            result = handler(**args)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "tool": name}
        return {"ok": True, "tool": name, "result": result}

    def _tool_get_objects(self, category=None, status=None):
        rows = [
            o
            for o in self.state.objects
            if (category is None or o["category"] == category)
            and (status is None or o["status"] == status)
        ]
        return {"objects": rows, "source": "scene_graph", "count": len(rows)}

    def _tool_get_relations(self, source=None, target=None, predicate=None, min_probability=0.0):
        rows = [
            r
            for r in self.state.relations
            if (source is None or r["source"] == source)
            and (target is None or r["target"] == target)
            and (predicate is None or r["predicate"] == predicate)
            and float(r.get("probability", 0.0)) >= float(min_probability)
        ]
        return {"relations": rows, "source": "scene_graph", "count": len(rows)}

    def _tool_get_facts(self, entity=None, attribute=None):
        rows = [
            f
            for f in self.state.facts
            if (entity is None or f["entity"] == entity)
            and (attribute is None or f["attribute"] == attribute)
        ]
        citations = [
            {"entity": f["entity"], "attribute": f["attribute"], "version": f["version"]}
            for f in rows
        ]
        return {"facts": rows, "citations": citations, "source": "fact_store", "count": len(rows)}

    def _tool_list_audit(self, limit=10):
        rows = self.state.audit[-int(limit) :]
        return {"audit": rows, "source": "fact_store.audit", "count": len(rows)}

    def _tool_get_uncertainty(self):
        return {"uncertainty": self.state.uncertainty, "source": "pss.uncertainty"}

    def _tool_update_fact(self, entity, attribute, value, reason=""):
        if self.facts is None:
            return {"ok": False, "error": "no fact store"}
        before = self.facts.version
        self.facts.update(entity, attribute, value, reason=reason or "language-head tool")
        return {
            "entity": entity,
            "attribute": attribute,
            "value": value,
            "version_before": int(before),
            "version_after": int(self.facts.version),
            "source": "fact_store",
        }

    def _tool_rollback_fact(self, version):
        if self.facts is None:
            return {"ok": False, "error": "no fact store"}
        self.facts.rollback(int(version))
        return {
            "rolled_back_to": int(version),
            "version": int(self.facts.version),
            "source": "fact_store",
        }


# ── LLM adapter / tool loop ─────────────────────────────────────────────────


SYSTEM_PROMPT = (
    "You are the language head of a PSS cognitive runtime. Answer only from "
    "STATE and tool results. Cite fact entity/attribute/version when you use a "
    "fact. If the answer is not in STATE or a tool result, say 'unknown'. Do "
    "not invent objects, relations, or versions. You may call a tool by "
    "emitting one JSON object: {\"tool\": \"name\", \"args\": {...}}. "
    "If TOOL_RESULTS is present, use them and give the final answer; do not "
    "repeat a tool call that has already been executed."
)


def parse_tool_call(text: str) -> dict | None:
    """Extract a JSON object with a ``tool`` key, including nested args."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for start in starts:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and "tool" in obj:
                        return obj
                    break
    return None


def build_prompt(state: PSSState, question: str, tools: ToolRegistry, tool_trace: list | None = None) -> str:
    tool_lines = [
        json.dumps(spec, ensure_ascii=False, sort_keys=True) for spec in tools.specs()
    ]
    parts = [
        "SYSTEM: " + SYSTEM_PROMPT,
        state.to_context(),
        "TOOLS: " + "\n".join(tool_lines),
    ]
    if tool_trace:
        parts.append(
            "TOOL_RESULTS: "
            + json.dumps(tool_trace[-4:], ensure_ascii=False, sort_keys=True)
        )
    parts.append("QUESTION: " + question)
    parts.append("ANSWER:")
    return "\n".join(parts)


class LLMLanguageHead:
    """Adapter for a frozen decoder plus the PSS tool loop.

    ``decoder(prompt) -> text`` is the only model interface. The head never
    trains it. If no decoder is supplied, it falls back to the template
    adapter for deterministic rendering.
    """

    def __init__(
        self,
        state: PSSState,
        facts: FactStore | None = None,
        decoder: Callable[[str], str] | None = None,
        allow_write: bool = False,
        max_tool_rounds: int = 4,
    ):
        self.state = state
        self.facts = facts
        self.decoder = decoder
        self.tools = ToolRegistry(state, facts=facts, allow_write=allow_write)
        self.max_tool_rounds = int(max_tool_rounds)
        self.trace: list[dict] = []

    def answer(self, question: str) -> dict:
        self.trace = []
        if self.decoder is None:
            return {
                "answer": self._template_answer(question),
                "tool_trace": [],
                "citations": [],
                "mode": "template",
            }
        prompt = build_prompt(self.state, question, self.tools)
        seen: set[str] = set()
        for _ in range(self.max_tool_rounds):
            text = self.decoder(prompt)
            call = parse_tool_call(text)
            if call is None:
                return {
                    "answer": text.strip(),
                    "tool_trace": self.trace,
                    "citations": self._citations(),
                    "mode": "decoder",
                }
            key = json.dumps(call, sort_keys=True, ensure_ascii=False)
            if key in seen:
                # The decoder repeated an executed call; converge on the result.
                last = self.trace[-1]["result"].get("result") if self.trace else None
                return {
                    "answer": self._render_tool_result(last),
                    "tool_trace": self.trace,
                    "citations": self._citations(),
                    "mode": "converged_on_repeated_call",
                }
            seen.add(key)
            result = self.tools.call(call.get("tool"), call.get("args") or {})
            self.trace.append({"call": call, "result": result})
            prompt = build_prompt(self.state, question, self.tools, self.trace)
        last = self.trace[-1]["result"].get("result") if self.trace else None
        return {
            "answer": self._render_tool_result(last),
            "tool_trace": self.trace,
            "citations": self._citations(),
            "mode": "tool_budget_exhausted",
        }

    @staticmethod
    def _render_tool_result(result) -> str:
        if not result:
            return "unknown"
        if isinstance(result, dict):
            if "version_after" in result:
                return f"已更新 {result.get('entity')}.{result.get('attribute')}="                        f"{result.get('value')}，version {result['version_after']}"
            if "rolled_back_to" in result:
                return f"已回滚到 v{result['rolled_back_to']}"
            if "audit" in result:
                return "审计：" + json.dumps(result["audit"], ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

    def _citations(self) -> list[dict]:
        out = []
        for step in self.trace:
            result = step["result"].get("result") or {}
            out.extend(result.get("citations") or [])
        return out

    def _template_answer(self, question: str) -> str:
        # Minimal deterministic fallback; the real path is the decoder.
        q = question.lower()
        if "audit" in q or "审计" in q:
            rows = self.tools.call("list_audit", {"limit": 5})["result"]["audit"]
            return "最近审计：" + json.dumps(rows, ensure_ascii=False)
        if "relation" in q or "关系" in q:
            rows = self.tools.call("get_relations", {})["result"]["relations"]
            return "关系：" + json.dumps(rows, ensure_ascii=False)
        if "fact" in q or "事实" in q:
            rows = self.tools.call("get_facts", {})["result"]["facts"]
            return "事实：" + json.dumps(rows, ensure_ascii=False)
        rows = self.tools.call("get_objects", {})["result"]["objects"]
        return "对象：" + json.dumps(rows, ensure_ascii=False)


class ToolAgent:
    """Deterministic stand-in for a decoder, used only by the harness.

    It maps a small set of question templates to tool calls and renders the
    result. It is not a language model; the harness measures the state/tool
    surface, not generation quality.
    """

    def __init__(self, state: PSSState, facts: FactStore | None = None, allow_write: bool = False):
        self.state = state
        self.facts = facts
        self.tools = ToolRegistry(state, facts=facts, allow_write=allow_write)

    def answer(self, question: str) -> dict:
        q = question.lower()
        trace = []
        if "改成" in q or "改为" in q or "update" in q:
            entity = self._extract_identifier(question)
            value = self._extract_value(question)
            if entity and value:
                result = self.tools.call(
                    "update_fact",
                    {"entity": entity, "attribute": "material", "value": value, "reason": "tool call"},
                )
                trace.append(
                    {
                        "call": {
                            "tool": "update_fact",
                            "args": {"entity": entity, "attribute": "material", "value": value},
                        },
                        "result": result,
                    }
                )
                payload = result.get("result", {})
                return {
                    "answer": (
                        f"已更新 {entity}.material={value}, "
                        f"version {payload.get('version_after', 'unknown')}."
                    ),
                    "tool_trace": trace,
                    "citations": [],
                    "mode": "tool_agent",
                }
        if "audit" in q or "审计" in q or "回滚" in q:
            if "回滚" in q or "rollback" in q:
                version = self._extract_version(question)
                if version is not None:
                    result = self.tools.call("rollback_fact", {"version": version})
                    trace.append({"call": {"tool": "rollback_fact", "args": {"version": version}}, "result": result})
            result = self.tools.call("list_audit", {"limit": 5})
            trace.append({"call": {"tool": "list_audit", "args": {"limit": 5}}, "result": result})
            rows = result["result"]["audit"]
            return {"answer": "审计记录：" + json.dumps(result["result"], ensure_ascii=False), "tool_trace": trace, "citations": [], "mode": "tool_agent"}
        if "关系" in q or "relation" in q:
            predicate = self._extract_predicate(question)
            args = {"predicate": predicate} if predicate else {}
            result = self.tools.call("get_relations", args)
            trace.append({"call": {"tool": "get_relations", "args": args}, "result": result})
            rows = result["result"]["relations"]
            return {"answer": "关系：" + json.dumps(result["result"], ensure_ascii=False), "tool_trace": trace, "citations": [], "mode": "tool_agent"}
        if (
            "事实" in q
            or "fact" in q
            or "material" in q
            or "材质" in q
            or "location" in q
            or "位置" in q
            or "在哪" in q
        ):
            entity = self._extract_identifier(question)
            args = {"entity": entity} if entity else {}
            result = self.tools.call("get_facts", args)
            trace.append({"call": {"tool": "get_facts", "args": args}, "result": result})
            return {
                "answer": "事实：" + json.dumps(result["result"], ensure_ascii=False),
                "tool_trace": trace,
                "citations": result["result"].get("citations", []),
                "mode": "tool_agent",
            }
        if "不确定" in q or "uncertain" in q:
            result = self.tools.call("get_uncertainty", {})
            trace.append({"call": {"tool": "get_uncertainty", "args": {}}, "result": result})
            return {"answer": "不确定项：" + json.dumps(result["result"], ensure_ascii=False), "tool_trace": trace, "citations": [], "mode": "tool_agent"}
        result = self.tools.call("get_objects", {})
        trace.append({"call": {"tool": "get_objects", "args": {}}, "result": result})
        return {"answer": "对象：" + json.dumps(result["result"], ensure_ascii=False), "tool_trace": trace, "citations": [], "mode": "tool_agent"}

    @staticmethod
    def _extract_predicate(question: str) -> str | None:
        for p in ("in", "contact", "near", "left", "right", "above", "below"):
            if p in question.lower():
                return p
        return None

    @staticmethod
    def _extract_identifier(question: str) -> str | None:
        match = re.search(r"\b([a-zA-Z]+_\d+)\b", question)
        return match.group(1) if match else None

    @staticmethod
    def _extract_version(question: str) -> int | None:
        match = re.search(r"(?:v|version|版本)\s*(\d+)", question.lower())
        return int(match.group(1)) if match else None

    @staticmethod
    def _extract_value(question: str) -> str | None:
        match = re.search(r"(?:改成|改为|to)\s*([a-zA-Z_]+)", question.lower())
        return match.group(1) if match else None


def verify_citations(answer: dict, state: PSSState) -> dict:
    """Check that cited facts exist and match the current state."""
    by_key = {(f["entity"], f["attribute"]): f for f in state.facts}
    ok, missing = 0, []
    for citation in answer.get("citations", []):
        key = (citation.get("entity"), citation.get("attribute"))
        if key in by_key:
            ok += 1
        else:
            missing.append(citation)
    return {
        "n_citations": len(answer.get("citations", [])),
        "verified": ok,
        "missing": missing,
        "coverage": ok / max(1, len(answer.get("citations", []))),
    }


def verify_answer_entities(answer_text: str, state: PSSState) -> dict:
    """Flag entity ids in the answer that do not exist in the structured state."""
    known = {o["id"] for o in state.objects} | {f["entity"] for f in state.facts}
    mentioned = set(re.findall(r"\b[a-zA-Z]+_\d+\b", answer_text))
    unsupported = sorted(mentioned - known)
    return {
        "mentioned": sorted(mentioned),
        "unsupported": unsupported,
        "supported_rate": 1.0 - len(unsupported) / max(1, len(mentioned)),
    }

def load_local_decoder(model_id: str | None = None, device: str = "cpu", max_new_tokens: int = 192):
    """Load a cached causal LM as a ``decoder(prompt) -> text`` callable.

    Returns ``None`` when no text model is present in the local HF cache. The
    runtime treats this as "real decoder unavailable" rather than pretending
    the tool agent is a language model.
    """
    candidates = (
        [model_id]
        if model_id
        else [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "HuggingFaceTB/SmolLM-360M-Instruct",
            "HuggingFaceTB/SmolLM-135M-Instruct",
            "google/flan-t5-small",
        ]
    )
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None
    for candidate in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                candidate, local_files_only=True
            ).to(device)
        except Exception:
            continue
        model.eval()

        def decoder(prompt: str, _tok=tokenizer, _model=model) -> str:
            text = prompt
            if getattr(_tok, "chat_template", None):
                text = _tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = _tok(text, return_tensors="pt").to(device)
            with torch.inference_mode():
                output = _model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    repetition_penalty=1.1,
                    pad_token_id=_tok.eos_token_id,
                )
            decoded = _tok.decode(
                output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            return decoded.strip()

        return decoder
    return None
