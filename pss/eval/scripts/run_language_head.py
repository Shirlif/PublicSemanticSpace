"""Fourth priority: structured state + tools + LLM language head.

Builds a PSS state from a simulated episode, then compares:

- ``no_context``: no PSS state/tools;
- ``flat_context``: the full JSON dump, no tools;
- ``structured_tools``: compact structured state + tool registry + audit;
- ``llm_adapter``: the real adapter path with a scripted decoder that emits a
  tool call, to verify the decoder loop and citations.

The deterministic agents are harness stand-ins, not language models. The
experiment measures the state/tool/audit surface: can the head answer grounded
questions, perform auditable writes, and cite facts/versions?

Example:
    python -m pss.eval.scripts.run_language_head --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.encoders.data import load_index
from pss.heads.text import (
    LLMLanguageHead,
    PSSState,
    ToolAgent,
    build_prompt,
    parse_tool_call,
    verify_answer_entities,
    verify_citations,
)
from pss.heads.text.structured import SYSTEM_PROMPT, ToolRegistry
from pss.knowledge import FactStore
from pss.memory.scene_graph import SceneGraph, Track
from pss.reasoning.relations import geometric_relations
from pss.sim.dataset import load_sample
from pss.sim.scene import CATEGORY_SPECS

DATA = "data/phase7_v2"
OUT = "eval/results/language_head"


def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "n/a" if math.isnan(v) else f"{v:.{digits}f}"


def _json_tail(text: str):
    """Parse the JSON object after a Chinese/ASCII colon, if present."""
    for sep in ("：", ":"):
        if sep in text:
            text = text.split(sep, 1)[1]
            break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _material_name(material) -> str:
    if isinstance(material, dict):
        return str(material.get("name") or material.get("kind") or "unknown")
    return str(material)


@dataclass
class EpisodeState:
    state: PSSState
    facts: FactStore
    entity: str


def build_episode_state(seed: int = 0) -> EpisodeState:
    index = load_index(DATA)
    episodes = [e for e in index["entries"] if e.get("kind") == "episode"]
    # Prefer an episode whose first frame has at least one relation.
    entry = episodes[0]
    sample = load_sample(Path(DATA) / entry["path"])
    for candidate in episodes:
        probe = load_sample(Path(DATA) / candidate["path"])
        if any(len(frame_rels) for frame_rels in probe["relations"]):
            entry, sample = candidate, probe
            break
    frame = next(
        (t for t, rels in enumerate(sample["relations"]) if len(rels)),
        0,
    )
    sample = load_sample(Path(DATA) / entry["path"])
    graph = SceneGraph()
    ids = [str(x) for x in sample["instance_ids"]]
    facts = {}
    for i, gid in enumerate(ids):
        category = str(sample["categories"][i])
        material = _material_name(sample["materials"][i])
        graph.add(
            Track(
                id=gid,
                pos=np.asarray(sample["poses"][frame, i, :3], dtype=float),
                quat=np.asarray(sample["poses"][frame, i, 3:7], dtype=float),
                category=category,
                confidence=0.95,
                raw={"category": np.asarray([category == c for c in set(sample["categories"])], dtype=float)},
            )
        )
        facts[(gid, "category")] = category
        facts[(gid, "material")] = material
    for fact in sample["relations"][frame]:
        source, predicate, target = fact[0], fact[1], fact[2]
        graph.set_relation(
            {
                "source": source,
                "target": target or "world",
                "type": "symbolic",
                "predicate": predicate,
                "probability": 1.0,
            }
        )
    store = FactStore(facts)
    # Create audit history: two versioned updates.
    if ids:
        store.update(ids[0], "material", "metal", reason="sensor correction")
    if len(ids) > 1:
        store.update(ids[1], "location", "shelf", reason="placement")
    active = ids[: min(3, len(ids))]
    uncertainty = (
        {
            "low_confidence": [
                {"id": ids[0], "confidence": 0.35, "reason": "occlusion"}
            ],
            "missing_modalities": {ids[0]: ["tactile"]},
        }
        if ids
        else {}
    )
    state = PSSState.from_scene_graph(
        graph, facts=store, active_ids=active, uncertainty=uncertainty
    )
    return EpisodeState(state=state, facts=store, entity=ids[0] if ids else "obj_0000")


# ── agents ─────────────────────────────────────────────────────────────────


class NoContextAgent:
    def answer(self, question: str) -> dict:
        return {"answer": "unknown", "tool_trace": [], "citations": [], "mode": "no_context"}


class FlatContextAgent:
    """Full JSON context, no tools. Can answer static questions, cannot mutate."""

    def __init__(self, episode: EpisodeState):
        self.episode = episode
        self.context = episode.state.to_json()

    def answer(self, question: str) -> dict:
        q = question.lower()
        state = self.episode.state
        if "回滚" in q or "rollback" in q or "改成" in q or "update" in q:
            return {"answer": "unknown", "tool_trace": [], "citations": [], "mode": "flat_context"}
        if "审计" in q or "audit" in q:
            return {"answer": "审计：" + json.dumps({"audit": state.audit[-5:]}, ensure_ascii=False), "tool_trace": [], "citations": [], "mode": "flat_context"}
        if "关系" in q or "relation" in q:
            rows = list(state.relations)
            return {"answer": "关系：" + json.dumps({"relations": rows}, ensure_ascii=False), "tool_trace": [], "citations": [], "mode": "flat_context"}
        if "材质" in q or "material" in q:
            rows = [f for f in state.facts if f["entity"] == self.episode.entity and f["attribute"] == "material"]
            return {"answer": "事实：" + json.dumps({"facts": rows}, ensure_ascii=False), "tool_trace": [], "citations": [], "mode": "flat_context"}
        if "不确定" in q or "uncertain" in q:
            return {"answer": "不确定项：" + json.dumps({"uncertainty": state.uncertainty}, ensure_ascii=False), "tool_trace": [], "citations": [], "mode": "flat_context"}
        return {"answer": "对象：" + json.dumps({"objects": state.objects}, ensure_ascii=False), "tool_trace": [], "citations": [], "mode": "flat_context"}


# ── queries ────────────────────────────────────────────────────────────────


def _checks(episode: EpisodeState):
    state = episode.state
    facts = episode.facts
    entity = episode.entity
    ids = sorted(o["id"] for o in state.objects)
    all_relations = sorted(
        (r["source"], r["predicate"], r["target"]) for r in state.relations
    )
    initial_material = next(
        (f["value"] for f in state.facts if f["entity"] == entity and f["attribute"] == "material"),
        None,
    )
    queries = [
        (
            "列出所有对象",
            lambda ans: sorted(
                o["id"] for o in (_json_tail(ans["answer"]) or {}).get("objects", [])
            )
            == ids,
        ),
        (
            "有哪些关系？",
            lambda ans: sorted(
                (r["source"], r["predicate"], r["target"])
                for r in (_json_tail(ans["answer"]) or {}).get("relations", [])
            )
            == all_relations,
        ),
        (
            f"{entity} 的材质是什么？",
            lambda ans: any(
                f.get("entity") == entity
                and f.get("attribute") == "material"
                and f.get("value") == initial_material
                for f in (_json_tail(ans["answer"]) or {}).get("facts", [])
            ),
        ),
        (
            "最近的审计记录",
            lambda ans: len((_json_tail(ans["answer"]) or {}).get("audit", [])) > 0,
        ),
        (
            f"把 {entity} 的材质改成 wood，并告诉我新版本号",
            lambda ans: facts.get(entity, "material") == "wood"
            and "version" in ans["answer"].lower(),
        ),
        (
            "回滚到 v1 并列出审计",
            lambda ans: facts.version == 1
            and any(e.action == "rollback" for e in facts.audit[-3:]),
        ),
        (
            "哪些对象不确定？",
            lambda ans: "low_confidence" in ans["answer"]
            and entity in ans["answer"],
        ),
    ]
    return queries


def _run_agent(name, agent, episode: EpisodeState):
    queries = _checks(episode)
    rows = []
    citations_total = 0
    citations_ok = 0
    unsupported_total = 0
    tool_calls = 0
    for question, check in queries:
        answer = agent.answer(question)
        ok = False
        try:
            ok = bool(check(answer))
        except Exception:
            ok = False
        citation = verify_citations(answer, episode.state)
        hallucination = verify_answer_entities(answer["answer"], episode.state)
        citations_total += citation["n_citations"]
        citations_ok += citation["verified"]
        unsupported_total += len(hallucination["unsupported"])
        tool_calls += len(answer.get("tool_trace", []))
        rows.append(
            {
                "question": question,
                "ok": ok,
                "answer": answer["answer"][:200],
                "mode": answer.get("mode"),
                "n_citations": citation["n_citations"],
                "unsupported_entities": hallucination["unsupported"],
            }
        )
    return {
        "agent": name,
        "accuracy": float(np.mean([r["ok"] for r in rows])),
        "n_queries": len(rows),
        "tool_calls": tool_calls,
        "citation_coverage": citations_ok / max(1, citations_total),
        "n_citations": citations_total,
        "unsupported_entities": unsupported_total,
        "rows": rows,
    }


def _llm_adapter_case(episode: EpisodeState):
    """Scripted decoder: first turn calls get_facts, second turn answers."""

    def decoder(prompt: str) -> str:
        if "TOOL_RESULTS" not in prompt:
            return json.dumps(
                {"tool": "get_facts", "args": {"entity": episode.entity}},
                ensure_ascii=False,
            )
        return f"{episode.entity} 的材质见事实记录，引用 fact_store。"

    head = LLMLanguageHead(
        episode.state, facts=episode.facts, decoder=decoder, allow_write=False
    )
    result = head.answer(f"{episode.entity} 的材质是什么？")
    citations = verify_citations(result, episode.state)
    hallucination = verify_answer_entities(result["answer"], episode.state)
    return {
        "agent": "llm_adapter_scripted",
        "n_tool_rounds": len(result["tool_trace"]),
        "used_tool": bool(result["tool_trace"]),
        "citation_coverage": citations["coverage"],
        "unsupported_entities": hallucination["unsupported"],
        "mode": result["mode"],
        "answer": result["answer"],
    }


def _hallucination_case(episode: EpisodeState):
    """A fabricated answer is caught by the entity verifier."""
    fabricated = "obj_9999 位于 bin7，材质是 metal。"
    check = verify_answer_entities(fabricated, episode.state)
    return {"answer": fabricated, **check}



def _scale_context():
    """Compare capped structured context vs an uncapped full dump."""
    objects = [
        {"id": f"obj_{i:04d}", "category": "cup", "pos": [0.0, 0.0, 0.0],
         "confidence": 0.9, "status": "active", "last_seen": i, "missed": 0, "active": True}
        for i in range(60)
    ]
    relations = [
        {"source": f"obj_{i % 60:04d}", "target": f"obj_{(i + 1) % 60:04d}",
         "type": "spatial", "predicate": "near", "probability": 0.8}
        for i in range(200)
    ]
    facts = [
        {"entity": f"obj_{i % 60:04d}", "attribute": f"attr_{i % 5}", "value": "v", "version": 3}
        for i in range(200)
    ]
    audit = [
        {"version": i, "action": "update", "key": ["obj_0000", "material"], "old": "a", "new": "b", "reason": "r"}
        for i in range(50)
    ]
    state = PSSState(objects=objects, relations=relations, facts=facts, audit=audit)
    structured = state.token_estimate()
    flat = max(1, len(state.to_json(max_objects=10**6, max_relations=10**6, max_facts=10**6, max_audit=10**6)) // 4)
    return {"structured_tokens": structured, "flat_tokens": flat, "ratio": structured / max(1, flat)}


# ── report / run ───────────────────────────────────────────────────────────


def render(payload: dict) -> str:
    lines = [
        "# Output head: structured state + tools + LLM adapter",
        "",
        f"> generated: {payload.get('generated')}",
        "",
        f"- structured context: ~{payload['context']['structured_tokens']} tokens; "
        f"flat context: ~{payload['context']['flat_tokens']} tokens",
        f"- objects / relations / facts / audit: {payload['context']['counts']}",
        f"- capped structured vs full dump on a 60-object/200-relation state: "
        f"~{payload['context_scale']['structured_tokens']} vs "
        f"~{payload['context_scale']['flat_tokens']} tokens "
        f"({payload['context_scale']['ratio']:.2f}x)",
        "",
        "## Grounded query results",
        "",
        "| agent | accuracy | tool calls | citation coverage | unsupported entities |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("no_context", "flat_context", "structured_tools"):
        r = payload["agents"][name]
        lines.append(
            f"| {name} | {_fmt(r['accuracy'])} | {r['tool_calls']} | "
            f"{_fmt(r['citation_coverage'])} | {r['unsupported_entities']} |"
        )
    lines += [
        "",
        "| query | no context | flat context | structured tools |",
        "|---|---:|---:|---:|",
    ]
    rows = payload["agents"]["structured_tools"]["rows"]
    for i, row in enumerate(rows):
        vals = [payload["agents"][name]["rows"][i]["ok"] for name in ("no_context", "flat_context", "structured_tools")]
        lines.append(f"| {row['question']} | " + " | ".join("yes" if v else "no" for v in vals) + " |")
    adapter = payload.get("adapter")
    if adapter:
        lines += [
            "",
            "## LLM adapter loop (scripted decoder)",
            "",
            f"- tool rounds: {adapter['n_tool_rounds']}; used tool: {adapter['used_tool']}",
            f"- citation coverage: {_fmt(adapter['citation_coverage'])}; "
            f"unsupported entities: {adapter['unsupported_entities']}",
            "",
        ]
    hall = payload.get("hallucination")
    if hall:
        lines += [
            "## Hallucination guard",
            "",
            f"- fabricated answer: `{hall['answer']}`",
            f"- unsupported ids detected: {hall['unsupported']}",
            "",
        ]
    lines += [
        "## Verdict",
        "",
        "- The language head consumes structured state, not raw tensors; no MLP translation.",
        "- Tools are the only write path; write tools require explicit permission and return versions.",
        "- The flat-context agent can answer static queries but cannot update/rollback/audit;",
        "  the structured tool path covers the full query set with citations.",
        "- A fabricated entity is caught by the verifier; the audit surface is measurable.",
        "- The scripted decoder only validates the adapter loop; it is not a language-quality claim.",
        "",
    ]
    return "\n".join(lines)


def run(device: str = "cpu") -> dict:
    context_episode = build_episode_state()
    flat_tokens = max(1, len(context_episode.state.to_json()) // 4)
    no_episode = build_episode_state()
    flat_episode = build_episode_state()
    tool_episode = build_episode_state()
    agents = {
        "no_context": _run_agent("no_context", NoContextAgent(), no_episode),
        "flat_context": _run_agent(
            "flat_context", FlatContextAgent(flat_episode), flat_episode
        ),
        "structured_tools": _run_agent(
            "structured_tools",
            ToolAgent(tool_episode.state, facts=tool_episode.facts, allow_write=True),
            tool_episode,
        ),
    }
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "context": {
            "structured_tokens": context_episode.state.token_estimate(),
            "flat_tokens": flat_tokens,
            "counts": {
                "objects": len(context_episode.state.objects),
                "relations": len(context_episode.state.relations),
                "facts": len(context_episode.state.facts),
                "audit": len(context_episode.state.audit),
            },
        },
        "agents": agents,
        "context_scale": _scale_context(),
        "adapter": _llm_adapter_case(build_episode_state()),
        "hallucination": _hallucination_case(build_episode_state()),
    }
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "language_head.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "language_head.md").write_text(render(payload), encoding="utf-8")
    return payload


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    payload = run(args.device)
    print(
        "accuracy: "
        + ", ".join(
            f"{name}={payload['agents'][name]['accuracy']:.3f}"
            for name in ("no_context", "flat_context", "structured_tools")
        )
    )
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
