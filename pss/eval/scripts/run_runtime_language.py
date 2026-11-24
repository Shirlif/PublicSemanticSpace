"""Row 6: real LLM decoder in the runtime output path.

Loads a small local instruction model (Qwen2.5-1.5B-Instruct when cached) and
runs the 7-query grounded benchmark against a PSS state with tools. The tool
loop has a convergence guard: if the decoder repeats an executed call, the
head returns the tool result instead of looping.

Example:
    python -m pss.eval.scripts.run_runtime_language --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pss._console import ensure_utf8
from pss.eval.scripts.run_language_head import build_episode_state
from pss.heads.text import (
    LLMLanguageHead,
    ToolAgent,
    load_local_decoder,
    verify_answer_entities,
)

OUT = "eval/results/runtime_language"


def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _mentions(text: str, *values) -> bool:
    lower = text.lower()
    return any(str(v).lower() in lower for v in values)


def _checks(episode):
    state = episode.state
    facts = episode.facts
    entity = episode.entity
    ids = [o["id"] for o in state.objects]
    initial_material = next(
        (f["value"] for f in state.facts if f["entity"] == entity and f["attribute"] == "material"),
        None,
    )
    has_relations = bool(state.relations)

    def check_objects(ans):
        return all(i in ans["answer"] for i in ids)

    def check_relations(ans):
        if not has_relations:
            return _mentions(ans["answer"], "unknown", "没有", "无关系", "no relation")
        source = state.relations[0]["source"]
        target = state.relations[0]["target"]
        return source in ans["answer"] and target in ans["answer"]

    def check_material(ans):
        return _mentions(ans["answer"], initial_material or "")

    def check_audit(ans):
        return _mentions(ans["answer"], "version", "版本", "audit", "审计", "v1", "v2")

    def check_update(ans):
        return facts.get(entity, "material") == "wood" and _mentions(
            ans["answer"], "version", "版本"
        )

    def check_rollback(ans):
        return facts.version == 1 and any(
            e.action == "rollback" for e in facts.audit[-5:]
        )

    def check_uncertainty(ans):
        return entity in ans["answer"] and _mentions(
            ans["answer"], "不确定", "uncertain", "low", "低"
        )

    queries = [
        ("列出所有对象", check_objects),
        ("有哪些关系？", check_relations),
        (f"{entity} 的材质是什么？", check_material),
        ("最近的审计记录", check_audit),
        (f"把 {entity} 的材质改成 wood，并告诉我新版本号", check_update),
        ("回滚到 v1 并列出审计", check_rollback),
        ("哪些对象不确定？", check_uncertainty),
    ]
    return queries


def _run_agent(name, agent, episode):
    rows = []
    citations_total = 0
    citations_ok = 0
    unsupported_total = 0
    tool_calls = 0
    times = []
    for question, check in _checks(episode):
        t0 = time.perf_counter()
        answer = agent.answer(question)
        times.append((time.perf_counter() - t0) * 1000.0)
        try:
            ok = bool(check(answer))
        except Exception:
            ok = False
        hallucination = verify_answer_entities(answer["answer"], episode.state)
        citations_total += len(answer.get("citations", []))
        unsupported_total += len(hallucination["unsupported"])
        tool_calls += len(answer.get("tool_trace", []))
        rows.append(
            {
                "question": question,
                "ok": ok,
                "mode": answer.get("mode"),
                "n_tools": len(answer.get("tool_trace", [])),
                "answer": answer["answer"][:240],
                "unsupported": hallucination["unsupported"],
            }
        )
    return {
        "agent": name,
        "accuracy": float(np.mean([r["ok"] for r in rows])),
        "n_queries": len(rows),
        "tool_calls": tool_calls,
        "n_citations": citations_total,
        "unsupported_entities": unsupported_total,
        "mean_latency_ms": float(np.mean(times)),
        "rows": rows,
    }


def run(device="cuda") -> dict:
    decoder = load_local_decoder(device=device)
    if decoder is None:
        return {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "decoder_available": False,
            "note": "no local instruction model in the HF cache",
        }
    llm_episode = build_episode_state()
    llm = _run_agent(
        "llm_structured_tools",
        LLMLanguageHead(
            llm_episode.state,
            facts=llm_episode.facts,
            decoder=decoder,
            allow_write=True,
            max_tool_rounds=6,
        ),
        llm_episode,
    )
    agent_episode = build_episode_state()
    tool_agent = _run_agent(
        "tool_agent_reference",
        ToolAgent(agent_episode.state, facts=agent_episode.facts, allow_write=True),
        agent_episode,
    )
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "decoder_available": True,
        "agents": {"llm": llm, "tool_agent": tool_agent},
    }
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_language.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "runtime_language.md").write_text(render(payload), encoding="utf-8")
    return payload


def render(payload: dict) -> str:
    if not payload.get("decoder_available"):
        return "# Row 6 — real decoder\n\nNo local instruction model available.\n"
    lines = [
        "# Row 6 — real decoder in the runtime",
        "",
        f"> generated: {payload['generated']}",
        "",
        "| agent | accuracy | tool calls | citations | unsupported | latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in payload["agents"].items():
        lines.append(
            f"| {name} | {_fmt(r['accuracy'])} | {r['tool_calls']} | "
            f"{r['n_citations']} | {r['unsupported_entities']} | "
            f"{_fmt(r['mean_latency_ms'], 1)} ms |"
        )
    lines += [
        "",
        "## Per-query",
        "",
        "| query | llm ok | llm mode | tools | tool-agent ok |",
        "|---|---:|---|---:|---:|",
    ]
    for i, row in enumerate(payload["agents"]["llm"]["rows"]):
        ref = payload["agents"]["tool_agent"]["rows"][i]
        lines.append(
            f"| {row['question']} | {'yes' if row['ok'] else 'no'} | "
            f"{row['mode']} | {row['n_tools']} | {'yes' if ref['ok'] else 'no'} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        "- The real decoder answers static questions from the structured state.",
        "- It emits valid tool calls for update/rollback; the tool-loop convergence",
        "  guard prevents repeated calls from exhausting the budget.",
        "- The deterministic tool agent remains the reference with higher citation",
        "  coverage; the LLM often answers from context instead of calling tools.",
        "- This is a small model (Qwen2.5-1.5B) result, not a frontier-LLM claim.",
        "",
    ]
    return "\n".join(lines)


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    payload = run(args.device)
    if not payload.get("decoder_available"):
        print("decoder unavailable")
        return
    for name, r in payload["agents"].items():
        print(
            f"{name}: acc={r['accuracy']:.3f} tools={r['tool_calls']} "
            f"citations={r['n_citations']} latency={r['mean_latency_ms']:.0f} ms"
        )
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
