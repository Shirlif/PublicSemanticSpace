# Phase 14 — Output head: structured state, tools, LLM adapter

> Date: 2026-11-24
> Scope: `PSS-认知层架构定位.md` §7, "第四优先：输出头升级"
> Data: `data/phase7_v2` episode (synthetic), synthetic scale state
> Result artifacts: `eval/results/language_head/`

---

## 1. What was implemented

The language head is an adapter over structured PSS state and tools, not an
MLP trained to translate tensors into sentences.

| Component | File | Purpose |
|---|---|---|
| `PSSState` | `pss/heads/text/structured.py` | objects, typed relations, facts, audit, memory, uncertainty |
| `to_context` / `token_estimate` | `pss/heads/text/structured.py` | compact, capped serialization |
| `ToolRegistry` | `pss/heads/text/structured.py` | 7 tools; write tools require explicit permission |
| `LLMLanguageHead` | `pss/heads/text/structured.py` | decoder + tool loop; parses tool calls, collects citations |
| `verify_citations` | `pss/heads/text/structured.py` | every cited fact/version must exist |
| `verify_answer_entities` | `pss/heads/text/structured.py` | flags fabricated entity ids |
| `ToolAgent` | `pss/heads/text/structured.py` | deterministic harness stand-in, not an LLM |
| experiment runner | `pss/eval/scripts/run_language_head.py` | grounded query benchmark |

Tools:

| tool | read/write | returns |
|---|---|---|
| `get_objects` | read | objects + source |
| `get_relations` | read | typed relations + source |
| `get_facts` | read | facts + citations (entity/attribute/version) |
| `list_audit` | read | recent audit entries |
| `get_uncertainty` | read | low-confidence objects / missing modalities |
| `update_fact` | write | version before/after |
| `rollback_fact` | write | rolled-back version |

The action/behavior heads are unchanged; Phase 8's `ActionHead` remains the
action path. This phase only replaces the language readout.

---

## 2. Grounded query benchmark

State: one phase7 episode, 4 objects, 1 relation, 9 facts, 2 audit entries,
with one low-confidence object.

| agent | accuracy | tool calls | citation coverage | unsupported entities |
|---|---:|---:|---:|---:|
| no context | 0.000 | 0 | 0.000 | 0 |
| flat JSON dump, no tools | 0.714 | 0 | 0.000 | 0 |
| structured state + tools | **1.000** | 8 | **1.000** | 0 |

Query breakdown:

| query | no context | flat context | structured tools |
|---|---:|---:|---:|
| list objects | no | yes | yes |
| list relations | no | yes | yes |
| object material | no | yes | yes |
| recent audit | no | yes | yes |
| update material + version | no | **no** | **yes** |
| rollback to v1 + audit | no | **no** | **yes** |
| uncertain objects | no | yes | yes |

The flat-context agent can answer static questions from the dump, but it has no
write path, no version, and no rollback. The tool path covers the full set and
returns citations.

---

## 3. LLM adapter and grounding

The adapter takes any `decoder(prompt) -> text` callable. The experiment uses a
scripted decoder that first emits a tool call and then answers:

- tool rounds: 1; used tool: `get_facts`;
- citation coverage: 1.000;
- unsupported entities: 0.

A fabricated answer (`obj_9999 位于 bin7`) is rejected by the entity verifier:
`unsupported = ["obj_9999"]`. The same check runs on every language output, so
hallucinated entity ids are measurable rather than invisible.

Write tools are gated by `allow_write=False` by default; the read-only registry
returns `write permission denied`, and the fact store is unchanged.

---

## 4. Context size

- Small real episode: structured 564 tokens vs flat 511 tokens. At this size
  there is nothing to compress; the difference is the tool/audit surface.
- Scaled state (60 objects, 200 relations, 200 facts, 50 audit entries):
  capped structured **3439** tokens vs full dump **12666** tokens (**0.27x**).
  The cap is part of `to_context`; counts in the context tell the decoder what
  was omitted, so it can call a tool to fetch the rest.

The architecture does not claim that a flat dump is unusable; it claims that a
capped structured state plus tools gives the decoder a bounded context and an
explicit way to ask for more.

---

## 5. Findings

1. **Language is an adapter, not an MLP translation.** The head consumes
   `PSSState` and tool results; no gradient path reaches the public space.
2. **Tools are the only write path.** Updates/rollbacks require permission,
   return versions, and appear in the audit trail.
3. **Grounding is measurable.** Citations verify against the fact store;
   fabricated entity ids are flagged. This is the audit surface the
   architecture asks for.
4. **Flat context is enough for static questions but not for audited actions.**
   The tool path is necessary for update/rollback/version queries.
5. **Context capping matters at scale** (0.27x on a 60-object state); on a tiny
   episode the sizes are comparable.
6. **Action/behavior heads are untouched.** The upgrade is isolated to the
   language readout.

---

## 6. Boundaries

- No language model is cached in this environment. The deterministic
  `ToolAgent` and the scripted decoder validate the **state/tool/audit**
  surface, not generation quality. They must not be reported as an LLM result.
- The benchmark has 7 queries on one episode; it is a harness check, not a
  language benchmark suite.
- The episode uses simulator GT relations/facts; perception errors are not the
  subject of this phase.
- The real decoder path is implemented (`decoder` callable + tool loop) but not
  exercised with a pretrained model.

---

## 7. Next steps

- Plug in a real decoder (local HF model or API) and rerun the same queries;
  measure answer accuracy and unsupported-entity rate with the same verifier.
- Add planner / active-set / write-policy tools to the registry.
- Include audit citations in the generated text, not only in the structured
  result.
- Add permission profiles (read-only, operator, admin) and tests for each.
- Evaluate on multi-turn clarification ("which cup?") using the structured
  state to disambiguate.

Reproduce:

```bash
python -m pss.eval.scripts.run_language_head --device cpu
```
