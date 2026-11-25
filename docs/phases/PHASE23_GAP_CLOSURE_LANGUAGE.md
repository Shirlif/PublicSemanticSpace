# Phase 23 — Gap closure 6: real local decoder in the runtime

> Date: 2026-11-24
> Scope: `PSS-认知层架构定位.md` §5, row 6 (language)
> Data: one phase7 episode, 7 grounded queries (synthetic)
> Result artifacts: `eval/results/runtime_language/`

---

## 1. Model

Downloaded and cached locally:

- **Qwen2.5-1.5B-Instruct** (~1.54B params); selected by `load_local_decoder`.
- Qwen2.5-0.5B-Instruct and SmolLM-135M-Instruct were tried; 0.5B emitted
  malformed tool names/args, 135M did not follow the task. They remain
  fallback candidates but are not used for the reported result.

The decoder runs with the tokenizer's chat template, `do_sample=False`,
`max_new_tokens=192`, about **4.0 s/query** on the GPU.

---

## 2. What changed

| Component | Change |
|---|---|
| `load_local_decoder` | picks Qwen2.5-1.5B; chat template; cached-only |
| `LLMLanguageHead` | tool-loop convergence guard: repeated executed calls return the tool result |
| `SYSTEM_PROMPT` | explicit "do not repeat a tool call; use TOOL_RESULTS" |
| `run_runtime_language.py` | 7-query grounded benchmark with the real decoder |
| reference | deterministic `ToolAgent` on the same queries |

Without the convergence guard, the 1.5B model repeated `update_fact`/
`rollback_fact` until the tool budget was exhausted and returned `unknown`.
The guard stops after the first repeated call and returns the tool result.

---

## 3. Results

| agent | accuracy | tool calls | citations | unsupported entities | latency |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B + tools | **0.857** | 5 | 0 | 0 | 4041 ms |
| deterministic tool agent | **1.000** | 8 | 2 | 0 | <1 ms |

Per-query:

| query | LLM ok | mode | tools | tool-agent ok |
|---|---:|---|---:|---:|
| list objects | **no** | decoder | 0 | yes |
| list relations | yes | decoder | 0 | yes |
| material | yes | decoder | 0 | yes |
| recent audit | yes | converged | 1 | yes |
| update material | yes | converged | 2 | yes |
| rollback to v1 | yes | converged | 2 | yes |
| uncertain objects | yes | decoder | 0 | yes |

- The real decoder answers static questions from the structured state and
  emits **valid** `update_fact` / `rollback_fact` calls.
- The one failure is object enumeration: the model listed 3 of 4 objects and
  omitted `plate_01`. It answered from context instead of calling
  `get_objects`.
- Citations are 0 because the model rarely calls read tools; it reads the
  state in the prompt. The deterministic tool agent cites facts and is the
  reference.

---

## 4. Findings

1. **The real decoder path works.** A small local model can answer grounded
   questions and operate the write/rollback tools; the runtime does not need a
   cloud API.
2. **The tool loop needs convergence protection.** Small models repeat calls;
   the guard makes the interaction terminate with the tool result instead of
   `unknown`.
3. **Small-model grounding is incomplete.** It omits an entity in enumeration
   and rarely cites facts. This is a model-quality boundary, not an interface
   failure.
4. **The deterministic tool agent is still the reliable reference** for the
   7-query set (1.000, citations 2).
5. **A stronger decoder or tool-first prompting** is the next lever for
   citation coverage and enumeration completeness.

---

## 5. Boundaries

- One synthetic episode, 7 queries, one model (1.5B); no multi-turn dialogue.
- 4.0 s/query on the GPU; not a production latency claim.
- The model is instruction-tuned; no fine-tuning on PSS state/tools.
- Citations are measured from tool results; context-only answers have none.
- No real sensor data.

---

## 6. Next row

**Row 7 (audit): permissions.** Add read-only / operator / admin profiles and
enforce them in `assert_fact`, `plan`, `act`, and `rollback`; keep the existing
version/rollback/audit trail.

Reproduce:

```bash
python -m pss.eval.scripts.run_runtime_language --device cuda
```
