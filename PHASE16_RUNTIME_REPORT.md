# Phase 16 — PSS cognitive runtime: layer alignment and closed loop

> Date: 2026-11-24
> Scope: `PSS-认知层架构定位.md` §2–§3, `PSS-认知运行时与Harness.md`
> Data: synthetic tabletop task (deterministic)
> Result artifacts: `eval/results/runtime/`

---

## 1. What this phase is

The five priorities built capabilities. This phase makes them a **runtime**:
one typed state, explicit layer boundaries, versioned/auditable transitions,
and a cross-layer closed loop. It does not add another model.

Mapping to the architecture document:

| Architecture layer | Runtime component | Owns |
|---|---|---|
| Perceptual semantic space | existing prototypes/projections | class vectors, reliability |
| Entity layer | `EntityLayer` | persistent IDs, object slots, attributes, confidence, provenance, lifecycle |
| Relation / fact layer | `RelationFactLayer` | typed relations (symbolic + learned), multi-source facts, conflicts, rule derivation |
| Uncertainty / conflict / abstention | `UncertaintyService` | object confidence, fact/relation conflicts, missing modalities, review/abstain |
| Logic / planning | `RuntimePlanner` | goal, A* plan, plan validation, replan |
| Output heads | `OutputRouter` | action proposal, language tools/citations, safety veto |
| Audit | `CognitiveState` + commit/rollback | versions, snapshots, before/after, permissions |

The runtime owns the state. Perception, relations, planning and language are
services that mutate or read it; none of them own it.

---

## 2. Closed-loop benchmark

A deterministic tabletop task:

1. three entities (cup, block, bottle) in regions A/B/D;
2. the bottle is occluded on the third frame, but the goal needs it;
3. a second source asserts a conflicting bottle location;
4. the goal is `cup → G`, `bottle → C`;
5. the runtime plans, executes kinematically, explains, writes a bad fact and
   rolls back.

Results:

| variant | entity persistence | plan | task success | conflict flagged | recommendation | rollback | explanation citations |
|---|---:|---:|---:|---:|---|---:|---:|
| **full** | **yes** | **yes (2)** | **yes** | **yes** | **review** | **yes** | **1** |
| `no_entity` | no | no (0) | no | no | accept | yes | 0 |
| `no_relations` | yes | yes (2) | yes | yes | review | yes | 1 |
| `no_uncertainty` | yes | yes (2) | yes | **no** | disabled | yes | 1 |
| `no_audit` | yes | yes (2) | yes | yes | review | **no** | 1 |
| `no_language` | yes | yes (2) | yes | yes | review | yes | **0** |

State size:

| variant | entities | relations | facts | audit | version |
|---|---:|---:|---:|---:|---:|
| full | 3 | 12 | 3 | 10 | 9 |
| no_entity | 2 | 2 | 2 | 5 | 4 |
| no_relations | 3 | 0 | 3 | 10 | 9 |
| no_uncertainty | 3 | 12 | 3 | 10 | 9 |
| no_audit | 3 | 12 | 3 | 0 | 0 |
| no_language | 3 | 12 | 3 | 10 | 9 |

---

## 3. Findings

### 3.1 The layer boundaries are real, not cosmetic

Each ablation fails exactly at its layer:

- `no_entity`: the occluded bottle is gone, so the goal that needs it cannot
  be planned. Persistent identity is doing real work.
- `no_uncertainty`: the conflicting fact is accepted silently; no review and
  no abstention. The conflict is invisible without the service.
- `no_audit`: the bad write cannot be rolled back; version stays 0.
- `no_language`: no grounded explanation/citations.
- `no_relations`: the simple location goal still succeeds (facts are enough),
  but the relation layer is empty (12 → 0). This is a **boundary**, not a
  defect: the current task does not need relations; a relational goal would.

### 3.2 The runtime is state-first, not model-first

The closed loop uses the existing SmallCNN/projection stack, the existing
tracker, the existing A* planner, and the language tool adapter. What makes it
a runtime is that:

- entities/relations/facts/goals/plans are typed and persistent;
- every mutation is versioned and auditable;
- conflict/uncertainty is a service over the state, not a prompt;
- outputs read the state and tools, not raw tensors.

This directly answers the harness-vs-runtime test in
`PSS-认知运行时与Harness.md`: the runtime owns state semantics and transitions.

### 3.3 What is still hand-written

- the planner operators (`move_*`) are hand-authored; the architecture target
  is learned/versioned operators;
- the relation layer in the benchmark is symbolic; the learned typed head is
  wired as an optional hook (`add_learned`) but not used in this loop;
- the output router uses the deterministic tool agent, not a real LLM;
- the active-set/write policies are not yet wired into the entity/relation
  layer defaults;
- uncertainty thresholds are fixed, not calibrated per episode.

These are the next integration tasks, not new models.

---

## 4. Architecture alignment

| Target in the document | Current runtime |
|---|---|
| typed, persistent state | `CognitiveState` (entities/relations/facts/goals/plans/uncertainty/audit) |
| state transitions | ingest / assert_fact / set_goal / plan / act / rollback |
| invariants | versions, snapshots, before/after audit, permission gate, safety veto |
| active-set / scheduling | entity slots (`EntityLayer.slots`), active IDs |
| learned write/read/reason | hooks exist (`add_learned`, `UncertaintyService`), not default |
| language as adapter | `OutputRouter` → `ToolAgent` / `LLMLanguageHead` |

The runtime now matches the document's structure. It does **not** yet match the
document's learned-computation ambition: most decisions are still hand-written
or rule-based. That is the honest remaining gap.

---

## 5. Boundaries

- Deterministic synthetic task; one episode, hand-authored operators.
- The relation layer is symbolic in this loop; the learned head is optional.
- The language path is the tool agent, not a pretrained decoder.
- The action is kinematic (region placement), not a physical push.
- No real sensor data, no sim2real.
- The uncertainty policy is a fixed threshold, not calibrated per episode.

---

## 6. Next steps

1. Wire the learned typed relation head into `EntityLayer.ingest` (already an
   optional `learned_head` hook) and test relational goals.
2. Wire the active-set/write policies into `EntityLayer.slots` and
   `RelationFactLayer.assert_fact` defaults.
3. Replace hand-authored `move_*` operators with versioned/learned operators
   and a world-model rollout for plan validation.
4. Plug a real `decoder` into `OutputRouter` and rerun the 7-query grounding
   benchmark inside the runtime.
5. Run the loop on `tracking_bench_long` (70-frame occlusion) and on the
   MuJoCo closed loop.
6. Add multi-episode statistics and latency; currently a single deterministic
   episode.

Reproduce:

```bash
python -m pss.eval.scripts.run_runtime
```
