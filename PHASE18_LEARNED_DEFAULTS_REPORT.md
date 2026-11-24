# Phase 18 — Learned modules as runtime defaults

> Date: 2026-11-24
> Scope: `PSS-优化任务清单.md` P0-11 next step; `PHASE17_RUNTIME_INTEGRATION_REPORT.md`
> Data: deterministic tabletop episode (synthetic)
> Result artifacts: `eval/results/runtime_defaults/`

---

## 1. What changed

The runtime now uses the learned modules by default when their checkpoints are
present:

| Default | Where it runs | Fallback |
|---|---|---|
| learned typed relations | `EntityLayer.ingest` writes relation logits into the scene graph | symbolic relation records |
| learned active-set policy | `EntityLayer.active_ids` / `slots` selects object slots | confidence top-k |
| learned write policy | `RelationFactLayer.assert_fact` gates facts | write-all |

`components()` reports which path is active, and every rejected write is
audited. The hand-written fallbacks stay available; disabling a default is a
constructor flag, not a code fork.

---

## 2. Results

One tabletop episode with an occluded bottle, a conflicting human fact, a
low-reliability noisy fact, and a high-confidence **distractor** track that no
observation supports.

| variant | learned relations | rejected facts | noisy fact rejected | slots | total relations | task | conflict | rollback | citations |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| **defaults** | **15** | **1** | **yes** | `obj_0001, obj_0002` | 27 | yes | yes | yes | 1 |
| no_learned_relations | 0 | 1 | yes | `obj_0001, obj_0002` | 12 | yes | yes | yes | 1 |
| no_active_policy | 15 | 1 | yes | `obj_distractor, obj_0001` | 27 | yes | yes | yes | 1 |
| no_write_policy | 15 | 0 | **no** | `obj_0001, obj_0002` | 27 | yes | yes | yes | 1 |
| symbolic_only | 0 | 0 | no | `obj_distractor, obj_0001` | 12 | yes | yes | yes | 1 |

Component flags:

| variant | relation head | active policy | write policy |
|---|---:|---:|---:|
| defaults | yes | yes | yes |
| no_learned_relations | no | yes | yes |
| no_active_policy | yes | no | yes |
| no_write_policy | yes | yes | no |
| symbolic_only | no | no | no |

---

## 3. Findings

### 3.1 The learned defaults change behavior, not the task contract

- **Relations**: 15 learned edges are added on top of 12 symbolic edges
  (27 vs 12 total). The task/plan/rollback/citation contract is unchanged.
- **Write gate**: the low-reliability noisy fact is rejected under the learned
  policy (1 rejection) and written when the policy is off. This is the intended
  boundary: the policy decides what enters memory, and the rejection is
  audited.
- **Active set**: the learned policy excludes the high-confidence distractor
  (`obj_distractor`) because no observation supports it; the confidence top-k
  fallback selects it. This is the clearest behavioral difference in the
  episode.

### 3.2 Defaults are reversible

Every learned default is a constructor flag. `components()` records the active
path, and the fallback is a first-class configuration, not dead code. This is
the runtime/harness distinction in practice: the state and transitions stay
the same; the policy layer is swappable.

### 3.3 The write policy needs application frequency

The learned write policy was trained with a per-key query-frequency feature.
When the runtime does not know that frequency, it passes `0.0` (unknown =
rare). That is conservative and rejects the noisy fact here, but an
application that knows a key is frequently queried should pass the real
frequency; otherwise a valid but rare fact can be rejected. This is a
calibration/interface detail, not a model failure.

### 3.4 The active policy is adapted, not retrained

The active-set policy was trained for pair matching (track vs observation
features). Using its max-over-observations score for slot selection is an
adaptation of the existing learned policy; it works on this episode but is not
a dedicated slot-retention model. A dedicated slot policy is the next step.

---

## 4. Boundaries

- One deterministic synthetic episode; the distractor is synthetic.
- The learned relation head is in-distribution; held-out relation
  generalization remains Phase 12's negative result.
- The write policy's frequency feature defaults to 0.0 when unknown.
- The active policy is reused from pair matching, not trained for slots.
- No real data, no LLM decoder, no physical execution.

---

## 5. Next steps

- Train a dedicated slot-retention policy on runtime episodes (labels: track
  needed for the next observation/goal) and replace the adapted pair policy.
- Calibrate the write policy's frequency feature per application/key.
- Run the defaults on `tracking_bench_long` and the MuJoCo closed loop.
- Add a held-out relation set to the runtime planner to measure the cost of
  learned-relation errors.
- Plug a real decoder into `OutputRouter` when weights are available.

Reproduce:

```bash
python -m pss.eval.scripts.run_runtime_defaults
```
