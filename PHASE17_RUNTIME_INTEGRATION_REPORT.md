# Phase 17 — Learned modules inside the runtime

> Date: 2026-11-24
> Scope: `PSS-认知层架构定位.md` §3.3 / §7, `PHASE16_RUNTIME_REPORT.md` next steps
> Data: synthetic tabletop, `tracking_bench_long` (synthetic)
> Result artifacts: `eval/results/runtime_phase17/`

---

## 1. What this phase integrates

Phase 16 built the runtime shell. This phase wires the learned modules into
the runtime's default paths and adds versioned logic:

| Integration | Change |
|---|---|
| Learned relational goal | `Goal.relation` + `RuntimePlanner.plan_relational` (symbolic or learned predicate) |
| Learned write gate | `PSSRuntime.assert_fact(..., gate=..., gate_features=...)`; `review`/`abstain` are audited skips |
| Versioned operators | `VersionedOperatorLibrary` with update/remove/rollback and `Plan.operator_version` |
| Real decoder interface | `OutputRouter(decoder=...)` + `load_local_decoder()`; records availability |
| Long occlusion | runtime loop on `tracking_bench_long` (70-frame occlusion) |

Files:

- `pss/runtime/operators.py` (new);
- `pss/runtime/state.py` (`Goal.relation`, `Plan.operator_version`);
- `pss/runtime/services.py` (`plan_relational`);
- `pss/runtime/runtime.py` (write gate, decoder router);
- `pss/heads/text/structured.py` (`load_local_decoder`);
- `pss/eval/scripts/run_runtime_phase17.py`.

---

## 2. Results

### 2.1 Learned relational goal

A region-relation head (`SpatialRelationHead`) is trained on noisy region
pairs; all-4 exact accuracy on the training distribution is **0.988**. The runtime planner consumes either the
exact predicate or the learned head.

| predicate | symbolic plan | symbolic success | learned plan | learned success |
|---|---:|---:|---:|---:|
| left | 2 | yes | 2 | yes |
| right | 0 | yes | 0 | yes |
| above | 1 | yes | 1 | yes |
| below | 0 | yes | 0 | yes |

The learned head reproduces the exact predicate on this in-distribution grid;
the plans are identical. This is an integration result, not a
generalization claim (Phase 12 already showed held-out region degradation).

### 2.2 Learned write gate

The runtime receives 300 fact observations over 20 keys; source reliability
varies and some observations are wrong.

| policy | accuracy | facts | rejected |
|---|---:|---:|---:|
| write all | 0.750 | 20 | 0 |
| learned gate | **0.850** | 20 | 77 |

The gate is the learned `WritePolicy` from Phase 13, called at the runtime
boundary. `skip`/`review`/`abstain` skip the write and leave a
`fact_rejected` audit entry; the fact store is smaller and more accurate.

### 2.3 Versioned operators

| version | plan | operator version |
|---|---|---:|
| v1 | `[move_cup_A_G]` | 1 |
| v2 (direct route removed) | `[move_cup_A_B, move_cup_B_G]` | 2 |
| v3 (rollback to v1) | `[move_cup_A_G]` | 1 |

Removing an operator changes the plan; rollback restores both the operator
set and the direct plan. Plans carry the operator version they were produced
under.

### 2.4 Long occlusion

`tracking_bench_long`, 12 test sequences, 70-frame full occlusion (pose noise
off for this identity check):

| metric | value |
|---|---:|
| ID retention | **1.000** |
| entity persistence | **1.000** |
| mean entities | 3.0 |
| mean active | 3.0 |

The runtime keeps the entity through the occlusion (historical track) and
re-IDs on reappearance. This matches the earlier P2-11 result and shows the
runtime uses the persistent entity layer, not a frame-local cache.

### 2.5 Real decoder

`load_local_decoder()` returns `None`: no text LLM is in the local HF cache.
`OutputRouter(decoder=...)` is wired and tested with a scripted decoder; the
phase records `decoder_available: false` rather than pretending the tool agent
is a language model.

---

## 3. Findings

1. **Learned relations can drive the planner**. The runtime planner accepts a
   predicate function; symbolic and learned relations produce the same plans
   in-distribution. The exact predicate remains the reference.
2. **The write policy belongs at the runtime boundary**. Gating facts on
   write improves accuracy from 0.75 to 0.85 and leaves an audit trail of
   rejected observations.
3. **Logic is now versioned**. Operators can be removed/updated and rolled
   back; plans record the operator version. This closes the "logic layer is
   versionable" gap from the architecture table.
4. **Long occlusion persistence works through the runtime**, not only in the
   standalone tracker.
5. **The real decoder is still absent.** The interface is wired; no language
   quality claim is made.

---

## 4. Boundaries

- The relational head is trained in-distribution on the same grid; held-out
  generalization is Phase 12's negative result, not retested here.
- The write stream is synthetic; source reliability is observable.
- Operators remain hand-authored; only their **versioning** is new.
- The long benchmark uses clean pose for identity; P2-11 covers noisy
  conditions.
- No real data, no sim2real, no physical execution.

---

## 5. Next steps

- Replace hand-authored `move_*` operators with learned/versioned operators
  and a world-model rollout validator.
- Use the held-out relational head inside the runtime and measure planner
  success under learned-relation errors.
- Wire the active-set policy into `EntityLayer.slots` (currently top-k by
  confidence).
- Plug a real decoder into `OutputRouter` and rerun the 7-query grounding
  benchmark inside the long-occlusion loop.
- Run the runtime on the MuJoCo closed loop (perception → plan → push →
  re-perceive → replan) and report physical success.

Reproduce:

```bash
python -m pss.eval.scripts.run_runtime_phase17 --device cpu --sequences 12
```
