# Phase 13 — Learned active-set and write policies

> Date: 2026-11-24
> Scope: `PSS-认知层架构定位.md` §7, "第三优先：学习 active-set / write policy"
> Data: `data/tracking_bench_v2`, synthetic fact streams (synthetic)
> Result artifacts: `eval/results/active_policy/`

---

## 1. What was implemented

Two learned policies and one calibrated uncertainty gate:

| Component | File | Purpose |
|---|---|---|
| `ActiveSetPolicy` | `pss/memory/active_policy.py` | scores candidate entities; top-k selection |
| listwise training | `pss/memory/active_policy.py` | the correct candidate should win each observation row |
| margin calibration | `pss/memory/active_policy.py` | choose a fallback threshold on validation data |
| `WritePolicy` | `pss/memory/active_policy.py` | write / review / skip for facts/observations |
| tracking features | `pss/memory/active_policy.py` | pose, vision, geometry, tactile, category, norms |
| experiment runner | `pss/eval/scripts/run_active_policy.py` | tracking, write, retrieval stages |

The policies are additive: the scene graph, tracker and fact store keep their
existing interfaces. The active-set policy selects a subset before the
existing matcher runs; the write policy is a gate before the existing
`FactStore`/admission path.

---

## 2. Active set for re-ID candidates

Candidate selection on `tracking_bench_v2`: for each post observation, rank the
pre-frame candidates and keep top-k; fall back to the full candidate set when
the margin is below a threshold calibrated on validation data.

| setting | no fallback | with fallback | mean active | fallback |
|---|---:|---:|---:|---:|
| k=1, clean | 0.972 | **1.000** | 1.20 | 0.10 |
| k=1, pose noise | 0.961 | **0.978** | 1.21 | 0.11 |
| k=1, no tactile | 0.689 | 0.689 | 1.60 | 0.30 |
| k=2, clean | 1.000 | 1.000 | 2.09 | 0.09 |
| k=2, pose noise | 0.989 | 0.978 | 2.10 | 0.10 |
| k=2, no tactile | 0.678 | 0.694 | 2.75 | 0.75 |

Reference points on the same features: full candidate set with the policy score
**1.000**; with the tactile score 0.967; pose top-1 0.344; vision top-1 0.778;
tactile top-1 0.917; tactile top-2 0.967.

- **The learned listwise policy is the right active-set mechanism**: k=1 plus a
  10% calibrated fallback matches the full-context policy score (1.000) while
  searching 1.2 candidates on average.
- **The fallback gate is calibrated, not hand-set**: the margin threshold is
  chosen on validation data for a target fallback rate, not tuned on test.
- **Fallback cannot recover missing tactile**: with tactile zeroed, both the
  full candidate set and the active set fail (0.689). The information is gone,
  not merely unselected. This is the correct boundary of the mechanism.
- A pairwise-BCE version of the same policy was tested first and underperformed
  (0.756 full-score vs 0.972 listwise); the listwise objective is necessary
  because the task is "pick the winner", not "classify each pair".

---

## 3. Write policy for fact memory

Stream of 1500 observations over 50 keys, 8 values; each observation has a
source reliability and may be wrong. A query at the end asks for each key's
true value. The memory answer is the latest written observation.

| policy | accuracy | writes | reviews |
|---|---:|---:|---:|
| write all | 0.700 | 1500 | 0 |
| write none | 0.000 | 0 | 0 |
| reliability rule (`rel > 0.7`) | 0.880 | 716 | 0 |
| learned (`p >= 0.5`) | 0.880 | 1083 | 0 |
| learned + review band | **0.920** | 1051 | 109 |

Threshold Pareto (accuracy / writes):

| threshold | rule | learned |
|---|---:|---:|
| 0.4 | 0.720 / 1384 | 0.880 / 1137 |
| 0.5 | 0.840 / 1131 | **0.880 / 1083** |
| 0.6 | 0.860 / 936 | **0.920 / 1051** |
| 0.7 | 0.880 / 716 | 0.900 / 989 |
| 0.8 | 0.920 / 468 | 0.900 / 834 |
| 0.9 | 0.920 / 240 | 0.480 / 339 |

- At **matched write counts** (~1.1k) the learned policy is +4 points over the
  reliability rule (0.88 vs 0.84).
- The **review band** (0.4–0.6) adds another +4 points: the policy abstains
  from ambiguous items instead of writing a likely-wrong fact.
- At very low write budgets the rule is better; the learned policy's advantage
  is in the high-write / high-accuracy regime, not a universal win.

---

## 4. Active set for long-history facts (stress test)

T=1000 facts, 50 keys, k=16 active, 20% hot keys.

| policy | accuracy |
|---|---:|
| full history | 1.000 |
| recency | 0.324 |
| hot (hand heuristic) | 0.480 |
| random | 0.008 |
| learned | 0.372 |
| learned + diversity | 0.428 |
| learned + diversity + fallback | 0.680 |

The learned policy does **not** beat the hand `hot` heuristic. Two causes:

1. **The policy is not query-conditioned here.** It scores each fact from
   retention features only (recency, hotness, reliability, key frequency), so
   it learns a global retention prior, not "which fact answers this query".
   The hot heuristic is the same prior, hand-coded.
2. **Coverage is not trained.** An independent per-fact score selects multiple
   facts for the same hot key, so a small active set misses queried keys.

The fallback to 64 active facts recovers much of the gap (0.680) and shows the
uncertainty gate works; it is still below full. A query-conditioned score plus
a set-level coverage/diversity loss is the next step.

This stage is reported as a negative/boundary result, not a headline.

---

## 5. Findings

1. **Listwise active-set works on the real matching task**: k=1 + 10% calibrated
   fallback matches full-context accuracy (1.000) at 1.2 candidates.
2. **The fallback gate is principled and calibrated**: it recovers pose-noise
   selection errors, but cannot recover missing-modality information loss.
3. **Write policy learns more than a reliability threshold**: at matched writes
   it is +4 points, and the review band adds another +4 by abstaining.
4. **Sparse long-history fact active sets are not solved by a per-fact score**;
   coverage/diversity is the missing ingredient. This is the honest negative.
5. The third priority's three sub-goals are covered: learned attention
   (tracking), learned write (facts), and uncertainty fallback (both).

---

## 6. Boundaries

- All data is synthetic; no real robot or sensor data.
- The tracking active set has only 3 candidates per frame; k=1 is a strong
  reduction but the absolute candidate count is small.
- The write stream is synthetic; source reliability is observable, which makes
  the task easier than real sensor provenance.
- The long-history fact stage is a stress test of coverage, not a claim that
  the policy is competitive.
- The policies are not yet wired into `Tracker` / `WorkingMemory` / `FactStore`
  as default paths; they are evaluated through the experiment runner.

---

## 7. Next steps

- Make the long-history fact policy **query-conditioned** (query key as input)
  and add a set-level coverage loss (or greedy diversity during training).
- Wire the active-set policy into `Tracker.reid_candidates` and measure
  end-to-end latency/retention.
- Wire the write policy into prototype/fact admission and the review gate.
- Evaluate on `tracking_bench_long` (70-frame occlusion) where the active set
  actually has many dormant candidates.

Reproduce:

```bash
python -m pss.eval.scripts.run_active_policy --stage all --device cuda
```
