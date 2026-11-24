# Phase 22 — Gap closure 5: learned heuristic + A* search

> Date: 2026-11-24
> Scope: `PSS-认知层架构定位.md` §5, row 5 (planning)
> Data: synthetic tabletop region domain (3 entities, 5 regions, distance costs)
> Result artifacts: `eval/results/runtime_planning/`

---

## 1. What changed

Row 5 target: **learning + search**. The search stays A*; this phase replaces
the hand `missing-facts` heuristic with a learned cost-to-go heuristic.

| Component | Change |
|---|---|
| `AStarPlanner` | accepts `heuristic(state, goal)` and `heuristic_weight` |
| `LearnedHeuristic` | `pss/reasoning/planner/learned.py`: MLP over per-entity current/goal region |
| dataset | random states/goals, optimal A* costs with distance-dependent move costs |
| `RuntimePlanner` | accepts `heuristic=`; reports `heuristic_name` |
| runtime summary | `planning.heuristic = learned / hand` |
| checkpoint | `checkpoints/planner/learned_heuristic.pt` |

The hand heuristic remains the default/fallback. A* semantics (goal check,
operator preconditions, plan validation) are unchanged.

---

## 2. Results

3 entities, 5 regions, 4000 train / 400 test states. Move cost depends on
region distance, so the hand `missing-facts` count is not cost-aware.

| planner | success | mean cost | expansions | latency | suboptimality |
|---|---:|---:|---:|---:|---:|
| hand heuristic | 1.000 | 2.114 | 6.29 | 0.13 ms | 0.000 |
| learned heuristic | 1.000 | 2.114 | **3.28** | 3.32 ms | ~0 |
| weighted learned (1.5) | 1.000 | 2.114 | **2.48** | 2.39 ms | ~0 |

- The learned heuristic **halves the expansions** (6.29 → 3.28) and weighted
  learned cuts them to 2.48, with the same cost/success.
- **Wall-clock is worse on this tiny domain** (0.13 → 3.3 ms): the MLP
  evaluation dominates when the state space is only 5³. The expansion win
  would matter at larger branching/state spaces, which this benchmark does not
  have.
- Runtime integration: `RuntimePlanner(heuristic=model.predict)` plans
  successfully and reports `planning.heuristic = "learned"`.

---

## 3. Findings

1. **A* remains the planner; only the heuristic is learned.** Plan validity and
   operator semantics are unchanged.
2. **Expansions drop without cost loss.** The learned heuristic is trained on
   optimal costs and is effectively admissible on this domain (suboptimality
   ~0).
3. **Latency is a real trade-off.** A learned heuristic is not free: on a tiny
   domain it is slower per plan. It should be evaluated on larger state spaces
   or with batched/GPU heuristic evaluation before claiming a speedup.
4. **The hand heuristic remains the default/fallback.** The runtime chooses
   explicitly; `planning.heuristic` is auditable.

---

## 4. Boundaries

- 3 entities / 5 regions; 5³ = 125 states; the expansion win is small.
- Distance-dependent operator costs; other cost structures may change the
  result.
- One seed, 400 test states; suboptimality is reported but not a guarantee.
- The learned heuristic is trained on A* labels from the same domain; no
  cross-domain generalization claim.

---

## 5. Next rows

- **Row 6 (language)**: blocked on LLM weights; the adapter and tool/audit
  surface are ready.
- **Row 7 (audit)**: add permission profiles (read-only / operator / admin)
  and enforce them in `assert_fact` / `plan` / `act` / `rollback`.

Reproduce:

```bash
python -m pss.eval.scripts.run_runtime_planning --device cpu --train 4000 --test 400
```
