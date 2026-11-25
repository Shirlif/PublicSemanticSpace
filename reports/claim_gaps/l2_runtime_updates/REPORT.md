# Claim-gap E19: L2 runtime mid-task updates

> seeds: 3

| scenario | plan changed | baseline goal | updated goal | baseline exec | updated exec |
|---|---:|---:|---:|---:|---:|
| kb_update | 100% | 0% | 100% | 100% | 100% |
| planner_swap | 100% | 0% | 100% | 0% | 100% |

## Scenarios

- `kb_update`: goal changes from G to B mid-task. The old plan moves to G and
  misses the new goal; `set_goal` + replan moves to B.
- `planner_swap`: the direct operator is blocked mid-task. The old plan fails
  its precondition; swapping `runtime.planner` to a detour library and
  replanning succeeds without restarting the runtime.

## Scope

This is runtime-level L2 evidence (state, facts, planning, audit), not a
sensory simulator. S4-PSS cannot test these because its observations are
oracle poses with no occlusion and no raw features to swap.

Conflict/abstention is covered by `run_runtime.py` (`no_uncertainty` drops
the conflict flag) and by E15 (modality conflict); L2 memory is E12.
