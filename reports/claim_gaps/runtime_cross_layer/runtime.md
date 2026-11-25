# PSS cognitive runtime: cross-layer closed loop

> generated: 2026-11-25 17:13:15 UTC

| variant | entity persistence | plan | task success | conflict flagged | recommendation | rollback | explanation citations |
|---|---:|---:|---:|---:|---|---:|---:|
| full | yes | yes (2) | yes | yes | review | yes | 1 |
| no_entity | no | no (0) | no | no | accept | yes | 0 |
| no_relations | yes | yes (2) | yes | yes | review | yes | 1 |
| no_uncertainty | yes | yes (2) | yes | no | disabled | yes | 1 |
| no_audit | yes | yes (2) | yes | yes | review | no | 1 |
| no_language | yes | yes (2) | yes | yes | review | yes | 0 |

## State size

| variant | entities | relations | facts | audit | version |
|---|---:|---:|---:|---:|---:|
| full | 3 | 12 | 3 | 10 | 9 |
| no_entity | 2 | 2 | 2 | 5 | 4 |
| no_relations | 3 | 0 | 3 | 10 | 9 |
| no_uncertainty | 3 | 12 | 3 | 10 | 9 |
| no_audit | 3 | 12 | 3 | 0 | 0 |
| no_language | 3 | 12 | 3 | 10 | 9 |

## What the ablations show

- `no_entity`: the occluded bottle is missing, so the goal that needs it cannot be planned.
- `no_relations`: the plan still succeeds from facts, but the relation layer is empty.
- `no_uncertainty`: the conflicting fact is not flagged and no review is recommended.
- `no_audit`: the bad write cannot be rolled back.
- `no_language`: no grounded explanation/citations.
