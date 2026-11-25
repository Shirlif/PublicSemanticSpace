# PSS interaction + negative-control scorecard — P3

- seeds: `[0, 1, 2, 3, 4]`

## Interactions

| metric | mean | min | max |
|---|---:|---:|---:|
| HC_ok | 1.0000 | 1.0000 | 1.0000 |
| HKC_ok | 1.0000 | 1.0000 | 1.0000 |
| HK_ok | 1.0000 | 1.0000 | 1.0000 |
| HK_rollback | 1.0000 | 1.0000 | 1.0000 |
| KC_ok | 1.0000 | 1.0000 | 1.0000 |
| interaction_pass_rate | 1.0000 | 1.0000 | 1.0000 |

## Negative controls

| id | name | expected | passed |
|---|---|---|---|
| N1 | flat KB | fail locality/rollback | ✅ |
| N2 | unbounded planner | fail budget adherence | ✅ |
| N3 | naive swap | accept invalid module (should fail) | ✅ |
| N4 | random expert capability | fail capability (near chance) | ✅ |
| N5 | random expert property | pass property contract | ✅ |
| N6 | no-op update | identity | ✅ |
| N7 | shuffled embeddings | capability fails, contract holds | ✅ |
| N8 | over-budget input | budget metric catches violation | ✅ |
