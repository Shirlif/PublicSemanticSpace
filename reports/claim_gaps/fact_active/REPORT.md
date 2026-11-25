# Claim-gap E4: query-conditioned fact active set

> T=1000, keys=50, values=16, seeds=3

| method | k=4 | k=8 | k=16 |
|---|---:|---:|---:|
| full | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| recency | 0.074 ± 0.015 | 0.259 ± 0.026 | 0.313 ± 0.028 |
| hot | 0.237 ± 0.020 | 0.421 ± 0.024 | 0.473 ± 0.015 |
| random | 0.001 ± 0.001 | 0.004 ± 0.003 | 0.008 ± 0.005 |
| learned_old | 0.241 ± 0.013 | 0.379 ± 0.112 | 0.604 ± 0.161 |
| query_recency | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| learned_query | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

## Verdict

| k | learned_query > hot | learned_query > old | query_recency > hot |
|---:|---|---|---|
| 4 | True | True | True |
| 8 | True | True | True |
| 16 | True | True | True |

The old negative result came from a query-independent scorer: one top-k set was
shared by every query. Query-conditioned selection is compared against the hot
heuristic and a query-recency rule; the latter attributes the gain to conditioning
rather than to learning. If query_recency equals learned_query, the honest
conclusion is 'the missing ingredient was query conditioning', not 'the learned
model is better'.
