# Claim-gap F9: multi-query fact selection under a shared budget

> generated: 2026-11-26 08:23:39 UTC
> T=1000, keys=50, values=16, seeds=3, Q=['2', '4', '8'], k=['2', '4', '8', '16', '32']

E4 saturated because its query answer was the latest fact for the key and
`query_recency` selected exactly those facts. Here the workload has `Q`
queries and one shared active set of `k` facts. A fact supports a query if
it has the query key, the resolved answer value, and a write time at or
before the query's `as_of`. Coverage is the fraction of queries with at
least one supporting fact.

## Reproduce

```bash
python -m pss.eval.scripts.run_gap_fact_multi_query --seeds 3 --qs 2,4,8 --ks 2,4,8,16,32 \
  --streams easy,conflict,duplicate,drift,hotkey,hard
```

Outputs: `reports/claim_gaps/fact_multi_query/REPORT.md` and
`reports/claim_gaps/fact_multi_query/metrics.json`.

## Streams

| stream | key skew | conflict | drift | duplicate | answer rule |
|---|---:|---:|---:|---:|---|
| easy | 0.0 | 0.00 | 0.00 | 0.00 | latest |
| conflict | 0.0 | 0.45 | 0.00 | 0.00 | majority |
| duplicate | 0.0 | 0.00 | 0.00 | 0.30 | latest |
| drift | 0.0 | 0.00 | 0.15 | 0.00 | latest |
| hotkey | 1.2 | 0.00 | 0.00 | 0.00 | latest |
| hard | 1.2 | 0.45 | 0.10 | 0.25 | majority |

`easy` reproduces the E4 regime. `conflict` mixes contradictory writes and
resolves the answer by reliability-weighted majority over the recent window.
`duplicate` repeats facts. `drift` changes a key's value over time.
`hotkey` draws both facts and queries from a Zipf key distribution.
`hard` combines conflict, drift, duplicates and hot-key skew.

## Coverage vs budget (Q=4)

### easy

| method | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| random | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.167 ± 0.236 | 0.333 ± 0.312 | 0.500 ± 0.204 |
| query_recency | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union_value | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union_filled | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| greedy_coverage | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

### conflict

| method | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| random | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.167 ± 0.236 | 0.167 ± 0.236 | 0.333 ± 0.312 |
| query_recency | 0.167 ± 0.236 | 0.333 ± 0.118 | 0.667 ± 0.236 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union | 0.083 ± 0.118 | 0.167 ± 0.236 | 0.750 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union_value | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union_filled | 0.083 ± 0.118 | 0.333 ± 0.236 | 0.750 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| greedy_coverage | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

### duplicate

| method | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| random | 0.000 ± 0.000 | 0.083 ± 0.118 | 0.000 ± 0.000 | 0.417 ± 0.312 | 0.500 ± 0.204 |
| query_recency | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union_value | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union_filled | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| greedy_coverage | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

### drift

| method | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| random | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.083 ± 0.118 | 0.250 ± 0.000 |
| query_recency | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.167 ± 0.236 |
| per_query_union | 0.417 ± 0.118 | 0.917 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union_value | 0.500 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union_filled | 0.417 ± 0.118 | 0.917 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| greedy_coverage | 0.500 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

### hotkey

| method | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| random | 0.167 ± 0.118 | 0.583 ± 0.312 | 0.583 ± 0.312 | 0.667 ± 0.312 | 0.667 ± 0.312 |
| query_recency | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union_value | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union_filled | 0.583 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| greedy_coverage | 0.667 ± 0.236 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

### hard

| method | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| random | 0.083 ± 0.118 | 0.000 ± 0.000 | 0.167 ± 0.118 | 0.250 ± 0.000 | 0.083 ± 0.118 |
| query_recency | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.250 ± 0.204 |
| per_query_union | 0.250 ± 0.000 | 0.583 ± 0.236 | 0.833 ± 0.118 | 0.917 ± 0.118 | 1.000 ± 0.000 |
| per_query_union_value | 0.500 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| per_query_union_filled | 0.250 ± 0.000 | 0.583 ± 0.236 | 0.833 ± 0.118 | 0.917 ± 0.118 | 1.000 ± 0.000 |
| greedy_coverage | 0.500 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

## Greedy coverage vs budget (all Q)

Absolute coverage of the coverage optimiser, i.e. the coverage-budget
curve. The delta tables below show where allocation beats per-query
selection.

### easy

| Q | k=2 | k=4 | k=8 | k=16 | k=32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 0.583 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 0.292 | 0.542 | 1.000 | 1.000 | 1.000 |

### conflict

| Q | k=2 | k=4 | k=8 | k=16 | k=32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 0.583 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 0.292 | 0.542 | 1.000 | 1.000 | 1.000 |

### duplicate

| Q | k=2 | k=4 | k=8 | k=16 | k=32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 0.583 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 0.292 | 0.542 | 1.000 | 1.000 | 1.000 |

### drift

| Q | k=2 | k=4 | k=8 | k=16 | k=32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 0.292 | 0.542 | 1.000 | 1.000 | 1.000 |

### hotkey

| Q | k=2 | k=4 | k=8 | k=16 | k=32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 0.542 | 0.792 | 1.000 | 1.000 | 1.000 |

### hard

| Q | k=2 | k=4 | k=8 | k=16 | k=32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 0.375 | 0.625 | 1.000 | 1.000 | 1.000 |

## Coverage optimisation vs per-query union

Delta = `greedy_coverage` - `per_query_union` at the same stream, Q and k.
Positive cells mean the coverage-aware selection covered more queries with
the same budget. `full` is omitted (it is 1.000 by construction).

### easy

| Q | k=2 | k=4 | k=8 | k=16 | k=32 |
|---:|---:|---:|---:|---:|---:|
| 2 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 4 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 8 | +0.042 | +0.000 | +0.000 | +0.000 | +0.000 |

### conflict

| Q | k=2 | k=4 | k=8 | k=16 | k=32 |
|---:|---:|---:|---:|---:|---:|
| 2 | +0.667 | +0.167 | +0.000 | +0.000 | +0.000 |
| 4 | +0.500 | +0.833 | +0.250 | +0.000 | +0.000 |
| 8 | +0.208 | +0.458 | +0.708 | +0.375 | +0.125 |

### duplicate

| Q | k=2 | k=4 | k=8 | k=16 | k=32 |
|---:|---:|---:|---:|---:|---:|
| 2 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 4 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 8 | +0.042 | +0.000 | +0.000 | +0.000 | +0.000 |

### drift

| Q | k=2 | k=4 | k=8 | k=16 | k=32 |
|---:|---:|---:|---:|---:|---:|
| 2 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 4 | +0.083 | +0.083 | +0.000 | +0.000 | +0.000 |
| 8 | +0.042 | +0.125 | +0.125 | +0.083 | +0.000 |

### hotkey

| Q | k=2 | k=4 | k=8 | k=16 | k=32 |
|---:|---:|---:|---:|---:|---:|
| 2 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 4 | +0.083 | +0.000 | +0.000 | +0.000 | +0.000 |
| 8 | +0.167 | +0.292 | +0.000 | +0.000 | +0.000 |

### hard

| Q | k=2 | k=4 | k=8 | k=16 | k=32 |
|---:|---:|---:|---:|---:|---:|
| 2 | +0.500 | +0.167 | +0.000 | +0.000 | +0.000 |
| 4 | +0.250 | +0.417 | +0.167 | +0.083 | +0.000 |
| 8 | +0.125 | +0.292 | +0.333 | +0.125 | +0.083 |

## Saturation

Smallest `k` at which a method reaches coverage 1.000, or `-` if it never
does in the swept range. `q_rec` is the E4 recency rule; `greedy` is the
coverage optimiser.

| stream | Q | q_rec | per_query | greedy | mean support |
|---|---:|---:|---:|---:|---:|
| easy | 2 | 2 | 2 | 2 | 21.7 |
| easy | 4 | 4 | 4 | 4 | 20.4 |
| easy | 8 | 8 | 8 | 8 | 21.3 |
| conflict | 2 | 8 | 8 | 2 | 12.3 |
| conflict | 4 | 16 | 16 | 4 | 10.7 |
| conflict | 8 | - | - | 8 | 10.7 |
| duplicate | 2 | 2 | 2 | 2 | 21.0 |
| duplicate | 4 | 4 | 4 | 4 | 20.0 |
| duplicate | 8 | 8 | 8 | 8 | 21.6 |
| drift | 2 | 32 | 2 | 2 | 5.0 |
| drift | 4 | - | 8 | 4 | 5.2 |
| drift | 8 | - | 32 | 8 | 5.2 |
| hotkey | 2 | 2 | 2 | 2 | 141.8 |
| hotkey | 4 | 4 | 4 | 4 | 150.4 |
| hotkey | 8 | 8 | 8 | 8 | 101.7 |
| hard | 2 | - | 8 | 2 | 14.0 |
| hard | 4 | - | 32 | 4 | 12.7 |
| hard | 8 | - | - | 8 | 9.0 |

## Redundancy (Q=4, k=8)

| stream | method | coverage | selected | wasted | duplicate | facts/covered | queries/fact |
|---|---|---:|---:|---:|---:|---:|---:|
| easy | full | 1.000 | 1000.0 | 0.926 | 0.950 | 250.00 | 0.08 |
| easy | random | 0.167 | 8.0 | 0.958 | 0.042 | 6.67 | 0.08 |
| easy | query_recency | 1.000 | 7.3 | 0.000 | 0.500 | 1.83 | 1.11 |
| easy | per_query_union | 1.000 | 7.3 | 0.000 | 0.500 | 1.83 | 1.11 |
| easy | per_query_union_value | 1.000 | 7.3 | 0.000 | 0.500 | 1.83 | 1.11 |
| easy | per_query_union_filled | 1.000 | 8.0 | 0.000 | 0.542 | 2.00 | 1.17 |
| easy | greedy_coverage | 1.000 | 8.0 | 0.542 | 0.000 | 2.00 | 0.50 |
| conflict | full | 1.000 | 1000.0 | 0.961 | 0.609 | 250.00 | 0.04 |
| conflict | random | 0.167 | 8.0 | 0.958 | 0.000 | 6.67 | 0.08 |
| conflict | query_recency | 0.667 | 7.3 | 0.611 | 0.056 | 3.17 | 0.50 |
| conflict | per_query_union | 0.750 | 7.3 | 0.639 | 0.000 | 2.44 | 0.42 |
| conflict | per_query_union_value | 1.000 | 7.3 | 0.000 | 0.500 | 1.83 | 1.11 |
| conflict | per_query_union_filled | 0.750 | 8.0 | 0.625 | 0.042 | 2.67 | 0.46 |
| conflict | greedy_coverage | 1.000 | 8.0 | 0.542 | 0.000 | 2.00 | 0.50 |
| duplicate | full | 1.000 | 1000.0 | 0.928 | 0.950 | 250.00 | 0.08 |
| duplicate | random | 0.000 | 8.0 | 1.000 | 0.000 | 8.00 | 0.00 |
| duplicate | query_recency | 1.000 | 7.3 | 0.000 | 0.500 | 1.83 | 1.11 |
| duplicate | per_query_union | 1.000 | 7.3 | 0.000 | 0.500 | 1.83 | 1.11 |
| duplicate | per_query_union_value | 1.000 | 7.3 | 0.000 | 0.500 | 1.83 | 1.11 |
| duplicate | per_query_union_filled | 1.000 | 8.0 | 0.000 | 0.542 | 2.00 | 1.17 |
| duplicate | greedy_coverage | 1.000 | 8.0 | 0.542 | 0.042 | 2.00 | 0.50 |
| drift | full | 1.000 | 1000.0 | 0.979 | 0.823 | 250.00 | 0.02 |
| drift | random | 0.000 | 8.0 | 1.000 | 0.000 | 8.00 | 0.00 |
| drift | query_recency | 0.000 | 7.3 | 1.000 | 0.458 | 7.33 | 0.00 |
| drift | per_query_union | 1.000 | 8.0 | 0.208 | 0.292 | 2.00 | 0.79 |
| drift | per_query_union_value | 1.000 | 8.0 | 0.042 | 0.458 | 2.00 | 0.96 |
| drift | per_query_union_filled | 1.000 | 8.0 | 0.208 | 0.292 | 2.00 | 0.79 |
| drift | greedy_coverage | 1.000 | 8.0 | 0.500 | 0.000 | 2.00 | 0.50 |
| hotkey | full | 1.000 | 1000.0 | 0.601 | 0.950 | 250.00 | 0.60 |
| hotkey | random | 0.583 | 8.0 | 0.667 | 0.292 | 4.67 | 0.42 |
| hotkey | query_recency | 1.000 | 6.7 | 0.000 | 0.500 | 1.67 | 1.33 |
| hotkey | per_query_union | 1.000 | 6.7 | 0.000 | 0.500 | 1.67 | 1.33 |
| hotkey | per_query_union_value | 1.000 | 6.7 | 0.000 | 0.500 | 1.67 | 1.33 |
| hotkey | per_query_union_filled | 1.000 | 8.0 | 0.000 | 0.583 | 2.00 | 1.50 |
| hotkey | greedy_coverage | 1.000 | 8.0 | 0.292 | 0.333 | 2.00 | 0.88 |
| hard | full | 1.000 | 1000.0 | 0.949 | 0.765 | 250.00 | 0.05 |
| hard | random | 0.167 | 8.0 | 0.917 | 0.000 | 8.00 | 0.08 |
| hard | query_recency | 0.000 | 6.7 | 1.000 | 0.125 | 6.67 | 0.00 |
| hard | per_query_union | 0.833 | 8.0 | 0.500 | 0.083 | 2.44 | 0.50 |
| hard | per_query_union_value | 1.000 | 8.0 | 0.125 | 0.375 | 2.00 | 0.88 |
| hard | per_query_union_filled | 0.833 | 8.0 | 0.500 | 0.083 | 2.44 | 0.50 |
| hard | greedy_coverage | 1.000 | 8.0 | 0.500 | 0.125 | 2.00 | 0.50 |

`full` is the reference row: keeping the whole stream trivially covers
everything but has maximal redundancy.

## Selection latency

Measured on the `hard` stream, 20 repeats;
mean / p95 milliseconds per selection call.

| setting | full | random | query_recency | per_query_union | per_query_union_value | per_query_union_filled | greedy_coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q=2,k=4 | 0.001/0.002 | 0.022/0.029 | 0.041/0.045 | 0.068/0.076 | 0.075/0.078 | 0.066/0.070 | 0.080/0.090 |
| Q=2,k=8 | 0.001/0.001 | 0.023/0.030 | 0.042/0.052 | 0.072/0.084 | 0.080/0.086 | 0.070/0.076 | 0.180/0.229 |
| Q=2,k=16 | 0.001/0.001 | 0.021/0.030 | 0.034/0.038 | 0.064/0.075 | 0.071/0.083 | 0.074/0.103 | 0.223/0.250 |
| Q=4,k=4 | 0.001/0.002 | 0.024/0.032 | 0.055/0.082 | 0.105/0.118 | 0.122/0.132 | 0.105/0.116 | 0.097/0.109 |
| Q=4,k=8 | 0.001/0.002 | 0.026/0.034 | 0.051/0.054 | 0.108/0.124 | 0.104/0.133 | 0.111/0.142 | 0.181/0.192 |
| Q=4,k=16 | 0.001/0.002 | 0.027/0.039 | 0.054/0.058 | 0.121/0.128 | 0.138/0.149 | 0.237/0.271 | 0.329/0.340 |
| Q=8,k=4 | 0.001/0.002 | 0.022/0.029 | 0.061/0.065 | 0.110/0.130 | 0.130/0.141 | 0.292/0.303 | 0.110/0.117 |
| Q=8,k=8 | 0.001/0.002 | 0.025/0.034 | 0.094/0.107 | 0.204/0.229 | 0.244/0.254 | 0.398/0.455 | 0.164/0.197 |
| Q=8,k=16 | 0.002/0.002 | 0.028/0.038 | 0.098/0.111 | 0.219/0.240 | 0.252/0.272 | 0.390/0.469 | 0.299/0.324 |

## Verdict

Coverage optimisation beats the value-blind per-query union in 32 setting(s):

- easy/Q=8/k=2 (0.292 vs 0.250)
- conflict/Q=2/k=2 (1.000 vs 0.333)
- conflict/Q=2/k=4 (1.000 vs 0.833)
- conflict/Q=4/k=2 (0.583 vs 0.083)
- conflict/Q=4/k=4 (1.000 vs 0.167)
- conflict/Q=4/k=8 (1.000 vs 0.750)
- conflict/Q=8/k=2 (0.292 vs 0.083)
- conflict/Q=8/k=4 (0.542 vs 0.083)
- conflict/Q=8/k=8 (1.000 vs 0.292)
- conflict/Q=8/k=16 (1.000 vs 0.625)
- conflict/Q=8/k=32 (1.000 vs 0.875)
- duplicate/Q=8/k=2 (0.292 vs 0.250)
- drift/Q=4/k=2 (0.500 vs 0.417)
- drift/Q=4/k=4 (1.000 vs 0.917)
- drift/Q=8/k=2 (0.292 vs 0.250)
- drift/Q=8/k=4 (0.542 vs 0.417)
- drift/Q=8/k=8 (1.000 vs 0.875)
- drift/Q=8/k=16 (1.000 vs 0.917)
- hotkey/Q=4/k=2 (0.667 vs 0.583)
- hotkey/Q=8/k=2 (0.542 vs 0.375)
- hotkey/Q=8/k=4 (0.792 vs 0.500)
- hard/Q=2/k=2 (1.000 vs 0.500)
- hard/Q=2/k=4 (1.000 vs 0.833)
- hard/Q=4/k=2 (0.500 vs 0.250)
- ... and 8 more

**Material hard-setting wins** (largest deltas vs the value-blind
union, which is the E4-style query-conditioned baseline):

- conflict/Q=4/k=4: 1.000 vs 0.167 (+0.833)
- conflict/Q=8/k=8: 1.000 vs 0.292 (+0.708)
- conflict/Q=2/k=2: 1.000 vs 0.333 (+0.667)
- hard/Q=2/k=2: 1.000 vs 0.500 (+0.500)
- conflict/Q=4/k=2: 0.583 vs 0.083 (+0.500)
- conflict/Q=8/k=4: 0.542 vs 0.083 (+0.458)

The small `easy`/`duplicate` Q=8, k=2 deltas come from chance key
collisions between queries, not from the conflict/drift/hot-key
mechanism; they are not the claimed effect.

Against `per_query_union_filled`, which spends the full budget k,
greedy still wins in 32 setting(s):

- easy/Q=8/k=2 (0.292 vs 0.250)
- conflict/Q=2/k=2 (1.000 vs 0.333)
- conflict/Q=2/k=4 (1.000 vs 0.833)
- conflict/Q=4/k=2 (0.583 vs 0.083)
- conflict/Q=4/k=4 (1.000 vs 0.333)
- conflict/Q=4/k=8 (1.000 vs 0.750)
- conflict/Q=8/k=2 (0.292 vs 0.083)
- conflict/Q=8/k=4 (0.542 vs 0.083)
- ... and 24 more

The gain is therefore not merely unused budget in the union.

Against the value-aware per-query union it wins in 9 setting(s) (budget allocation / shared support):

- easy/Q=8/k=2 (0.292 vs 0.250)
- conflict/Q=8/k=2 (0.292 vs 0.250)
- duplicate/Q=8/k=2 (0.292 vs 0.250)
- drift/Q=8/k=2 (0.292 vs 0.250)
- hotkey/Q=4/k=2 (0.667 vs 0.583)
- hotkey/Q=8/k=2 (0.542 vs 0.375)
- hotkey/Q=8/k=4 (0.792 vs 0.500)
- hard/Q=8/k=2 (0.375 vs 0.333)
- hard/Q=8/k=4 (0.625 vs 0.500)

### Saturation condition

Coverage is capped at 1.0, so it saturates whenever every query already
has one selected supporting fact. Concretely: if `k >= Q` and the
method allocates at least one slot per query, the only way to stay below
1.0 is to pick a non-supporting fact. That is exactly what the value-blind
rules do on `conflict` and `hard`; on `easy`, `duplicate` and `hotkey` the
value is stable, so any key-matched fact supports the query and
`query_recency` returns to 1.000 for `k >= Q`. On `drift`, recency that
ignores `as_of` picks future facts with the wrong value, so it does not
saturate. When `k < Q`, per-query allocation gives some queries zero slots
and coverage is bounded by `k / Q` unless one fact supports several
queries (shared support under hot-key skew).

### Limitations

- The stream is synthetic (T=1000, 50 keys, 16 values); the absolute
  numbers are not a claim about real fact stores.
- Coverage only asks for one supporting fact per query. It does not measure
  whether the selected fact is sufficient to answer, nor answer accuracy.
- `greedy_coverage` uses the support relation induced by the query and the
  stream's answer rule; it is not learning a relevance model. A deployable
  version still needs a value-resolution / relevance classifier.
- Latency is single-process CPU numpy on a synthetic stream; it does not
  include query parsing or support construction.
