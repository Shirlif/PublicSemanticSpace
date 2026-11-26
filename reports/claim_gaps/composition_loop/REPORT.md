# Minimal composition loop

> Generated: 2026-11-26 16:55:26 UTC
> Tasks: 25 laundry + 19 spatial + 6 selection

Strategies: `direct` (no composition), `frame_blind` (untyped fixed world-frame pipeline), 
`typed` (best-first with signatures), `typed_cache` (typed + verified plan reuse), `typed_warm` (second pass with the populated cache).

## Overall

| strategy | success | false claims | mean steps | mean expansions | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| direct | 0.300 | 0.460 | 1.140 | 0.000 | 0.009 | 0.083 |
| frame_blind | 0.420 | 0.340 | 1.140 | 0.000 | 0.008 | 0.138 |
| typed | 1.000 | 0.000 | 2.680 | 20.440 | 0.257 | 1.253 |
| typed_cache | 1.000 | 0.000 | 2.680 | 6.620 | 0.049 | 1.020 |
| typed_warm | 1.000 | 0.000 | 2.680 | 3.160 | 0.039 | 0.371 |

## By domain

| strategy | domain | n | success | false claims |
|---|---|---:|---:|---:|
| direct | laundry | 25 | 0.520 | 0.000 |
| direct | spatial | 19 | 0.053 | 0.947 |
| direct | selection | 6 | 0.167 | 0.833 |
| frame_blind | laundry | 25 | 0.520 | 0.000 |
| frame_blind | spatial | 19 | 0.368 | 0.632 |
| frame_blind | selection | 6 | 0.167 | 0.833 |
| typed | laundry | 25 | 1.000 | 0.000 |
| typed | spatial | 19 | 1.000 | 0.000 |
| typed | selection | 6 | 1.000 | 0.000 |
| typed_cache | laundry | 25 | 1.000 | 0.000 |
| typed_cache | spatial | 19 | 1.000 | 0.000 |
| typed_cache | selection | 6 | 1.000 | 0.000 |
| typed_warm | laundry | 25 | 1.000 | 0.000 |
| typed_warm | spatial | 19 | 1.000 | 0.000 |
| typed_warm | selection | 6 | 1.000 | 0.000 |

## Frame-sensitive subset (spatial, non-world goals)

These tasks were filtered so the first world-frame placement does not satisfy the goal.

| strategy | n | success | false claims |
|---|---:|---:|---:|
| direct | 12 | 0.000 | 1.000 |
| frame_blind | 12 | 0.000 | 1.000 |
| typed | 12 | 1.000 | 0.000 |
| typed_cache | 12 | 1.000 | 0.000 |
| typed_warm | 12 | 1.000 | 0.000 |

## Cache

- hits: 77, stale: 11, misses: 12, stores: 100
- verified writes: 100, rejected attempts: 0, entries: 12, coverage: 1.000
- mean expansions: typed 20.440 -> cache 6.620 (0.676) -> warm 3.160 (0.845)

## Decision

- `direct` succeeds 0.300 with 0.460 false claims.
- `frame_blind` succeeds 0.420 with 0.340 false claims.
- `typed` succeeds 1.000 with 0.000 false claims.
- The verified cache reaches coverage 1.000 after one pass and cuts mean expansions from 20.440 to 3.160 on the warm pass (0.845), with 11 stale replays rejected.
- Conclusion: the typed layer is viable.  The verifier must stay the goal test, and cache replay must re-check applicability and the goal.
