# PSS property scorecard — L2 bundle replay

- bundles: `bundles\dev`
- traces: 14
- bounds: entities ≤ 50, latency ≤ 10.0 ms

| trace | rows | BAR | p99 ms | max entities | max latency ms | max expansions | max facts |
|---|---:|---:|---:|---:|---:|---:|---:|
| b4bb5f0e/seed0 | 95 | 1.000 | 0.308 | 4 | 0.897 | 0 | 0 |
| b4bb5f0e/seed0 | 100 | 0.995 | 2.521 | 7 | 52.462 | 0 | 0 |
| 37f0c8b8/seed0 | 100 | 1.000 | 1.811 | 13 | 2.017 | 0 | 0 |
| 18c18c5e/seed0 | 100 | 1.000 | 3.175 | 20 | 3.302 | 0 | 0 |
| 18c18c5e/seed1 | 100 | 0.995 | 4.995 | 20 | 63.481 | 0 | 0 |
| 18c18c5e/seed2 | 100 | 1.000 | 3.256 | 20 | 3.575 | 0 | 0 |
| 18c18c5e/seed3 | 100 | 1.000 | 3.501 | 20 | 4.314 | 0 | 0 |
| 18c18c5e/seed4 | 100 | 1.000 | 3.412 | 20 | 3.853 | 0 | 0 |
| 16ea9f38/seed0 | 100 | 1.000 | 0.358 | 27 | 0.374 | 0 | 0 |
| 16ea9f38/seed0 | 100 | 1.000 | 1.090 | 4 | 2.019 | 0 | 0 |
| 16ea9f38/seed0 | 100 | 1.000 | 2.465 | 5 | 5.383 | 0 | 0 |
| 3d49ce1b/seed0 | 100 | 1.000 | 0.283 | 8 | 0.302 | 0 | 0 |
| 3d49ce1b/seed0 | 100 | 1.000 | 0.241 | 4 | 0.257 | 0 | 0 |
| 66399bd6/seed0 | 50 | 1.000 | 0.850 | 10 | 0.874 | 0 | 0 |

## Aggregate

- BAR mean/min: 0.9993 / 0.9950
- p99 distribution: {'n': 14, 'mean': 2.018970644078675, 'median': 2.13847649618402, 'p95': 4.023537941830107, 'p99': 4.800603582611337, 'min': 0.2412600022216794, 'max': 4.994869992806645}
- CAT mean: 0.0000

> Bundle traces do not exercise hot-swap or KB edits; those are covered by L1.
