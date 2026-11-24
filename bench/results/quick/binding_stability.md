# Binding stability report (Phase 0)

- generated: 2026-11-20 03:23:37 UTC
- trials: roundtrip=8, capacity=4
- runtime: 0.0s

## 1. Slot orthogonality (K=4)

Max absolute off-diagonal entry of the slot Gram matrix. Hadamard slots are exactly orthogonal; Gaussian slots are only approximately so.

| scheme | D | max off-diagonal |
|---|---:|---:|
| hadamard | 128 | 1.11e-16 |
| rademacher | 128 | 0.1406 |
| gaussian | 128 | 0.1302 |

## 2. Round-trip error under noise and repeated cycles

Relative error `||unbind(bind(x,s),s) - x|| / ||x||`, with Gaussian noise added to the bound vector each cycle at `noise * RMS(bound)`.

| scheme | D | noise | cycles | mean err | p95 err | mean norm drift |
|---|---:|---:|---:|---:|---:|---:|
| hadamard | 128 | 0.00 | 1 | 4.95e-17 | 6.49e-17 | 4.16e-17 |
| hadamard | 128 | 0.00 | 10 | 4.53e-17 | 6.22e-17 | 1.39e-17 |
| hadamard | 128 | 0.05 | 1 | 0.0497 | 0.0538 | 0.0030 |
| hadamard | 128 | 0.05 | 10 | 0.1587 | 0.1675 | 0.0098 |
| rademacher | 128 | 0.00 | 1 | 5.24e-17 | 6.93e-17 | 8.33e-17 |
| rademacher | 128 | 0.00 | 10 | 4.74e-17 | 5.73e-17 | 4.16e-17 |
| rademacher | 128 | 0.05 | 1 | 0.0479 | 0.0502 | 0.0024 |
| rademacher | 128 | 0.05 | 10 | 0.1557 | 0.1749 | 0.0192 |
| gaussian | 128 | 0.00 | 1 | 4.98e-17 | 5.56e-17 | 2.78e-17 |
| gaussian | 128 | 0.00 | 10 | 4.65e-17 | 5.58e-17 | 5.55e-17 |
| gaussian | 128 | 0.05 | 1 | 4.2406 | 17.7448 | 3.6066 |
| gaussian | 128 | 0.05 | 10 | 2.2928 | 3.2630 | 1.5544 |

## 3. Superposition capacity (top-1 retrieval)

`K` facts share one memory vector: `memory = sum_i bind(f_i, s_i)`. Each fact is retrieved by `unbind(memory, s_i)` and ranked by cosine.

| scheme | D | K | noise | top-1 accuracy |
|---|---:|---:|---:|---:|
| hadamard | 128 | 5 | 0.00 | 1.000 |
| hadamard | 128 | 5 | 0.05 | 1.000 |
| hadamard | 128 | 20 | 0.00 | 0.725 |
| hadamard | 128 | 20 | 0.05 | 0.687 |
| rademacher | 128 | 5 | 0.00 | 1.000 |
| rademacher | 128 | 5 | 0.05 | 1.000 |
| rademacher | 128 | 20 | 0.00 | 0.675 |
| rademacher | 128 | 20 | 0.05 | 0.725 |
| gaussian | 128 | 5 | 0.00 | 0.300 |
| gaussian | 128 | 5 | 0.05 | 0.350 |
| gaussian | 128 | 20 | 0.00 | 0.087 |
| gaussian | 128 | 20 | 0.05 | 0.088 |

## Notes

- Hadamard and Rademacher slots have `|s_i| = 1/sqrt(D)` for every component, so division never amplifies near-zero components.
- Gaussian unit vectors contain near-zero components; those components amplify noise, which is why the Gaussian row degrades fastest.
- Capacity is approximate by construction: cross-talk grows with `K`, and orthogonal slots do not make element-wise binding orthogonal.
- Contract v0 therefore uses Hadamard slots with a power-of-two `D` (128/256/512).
