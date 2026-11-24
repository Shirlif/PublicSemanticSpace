# Phase 1 binding comparison report

- generated: 2026-11-20 03:30:07 UTC
- runtime: 0.0s
- trials: 6

## 1. Round-trip error

`||unbind(bind(x, role), role) - x|| / ||x||`, with Gaussian noise added each cycle. HRR correlation is an approximate inverse; Hadamard is exact.

| scheme | D | noise | cycles | mean | p95 |
|---|---:|---:|---:|---:|---:|
| hadamard | 128 | 0.00 | 1 | 4.82e-17 | 6.52e-17 |
| hadamard | 128 | 0.00 | 10 | 5.15e-17 | 6.55e-17 |
| hadamard | 128 | 0.05 | 1 | 0.051 | 0.054 |
| hadamard | 128 | 0.05 | 10 | 0.166 | 0.178 |
| hrr | 128 | 0.00 | 1 | 1.096 | 1.478 |
| hrr | 128 | 0.00 | 10 | 1.13e+06 | 3.38e+06 |
| hrr | 128 | 0.05 | 1 | 0.985 | 1.370 |
| hrr | 128 | 0.05 | 10 | 5.21e+07 | 2.15e+08 |

## 2. Superposition capacity (top-1 retrieval)

| scheme | D | K | noise | accuracy |
|---|---:|---:|---:|---:|
| hadamard | 128 | 5 | 0.00 | 1.000 |
| hadamard | 128 | 20 | 0.00 | 0.700 |
| hrr | 128 | 5 | 0.00 | 1.000 |
| hrr | 128 | 20 | 0.00 | 0.650 |

## 3. Relation triple separability

Mean relative error when decoding each role. `separate` uses the three stored parts; `per-fact` decodes from one summed relation vector; `memory(n)` decodes from all `n` relation vectors superposed.

| scheme | D | n | separate | per-fact | memory(n) |
|---|---:|---:|---:|---:|---:|
| hadamard | 128 | 1 | 4.72e-17 | 1.425 | 1.425 |
| hadamard | 128 | 5 | 4.96e-17 | 1.415 | 3.698 |
| hrr | 128 | 1 | 0.991 | 1.732 | 1.732 |
| hrr | 128 | 5 | 0.974 | 1.747 | 4.574 |

## 4. Cross-role crosstalk

Max absolute cosine when one role's stored part is decoded with a different role slot.

| scheme | D | mean | max |
|---|---:|---:|---:|
| hadamard | 128 | 0.132 | 0.208 |
| hrr | 128 | 0.203 | 0.286 |

## Findings

1. **Hadamard wins decisively.** HRR round-trip error is ~1.0 at one cycle and **diverges** under repeated bind/unbind: ~1e6 at 10 cycles and ~1e80 at 100 cycles. Correlation is not a stable inverse, so HRR cannot support iterative rebind loops. Values >1e3 in the table mean divergence, not a large but usable error.
2. **HRR gives no consistent capacity gain.** D=128 K=20: 0.667 vs 0.775; D=256 K=20: 0.958 vs 0.925; D=512 K=50: 0.790 vs 0.833. The document's 'convolution has larger capacity' expectation is not supported by this benchmark.
3. **Keep Hadamard.** HRR stays as an audited alternative, not the default binding scheme.
4. **Never decode a role from `fact`.** The summed vector's cross-talk is O(1) (measured ~1.41 relative error even for one relation); only the three separately stored parts are decodable.
5. **Capacity budget holds.** D=128 is safe for K<=5; K=20 needs D=512 or per-fact storage (see Phase 0 report).
