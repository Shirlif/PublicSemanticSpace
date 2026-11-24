# Phase 1 binding comparison report

- generated: 2026-11-20 03:30:07 UTC
- runtime: 1.9s
- trials: 12

## 1. Round-trip error

`||unbind(bind(x, role), role) - x|| / ||x||`, with Gaussian noise added each cycle. HRR correlation is an approximate inverse; Hadamard is exact.

| scheme | D | noise | cycles | mean | p95 |
|---|---:|---:|---:|---:|---:|
| hadamard | 128 | 0.00 | 1 | 4.99e-17 | 6.85e-17 |
| hadamard | 128 | 0.00 | 10 | 4.85e-17 | 6.37e-17 |
| hadamard | 128 | 0.00 | 100 | 5.50e-17 | 7.28e-17 |
| hadamard | 128 | 0.01 | 1 | 0.010 | 0.011 |
| hadamard | 128 | 0.01 | 10 | 0.032 | 0.035 |
| hadamard | 128 | 0.01 | 100 | 0.099 | 0.108 |
| hadamard | 128 | 0.05 | 1 | 0.048 | 0.054 |
| hadamard | 128 | 0.05 | 10 | 0.157 | 0.169 |
| hadamard | 128 | 0.05 | 100 | 0.547 | 0.595 |
| hadamard | 256 | 0.00 | 1 | 0 | 0 |
| hadamard | 256 | 0.00 | 10 | 0 | 0 |
| hadamard | 256 | 0.00 | 100 | 0 | 0 |
| hadamard | 256 | 0.01 | 1 | 0.010 | 0.011 |
| hadamard | 256 | 0.01 | 10 | 0.032 | 0.034 |
| hadamard | 256 | 0.01 | 100 | 0.100 | 0.106 |
| hadamard | 256 | 0.05 | 1 | 0.049 | 0.052 |
| hadamard | 256 | 0.05 | 10 | 0.157 | 0.166 |
| hadamard | 256 | 0.05 | 100 | 0.535 | 0.570 |
| hadamard | 512 | 0.00 | 1 | 5.22e-17 | 5.94e-17 |
| hadamard | 512 | 0.00 | 10 | 5.37e-17 | 5.91e-17 |
| hadamard | 512 | 0.00 | 100 | 4.83e-17 | 5.39e-17 |
| hadamard | 512 | 0.01 | 1 | 0.010 | 0.010 |
| hadamard | 512 | 0.01 | 10 | 0.031 | 0.032 |
| hadamard | 512 | 0.01 | 100 | 0.100 | 0.105 |
| hadamard | 512 | 0.05 | 1 | 0.050 | 0.051 |
| hadamard | 512 | 0.05 | 10 | 0.160 | 0.164 |
| hadamard | 512 | 0.05 | 100 | 0.523 | 0.548 |
| hrr | 128 | 0.00 | 1 | 1.083 | 1.749 |
| hrr | 128 | 0.00 | 10 | 2.61e+06 | 1.35e+07 |
| hrr | 128 | 0.00 | 100 | 2.31e+80 | 1.25e+81 |
| hrr | 128 | 0.01 | 1 | 0.959 | 1.188 |
| hrr | 128 | 0.01 | 10 | 4.64e+06 | 2.19e+07 |
| hrr | 128 | 0.01 | 100 | 1.37e+84 | 7.38e+84 |
| hrr | 128 | 0.05 | 1 | 1.019 | 1.280 |
| hrr | 128 | 0.05 | 10 | 5.08e+06 | 2.52e+07 |
| hrr | 128 | 0.05 | 100 | 6.61e+83 | 3.57e+84 |
| hrr | 256 | 0.00 | 1 | 1.009 | 1.210 |
| hrr | 256 | 0.00 | 10 | 2.05e+07 | 1.05e+08 |
| hrr | 256 | 0.00 | 100 | 4.27e+89 | 2.31e+90 |
| hrr | 256 | 0.01 | 1 | 0.972 | 1.189 |
| hrr | 256 | 0.01 | 10 | 1.50e+07 | 5.43e+07 |
| hrr | 256 | 0.01 | 100 | 5.86e+85 | 3.18e+86 |
| hrr | 256 | 0.05 | 1 | 1.042 | 1.323 |
| hrr | 256 | 0.05 | 10 | 4.66e+08 | 2.50e+09 |
| hrr | 256 | 0.05 | 100 | 6.49e+83 | 3.50e+84 |
| hrr | 512 | 0.00 | 1 | 0.998 | 1.178 |
| hrr | 512 | 0.00 | 10 | 6.30e+07 | 3.28e+08 |
| hrr | 512 | 0.00 | 100 | 6.99e+89 | 3.78e+90 |
| hrr | 512 | 0.01 | 1 | 0.977 | 1.083 |
| hrr | 512 | 0.01 | 10 | 1.23e+07 | 5.06e+07 |
| hrr | 512 | 0.01 | 100 | 2.74e+84 | 1.50e+85 |
| hrr | 512 | 0.05 | 1 | 0.959 | 1.176 |
| hrr | 512 | 0.05 | 10 | 3.39e+07 | 1.59e+08 |
| hrr | 512 | 0.05 | 100 | 5.85e+88 | 3.31e+89 |

## 2. Superposition capacity (top-1 retrieval)

| scheme | D | K | noise | accuracy |
|---|---:|---:|---:|---:|
| hadamard | 128 | 5 | 0.00 | 1.000 |
| hadamard | 128 | 5 | 0.05 | 1.000 |
| hadamard | 128 | 20 | 0.00 | 0.775 |
| hadamard | 128 | 20 | 0.05 | 0.808 |
| hadamard | 128 | 50 | 0.00 | 0.290 |
| hadamard | 128 | 50 | 0.05 | 0.297 |
| hadamard | 128 | 100 | 0.00 | 0.102 |
| hadamard | 128 | 100 | 0.05 | 0.088 |
| hadamard | 256 | 5 | 0.00 | 1.000 |
| hadamard | 256 | 5 | 0.05 | 1.000 |
| hadamard | 256 | 20 | 0.00 | 0.925 |
| hadamard | 256 | 20 | 0.05 | 0.967 |
| hadamard | 256 | 50 | 0.00 | 0.497 |
| hadamard | 256 | 50 | 0.05 | 0.513 |
| hadamard | 256 | 100 | 0.00 | 0.205 |
| hadamard | 256 | 100 | 0.05 | 0.197 |
| hadamard | 512 | 5 | 0.00 | 1.000 |
| hadamard | 512 | 5 | 0.05 | 1.000 |
| hadamard | 512 | 20 | 0.00 | 1.000 |
| hadamard | 512 | 20 | 0.05 | 1.000 |
| hadamard | 512 | 50 | 0.00 | 0.833 |
| hadamard | 512 | 50 | 0.05 | 0.773 |
| hadamard | 512 | 100 | 0.00 | 0.415 |
| hadamard | 512 | 100 | 0.05 | 0.400 |
| hrr | 128 | 5 | 0.00 | 1.000 |
| hrr | 128 | 5 | 0.05 | 1.000 |
| hrr | 128 | 20 | 0.00 | 0.667 |
| hrr | 128 | 20 | 0.05 | 0.700 |
| hrr | 128 | 50 | 0.00 | 0.290 |
| hrr | 128 | 50 | 0.05 | 0.230 |
| hrr | 128 | 100 | 0.00 | 0.108 |
| hrr | 128 | 100 | 0.05 | 0.092 |
| hrr | 256 | 5 | 0.00 | 1.000 |
| hrr | 256 | 5 | 0.05 | 1.000 |
| hrr | 256 | 20 | 0.00 | 0.958 |
| hrr | 256 | 20 | 0.05 | 0.908 |
| hrr | 256 | 50 | 0.00 | 0.483 |
| hrr | 256 | 50 | 0.05 | 0.500 |
| hrr | 256 | 100 | 0.00 | 0.207 |
| hrr | 256 | 100 | 0.05 | 0.200 |
| hrr | 512 | 5 | 0.00 | 1.000 |
| hrr | 512 | 5 | 0.05 | 1.000 |
| hrr | 512 | 20 | 0.00 | 1.000 |
| hrr | 512 | 20 | 0.05 | 1.000 |
| hrr | 512 | 50 | 0.00 | 0.790 |
| hrr | 512 | 50 | 0.05 | 0.793 |
| hrr | 512 | 100 | 0.00 | 0.418 |
| hrr | 512 | 100 | 0.05 | 0.405 |

## 3. Relation triple separability

Mean relative error when decoding each role. `separate` uses the three stored parts; `per-fact` decodes from one summed relation vector; `memory(n)` decodes from all `n` relation vectors superposed.

| scheme | D | n | separate | per-fact | memory(n) |
|---|---:|---:|---:|---:|---:|
| hadamard | 128 | 1 | 4.93e-17 | 1.417 | 1.417 |
| hadamard | 128 | 5 | 5.02e-17 | 1.412 | 3.712 |
| hadamard | 128 | 10 | 5.06e-17 | 1.417 | 5.609 |
| hadamard | 128 | 20 | 5.00e-17 | 1.414 | 7.922 |
| hadamard | 256 | 1 | 0 | 1.426 | 1.426 |
| hadamard | 256 | 5 | 0 | 1.421 | 3.830 |
| hadamard | 256 | 10 | 0 | 1.415 | 5.462 |
| hadamard | 256 | 20 | 0 | 1.415 | 7.607 |
| hadamard | 512 | 1 | 5.20e-17 | 1.413 | 1.413 |
| hadamard | 512 | 5 | 5.13e-17 | 1.413 | 3.765 |
| hadamard | 512 | 10 | 5.12e-17 | 1.414 | 5.433 |
| hadamard | 512 | 20 | 5.11e-17 | 1.413 | 7.723 |
| hrr | 128 | 1 | 0.980 | 1.689 | 1.689 |
| hrr | 128 | 5 | 0.988 | 1.712 | 4.381 |
| hrr | 128 | 10 | 0.968 | 1.704 | 5.997 |
| hrr | 128 | 20 | 0.949 | 1.684 | 8.582 |
| hrr | 256 | 1 | 0.982 | 1.724 | 1.724 |
| hrr | 256 | 5 | 0.992 | 1.707 | 4.393 |
| hrr | 256 | 10 | 0.973 | 1.737 | 6.131 |
| hrr | 256 | 20 | 1.011 | 1.747 | 8.625 |
| hrr | 512 | 1 | 1.016 | 1.782 | 1.782 |
| hrr | 512 | 5 | 0.976 | 1.712 | 4.343 |
| hrr | 512 | 10 | 0.986 | 1.713 | 6.076 |
| hrr | 512 | 20 | 0.978 | 1.714 | 8.671 |

## 4. Cross-role crosstalk

Max absolute cosine when one role's stored part is decoded with a different role slot.

| scheme | D | mean | max |
|---|---:|---:|---:|
| hadamard | 128 | 0.110 | 0.153 |
| hadamard | 256 | 0.083 | 0.129 |
| hadamard | 512 | 0.049 | 0.090 |
| hrr | 128 | 0.192 | 0.310 |
| hrr | 256 | 0.180 | 0.239 |
| hrr | 512 | 0.136 | 0.218 |

## Findings

1. **Hadamard wins decisively.** HRR round-trip error is ~1.0 at one cycle and **diverges** under repeated bind/unbind: ~1e6 at 10 cycles and ~1e80 at 100 cycles. Correlation is not a stable inverse, so HRR cannot support iterative rebind loops. Values >1e3 in the table mean divergence, not a large but usable error.
2. **HRR gives no consistent capacity gain.** D=128 K=20: 0.667 vs 0.775; D=256 K=20: 0.958 vs 0.925; D=512 K=50: 0.790 vs 0.833. The document's 'convolution has larger capacity' expectation is not supported by this benchmark.
3. **Keep Hadamard.** HRR stays as an audited alternative, not the default binding scheme.
4. **Never decode a role from `fact`.** The summed vector's cross-talk is O(1) (measured ~1.41 relative error even for one relation); only the three separately stored parts are decodable.
5. **Capacity budget holds.** D=128 is safe for K<=5; K=20 needs D=512 or per-fact storage (see Phase 0 report).
