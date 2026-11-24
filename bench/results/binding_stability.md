# Binding stability report (Phase 0)

- generated: 2026-11-20 03:23:56 UTC
- trials: roundtrip=24, capacity=8
- runtime: 3.0s

## 1. Slot orthogonality (K=4)

Max absolute off-diagonal entry of the slot Gram matrix. Hadamard slots are exactly orthogonal; Gaussian slots are only approximately so.

| scheme | D | max off-diagonal |
|---|---:|---:|
| hadamard | 128 | 1.11e-16 |
| hadamard | 256 | 0 |
| hadamard | 512 | 1.11e-16 |
| rademacher | 128 | 0.1406 |
| rademacher | 256 | 0.1016 |
| rademacher | 512 | 0.0547 |
| gaussian | 128 | 0.1302 |
| gaussian | 256 | 0.1351 |
| gaussian | 512 | 0.0827 |

## 2. Round-trip error under noise and repeated cycles

Relative error `||unbind(bind(x,s),s) - x|| / ||x||`, with Gaussian noise added to the bound vector each cycle at `noise * RMS(bound)`.

| scheme | D | noise | cycles | mean err | p95 err | mean norm drift |
|---|---:|---:|---:|---:|---:|---:|
| hadamard | 128 | 0.00 | 1 | 4.92e-17 | 6.79e-17 | 3.70e-17 |
| hadamard | 128 | 0.00 | 10 | 5.25e-17 | 6.88e-17 | 5.09e-17 |
| hadamard | 128 | 0.00 | 100 | 5.25e-17 | 6.71e-17 | 4.63e-17 |
| hadamard | 128 | 0.01 | 1 | 0.0100 | 0.0107 | 0.0006 |
| hadamard | 128 | 0.01 | 10 | 0.0315 | 0.0338 | 0.0022 |
| hadamard | 128 | 0.01 | 100 | 0.1001 | 0.1133 | 0.0087 |
| hadamard | 128 | 0.05 | 1 | 0.0509 | 0.0602 | 0.0042 |
| hadamard | 128 | 0.05 | 10 | 0.1598 | 0.1754 | 0.0120 |
| hadamard | 128 | 0.05 | 100 | 0.5397 | 0.5892 | 0.1388 |
| hadamard | 128 | 0.10 | 1 | 0.0976 | 0.1096 | 0.0080 |
| hadamard | 128 | 0.10 | 10 | 0.3151 | 0.3424 | 0.0473 |
| hadamard | 128 | 0.10 | 100 | 1.3284 | 1.5196 | 0.6789 |
| hadamard | 256 | 0.00 | 1 | 0 | 0 | 4.16e-17 |
| hadamard | 256 | 0.00 | 10 | 0 | 0 | 3.70e-17 |
| hadamard | 256 | 0.00 | 100 | 0 | 0 | 3.24e-17 |
| hadamard | 256 | 0.01 | 1 | 0.0101 | 0.0107 | 0.0005 |
| hadamard | 256 | 0.01 | 10 | 0.0315 | 0.0343 | 0.0018 |
| hadamard | 256 | 0.01 | 100 | 0.1000 | 0.1056 | 0.0057 |
| hadamard | 256 | 0.05 | 1 | 0.0501 | 0.0534 | 0.0025 |
| hadamard | 256 | 0.05 | 10 | 0.1584 | 0.1674 | 0.0137 |
| hadamard | 256 | 0.05 | 100 | 0.5289 | 0.5694 | 0.1267 |
| hadamard | 256 | 0.10 | 1 | 0.0994 | 0.1052 | 0.0053 |
| hadamard | 256 | 0.10 | 10 | 0.3237 | 0.3534 | 0.0558 |
| hadamard | 256 | 0.10 | 100 | 1.2981 | 1.3907 | 0.6245 |
| hadamard | 512 | 0.00 | 1 | 5.13e-17 | 6.11e-17 | 4.63e-17 |
| hadamard | 512 | 0.00 | 10 | 5.21e-17 | 6.11e-17 | 3.24e-17 |
| hadamard | 512 | 0.00 | 100 | 5.12e-17 | 5.83e-17 | 2.31e-17 |
| hadamard | 512 | 0.01 | 1 | 0.0099 | 0.0105 | 0.0004 |
| hadamard | 512 | 0.01 | 10 | 0.0317 | 0.0331 | 0.0007 |
| hadamard | 512 | 0.01 | 100 | 0.0997 | 0.1041 | 0.0058 |
| hadamard | 512 | 0.05 | 1 | 0.0498 | 0.0521 | 0.0017 |
| hadamard | 512 | 0.05 | 10 | 0.1588 | 0.1660 | 0.0106 |
| hadamard | 512 | 0.05 | 100 | 0.5340 | 0.5557 | 0.1384 |
| hadamard | 512 | 0.10 | 1 | 0.1007 | 0.1056 | 0.0044 |
| hadamard | 512 | 0.10 | 10 | 0.3232 | 0.3326 | 0.0464 |
| hadamard | 512 | 0.10 | 100 | 1.2801 | 1.3328 | 0.6147 |
| rademacher | 128 | 0.00 | 1 | 5.14e-17 | 6.72e-17 | 4.63e-17 |
| rademacher | 128 | 0.00 | 10 | 5.03e-17 | 6.76e-17 | 3.70e-17 |
| rademacher | 128 | 0.00 | 100 | 5.21e-17 | 6.73e-17 | 6.01e-17 |
| rademacher | 128 | 0.01 | 1 | 0.0098 | 0.0107 | 0.0007 |
| rademacher | 128 | 0.01 | 10 | 0.0315 | 0.0358 | 0.0022 |
| rademacher | 128 | 0.01 | 100 | 0.1005 | 0.1107 | 0.0089 |
| rademacher | 128 | 0.05 | 1 | 0.0489 | 0.0537 | 0.0026 |
| rademacher | 128 | 0.05 | 10 | 0.1599 | 0.1757 | 0.0144 |
| rademacher | 128 | 0.05 | 100 | 0.5376 | 0.5979 | 0.1199 |
| rademacher | 128 | 0.10 | 1 | 0.0993 | 0.1084 | 0.0076 |
| rademacher | 128 | 0.10 | 10 | 0.3227 | 0.3565 | 0.0573 |
| rademacher | 128 | 0.10 | 100 | 1.2924 | 1.5242 | 0.6167 |
| rademacher | 256 | 0.00 | 1 | 0 | 0 | 5.09e-17 |
| rademacher | 256 | 0.00 | 10 | 0 | 0 | 3.70e-17 |
| rademacher | 256 | 0.00 | 100 | 0 | 0 | 6.48e-17 |
| rademacher | 256 | 0.01 | 1 | 0.0099 | 0.0104 | 0.0006 |
| rademacher | 256 | 0.01 | 10 | 0.0316 | 0.0334 | 0.0018 |
| rademacher | 256 | 0.01 | 100 | 0.1008 | 0.1082 | 0.0061 |
| rademacher | 256 | 0.05 | 1 | 0.0502 | 0.0535 | 0.0030 |
| rademacher | 256 | 0.05 | 10 | 0.1598 | 0.1712 | 0.0129 |
| rademacher | 256 | 0.05 | 100 | 0.5333 | 0.5793 | 0.1425 |
| rademacher | 256 | 0.10 | 1 | 0.1006 | 0.1076 | 0.0076 |
| rademacher | 256 | 0.10 | 10 | 0.3238 | 0.3464 | 0.0518 |
| rademacher | 256 | 0.10 | 100 | 1.3060 | 1.4364 | 0.6427 |
| rademacher | 512 | 0.00 | 1 | 5.06e-17 | 5.49e-17 | 2.31e-17 |
| rademacher | 512 | 0.00 | 10 | 4.88e-17 | 5.43e-17 | 1.85e-17 |
| rademacher | 512 | 0.00 | 100 | 5.10e-17 | 5.60e-17 | 3.70e-17 |
| rademacher | 512 | 0.01 | 1 | 0.0099 | 0.0102 | 0.0003 |
| rademacher | 512 | 0.01 | 10 | 0.0321 | 0.0336 | 0.0010 |
| rademacher | 512 | 0.01 | 100 | 0.1004 | 0.1040 | 0.0063 |
| rademacher | 512 | 0.05 | 1 | 0.0501 | 0.0521 | 0.0024 |
| rademacher | 512 | 0.05 | 10 | 0.1593 | 0.1655 | 0.0121 |
| rademacher | 512 | 0.05 | 100 | 0.5413 | 0.5745 | 0.1430 |
| rademacher | 512 | 0.10 | 1 | 0.1001 | 0.1044 | 0.0055 |
| rademacher | 512 | 0.10 | 10 | 0.3267 | 0.3378 | 0.0515 |
| rademacher | 512 | 0.10 | 100 | 1.3096 | 1.3832 | 0.6442 |
| gaussian | 128 | 0.00 | 1 | 4.01e-17 | 5.47e-17 | 1.85e-17 |
| gaussian | 128 | 0.00 | 10 | 3.93e-17 | 5.32e-17 | 3.70e-17 |
| gaussian | 128 | 0.00 | 100 | 4.22e-17 | 5.64e-17 | 3.70e-17 |
| gaussian | 128 | 0.01 | 1 | 0.3184 | 1.8078 | 0.1277 |
| gaussian | 128 | 0.01 | 10 | 1.1162 | 4.3819 | 0.7253 |
| gaussian | 128 | 0.01 | 100 | 3.9500 | 16.7942 | 3.2732 |
| gaussian | 128 | 0.05 | 1 | 2.1635 | 7.9673 | 1.6606 |
| gaussian | 128 | 0.05 | 10 | 4.1342 | 11.1420 | 3.3646 |
| gaussian | 128 | 0.05 | 100 | 15.7722 | 52.5324 | 14.8251 |
| gaussian | 128 | 0.10 | 1 | 2.1013 | 8.8649 | 1.4471 |
| gaussian | 128 | 0.10 | 10 | 10.1577 | 32.5963 | 9.3043 |
| gaussian | 128 | 0.10 | 100 | 26.4602 | 80.0649 | 25.5007 |
| gaussian | 256 | 0.00 | 1 | 4.18e-17 | 5.93e-17 | 3.70e-17 |
| gaussian | 256 | 0.00 | 10 | 4.19e-17 | 5.31e-17 | 6.01e-17 |
| gaussian | 256 | 0.00 | 100 | 0.0003 | 6.26e-17 | 0.0000 |
| gaussian | 256 | 0.01 | 1 | 0.3452 | 0.9599 | 0.0865 |
| gaussian | 256 | 0.01 | 10 | 5.2009 | 4.5214 | 4.6459 |
| gaussian | 256 | 0.01 | 100 | 3.0768 | 11.0753 | 2.3565 |
| gaussian | 256 | 0.05 | 1 | 16.3116 | 149.4040 | 15.7120 |
| gaussian | 256 | 0.05 | 10 | 3.1461 | 9.5674 | 2.3881 |
| gaussian | 256 | 0.05 | 100 | 15.5952 | 38.0237 | 14.6577 |
| gaussian | 256 | 0.10 | 1 | 3.0915 | 11.3390 | 2.3912 |
| gaussian | 256 | 0.10 | 10 | 10.0360 | 41.3032 | 9.1442 |
| gaussian | 256 | 0.10 | 100 | 112.3144 | 116.2482 | 111.3423 |
| gaussian | 512 | 0.00 | 1 | 4.07e-17 | 4.65e-17 | 3.70e-17 |
| gaussian | 512 | 0.00 | 10 | 4.16e-17 | 4.86e-17 | 2.31e-17 |
| gaussian | 512 | 0.00 | 100 | 4.29e-17 | 5.08e-17 | 4.16e-17 |
| gaussian | 512 | 0.01 | 1 | 0.4036 | 0.7743 | 0.1190 |
| gaussian | 512 | 0.01 | 10 | 2.6544 | 10.0962 | 2.1177 |
| gaussian | 512 | 0.01 | 100 | 9.7818 | 26.3198 | 8.9517 |
| gaussian | 512 | 0.05 | 1 | 2.0026 | 5.6579 | 1.3532 |
| gaussian | 512 | 0.05 | 10 | 7.8080 | 29.0700 | 6.9348 |
| gaussian | 512 | 0.05 | 100 | 21.0969 | 77.5534 | 20.1422 |
| gaussian | 512 | 0.10 | 1 | 16.2206 | 60.7829 | 15.3888 |
| gaussian | 512 | 0.10 | 10 | 60.9092 | 257.0134 | 59.9910 |
| gaussian | 512 | 0.10 | 100 | 57.6886 | 150.3682 | 56.6933 |

## 3. Superposition capacity (top-1 retrieval)

`K` facts share one memory vector: `memory = sum_i bind(f_i, s_i)`. Each fact is retrieved by `unbind(memory, s_i)` and ranked by cosine.

| scheme | D | K | noise | top-1 accuracy |
|---|---:|---:|---:|---:|
| hadamard | 128 | 5 | 0.00 | 1.000 |
| hadamard | 128 | 5 | 0.05 | 1.000 |
| hadamard | 128 | 20 | 0.00 | 0.769 |
| hadamard | 128 | 20 | 0.05 | 0.662 |
| hadamard | 128 | 50 | 0.00 | 0.285 |
| hadamard | 128 | 50 | 0.05 | 0.302 |
| hadamard | 128 | 100 | 0.00 | 0.100 |
| hadamard | 128 | 100 | 0.05 | 0.091 |
| hadamard | 256 | 5 | 0.00 | 1.000 |
| hadamard | 256 | 5 | 0.05 | 1.000 |
| hadamard | 256 | 20 | 0.00 | 0.944 |
| hadamard | 256 | 20 | 0.05 | 0.919 |
| hadamard | 256 | 50 | 0.00 | 0.512 |
| hadamard | 256 | 50 | 0.05 | 0.518 |
| hadamard | 256 | 100 | 0.00 | 0.215 |
| hadamard | 256 | 100 | 0.05 | 0.198 |
| hadamard | 512 | 5 | 0.00 | 1.000 |
| hadamard | 512 | 5 | 0.05 | 1.000 |
| hadamard | 512 | 20 | 0.00 | 1.000 |
| hadamard | 512 | 20 | 0.05 | 0.994 |
| hadamard | 512 | 50 | 0.00 | 0.782 |
| hadamard | 512 | 50 | 0.05 | 0.802 |
| hadamard | 512 | 100 | 0.00 | 0.405 |
| hadamard | 512 | 100 | 0.05 | 0.436 |
| rademacher | 128 | 5 | 0.00 | 1.000 |
| rademacher | 128 | 5 | 0.05 | 1.000 |
| rademacher | 128 | 20 | 0.00 | 0.669 |
| rademacher | 128 | 20 | 0.05 | 0.731 |
| rademacher | 128 | 50 | 0.00 | 0.273 |
| rademacher | 128 | 50 | 0.05 | 0.278 |
| rademacher | 128 | 100 | 0.00 | 0.106 |
| rademacher | 128 | 100 | 0.05 | 0.098 |
| rademacher | 256 | 5 | 0.00 | 1.000 |
| rademacher | 256 | 5 | 0.05 | 1.000 |
| rademacher | 256 | 20 | 0.00 | 0.906 |
| rademacher | 256 | 20 | 0.05 | 0.938 |
| rademacher | 256 | 50 | 0.00 | 0.502 |
| rademacher | 256 | 50 | 0.05 | 0.505 |
| rademacher | 256 | 100 | 0.00 | 0.172 |
| rademacher | 256 | 100 | 0.05 | 0.190 |
| rademacher | 512 | 5 | 0.00 | 1.000 |
| rademacher | 512 | 5 | 0.05 | 1.000 |
| rademacher | 512 | 20 | 0.00 | 1.000 |
| rademacher | 512 | 20 | 0.05 | 1.000 |
| rademacher | 512 | 50 | 0.00 | 0.840 |
| rademacher | 512 | 50 | 0.05 | 0.805 |
| rademacher | 512 | 100 | 0.00 | 0.411 |
| rademacher | 512 | 100 | 0.05 | 0.406 |
| gaussian | 128 | 5 | 0.00 | 0.375 |
| gaussian | 128 | 5 | 0.05 | 0.475 |
| gaussian | 128 | 20 | 0.00 | 0.044 |
| gaussian | 128 | 20 | 0.05 | 0.044 |
| gaussian | 128 | 50 | 0.00 | 0.033 |
| gaussian | 128 | 50 | 0.05 | 0.035 |
| gaussian | 128 | 100 | 0.00 | 0.015 |
| gaussian | 128 | 100 | 0.05 | 0.011 |
| gaussian | 256 | 5 | 0.00 | 0.350 |
| gaussian | 256 | 5 | 0.05 | 0.275 |
| gaussian | 256 | 20 | 0.00 | 0.075 |
| gaussian | 256 | 20 | 0.05 | 0.081 |
| gaussian | 256 | 50 | 0.00 | 0.018 |
| gaussian | 256 | 50 | 0.05 | 0.035 |
| gaussian | 256 | 100 | 0.00 | 0.011 |
| gaussian | 256 | 100 | 0.05 | 0.013 |
| gaussian | 512 | 5 | 0.00 | 0.400 |
| gaussian | 512 | 5 | 0.05 | 0.250 |
| gaussian | 512 | 20 | 0.00 | 0.050 |
| gaussian | 512 | 20 | 0.05 | 0.044 |
| gaussian | 512 | 50 | 0.00 | 0.020 |
| gaussian | 512 | 50 | 0.05 | 0.025 |
| gaussian | 512 | 100 | 0.00 | 0.013 |
| gaussian | 512 | 100 | 0.05 | 0.010 |

## Notes

- Hadamard and Rademacher slots have `|s_i| = 1/sqrt(D)` for every component, so division never amplifies near-zero components.
- Gaussian unit vectors contain near-zero components; those components amplify noise, which is why the Gaussian row degrades fastest.
- Capacity is approximate by construction: cross-talk grows with `K`, and orthogonal slots do not make element-wise binding orthogonal.
- Contract v0 therefore uses Hadamard slots with a power-of-two `D` (128/256/512).
