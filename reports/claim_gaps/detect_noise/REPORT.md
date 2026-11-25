# Claim-gap E7: binding under non-oracle detection noise

> data: `data\tracking_bench_long`, sequences: 60

| condition | miss rate | pose sigma | ID retention | switches | reappearance |
|---|---:|---:|---:|---:|---:|
| oracle | 0.00 | 0.000 | 0.989 | 0.02 | 0.978 |
| miss10_sigma2cm | 0.10 | 0.020 | 0.975 | 0.06 | 0.965 |
| miss20_sigma5cm | 0.20 | 0.050 | 0.871 | 0.58 | 0.907 |
| miss30_sigma8cm | 0.30 | 0.080 | 0.697 | 2.06 | 0.756 |

## Per subset

| condition | mixed | samecat | near-identical |
|---|---:|---:|---:|
| oracle | 1.000 | 1.000 | 0.956 |
| miss10_sigma2cm | 0.991 | 1.000 | 0.918 |
| miss20_sigma5cm | 0.905 | 0.828 | 0.844 |
| miss30_sigma8cm | 0.699 | 0.678 | 0.711 |

## How to read this

This is a stress test of the state layer's input, not a detector claim.
A large drop at high miss rates means the binding contract is oracle-dependent;
a flat curve means the tracker tolerates detector noise.
