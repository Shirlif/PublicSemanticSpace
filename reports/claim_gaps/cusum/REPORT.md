# Gap: CUSUM detection across streaming windows

> generated: 2026-11-27 03:59:19 UTC
> source: real
> reference=0.9575  slack=0.0089  threshold=0.0894  (calibrated on 50 clean windows)

| scenario | windows | global alarms | global first | per-class alarms | per-class first |
|---|---:|---:|---:|---:|---:|
| clean | 20 | 0 | None | 0 | None |
| random_label_flip@0.05 | 20 | 6 | 2 | 55 | 1 |
| targeted_agree@0.10 | 20 | 3 | 10 | 0 | None |
| targeted_single_class@0.20 | 20 | 3 | 7 | 32 | 1 |
| drift_sigma@0.30 | 20 | 0 | None | 0 | None |

A single window cannot separate the targeted attacks from noise; the
question is how many windows the CUSUM needs, and whether clean/drift
stay silent.
