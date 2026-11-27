# Gap: CUSUM detection across streaming windows

> generated: 2026-11-27 03:19:05 UTC
> source: synthetic
> reference=0.9645  slack=0.0201  threshold=0.1005  (calibrated on 50 clean windows)

| scenario | windows | alarms | first alarm | detected |
|---|---:|---:|---:|---|
| clean | 20 | 0 | None | False |
| random_label_flip@0.05 | 20 | 5 | 4 | True |
| targeted_agree@0.10 | 20 | 2 | 7 | True |
| targeted_single_class@0.20 | 20 | 1 | 15 | True |
| drift_sigma@0.30 | 20 | 0 | None | False |

A single window cannot separate the targeted attacks from noise; the
question is how many windows the CUSUM needs, and whether clean/drift
stay silent.
