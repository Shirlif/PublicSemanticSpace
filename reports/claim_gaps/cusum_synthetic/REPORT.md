# Gap: CUSUM detection across streaming windows

> generated: 2026-11-27 04:10:19 UTC
> source: synthetic
> reference=0.9645  slack=0.0080  threshold=0.0804  (calibrated on 50 clean windows)

| scenario | windows | global first | per-class first | first quarantine | quarantined at end |
|---|---:|---:|---:|---:|---|
| clean | 20 | None | None | None | [] |
| random_label_flip@0.05 | 20 | 5 | 1 | None | [] |
| targeted_agree@0.10 | 20 | 4 | None | None | [] |
| targeted_single_class@0.20 | 20 | 11 | 5 | None | [] |
| drift_sigma@0.30 | 20 | None | None | None | [] |
| source_outlier_stream | 20 | 3 | 3 | 1 | ['bad'] |

A single window cannot separate the targeted attacks from noise; the
question is how many windows the CUSUM needs, and whether clean/drift
stay silent.
