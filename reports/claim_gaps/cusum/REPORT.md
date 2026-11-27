# Gap: CUSUM detection across streaming windows

> generated: 2026-11-27 04:10:31 UTC
> source: real
> reference=0.9575  slack=0.0089  threshold=0.0894  (calibrated on 50 clean windows)

| scenario | windows | global first | per-class first | first quarantine | quarantined at end |
|---|---:|---:|---:|---:|---|
| clean | 20 | None | None | None | [] |
| random_label_flip@0.05 | 20 | 2 | 1 | None | [] |
| targeted_agree@0.10 | 20 | 10 | None | None | [] |
| targeted_single_class@0.20 | 20 | 7 | 1 | None | [] |
| drift_sigma@0.30 | 20 | None | None | None | [] |
| source_outlier_stream | 20 | 1 | 1 | 1 | ['bad'] |

A single window cannot separate the targeted attacks from noise; the
question is how many windows the CUSUM needs, and whether clean/drift
stay silent.
