# Gap: CUSUM detection across streaming windows

> generated: 2026-11-27 09:50:38 UTC
> source: real
> reference=0.9575  slack=0.0089  threshold=0.0894  (calibrated on 50 clean windows)
> confidence gap: reference=-0.0058  slack=0.0088  threshold=0.0882

| scenario | windows | global first | per-class first | confidence first | first quarantine | quarantined at end |
|---|---:|---:|---:|---:|---:|---|
| clean | 20 | None | None | None | None | [] |
| random_label_flip@0.05 | 20 | 2 | 1 | None | None | [] |
| targeted_agree@0.10 | 20 | 10 | None | None | None | [] |
| targeted_single_class@0.20 | 20 | 7 | 1 | None | None | [] |
| drift_sigma@0.30 | 20 | None | None | None | None | [] |
| source_outlier_stream | 20 | 1 | 1 | None | 1 | ['bad'] |

A single window cannot separate the targeted attacks from noise; the
question is how many windows the CUSUM needs, and whether clean/drift
stay silent.  `confidence first` is the independent signed-gap CUSUM
on the confidence map.  At this scale it stays silent: its clean
window-to-window noise (sd ~2.2pp) exceeds the per-window attack
signal (0.16-0.48pp), so the label/consensus detectors remain the
effective ones and the gap CUSUM is a diagnostic, not a detector.
