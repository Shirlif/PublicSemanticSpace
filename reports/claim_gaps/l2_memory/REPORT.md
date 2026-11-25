# Claim-gap E12: L2 bundle replay under cognitive ablations

> bundles: 9 under `bundles\dev`

## Overall

| variant | n | ID retention | switches | reappearance | facts mean | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| full | 9 | 0.889 | 0.00 | 0.889 | 0.00 | 5.002 |
| no_memory | 9 | 0.615 | 0.96 | 0.873 | 0.00 | 5.431 |
| no_entity | 9 | 0.213 | 60.19 | 0.111 | 0.00 | 2.293 |
| no_relations | 9 | 0.889 | 0.00 | 0.889 | 0.00 | 5.554 |

## By bundle group

| group | variant | ID retention | switches | reappearance | facts |
|---|---|---:|---:|---:|---:|
| s1_occlusion | full | 0.000 | 0.00 | 0.000 | 0.00 |
| s1_occlusion | no_memory | 0.000 | 0.00 | 0.000 | 0.00 |
| s1_occlusion | no_entity | 0.000 | 0.00 | 0.000 | 0.00 |
| s1_occlusion | no_relations | 0.000 | 0.00 | 0.000 | 0.00 |
| s1_occlusion_20 | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_occlusion_20 | no_memory | 0.602 | 1.39 | 0.984 | 0.00 |
| s1_occlusion_20 | no_entity | 0.011 | 87.37 | 0.000 | 0.00 |
| s1_occlusion_20 | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_occlusion_noisy | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_occlusion_noisy | no_memory | 0.920 | 0.70 | 0.945 | 0.00 |
| s1_occlusion_noisy | no_entity | 0.854 | 12.89 | 0.000 | 0.00 |
| s1_occlusion_noisy | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_occlusion_oracle | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_occlusion_oracle | no_memory | 0.608 | 1.00 | 0.989 | 0.00 |
| s1_occlusion_oracle | no_entity | 0.011 | 92.00 | 0.000 | 0.00 |
| s1_occlusion_oracle | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_occlusion_real | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_occlusion_real | no_memory | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_occlusion_real | no_entity | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_occlusion_real | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |

## How to read this

The replay consumes only `obs_detections.parquet`; it never reads GT for
association. GT is used after the fact for scoring. `no_memory` and
`no_entity` are architecture ablations on the same detection stream, so the
difference is the state layer, not perception. `no_relations` should keep
identity intact while `facts_n` drops; that is the relation-layer control.
