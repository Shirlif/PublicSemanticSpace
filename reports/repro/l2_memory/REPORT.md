# Claim-gap E12: L2 bundle replay under cognitive ablations

> bundles: 33 under `bundles/repro`

## Overall

| variant | n | ID retention | switches | reappearance | facts mean | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| full | 33 | 0.576 | 0.00 | 0.576 | 0.00 | 0.901 |
| no_memory | 33 | 0.476 | 0.32 | 0.571 | 0.00 | 0.843 |
| no_entity | 33 | 0.156 | 27.33 | 0.121 | 0.00 | 0.527 |
| no_relations | 33 | 0.576 | 0.00 | 0.576 | 0.00 | 0.853 |

## Valid bundles only

Bundles where the full runtime has a non-empty identity trace
(19 of 33). Zero-retention bundles are excluded
because they carry no identity signal to ablate.

| variant | n | ID retention | switches |
|---|---:|---:|---:|
| full | 19 | 1.000 | 0.00 |
| no_memory | 19 | 0.827 | 0.56 |
| no_entity | 19 | 0.270 | 47.47 |
| no_relations | 19 | 1.000 | 0.00 |

## By bundle group

| group | variant | ID retention | switches | reappearance | facts |
|---|---|---:|---:|---:|---:|
| g1_load | full | 0.000 | 0.00 | 0.000 | 0.00 |
| g1_load | no_memory | 0.000 | 0.00 | 0.000 | 0.00 |
| g1_load | no_entity | 0.000 | 0.00 | 0.000 | 0.00 |
| g1_load | no_relations | 0.000 | 0.00 | 0.000 | 0.00 |
| s1_camera_motion | full | 0.000 | 0.00 | 0.000 | 0.00 |
| s1_camera_motion | no_memory | 0.000 | 0.00 | 0.000 | 0.00 |
| s1_camera_motion | no_entity | 0.000 | 0.00 | 0.000 | 0.00 |
| s1_camera_motion | no_relations | 0.000 | 0.00 | 0.000 | 0.00 |
| s1_camera_oracle | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_camera_oracle | no_memory | 0.546 | 1.00 | 0.987 | 0.00 |
| s1_camera_oracle | no_entity | 0.014 | 74.25 | 0.000 | 0.00 |
| s1_camera_oracle | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_camera_real | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_camera_real | no_memory | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_camera_real | no_entity | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_camera_real | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_hard | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_hard | no_memory | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_hard | no_entity | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_hard | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |
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
| s2_twins | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s2_twins | no_memory | 1.000 | 0.00 | 1.000 | 0.00 |
| s2_twins | no_entity | 1.000 | 0.00 | 1.000 | 0.00 |
| s2_twins | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |
| s2_twins_oracle | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s2_twins_oracle | no_memory | 0.629 | 1.00 | 0.989 | 0.00 |
| s2_twins_oracle | no_entity | 0.011 | 92.00 | 0.000 | 0.00 |
| s2_twins_oracle | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |
| s3_relations_ood | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s3_relations_ood | no_memory | 1.000 | 0.00 | 1.000 | 0.00 |
| s3_relations_ood | no_entity | 0.020 | 49.00 | 0.000 | 0.00 |
| s3_relations_ood | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |
| s4_g1_pick_place | full | 0.000 | 0.00 | 0.000 | 0.00 |
| s4_g1_pick_place | no_memory | 0.000 | 0.00 | 0.000 | 0.00 |
| s4_g1_pick_place | no_entity | 0.000 | 0.00 | 0.000 | 0.00 |
| s4_g1_pick_place | no_relations | 0.000 | 0.00 | 0.000 | 0.00 |
| s6_memory | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s6_memory | no_memory | 1.000 | 0.00 | 1.000 | 0.00 |
| s6_memory | no_entity | 0.033 | 29.00 | 0.000 | 0.00 |
| s6_memory | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |

## How to read this

The replay consumes only `obs_detections.parquet`; it never reads GT for
association. GT is used after the fact for scoring. `no_memory` and
`no_entity` are architecture ablations on the same detection stream, so the
difference is the state layer, not perception. `no_relations` should keep
identity intact while `facts_n` drops; that is the relation-layer control.
