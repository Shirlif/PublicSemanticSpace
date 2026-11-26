# L2 perception subset: detector embeddings, no memory ablation

> bundles: 4 under `bundles/repro`

## Overall

| variant | n | ID retention | switches | reappearance | facts mean | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| full | 4 | 1.000 | 0.00 | 1.000 | 0.00 | 1.431 |
| no_memory | 4 | 1.000 | 0.00 | 1.000 | 0.00 | 1.269 |
| no_entity | 4 | 1.000 | 0.00 | 1.000 | 0.00 | 1.059 |
| no_relations | 4 | 1.000 | 0.00 | 1.000 | 0.00 | 1.307 |

## Valid bundles only

Bundles where the full runtime has a non-empty identity trace
(4 of 4). Zero-retention bundles are excluded
because they carry no identity signal to ablate.

| variant | n | ID retention | switches |
|---|---:|---:|---:|
| full | 4 | 1.000 | 0.00 |
| no_memory | 4 | 1.000 | 0.00 |
| no_entity | 4 | 1.000 | 0.00 |
| no_relations | 4 | 1.000 | 0.00 |

## By bundle group

| group | variant | ID retention | switches | reappearance | facts |
|---|---|---:|---:|---:|---:|
| s1_camera_real | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_camera_real | no_memory | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_camera_real | no_entity | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_camera_real | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_hard | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_hard | no_memory | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_hard | no_entity | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_hard | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_occlusion_real | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_occlusion_real | no_memory | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_occlusion_real | no_entity | 1.000 | 0.00 | 1.000 | 0.00 |
| s1_occlusion_real | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |
| s2_twins | full | 1.000 | 0.00 | 1.000 | 0.00 |
| s2_twins | no_memory | 1.000 | 0.00 | 1.000 | 0.00 |
| s2_twins | no_entity | 1.000 | 0.00 | 1.000 | 0.00 |
| s2_twins | no_relations | 1.000 | 0.00 | 1.000 | 0.00 |

## How to read this

The replay consumes only `obs_detections.parquet`; it never reads GT for
association. GT is used after the fact for scoring. `no_memory` and
`no_entity` are architecture ablations on the same detection stream, so the
difference is the state layer, not perception. `no_relations` should keep
identity intact while `facts_n` drops; that is the relation-layer control.
