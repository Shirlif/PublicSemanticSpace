# Claim-gap E11: contract across expert families

> base: `checkpoints/expert_v1` + `checkpoints/projection_v1`
> ks: [0, 1, 5, 20]

| family | samples | chance | vision ZCR | tactile ZCR | vision k=1 | tactile k=1 |
|---|---:|---:|---:|---:|---:|---:|
| category_pretrain | 60 | 0.167 | 0.200 | 0.067 | 1.000 | 0.400 |
| category_pretrain_instance | 60 | 0.167 | 0.167 | 0.067 | 0.900 | 0.350 |
| category_pretrain_instance8 | 120 | 0.167 | 0.192 | 0.100 | 0.908 | 0.325 |

## Full calibration curves

### category_pretrain

| k | vision | tactile |
|---:|---:|---:|
| 0 | 0.200 | 0.067 |
| 1 | 1.000 | 0.400 |
| 5 | 0.950 | 0.317 |
| 20 | 1.000 | 0.333 |

### category_pretrain_instance

| k | vision | tactile |
|---:|---:|---:|
| 0 | 0.167 | 0.067 |
| 1 | 0.900 | 0.350 |
| 5 | 0.967 | 0.267 |
| 20 | 1.000 | 0.267 |

### category_pretrain_instance8

| k | vision | tactile |
|---:|---:|---:|
| 0 | 0.192 | 0.100 |
| 1 | 0.908 | 0.325 |
| 5 | 0.933 | 0.342 |
| 20 | 0.992 | 0.275 |

## How to read this

Same-modality swap should recover with a little calibration (k=1) regardless
of family. Tactile is expected to stay limited by the tactile expert itself.
A family where vision does not recover means the frozen space is not a
calibratable interface for that family.
