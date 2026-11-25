# Claim-gap E14: per-modality space ablation

> data: `data/expert_v1`, split: test, classes: 6, chance: 0.167

| variant | accuracy | cross-modal R@1 |
|---|---:|---:|
| full per-modality | 0.969 | 0.510 |
| cross-modal sum | 0.969 | - |
| random projection | 0.969 | 0.115 |

## How to read this

Closed-set accuracy is invariant across the three variants because the
prototype bank is re-fit in each space; that is not where the learned
alignment shows up. It shows up in cross-modal retrieval: trained
vision-to-tactile R@1 is far above the random-projection control. So the
honest claim is 'per-modality projection provides cross-modal alignment',
not 'per-modality projection improves closed-set accuracy'.
