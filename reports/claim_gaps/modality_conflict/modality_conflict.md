# Modality conflict: PSS fixed fusion vs transformer attention

- data: `data/category_pretrain`
- seen test samples: 192
- PSS weights: {'vision': 1.0, 'tactile': 0.5, 'geometry': 1.0}

| condition | PSS fixed | PSS abstain (cov/acc) | PSS reliable (cov/acc) | PSS drop | Transformer |
|---|---:|---:|---:|---:|---:|
| clean | 0.979 | 0.85/0.982 | 0.74/0.993 | 0.979 | 0.976 ± 0.006 |
| tactile_shuffle | 0.974 | 0.85/0.982 | 0.74/0.993 | 0.974 | 0.960 ± 0.005 |
| tactile_noise_0.5 | 0.979 | 0.85/0.982 | 0.74/0.993 | 0.979 | 0.977 ± 0.005 |
| tactile_noise_1.0 | 0.979 | 0.85/0.982 | 0.74/0.993 | 0.979 | 0.974 ± 0.004 |
| tactile_zero | 0.979 | 0.85/0.982 | 0.74/0.993 | 0.979 | 0.964 ± 0.004 |
| vision_shuffle | 0.474 | 0.08/1.000 | 0.07/1.000 | 0.474 | 0.153 ± 0.009 |
| geometry_shuffle | 0.458 | 0.07/0.923 | 0.05/1.000 | 0.458 | 0.734 ± 0.004 |
| vision_zero | 0.865 | 1.00/0.865 | 0.82/0.904 | 0.865 | 0.635 ± 0.041 |
| geometry_zero | 0.979 | 1.00/0.979 | 0.97/0.989 | 0.979 | 0.910 ± 0.024 |

Both models see clean training data; only the test modality is corrupted.
Conflict-aware PSS uses per-modality top-1 disagreement; abstain drops conflicted samples, drop re-fuses without disagreeing modalities.
