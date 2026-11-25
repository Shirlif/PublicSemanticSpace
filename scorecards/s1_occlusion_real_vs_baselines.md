# s1_occlusion_real vs baselines

- episodes: 1  detector_tier: frozen_real

| method | IDF1 | retention | switches | duplicate | reappearance |
|---|---|---|---|---|---|
| full | 0.8942 | 1.0000 | 0.00 | 0.0000 | 1.0000 |
| no_entity | 0.0096 | 0.0108 | 92.00 | 0.9892 | 0.0000 |
| B1_text | 0.8942 | 1.0000 | 0.00 | 0.0000 | 1.0000 |
| B2_sort | 0.8942 | 1.0000 | 0.00 | 0.0000 | 1.0000 |

## full − baseline (paired bootstrap, 95% CI)

- no_entity / idf1: +0.8846 [+0.8846, +0.8846]
- no_entity / id_retention: +0.9892 [+0.9892, +0.9892]
- no_entity / reappearance_rate: +1.0000 [+1.0000, +1.0000]
- B1_text / idf1: +0.0000 [+0.0000, +0.0000]
- B1_text / id_retention: +0.0000 [+0.0000, +0.0000]
- B1_text / reappearance_rate: +0.0000 [+0.0000, +0.0000]
- B2_sort / idf1: +0.0000 [+0.0000, +0.0000]
- B2_sort / id_retention: +0.0000 [+0.0000, +0.0000]
- B2_sort / reappearance_rate: +0.0000 [+0.0000, +0.0000]
