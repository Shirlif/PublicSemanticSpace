# s1_occlusion_20 vs baselines

- episodes: 5  detector_tier: oracle

| method | IDF1 | retention | switches | duplicate | reappearance |
|---|---|---|---|---|---|
| full | 1.0000 | 1.0000 | 0.00 | 0.0000 | 1.0000 |
| no_entity | 0.0113 | 0.0114 | 87.37 | 0.9887 | 0.0000 |
| B1_text | 0.6038 | 0.6015 | 1.39 | 0.5816 | 0.0000 |
| B2_sort | 1.0000 | 1.0000 | 0.00 | 0.0000 | 1.0000 |

## full − baseline (paired bootstrap, 95% CI)

- no_entity / idf1: +0.9887 [+0.9887, +0.9887]
- no_entity / id_retention: +0.9886 [+0.9886, +0.9887]
- no_entity / reappearance_rate: +1.0000 [+1.0000, +1.0000]
- B1_text / idf1: +0.3962 [+0.3923, +0.3996]
- B1_text / id_retention: +0.3985 [+0.3947, +0.4021]
- B1_text / reappearance_rate: +1.0000 [+1.0000, +1.0000]
- B2_sort / idf1: +0.0000 [+0.0000, +0.0000]
- B2_sort / id_retention: +0.0000 [+0.0000, +0.0000]
- B2_sort / reappearance_rate: +0.0000 [+0.0000, +0.0000]
