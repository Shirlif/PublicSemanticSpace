# s1_camera_real vs baselines

- episodes: 1  detector_tier: frozen_real

| method | IDF1 | retention | switches | duplicate | reappearance |
|---|---|---|---|---|---|
| full | 0.7406 | 0.8725 | 0.75 | 0.4000 | 0.7500 |
| no_entity | 0.0137 | 0.0181 | 59.25 | 0.9834 | 0.0000 |
| B1_text | 0.5597 | 0.6120 | 2.00 | 0.6364 | 0.2500 |
| B2_sort | 0.5631 | 0.6195 | 2.75 | 0.6364 | 0.2500 |

## full − baseline (paired bootstrap, 95% CI)

- no_entity / idf1: +0.7270 [+0.7270, +0.7270]
- no_entity / id_retention: +0.8544 [+0.8544, +0.8544]
- no_entity / reappearance_rate: +0.7500 [+0.7500, +0.7500]
- B1_text / idf1: +0.1809 [+0.1809, +0.1809]
- B1_text / id_retention: +0.2604 [+0.2604, +0.2604]
- B1_text / reappearance_rate: +0.5000 [+0.5000, +0.5000]
- B2_sort / idf1: +0.1775 [+0.1775, +0.1775]
- B2_sort / id_retention: +0.2530 [+0.2530, +0.2530]
- B2_sort / reappearance_rate: +0.5000 [+0.5000, +0.5000]
