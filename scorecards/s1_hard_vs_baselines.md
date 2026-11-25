# s1_hard vs baselines

- episodes: 1  detector_tier: frozen_real

| method | IDF1 | retention | switches | duplicate | reappearance |
|---|---|---|---|---|---|
| full | 0.5874 | 0.7088 | 0.92 | 0.7692 | 0.2308 |
| no_entity | 0.0232 | 0.0272 | 35.92 | 0.9729 | 0.0000 |
| B1_text | 0.5179 | 0.6032 | 1.00 | 0.5000 | 0.0769 |
| B2_sort | 0.5179 | 0.6032 | 1.00 | 0.5217 | 0.0769 |

## full − baseline (paired bootstrap, 95% CI)

- no_entity / idf1: +0.5643 [+0.5643, +0.5643]
- no_entity / id_retention: +0.6816 [+0.6816, +0.6816]
- no_entity / reappearance_rate: +0.2308 [+0.2308, +0.2308]
- B1_text / idf1: +0.0696 [+0.0696, +0.0696]
- B1_text / id_retention: +0.1056 [+0.1056, +0.1056]
- B1_text / reappearance_rate: +0.1538 [+0.1538, +0.1538]
- B2_sort / idf1: +0.0696 [+0.0696, +0.0696]
- B2_sort / id_retention: +0.1056 [+0.1056, +0.1056]
- B2_sort / reappearance_rate: +0.1538 [+0.1538, +0.1538]
