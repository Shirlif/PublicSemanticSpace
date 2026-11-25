# s2_twins_oracle scorecard

- device_class: `dev`
- detector_tier: `oracle`
- episodes: 1  seeds: [0]
- spec hashes: `sha256:3d49ce1bca786b2706c80cf76d7723c8d563c33d04c7b2351616913ea0bfdd18`
- caveat: oracle tier is a calibration upper bound (perfect GT identity/pose); it is not capability evidence

| metric | mean | std | 95% CI | min | max |
|---|---|---|---|---|---|
| idf1 | 1.0000 | 0.0000 | ±0.0000 | 1.0000 | 1.0000 |
| id_retention | 1.0000 | 0.0000 | ±0.0000 | 1.0000 | 1.0000 |
| id_switches | 0.0000 | 0.0000 | ±0.0000 | 0.0000 | 0.0000 |
| duplicate_rate | 0.0000 | 0.0000 | ±0.0000 | 0.0000 | 0.0000 |
| reappearance_rate | 1.0000 | 0.0000 | ±0.0000 | 1.0000 | 1.0000 |
| slot_purity | 1.0000 | 0.0000 | ±0.0000 | 1.0000 | 1.0000 |

