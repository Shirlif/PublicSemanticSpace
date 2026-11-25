# s1_camera_oracle scorecard

- device_class: `dev`
- detector_tier: `oracle`
- episodes: 1  seeds: [0]
- spec hashes: `sha256:b4bb5f0ef7cfa22152e853a76ef3ac4e9ad500824a19cee7505cb411bf06dc5d`
- caveat: oracle tier is a calibration upper bound (perfect GT identity/pose); it is not capability evidence

| metric | mean | std | 95% CI | min | max |
|---|---|---|---|---|---|
| idf1 | 1.0000 | 0.0000 | ±0.0000 | 1.0000 | 1.0000 |
| id_retention | 1.0000 | 0.0000 | ±0.0000 | 1.0000 | 1.0000 |
| id_switches | 0.0000 | 0.0000 | ±0.0000 | 0.0000 | 0.0000 |
| duplicate_rate | 0.0000 | 0.0000 | ±0.0000 | 0.0000 | 0.0000 |
| reappearance_rate | 1.0000 | 0.0000 | ±0.0000 | 1.0000 | 1.0000 |
| slot_purity | 1.0000 | 0.0000 | ±0.0000 | 1.0000 | 1.0000 |

