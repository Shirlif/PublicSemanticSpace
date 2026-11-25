# Claim-gap E6: retrain-all proxy

> alt data: `data/category_pretrain`, alt experts: `eval/results/_retrain_experts_30ep`
> train/test: 1152/192, classes: 16, chance: 0.062

| method | vision ZCR | tactile ZCR | cost |
|---|---:|---:|---:|
| frozen heads + alt prototypes | 0.953 | 0.234 | 2.87 ms |
| 1-shot ridge calibration | 0.927 | 0.208 | 109.51-102.67 ms |
| retrained projection + prototypes | 0.969 | 0.224 | 13.41 s (+ expert 24.60 s) |

## How to read this

This is the honest middle ground between the oracle A5 and the 1-shot
calibration.

This run also retrains the expert encoders (24.6 s), then the projection heads and prototypes.

On this family, rebuilding prototypes on frozen heads recovers most of the
gap; retraining the projections adds a small gain at tens of seconds; the
1-shot ridge map is worse than either because the prototypes are already in
the target family. Practical reading: prototype rebuild is the dominant
cheap step, projection retraining is optional, and calibration is not a
universal fix. The cross-family zero-shot failure against the *base bank*
is measured separately in E11.

Retrains projection heads + prototypes only; frozen expert encoders are not retrained, so this is a lower bound on retrain-all cost. The frozen condition rebuilds prototypes on the target family, so it is not the base-bank zero-shot condition (that is the E11 family swap).
