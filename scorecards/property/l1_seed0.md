# PSS property scorecard — L1

- seed: `0`
- data: `data/expert_v1`
- experts: `checkpoints\expert_v1`
- samples: `96`

## Swap metrics (real embeddings)

| metric | value |
|---|---:|
| SSR_L1 | 1 |
| ZCR_L1 | 0.65 |
| ZCR_same_modality_L1 | 0.84375 |
| ZCR_ratio_Q1 | 0.967742 |
| ZCR_ratio_mean | 1.01206 |
| INT_L1 | 0 |
| RBE_L1 | 1 |
| RCR_L1 | 1 |
| swap_latency_ms_L1 | 0.6273 |

## Quality sweep

| level | target acc | ZCR | ratio vs expected | rollback | swap ok |
|---|---:|---:|---:|---:|---|
| Q0 | 1.00 | 0.9688 | 0.969 | 1.00 | True |
| Q1 | 0.90 | 0.8438 | 0.968 | 1.00 | True |
| Q2 | 0.75 | 0.7500 | 1.032 | 1.00 | True |
| Q3 | 0.55 | 0.5521 | 1.036 | 1.00 | True |
| Q4 | 0.00 | 0.1354 | 135416666.667 | 1.00 | True |

## Expert-family swap (frozen shared space, H7)

- alt data: `data/category_pretrain`
- alt experts: `checkpoints\category_pretrain_experts`
- samples: 60
- vision ZCR: 0.2
- tactile ZCR: 0.06666666666666667
- chance: 0.16666666666666666

## Few-shot calibration (H7 follow-up)

- test samples: 60
- chance: 0.16666666666666666

| k | vision ZCR | tactile ZCR |
|---:|---:|---:|
| 0 | 0.2000 | 0.0667 |
| 1 | 1.0000 | 0.4000 |
| 5 | 0.9500 | 0.3167 |
| 20 | 1.0000 | 0.3333 |
