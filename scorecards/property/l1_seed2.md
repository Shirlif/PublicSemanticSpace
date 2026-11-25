# PSS property scorecard — L1

- seed: `2`
- data: `data/expert_v1`
- experts: `checkpoints\expert_v1`
- samples: `96`

## Swap metrics (real embeddings)

| metric | value |
|---|---:|
| SSR_L1 | 1 |
| ZCR_L1 | 0.664583 |
| ZCR_same_modality_L1 | 0.895833 |
| ZCR_ratio_Q1 | 1.02748 |
| ZCR_ratio_mean | 1.03849 |
| INT_L1 | 0 |
| RBE_L1 | 1 |
| RCR_L1 | 1 |
| swap_latency_ms_L1 | 0.11506 |

## Quality sweep

| level | target acc | ZCR | ratio vs expected | rollback | swap ok |
|---|---:|---:|---:|---:|---|
| Q0 | 1.00 | 0.9688 | 0.969 | 1.00 | True |
| Q1 | 0.90 | 0.8958 | 1.027 | 1.00 | True |
| Q2 | 0.75 | 0.7500 | 1.032 | 1.00 | True |
| Q3 | 0.55 | 0.5625 | 1.056 | 1.00 | True |
| Q4 | 0.00 | 0.1458 | 145833333.333 | 1.00 | True |
