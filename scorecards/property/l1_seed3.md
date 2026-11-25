# PSS property scorecard — L1

- seed: `3`
- data: `data/expert_v1`
- experts: `checkpoints\expert_v1`
- samples: `96`

## Swap metrics (real embeddings)

| metric | value |
|---|---:|
| SSR_L1 | 1 |
| ZCR_L1 | 0.6875 |
| ZCR_same_modality_L1 | 0.916667 |
| ZCR_ratio_Q1 | 1.05137 |
| ZCR_ratio_mean | 1.09815 |
| INT_L1 | 0 |
| RBE_L1 | 1 |
| RCR_L1 | 1 |
| swap_latency_ms_L1 | 0.10118 |

## Quality sweep

| level | target acc | ZCR | ratio vs expected | rollback | swap ok |
|---|---:|---:|---:|---:|---|
| Q0 | 1.00 | 0.9688 | 0.969 | 1.00 | True |
| Q1 | 0.90 | 0.9167 | 1.051 | 1.00 | True |
| Q2 | 0.75 | 0.7917 | 1.090 | 1.00 | True |
| Q3 | 0.55 | 0.6146 | 1.153 | 1.00 | True |
| Q4 | 0.00 | 0.1458 | 145833333.333 | 1.00 | True |
