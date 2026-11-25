# P2: open-set rejection

- data: `data/category_pretrain`
- seen train / val / test: 1152 / 192 / 192
- unseen test: 384

| method | AUROC | unseen reject @ 95% seen accept | seen top-1 |
|---|---:|---:|---:|
| PSS prototypes | 0.993 | 0.958 | 0.979 |
| Transformer softmax | 0.885 ± 0.028 | 0.514 | 0.976 |
| Transformer energy | 0.937 ± 0.021 | 0.753 | 0.976 |

Both methods calibrate their threshold on seen val to 95% acceptance.
