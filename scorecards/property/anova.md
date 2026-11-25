# Two-way variance decomposition

- design: 5 quality x 3 architecture x 3 seeds
- A4 = cross-family zero-shot; A5 = oracle/retrain upper bound

| metric | SS(expert) | SS(arch) | SS(interaction) | R² | Var share (expert/arch/interaction/error) |
|---|---:|---:|---:|---:|---|
| SSR | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 / 0.000 / 0.000 / 0.000 |
| ZCR | 0.4255 | 4.8444 | 0.8511 | 0.9988 | 0.056 / 0.773 / 0.169 / 0.001 |
| INT | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 / 0.000 / 0.000 / 0.000 |
| RBE | 0.0000 | 10.0000 | 0.0000 | 1.0000 | 0.000 / 1.000 / 0.000 / 0.000 |

读法：SS/方差份额按 balanced two-way ANOVA 计算；A4/A5 是负控制/上界，不是真实训练运行。
