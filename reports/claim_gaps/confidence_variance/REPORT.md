# Gap: confidence-gap noise vs window size and map family

> generated: 2026-11-27 09:22:28 UTC
> source: real
> pools: {'fit': 96, 'val': 192, 'train': 1152}  gate: 64  test: 192  resamples: 30

`sd` is the clean window-to-window standard deviation of the signed-gap
change on the frozen gate set. `signal` is the mean shift under the
attack, relative to the clean mean; `snr = |signal| / sd`. A viable map
needs `sd < 0.5pp`, clean ECE <= 0.03, and `snr >= 0.6` (the CUSUM slack is 0.4 sd, so a weaker signal never accumulates).
Each pool is tested at half and full size. `fit` is the production
calibration window; `val` is the whole validation split (using it all
would need a larger labelled pool); `train` is optimistic, because the
experts were fitted on it.

| variant | n | pool | sd(pp) | ECE | signal single(pp) | snr single | signal agree(pp) | snr agree | viable |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| pav4 | 48 | fit | 0.943 | 0.0153 | -0.002 | 0.00 | +0.237 | 0.25 | False |
| pav4 | 96 | fit | 0.888 | 0.0126 | +0.179 | 0.20 | +0.350 | 0.39 | False |
| pav6 | 48 | fit | 1.000 | 0.0168 | +0.056 | 0.06 | +0.341 | 0.34 | False |
| pav6 | 96 | fit | 1.011 | 0.0148 | +0.174 | 0.17 | +0.286 | 0.28 | False |
| pav6_shrink75 | 48 | fit | 1.046 | 0.0195 | +0.014 | 0.01 | +0.179 | 0.17 | False |
| pav6_shrink75 | 96 | fit | 0.682 | 0.0177 | +0.075 | 0.11 | +0.249 | 0.36 | False |
| constant | 48 | fit | 1.491 | 0.0195 | +0.123 | 0.08 | +0.167 | 0.11 | False |
| constant | 96 | fit | 0.529 | 0.0163 | +0.073 | 0.14 | +0.262 | 0.49 | False |
| pav4 | 96 | val | 0.953 | 0.0145 | +0.072 | 0.08 | +0.319 | 0.34 | False |
| pav4 | 192 | val | 0.565 | 0.0140 | +0.078 | 0.14 | +0.408 | 0.72 | False |
| pav6 | 96 | val | 0.820 | 0.0134 | +0.134 | 0.16 | +0.385 | 0.47 | False |
| pav6 | 192 | val | 0.600 | 0.0109 | +0.098 | 0.16 | +0.425 | 0.71 | False |
| pav6_shrink75 | 96 | val | 0.852 | 0.0146 | +0.022 | 0.03 | +0.428 | 0.50 | False |
| pav6_shrink75 | 192 | val | 0.517 | 0.0159 | +0.061 | 0.12 | +0.297 | 0.57 | False |
| constant | 96 | val | 0.691 | 0.0172 | +0.054 | 0.08 | +0.350 | 0.51 | False |
| constant | 192 | val | 0.656 | 0.0144 | +0.055 | 0.08 | +0.396 | 0.60 | False |
| pav4 | 576 | train | 0.054 | 0.0255 | -0.034 | 0.63 | -0.020 | 0.37 | True |
| pav4 | 1152 | train | 0.022 | 0.0256 | -0.029 | 1.32 | -0.014 | 0.64 | True |
| pav6 | 576 | train | 0.045 | 0.0256 | -0.050 | 1.09 | -0.025 | 0.55 | True |
| pav6 | 1152 | train | 0.026 | 0.0255 | -0.041 | 1.55 | -0.017 | 0.66 | True |
| pav6_shrink75 | 576 | train | 0.036 | 0.0255 | -0.038 | 1.05 | -0.009 | 0.24 | True |
| pav6_shrink75 | 1152 | train | 0.028 | 0.0257 | -0.033 | 1.18 | -0.016 | 0.57 | True |
| constant | 576 | train | 0.034 | 0.0257 | -0.036 | 1.05 | -0.015 | 0.46 | True |
| constant | 1152 | train | 0.019 | 0.0257 | -0.028 | 1.47 | -0.014 | 0.74 | True |

## Verdict

No production-representative pool (fit/val) reaches the noise and SNR targets while keeping clean ECE under the ceiling. The noise is bound by the size of the labelled pool, not by the map family.
For reference, the train pool (optimistic: the experts were fitted on it) reaches sd 0.026pp at n=1152 with clean ECE 0.0255.
