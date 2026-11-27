# Gap: sample-size sweep

> generated: 2026-11-27 03:56:17 UTC
> source: synthetic
> classes: 8  pool: 112  seeds: 5  draws: 40

CUSUM slack is 0.4 within-window sd, so `snr < 0.4` means the streaming
detector cannot separate the attack from window noise at that size.

| window | per class | ECE(map) | flip10 detect | flip20 detect | agreement sd | agree snr | single snr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 3.0 | 0.0259+/-0.0212 | 0.60 | 1.00 | 0.0388 | 0.94 | 0.70 |
| 48 | 6.0 | 0.0206+/-0.0108 | 0.60 | 1.00 | 0.0192 | 2.20 | 1.39 |
| 96 | 12.0 | 0.0150+/-0.0055 | 0.40 | 1.00 | 0.0097 | 4.58 | 2.60 |
| 112 | 14.0 | 0.0073+/-0.0043 | 0.00 | 1.00 | 0.0176 | 2.75 | 1.51 |
