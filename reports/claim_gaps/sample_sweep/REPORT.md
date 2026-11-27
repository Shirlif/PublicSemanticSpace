# Gap: sample-size sweep

> generated: 2026-11-27 03:56:32 UTC
> source: real
> classes: 16  pool: 128  seeds: 5  draws: 40

CUSUM slack is 0.4 within-window sd, so `snr < 0.4` means the streaming
detector cannot separate the attack from window noise at that size.

| window | per class | ECE(map) | flip10 detect | flip20 detect | agreement sd | agree snr | single snr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 1.5 | 0.0418+/-0.0433 | 1.00 | 1.00 | 0.0395 | 0.26 | 0.16 |
| 48 | 3.0 | 0.0193+/-0.0177 | 0.80 | 1.00 | 0.0231 | 0.88 | 0.38 |
| 96 | 6.0 | 0.0180+/-0.0041 | 0.40 | 1.00 | 0.0119 | 1.60 | 0.74 |
| 128 | 8.0 | 0.0087+/-0.0062 | 0.40 | 1.00 | 0.0170 | 1.05 | 0.63 |
