# Gap: calibration robustness pipeline (Wave 1)

> generated: 2026-11-27 09:50:21 UTC
> source: real
> scenarios: 18

## Pre-registration

- ok: **False**
  - targeted_agree@0.10.detected: expected True, got False
  - targeted_agree@0.10.confidence_map_accept: expected False, got True
  - targeted_single_class@0.20.detected: expected True, got False
  - targeted_single_class@0.20.confidence_map_accept: expected False, got True
  - drift_sigma@0.30.recovery_samples: expected <= 25, got 28.8000
  - label_shift@4of16.recovery_samples: expected <= 40, got 85.0000

## State thresholds (calibrated on the clean stream)

| threshold | value |
|---|---:|
| `abstain_release_windows` | 3 |
| `gate_fail_abstain` | 3 |
| `quarantine_recover_windows` | 3 |
| `theta_cross` | 0.8 |
| `theta_drift` | 0.03 |
| `theta_global_ok` | 0.9060416666666667 |
| `theta_inflation` | 0.15 |
| `theta_label` | 0.85 |
| `theta_spread` | 0.18 |

## Clean-stream false alarms

- windows: **100** (gate forced on every window)
- gate rejection rate: **0.080** (target < 0.05 -> FAIL)
- state alarm rate: **0.030**
- reasons: {'no adaptation gain': 6, 'frozen by quarantine': 2}

## Scenarios

`ECE` is measured on the scenario's own test set; for drift and label-shift
scenarios that set is transformed the same way as the calibration window, so
an adaptation failure is visible instead of hidden by a clean test set.
`ECE(no adapt)` is the incumbent calibration on the same set: when the two are
equal, the pipeline did not adapt at all.

| scenario | eval | state | detected | accept | ECE(T) | ECE(map) | coverage | set size | ECE(no adapt) | recovery | reason |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| clean | clean_test | normal | False | True | 0.0208 | 0.0151 | 0.911 | 0.92 | 0.0208 | — | ok |
| random_label_flip@0.05 | clean_test | normal | True | True | 0.0208 | 0.0139 | 0.938 | 0.95 | 0.0208 | — | ok |
| random_label_flip@0.10 | clean_test | normal | True | True | 0.0208 | 0.0109 | 1.000 | 1.24 | 0.0208 | — | ok |
| random_label_flip@0.20 | clean_test | quarantine | True | False | 0.0208 | 0.0151 | 0.911 | 0.92 | 0.0208 | — | frozen by quarantine |
| random_label_flip@0.40 | clean_test | quarantine | True | False | 0.0208 | 0.0151 | 0.911 | 0.92 | 0.0208 | — | frozen by quarantine |
| embedding_replace@0.10 | clean_test | normal | True | True | 0.0208 | 0.0116 | 1.000 | 1.22 | 0.0208 | — | ok |
| embedding_replace@0.20 | clean_test | quarantine | True | False | 0.0208 | 0.0151 | 0.911 | 0.92 | 0.0208 | — | frozen by quarantine |
| embedding_replace@0.40 | clean_test | quarantine | True | False | 0.0208 | 0.0151 | 0.911 | 0.92 | 0.0208 | — | frozen by quarantine |
| duplicate_inject@0.20 | clean_test | normal | True | True | 0.0209 | 0.0201 | 0.922 | 0.93 | 0.0208 | — | ok |
| duplicate_inject@0.40 | clean_test | normal | True | True | 0.0208 | 0.0075 | 0.922 | 0.93 | 0.0208 | — | ok |
| targeted_agree@0.10 | clean_test | normal | False | True | 0.0208 | 0.0169 | 0.911 | 0.92 | 0.0208 | — | ok |
| targeted_single_class@0.20 | clean_test | normal | False | True | 0.0201 | 0.0257 | 0.917 | 0.92 | 0.0208 | — | ok |
| source_outlier@0.30 | clean_test | quarantine | True | False | 0.0208 | 0.0151 | 0.911 | 0.92 | 0.0208 | — | frozen by quarantine |
| drift_sigma@0.10 | drifted_test | recalibrate | True | True | 0.0559 | 0.0007 | 0.917 | 0.92 | 0.0558 | 29 | ok |
| drift_sigma@0.20 | drifted_test | recalibrate | True | True | 0.1198 | 0.0349 | 0.891 | 0.94 | 0.1186 | 29 | ok |
| drift_sigma@0.30 | drifted_test | recalibrate | True | False | 0.1500 | 0.0976 | 0.906 | 1.16 | 0.1500 | 29 | no adaptation gain |
| ood_unseen | clean_test | quarantine | True | False | 0.0208 | 0.0151 | 0.911 | 0.92 | 0.0208 | — | frozen by quarantine |
| label_shift@4of16 | label_shift_test | recalibrate | True | True | 0.0307 | 0.0292 | 0.979 | 1.00 | 0.0318 | 85 | ok |

## Confidence-channel gate (P0)

The gate set is the confidence channel's standard answer; the
tolerances are calibrated on clean windows.
`gap_tolerance` = 0.0323,
`ece_tolerance` = 0.0307,
`z` = 2.0.
`gap` is mean confidence minus accuracy; `dGap(gate)` is |gap|
candidate minus incumbent; `z` is that difference over its paired
standard error. `gate ok` = the candidate is within the calibrated
band on the frozen reference; `improve(hold)` = |gap| reduction on the
recent trusted slice; `adapt ok` = that reduction is significant.
`would accept` = keep when `gate ok` OR `adapt ok` (a real drift update
must not be blocked by the stale reference). `map used` shows the
enforced choice; `--report-only-confidence-gate` disables rejection.

| scenario | n | gap(inc) | gap(cand) | dGap(gate) | z | gate ok | improve(hold) | adapt ok | strict | would accept | map used |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| clean | 64 | 0.0222 | 0.0222 | +0.0000 | +0.0 | True | -0.0000 | False | True | True | candidate |
| random_label_flip@0.05 | 64 | 0.0222 | 0.0213 | -0.0010 | -1.0 | True | +0.0014 | False | True | True | candidate |
| random_label_flip@0.10 | 64 | 0.0222 | 0.0188 | -0.0034 | -3.4 | True | +0.0048 | True | True | True | candidate |
| random_label_flip@0.20 | 0 | — | — | — | — | False | — | False | False | False | incumbent |
| random_label_flip@0.40 | 0 | — | — | — | — | False | — | False | False | False | incumbent |
| embedding_replace@0.10 | 64 | 0.0222 | 0.0190 | -0.0032 | -3.2 | True | +0.0027 | True | True | True | candidate |
| embedding_replace@0.20 | 0 | — | — | — | — | False | — | False | False | False | incumbent |
| embedding_replace@0.40 | 0 | — | — | — | — | False | — | False | False | False | incumbent |
| duplicate_inject@0.20 | 64 | 0.0222 | 0.0222 | -0.0000 | -0.0 | True | +0.0002 | False | True | True | candidate |
| duplicate_inject@0.40 | 64 | 0.0222 | 0.0157 | -0.0065 | -4.0 | True | +0.0061 | True | True | True | candidate |
| targeted_agree@0.10 | 64 | 0.0222 | 0.0238 | +0.0016 | +1.6 | True | -0.0022 | False | True | True | candidate |
| targeted_single_class@0.20 | 64 | 0.0222 | 0.0270 | +0.0048 | +4.6 | True | -0.0061 | False | True | True | candidate |
| source_outlier@0.30 | 0 | — | — | — | — | False | — | False | False | False | incumbent |
| drift_sigma@0.10 | 64 | 0.0222 | 0.0061 | -0.0161 | -9.6 | True | -0.0022 | False | True | True | candidate |
| drift_sigma@0.20 | 64 | 0.0220 | 0.0312 | +0.0093 | +5.5 | True | +0.0332 | True | True | True | candidate |
| drift_sigma@0.30 | 64 | 0.0222 | 0.0312 | +0.0090 | +5.4 | True | +0.0685 | True | True | True | candidate |
| ood_unseen | 0 | — | — | — | — | False | — | False | False | False | incumbent |
| label_shift@4of16 | 64 | 0.0285 | -0.0073 | -0.0212 | -5.4 | True | +0.0222 | False | True | True | candidate |

## How to read

- `detected` = the audit flagged mass above the clean floor, or the state is not NORMAL.
- `accept_update` is the gate decision. Low poison fractions may accept because the
  weighted fit already neutralised the rows; the guarantee there is `ECE` staying flat.
- `recovery` is the number of streamed samples after a clean warm-up until the
  state leaves NORMAL (F6-comparable detection latency). `—` means not asserted.
- `signed gap = mean_confidence - accuracy`: negative is underconfident.
- `dGap` in the confidence-channel gate is |mean confidence - accuracy|
  candidate minus incumbent on the same samples: negative means the
  candidate is closer to the truth.
