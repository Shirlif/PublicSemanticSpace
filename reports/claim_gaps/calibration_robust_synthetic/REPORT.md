# Gap: calibration robustness pipeline (Wave 1)

> generated: 2026-11-26 17:15:25 UTC
> source: synthetic
> scenarios: 18

## Pre-registration

- ok: **False**
  - targeted_agree@0.10.detected: expected True, got False
  - targeted_agree@0.10.accept_update: expected False, got True

## Scenarios

`ECE` is measured on the scenario's own test set; for drift and label-shift
scenarios that set is transformed the same way as the calibration window, so
an adaptation failure is visible instead of hidden by a clean test set.
`ECE(no adapt)` is the incumbent calibration on the same set: when the two are
equal, the pipeline did not adapt at all.

| scenario | eval | state | detected | accept | suspect | cross | label | drift | ECE | ECE(clean) | ECE(no adapt) | reason |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| clean | clean_test | normal | False | True | 0.037 | 1.000 | 0.963 | 0.022 | 0.0158 | 0.0158 | 0.0158 | ok |
| random_label_flip@0.05 | clean_test | normal | True | True | 0.088 | 1.000 | 0.912 | 0.022 | 0.0158 | 0.0158 | 0.0158 | ok |
| random_label_flip@0.10 | clean_test | normal | True | True | 0.137 | 1.000 | 0.863 | 0.022 | 0.0158 | 0.0158 | 0.0158 | ok |
| random_label_flip@0.20 | clean_test | quarantine | True | False | 0.238 | 1.000 | 0.762 | 0.022 | 0.0158 | 0.0158 | 0.0158 | frozen by quarantine |
| random_label_flip@0.40 | clean_test | quarantine | True | False | 0.425 | 1.000 | 0.575 | 0.022 | 0.0158 | 0.0158 | 0.0158 | frozen by quarantine |
| embedding_replace@0.10 | clean_test | normal | True | True | 0.125 | 1.000 | 0.875 | 0.022 | 0.0158 | 0.0158 | 0.0158 | ok |
| embedding_replace@0.20 | clean_test | quarantine | True | False | 0.225 | 1.000 | 0.775 | 0.007 | 0.0158 | 0.0158 | 0.0158 | frozen by quarantine |
| embedding_replace@0.40 | clean_test | quarantine | True | False | 0.363 | 1.000 | 0.637 | 0.024 | 0.0158 | 0.0158 | 0.0158 | frozen by quarantine |
| duplicate_inject@0.20 | clean_test | recalibrate | True | True | 0.042 | 1.000 | 0.958 | 0.049 | 0.0158 | 0.0158 | 0.0158 | ok |
| duplicate_inject@0.40 | clean_test | normal | True | True | 0.049 | 1.000 | 0.955 | 0.017 | 0.0158 | 0.0158 | 0.0158 | ok |
| targeted_agree@0.10 | clean_test | normal | False | True | 0.000 | 1.000 | 1.000 | 0.022 | 0.0158 | 0.0158 | 0.0158 | ok |
| targeted_single_class@0.20 | clean_test | quarantine | True | False | 0.062 | 1.000 | 0.938 | 0.022 | 0.0158 | 0.0158 | 0.0158 | frozen by quarantine |
| source_outlier@0.30 | clean_test | quarantine | True | False | 0.381 | 1.000 | 0.625 | 0.022 | 0.0158 | 0.0158 | 0.0158 | frozen by quarantine |
| drift_sigma@0.10 | drifted_test | recalibrate | True | True | 0.025 | 1.000 | 0.975 | 0.049 | 0.0288 | 0.0158 | 0.0288 | ok |
| drift_sigma@0.20 | drifted_test | recalibrate | True | True | 0.069 | 1.000 | 0.950 | 0.100 | 0.0339 | 0.0158 | 0.0339 | ok |
| drift_sigma@0.30 | drifted_test | recalibrate | True | True | 0.244 | 1.000 | 0.762 | 0.187 | 0.0660 | 0.0158 | 0.0660 | ok |
| ood_unseen | clean_test | quarantine | True | False | 0.000 | 1.000 | 0.000 | 1.000 | 0.0158 | 0.0158 | 0.0158 | frozen by quarantine |
| label_shift@4of16 | label_shift_test | recalibrate | True | True | 0.050 | 1.000 | 0.950 | 0.500 | 0.0249 | 0.0158 | 0.0249 | ok |

## How to read

- `detected` = the audit flagged mass above the clean floor, or the state is not NORMAL.
- `accept_update` is the gate decision. Low poison fractions may accept because the
  weighted fit already neutralised the rows; the guarantee there is `ECE` staying flat.
- `signed gap = mean_confidence - accuracy`: negative is underconfident.
