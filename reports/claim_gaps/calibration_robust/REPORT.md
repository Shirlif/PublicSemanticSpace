# Gap: calibration robustness pipeline (Wave 1)

> generated: 2026-11-27 04:02:25 UTC
> source: real
> scenarios: 18

## Pre-registration

- ok: **False**
  - targeted_agree@0.10.detected: expected True, got False
  - targeted_single_class@0.20.detected: expected True, got False

## Scenarios

`ECE` is measured on the scenario's own test set; for drift and label-shift
scenarios that set is transformed the same way as the calibration window, so
an adaptation failure is visible instead of hidden by a clean test set.
`ECE(no adapt)` is the incumbent calibration on the same set: when the two are
equal, the pipeline did not adapt at all.

| scenario | eval | state | detected | accept | ECE(T) | ECE(map) | coverage | set size | ECE(no adapt) | reason |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| clean | clean_test | normal | False | True | 0.0208 | 0.0151 | 0.911 | 0.92 | 0.0208 | ok |
| random_label_flip@0.05 | clean_test | normal | True | True | 0.0208 | 0.0139 | 0.938 | 0.95 | 0.0208 | ok |
| random_label_flip@0.10 | clean_test | normal | True | True | 0.0208 | 0.0109 | 1.000 | 1.24 | 0.0208 | ok |
| random_label_flip@0.20 | clean_test | quarantine | True | False | 0.0208 | 0.0151 | 0.911 | 0.92 | 0.0208 | frozen by quarantine |
| random_label_flip@0.40 | clean_test | quarantine | True | False | 0.0208 | 0.0151 | 0.911 | 0.92 | 0.0208 | frozen by quarantine |
| embedding_replace@0.10 | clean_test | normal | True | True | 0.0208 | 0.0116 | 1.000 | 1.22 | 0.0208 | ok |
| embedding_replace@0.20 | clean_test | quarantine | True | False | 0.0208 | 0.0151 | 0.911 | 0.92 | 0.0208 | frozen by quarantine |
| embedding_replace@0.40 | clean_test | quarantine | True | False | 0.0208 | 0.0151 | 0.911 | 0.92 | 0.0208 | frozen by quarantine |
| duplicate_inject@0.20 | clean_test | normal | True | True | 0.0209 | 0.0201 | 0.922 | 0.93 | 0.0208 | ok |
| duplicate_inject@0.40 | clean_test | normal | True | True | 0.0208 | 0.0075 | 0.922 | 0.93 | 0.0208 | ok |
| targeted_agree@0.10 | clean_test | normal | False | True | 0.0208 | 0.0169 | 0.911 | 0.92 | 0.0208 | ok |
| targeted_single_class@0.20 | clean_test | normal | False | True | 0.0201 | 0.0257 | 0.917 | 0.92 | 0.0208 | ok |
| source_outlier@0.30 | clean_test | quarantine | True | False | 0.0208 | 0.0151 | 0.911 | 0.92 | 0.0208 | frozen by quarantine |
| drift_sigma@0.10 | drifted_test | recalibrate | True | True | 0.0559 | 0.0007 | 0.917 | 0.92 | 0.0558 | ok |
| drift_sigma@0.20 | drifted_test | recalibrate | True | True | 0.1198 | 0.0349 | 0.891 | 0.94 | 0.1186 | ok |
| drift_sigma@0.30 | drifted_test | recalibrate | True | False | 0.1500 | 0.0976 | 0.906 | 1.16 | 0.1500 | no adaptation gain |
| ood_unseen | clean_test | quarantine | True | False | 0.0208 | 0.0151 | 0.911 | 0.92 | 0.0208 | frozen by quarantine |
| label_shift@4of16 | label_shift_test | recalibrate | True | True | 0.0307 | 0.0292 | 0.979 | 1.00 | 0.0318 | ok |

## How to read

- `detected` = the audit flagged mass above the clean floor, or the state is not NORMAL.
- `accept_update` is the gate decision. Low poison fractions may accept because the
  weighted fit already neutralised the rows; the guarantee there is `ECE` staying flat.
- `signed gap = mean_confidence - accuracy`: negative is underconfident.
