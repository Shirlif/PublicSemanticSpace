# Claim-gap F6: online calibration + poisoning negative control

> generated: 2026-11-26 07:45:02 UTC
> data: `D:/PublicSemanticSpace/data/category_pretrain`
> experts: `D:/PublicSemanticSpace/checkpoints/category_pretrain_experts`; projections: `D:/PublicSemanticSpace/checkpoints/category_pretrain_projection`
> device: `cuda`; modalities: vision, tactile, geometry
> classes: 16; train/val/test/unseen rows: 1152/192/192/384
> seeds: [0, 1, 2]; drift warmup: 32; drift window: 64

The available bundle `obs_detections.parquet` embeddings are 384-d, but
every emb-bearing bundle is single-class, so they cannot support
multi-class temperature/reliability calibration. The real source here is
the existing expert zoo on `category_pretrain`.

## One-shot reference

The existing `calibrate_fusion` on the full validation split. This is
the reference the streaming calibrator must recover.

| modality | T | reliability | val acc | test acc | test ECE | test NLL |
|---|---:|---:|---:|---:|---:|---:|
| vision | 0.068 | 0.950 | 0.953 | 0.969 | 0.028 | 0.074 |
| tactile | 0.175 | 0.239 | 0.286 | 0.203 | 0.062 | 2.204 |
| geometry | 0.050 | 0.889 | 0.896 | 0.870 | 0.023 | 0.374 |
| fused | — | — | — | 0.964 | 0.045 | 0.110 |

## Streaming calibration on a clean stream

Full val split streamed one sample at a time (window 192).
Feeding the whole split in one batch reproduces the one-shot record; this
run checks that the stateful path does not lose calibration quality.

| modality | T | reliability | test acc | test ECE | test NLL |
|---|---:|---:|---:|---:|---:|
| vision | 0.068 | 0.950 | 0.969 | 0.028 | 0.074 |
| tactile | 0.175 | 0.239 | 0.203 | 0.062 | 2.204 |
| geometry | 0.050 | 0.889 | 0.870 | 0.023 | 0.374 |
| fused | — | — | 0.964 | 0.045 | 0.110 |

### Rolling metrics (clean stream)

Checkpoints from the stateful run; `T` and `rel` are the current
temperature and reliability, `acc`/`ECE` are measured on the rolling
window. The drift score stays at the clean level throughout.

| n | score | action | vision acc | vision ECE | vision T | vision rel | geometry acc | geometry ECE | geometry T | geometry rel |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 33 | 0.293 | continue | 0.879 | 0.086 | 0.059 | 0.871 | 0.879 | 0.102 | 0.050 | 0.871 |
| 65 | 0.477 | continue | 0.938 | 0.045 | 0.050 | 0.934 | 0.862 | 0.070 | 0.050 | 0.852 |
| 97 | 0.430 | continue | 0.928 | 0.041 | 0.074 | 0.923 | 0.866 | 0.064 | 0.050 | 0.857 |
| 129 | 0.359 | continue | 0.930 | 0.038 | 0.077 | 0.926 | 0.884 | 0.052 | 0.050 | 0.876 |
| 161 | 0.354 | continue | 0.944 | 0.021 | 0.072 | 0.940 | 0.882 | 0.050 | 0.050 | 0.874 |
| 192 | 0.611 | continue | 0.953 | 0.020 | 0.068 | 0.950 | 0.896 | 0.048 | 0.050 | 0.889 |

## Drift detection

The drift score is the maximum of four normalised deviations from the
clean reference: rolling accuracy drop, ECE increase, reliability drop
and class-coverage drop. Thresholds are the 99th percentile of clean
stream deviations, floored at the values below.

| signal | threshold |
|---|---:|
| accuracy drop | 0.172 |
| ECE increase | 0.100 |
| reliability drop | 0.183 |
| class-coverage drop | 0.500 |
| score (drift / abstain) | 1.000 / 2.000 |

Threshold source: 0.990 quantile of 480 clean stream steps, floored at accuracy 0.100, ECE 0.100, reliability 0.100, coverage 0.500.

Clean-stream false-positive rate: **0.033** (max score 1.000); held-out clean seeds: **0.008** (max score 1.000).

### Covariate drift (Gaussian noise on real embeddings)

| noise σ | detection | abstain | latency (samples) | recalibrations | max score | acc drop | ECE increase | adapted test ECE | clean test ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.000 | 0.000 | n/a | 0.0 | 0.818 | 0.141 | 0.049 | 0.036 | 0.030 |
| 0.10 | 0.667 | 0.000 | 109.0 | 1.0 | 0.973 | 0.161 | 0.081 | 0.025 | 0.030 |
| 0.20 | 1.000 | 0.000 | 36.7 | 1.7 | 1.051 | 0.172 | 0.099 | 0.025 | 0.030 |
| 0.30 | 1.000 | 0.000 | 23.7 | 3.3 | 1.069 | 0.182 | 0.105 | 0.022 | 0.030 |
| 0.50 | 1.000 | 0.000 | 12.7 | 4.7 | 1.139 | 0.179 | 0.112 | 0.031 | 0.030 |

### Rolling metrics under covariate drift (σ=0.30)

Same stateful signals as the clean table, but the stream is perturbed.
The first non-`continue` action is the trigger point.

| k | score | action | vision acc | vision ECE | vision T | vision rel | geometry acc | geometry ECE | geometry T | geometry rel |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.190 | continue | 0.879 | 0.081 | 0.055 | 0.871 | 0.879 | 0.095 | 0.050 | 0.871 |
| 32 | 0.584 | continue | 0.859 | 0.102 | 0.050 | 0.850 | 0.828 | 0.140 | 0.050 | 0.817 |
| **43** | 1.010 | **recalibrate** | 0.812 | 0.120 | 0.050 | 0.800 | 0.812 | 0.183 | 0.050 | 0.800 |
| 64 | 0.636 | continue | 0.734 | 0.131 | 0.050 | 0.717 | 0.703 | 0.220 | 0.050 | 0.683 |
| 96 | 0.273 | continue | 0.719 | 0.103 | 0.050 | 0.700 | 0.594 | 0.155 | 0.050 | 0.567 |
| 128 | 0.909 | continue | 0.812 | 0.211 | 0.050 | 0.800 | 0.516 | 0.120 | 0.050 | 0.483 |
| 159 | 0.727 | continue | 0.703 | 0.208 | 0.050 | 0.683 | 0.578 | 0.125 | 0.050 | 0.550 |

### Out-of-distribution and label-shift streams

| stream | detection | abstain | recalibrate | latency | coverage drop | max score |
|---|---:|---:|---:|---:|---:|---:|
| unseen categories (OOD) | 1.000 | 1.000 | — | 6.3 | 0.413 | 1.086 |
| 4 of 16 classes (label shift) | 1.000 | — | 1.000 | 40.0 | 0.500 | 1.102 |

A drift flag triggers `recalibrate()` (refit on the current window and
accept it as the new reference). A score at the abstain threshold, or
accuracy below half of the reference, triggers `abstain`: the calibration
is not accepted and the query is treated as unknown.

## Poisoned calibration negative control

Poisoning is applied to the real val calibration set. The audit compares
label mismatch, margin violation, near-duplicate and centroid-outlier
rates against a clean baseline; this is why a weak modality such as
tactile is not flagged merely for being weak.

Audit thresholds (excess over clean baseline): label mismatch 0.050, margin violation 0.050, duplicate 0.050, outlier 0.050; abstain at flag rate 0.300 or label mismatch 0.300.

### Detection and abstention

| poison | fraction | detection | abstain | recalibrate | flagged fraction |
|---|---:|---:|---:|---:|---:|
| label_flip | 0.05 | 0.667 | 0.000 | 0.667 | 0.226 |
| label_flip | 0.10 | 1.000 | 0.000 | 1.000 | 0.226 |
| label_flip | 0.20 | 1.000 | 0.000 | 1.000 | 0.309 |
| label_flip | 0.40 | 1.000 | 1.000 | 0.000 | 0.476 |
| label_flip | 0.60 | 1.000 | 1.000 | 0.000 | 0.651 |
| embedding_replace | 0.05 | 1.000 | 0.000 | 1.000 | 0.193 |
| embedding_replace | 0.10 | 1.000 | 0.000 | 1.000 | 0.226 |
| embedding_replace | 0.20 | 1.000 | 0.000 | 1.000 | 0.309 |
| embedding_replace | 0.40 | 1.000 | 1.000 | 0.000 | 0.467 |
| embedding_replace | 0.60 | 1.000 | 1.000 | 0.000 | 0.646 |
| duplicate_inject | 0.05 | 1.000 | 0.000 | 1.000 | 0.287 |
| duplicate_inject | 0.10 | 1.000 | 0.000 | 1.000 | 0.355 |
| duplicate_inject | 0.20 | 1.000 | 0.333 | 0.667 | 0.470 |
| duplicate_inject | 0.40 | 1.000 | 1.000 | 0.000 | 0.672 |
| duplicate_inject | 0.60 | 1.000 | 1.000 | 0.000 | 0.793 |

`flagged fraction` is the fraction dropped by the reason-specific audit
mask. It includes the clean baseline error tail (vision/geometry
misclassifications and near-duplicate views), so it is not the poison
rate.

### Degradation when the audit is bypassed

These are the test metrics when the poisoned set is used to fit the
one-shot calibration anyway. This is the curve an undetected poison would
produce. `robust` drops the flagged samples and refits.

**label_flip**

| fraction | fused acc | fused ECE | fused NLL | vision NLL | robust fused NLL | robust kept |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.964 | 0.045 | 0.110 | 0.074 | 0.110 | 192 |
| 0.05 | 0.969 | 0.215 | 0.294 | 0.200 | 0.085 | 149 |
| 0.10 | 0.974 | 0.325 | 0.458 | 0.312 | 0.086 | 149 |
| 0.20 | 0.974 | 0.456 | 0.681 | 0.490 | 0.086 | 133 |
| 0.40 | 0.974 | 0.648 | 1.138 | 0.906 | 0.086 | 101 |
| 0.60 | 0.974 | 0.788 | 1.690 | 1.446 | 0.086 | 67 |

**embedding_replace**

| fraction | fused acc | fused ECE | fused NLL | vision NLL | robust fused NLL | robust kept |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.964 | 0.045 | 0.110 | 0.074 | 0.110 | 192 |
| 0.05 | 0.969 | 0.207 | 0.288 | 0.193 | 0.096 | 155 |
| 0.10 | 0.974 | 0.304 | 0.426 | 0.291 | 0.086 | 149 |
| 0.20 | 0.974 | 0.473 | 0.715 | 0.519 | 0.086 | 133 |
| 0.40 | 0.974 | 0.659 | 1.169 | 0.919 | 0.086 | 102 |
| 0.60 | 0.974 | 0.779 | 1.644 | 1.424 | 0.086 | 68 |

**duplicate_inject**

| fraction | fused acc | fused ECE | fused NLL | vision NLL | robust fused NLL | robust kept |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.964 | 0.045 | 0.110 | 0.074 | 0.110 | 192 |
| 0.05 | 0.965 | 0.045 | 0.109 | 0.073 | 0.117 | 144 |
| 0.10 | 0.965 | 0.046 | 0.110 | 0.073 | 0.117 | 136 |
| 0.20 | 0.969 | 0.046 | 0.108 | 0.073 | 0.118 | 122 |
| 0.40 | 0.965 | 0.046 | 0.111 | 0.074 | 0.116 | 88 |
| 0.60 | 0.964 | 0.047 | 0.115 | 0.076 | 0.094 | 64 |

## Cost-performance Pareto

The streaming calibrator is fed `N` class-balanced samples from the val
split. `recovery` is one-shot test metric / N-sample test metric (NLL and
ECE: higher is better; accuracy: N-sample / one-shot). `fit ms` is the
cumulative calibration-update time for N samples; feature extraction and
report generation are excluded.

| N | samples/class | fit ms | fused acc | fused ECE | fused NLL | vision NLL | geometry NLL | tactile NLL | min NLL recovery | ECE recovery |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 1.9 | 0.970 | 0.037 | 0.089 | 0.067 | 0.374 | 2.828 | 0.786 | 0.259 |
| 2 | 1.0 | 3.3 | 0.972 | 0.033 | 0.083 | 0.067 | 0.387 | 2.753 | 0.804 | 0.399 |
| 4 | 1.0 | 6.1 | 0.965 | 0.050 | 0.128 | 0.108 | 0.374 | 2.490 | 0.865 | 0.572 |
| 8 | 1.0 | 14.5 | 0.974 | 0.028 | 0.083 | 0.067 | 0.374 | 2.265 | 0.973 | 0.999 |
| 16 | 1.0 | 27.1 | 0.972 | 0.049 | 0.104 | 0.076 | 0.374 | 2.444 | 0.913 | 0.606 |
| 32 | 2.0 | 61.6 | 0.974 | 0.030 | 0.087 | 0.067 | 0.374 | 2.208 | 0.998 | 1.000 |
| 64 | 4.0 | 128.5 | 0.964 | 0.053 | 0.118 | 0.080 | 0.374 | 2.217 | 0.928 | 1.000 |
| 96 | 6.0 | 201.6 | 0.964 | 0.047 | 0.113 | 0.075 | 0.374 | 2.210 | 0.978 | 0.941 |
| 192 | 12.0 | 551.1 | 0.964 | 0.045 | 0.110 | 0.074 | 0.374 | 2.204 | 1.000 | 1.000 |

The one-shot fused ECE is worse than the vision-only ECE (see the
one-shot table): the existing fusion gives weight to the weak tactile
expert. A small streaming estimate can therefore have a recovery ratio
above 1.0 (it is better on test than the full-set one-shot), which is why
the accuracy criterion saturates at N=1 and the NLL criterion is the
meaningful one.


At N=8 the per-seed minimum all-modality NLL recovery is **0.960**, so the 90% target is met in every seed, not only on average.

Minimal N for a 90% recovery of the one-shot record:

- all modalities, test NLL: **8**
- fused test NLL: **1**
- all modalities, ECE: **8**
- all modalities, accuracy: **1**

The NLL criterion is the strict one: accuracy saturates quickly because
temperature scaling is monotone within a modality, while NLL and ECE
measure whether the fitted temperature/reliability are actually right.

## Synthetic Gaussian supplement

> gaussian(seed=0, classes=8, dim=32); counts: {'train': 320, 'val': 160, 'test': 160, 'unseen': 40, 'classes': 8}

| condition | detection | action | latency | score |
|---|---:|---|---:|---:|
| noise σ=0.5 | True | recalibrate | 10.0 | 1.270 |
| noise σ=1.0 | True | recalibrate | 4.0 | 1.293 |
| noise σ=2.0 | True | recalibrate | 1.0 | 1.328 |
| label_flip p=0.30 | True | recalibrate | — | — |
| embedding_replace p=0.30 | True | abstain | — | — |
| duplicate_inject p=0.30 | True | abstain | — | — |

The synthetic context is a controlled sanity check, not the primary
evidence; the real expert-zoo stream above is the claim-bearing one.

## Verdict

- **Recovery**: N=**8** class-balanced samples recover ≥90% of the
  one-shot test NLL on all modalities (at N=8: vision=1.105, tactile=0.973, geometry=1.000; fused
  recovery 1.330; per-seed minimum 0.960).
  Accuracy recovers at N=1; ECE at N=8.
- **Poison detection at p=0.05**: label flip 0.667, embedding replacement 1.000, duplicate injection 1.000. At p=0.10 all three are 1.000.
- **Abstention at p=0.40**: label flip 1.000, embedding replacement 1.000, duplicate injection 1.000.
- **Drift**: OOD detection 1.000 / abstain 1.000 (latency 6.3); label shift 1.000 (latency 40.0); noise detection σ=0.05 0.000, σ=0.10 0.667, σ=0.20 1.000, σ=0.50 1.000; clean FPR 0.033 (held-out 0.008).

## Failure cases and reading notes

- **Tactile is the Pareto bottleneck in this run.** The tactile expert
  is weak (val accuracy near 0.29); its temperature is noisy with very
  few samples and dominates the minimal-N requirement. Vision and
  geometry recover with one or two samples.
- **Five-percent label flips can be missed.** Detection at p=0.05 is
  0.667 across seeds. When missed, the bypass row
  shows fused NLL rising from 0.110 to 0.294 and ECE from 0.045 to 0.215. Embedding
  replacement and duplicate injection are detected at p=0.05 in every
  seed.
- **Duplicate injection is the easiest to detect and the least damaging
  when missed.** Exact copies raise the duplicate rate sharply, but a
  random duplicate does not move the class-conditional accuracy much;
  the bypass NLL stays near the clean value.
- **Label shift is detected by class coverage, not accuracy.** Streaming
  a subset of classes keeps accuracy high; the coverage signal is what
  fires, and recalibration adapts the reference to the narrower stream.
- **Covariate noise degrades accuracy, but recalibration still adapts
  the scale.** The adapted ECE is 0.025 at σ=0.10 and 0.022
  at σ=0.30 (clean streaming reference 0.030), while
  accuracy keeps falling.
  At σ=0.05 the drift is below threshold in every seed (false negative).
- **Drift thresholds are data-driven and floored.** The clean stream
  raised the accuracy (0.172) and reliability
  (0.183) thresholds above the 0.10 floors;
  ECE (0.100) and coverage
  (0.500) stayed at the floors. Clean FPR is
  0.033 on the derivation seeds and
  0.008 on held-out clean
  seeds.

## Reproduction

```bash
python -m pss.eval.scripts.run_gap_online_calibration \
  --data D:/PublicSemanticSpace/data/category_pretrain \
  --experts D:/PublicSemanticSpace/checkpoints/category_pretrain_experts \
  --projections D:/PublicSemanticSpace/checkpoints/category_pretrain_projection \
  --seeds 0,1,2 \
  --out reports/claim_gaps/online_calibration
```
