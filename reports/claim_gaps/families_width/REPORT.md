# Claim-gap F4: heterogeneous-width expert adapter

> data: `data/category_pretrain`
> base experts: `checkpoints/expert_v1`; WT-MM: `checkpoints/tactile_wtmm`
> prototypes: `checkpoints/prototypes_v1`; device: `cuda`
> ks: [0, 1, 5, 20]; seed: 0; head epochs: 80

## Widths

| expert | modality | native width | head-fit | calib/class | test |
|---|---:|---:|---:|---:|---:|
| vision_dinov2_384 | vision | 384 | 180 | 36 | 60 |
| vision_clip_768 | vision | 768 | 180 | 36 | 60 |
| tactile_wtmm_64 | tactile | 64 | 180 | 36 | 60 |

## Calibration curves: ridge into the frozen public space

Same protocol as E11: fit a linear map from the expert's projected features
to the frozen prototype vectors with k examples per class.

| expert | width | k=0 | k=1 | k=5 | k=20 |
|---|---:|---:|---:|---:|---:|
| vision_dinov2_384 | 384 | 0.167 | 0.983 | 0.983 | 0.983 |
| vision_clip_768 | 768 | 0.000 | 1.000 | 1.000 | 1.000 |
| tactile_wtmm_64 | 64 | 0.200 | 0.383 | 0.333 | 0.433 |

## Few-shot prototype curves: independent head + calibration set

k>=1 builds prototypes from k calibration examples per class in the
expert's own projected space; k=0 is the frozen-bank zero-shot reference.

| expert | width | k=0 | k=1 | k=5 | k=20 |
|---|---:|---:|---:|---:|---:|
| vision_dinov2_384 | 384 | 0.167 | 0.983 | 0.983 | 0.983 |
| vision_clip_768 | 768 | 0.000 | 1.000 | 1.000 | 1.000 |
| tactile_wtmm_64 | 64 | 0.200 | 0.450 | 0.533 | 0.617 |

## Admission contract across widths

All three probes use the expert's held-out samples and the contract's own
data-driven threshold (`calibrated_thresholds`). Expected: near-duplicate
= reject, novel class = accept, insufficient = review.

| expert | width | near-duplicate | novel class | insufficient | near dist | novel dist | threshold |
|---|---:|---|---|---|---:|---:|---:|
| vision_dinov2_384 | 384 | reject | accept | review | 0.001 | 0.755 | 0.329 |
| vision_clip_768 | 768 | reject | accept | review | 0.001 | 0.775 | 0.360 |
| tactile_wtmm_64 | 64 | reject | accept | review | 0.003 | 0.023 | 0.010 |

## Verdict

- Required widths: [64, 384, 768]
- Vision k=1 >= 0.90: **yes** (vision_dinov2_384=0.983, vision_clip_768=1.000)
- Admission contract consistent across required widths: **yes**

| expert | near-duplicate | novel class | insufficient |
|---|---|---|---|
| vision_dinov2_384 | reject | accept | review |
| vision_clip_768 | reject | accept | review |
| tactile_wtmm_64 | reject | accept | review |

## How to read this

- The projection head is trained on one disjoint subset of each expert's
  train split; the calibration set is the other subset.  No test sample
  enters either.
- `k=0` uses the frozen base codebook with the new head, so a low value is
  expected: an independently trained head does not land in the shared
  public space without calibration.  `k>=1` is the calibrated condition.
- The admission bank is built in the expert's own projected space from the
  full train split; the probes use val+test samples of the near class
  (`block`, in the bank) and a far class (`plate`, removed from the bank).
- WT-MM is a domain bridge: a real-GelSight CNN applied to upsampled 8x8
  synthetic tactile.  Its curves are expected to stay well below vision.
- `vision_dinov2_384` uses `facebook/dinov2-small`; `vision_clip_768` uses
  `openai/clip-vit-base-patch16` (768-d), not the registry default ViT-L/14.
  Both must be present in the local HF cache; pass `--allow-download` to
  fetch a missing snapshot.
- Runtime code is untouched; only frozen features and offline heads are
  used.

## Reproduction

```bash
python -m pss.eval.scripts.run_gap_families_width \
  --data data/category_pretrain \
  --base-experts checkpoints/expert_v1 \
  --wtmm checkpoints/tactile_wtmm \
  --prototypes checkpoints/prototypes_v1 \
  --ks 0,1,5,20 \
  --out reports/claim_gaps/families_width
```
