# Phase 11 — Encoder priority: frozen backbones + calibrated projections

> Date: 2026-11-24
> Scope: `PSS-认知层架构定位.md` §7, "第一优先：换编码器，不动 PSS 接口"
> Data: `data/category_pretrain`, `data/tracking_bench_v2` (synthetic)
> Result artifacts: `eval/results/encoder_priority/`

---

## 1. What was implemented

The PSS interfaces are unchanged: `Observation`, the 128-d public space,
`entity_token`, the token layout, and every cost matrix keep their existing
shapes. Only the frozen encoder that feeds each 128-d projection head changes.

| Component | File | Purpose |
|---|---|---|
| CLIP vision tower | `pss/encoders/vision/clip.py` | frozen `openai/clip-vit-large-patch14`, same `encode()` contract |
| DINOv2 patch tokens | `pss/encoders/vision/dinov2.py` | `pool="cls"|"patch_mean"`, `encode_patches()`, internal batching |
| VLAD aggregation | `pss/encoders/vision/local.py` | patch tokens → fixed-size VLAD; codebook fitted on train only |
| Conv tactile encoder | `pss/encoders/tactile/conv.py` | 8×8/12×32 contact-grid CNN with a residual skip |
| Encoder registry | `pss/encoders/registry.py` | swap backbones by name (`smallcnn`, `dinov2`, `dinov2_vlad`, `clip`, `mlp`, `conv`, `wtmm`) |
| Calibrated projection | `pss/projection/heads.py` | optional PCA-whitening front-end + linear residual skip |
| Reliability / temperature | `pss/projection/calibration.py` | per-modality temperature, reliability weights, calibrated scores |
| Experiment runner | `pss/eval/scripts/run_encoder_priority.py` | trains variants; runs open-set, re-ID, P5b |
| Conv tactile training | `--tactile-arch conv` in `train_experts.py` | writes `tactile_conv.pt` |

`ProjectionHead` stays backward compatible: checkpoints without the new keys
load as plain MLP heads. The new head is opt-in via `whiten_dim`,
`whiten_modalities` and `residual`.

---

## 2. Experiment matrix

All variants use the same 128-d public space and the same downstream code.
The projection ablations keep the existing SmallCNN + tactile MLP and change
only the head, isolating alignment/calibration from the encoder.

| variant | vision encoder | tactile encoder | projection head |
|---|---|---|---|
| `base` | SmallCNN (task-trained) | tactile MLP | existing plain MLP heads |
| `residual_only` | SmallCNN | tactile MLP | linear residual skip, no whitening |
| `whiten_only` | SmallCNN | tactile MLP | PCA-whiten vision/geometry, no residual |
| `calibrated` | SmallCNN | tactile MLP | whiten vision/geometry + residual |
| `dinov2_plain` | frozen DINOv2-small CLS | tactile MLP | uncalibrated plain MLP |
| `dinov2` | frozen DINOv2-small CLS | tactile MLP | whiten vision/geometry + residual |
| `dinov2_vlad` | DINOv2 patch tokens + VLAD (K=16) | tactile MLP | whiten vision/geometry + residual |
| `conv_tactile` | SmallCNN | residual conv tactile | whiten vision/geometry + residual |
| `clip` | frozen CLIP ViT-L/14 | tactile MLP | not run — weights absent from the local cache |

Protocol: projections trained for 120 epochs on the category train split;
re-ID uses `tracking_bench_v2` with the previous Hungarian + assignment-loss
protocol (5 seeds, 200 epochs; VLAD 3 seeds because each run is much slower);
open-set uses the Phase 5 prototype bank; P5b uses the perceived relational
benchmark (2 seeds).

---

## 3. Results

| variant | closed-set test ↑ | open-set AUROC ↑ | calibrated AUROC | re-ID overall ↑ | re-ID near-identical ↑ | P5b PSS ID / OOD | P5b TR ID / OOD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base` | 0.979 | 0.993 | 0.959 | **0.997 ± 0.007** | 0.987 | 1.000 / 1.000 | 1.000 / 0.400 |
| `residual_only` | 0.984 | **0.998** | 0.979 | 0.989 ± 0.012 | 0.956 | — | — |
| `whiten_only` | 0.990 | 0.969 | 0.981 | 0.948 ± 0.029 | 0.791 | — | — |
| `calibrated` | **0.995** | 0.977 | 0.986 | 0.989 ± 0.010 | 0.956 | 1.000 / 1.000 | 1.000 / 0.400 |
| `dinov2_plain` | 0.974 | 0.976 | 0.952 | 0.963 ± 0.035 | 0.880 | — | — |
| `dinov2` | 0.969 | 0.897 | 0.950 | 0.993 ± 0.005 | 0.982 | 1.000 / 1.000 | 1.000 / 0.400 |
| `dinov2_vlad` | 0.964 | 0.841 | 0.966 | **1.000 ± 0.000** (3 seeds) | **1.000** | 1.000 / 1.000 | 1.000 / 0.400 |
| `conv_tactile` | **0.995** | 0.973 | 0.976 | 0.991 ± 0.013 | **1.000** | 1.000 / 1.000 | 1.000 / 0.400 |

"Calibrated AUROC" uses per-modality temperatures and reliability weights
fitted on the validation split. It is a secondary diagnostic, not a
replacement for the bank's open-set score.

Per-modality calibration for the base encoders (validation split):

| modality | temperature | reliability | val accuracy |
|---|---:|---:|---:|
| vision | 0.24 | 0.968 | 0.970 |
| tactile | 1.11 | 0.188 | 0.240 |
| geometry | 1.00 | 0.905 | 0.911 |

---

## 4. Findings

### 4.1 Encoder swap alone does not beat the task-trained encoders

Frozen DINOv2 with an uncalibrated head is close to but below the task-trained
SmallCNN (0.974 closed-set / 0.976 open-set vs 0.979 / 0.993). VLAD is lower
(0.964 / 0.841). No frozen encoder is a free recognition win on this
benchmark, which reproduces the earlier expert-zoo ranking and supports the
architecture document's claim that PSS should not compete with the perception
layer.

### 4.2 The projection head was the real bottleneck for frozen features

The same DINOv2 encoder swings by 3 re-ID points depending only on the
projection head:

| DINOv2 head | re-ID overall | near-identical | closed-set | open-set |
|---|---:|---:|---:|---:|
| plain MLP (`dinov2_plain`) | 0.963 ± 0.035 | 0.880 | 0.974 | 0.976 |
| whiten + residual (`dinov2`) | **0.993 ± 0.005** | **0.982** | 0.969 | 0.897 |

The projection was the bottleneck for *re-ID*, not for *recognition*: the
plain head is actually better on open-set (0.976 vs 0.897). The calibrated
head is a re-ID-specific intervention, not a universal improvement. The plain
head is also noisier across seeds (±0.035 vs ±0.005), so treat the gap as
suggestive rather than a precise effect size. The prior 0.903 DINOv2 figure
came from a different token set (no tactile block) and is not directly
comparable to this table.

### 4.3 No single projection change dominates; the effect is task-dependent

The ablations on the *unchanged* SmallCNN + tactile MLP isolate the two pieces:

| head change | closed-set | open-set | re-ID | near-identical |
|---|---:|---:|---:|---:|
| none (`base`) | 0.979 | 0.993 | **0.997** | 0.987 |
| residual only | 0.984 | **0.998** | 0.989 | 0.956 |
| whitening only | 0.990 | 0.969 | 0.948 | 0.791 |
| whitening + residual | **0.995** | 0.977 | 0.989 | 0.956 |

- **Residual skip is the safe recognition piece**: it improves closed-set
  (+0.5) and open-set (+0.5) with no whitening.
- **Whitening buys closed-set but costs open-set and identity**: whitening
  alone reaches 0.990 closed-set but drops open-set to 0.969 and
  near-identical re-ID to 0.791.
- **Combining both** maximizes closed-set (0.995) and recovers re-ID to the
  residual-only level, but open-set stays below base (0.977 vs 0.993).
- For the already-tuned SmallCNN path, every head change slightly *lowers*
  overall re-ID; the gains are in recognition.

The practical recommendation is task-dependent: keep the existing base
projection for the re-ID path, and use a residual-only (or
whitening+residual) head for recognition when closed-set/open-set matter. A
single head is a compromise.

### 4.4 Conv tactile helps identity, not category recognition

The conv tactile encoder is neutral on category recognition (0.995 closed-set
matches `calibrated`, both driven by the calibrated vision projection) but
gives **1.000 near-identical re-ID** and 0.991 overall, versus 0.987 / 0.997
for the base MLP. The synthetic tactile pattern is already close to linearly
separable; the remaining headroom is identity preservation, not encoder
capacity.

### 4.5 P5b is encoder-invariant by construction

All variants produce **PSS 1.000 ID / 1.000 OOD** and **transformer 1.000 ID /
0.400 OOD**, but this is not evidence that P5b measures the relational layer
against perception. The P5b perception path loads no prototype bank
(`prototype_dir=None`), so the tracker receives GT categories; its
`TrackerConfig` sets `w_category=0.0`; and the planner state is derived from
tracked **positions** mapped to regions. The vision encoder never influences
the P5b state.

This round cannot use P5b as an encoder result. The earlier P5b conclusion
(PSS 1.000 vs transformer OOD 0.400) still stands as a symbolic/relational
result, but making P5b encoder-sensitive requires a goal that depends on
perceived category/identity (for example "grasp the cup"), not just position.

### 4.6 CLIP is implemented but not evaluated

The local Hugging Face cache contains only the CLIP config, not the 1.7 GB
weights. `clip.py` is implemented and wired into the registry; the experiment
skips it explicitly and records the reason. With the weights present,
`run_encoder_priority.py --only clip` runs the same pipeline unchanged.

---

## 5. Boundaries

- All data is synthetic. This is not sim2real evidence.
- Re-ID uses one benchmark and 5 seeds (VLAD: 3). The ± values are seed
  noise, not confidence intervals over datasets.
- The near-identical subset is small (15 test sequences), so 1.000 deserves
  caution.
- P5b is the perceived relational benchmark; no pushes are executed. It is
  encoder-invariant here, so this round provides no P5b×encoder evidence.
- The calibrated open-set score does not beat the bank's tuned-weights score
  (0.959 vs 0.993 on base). Temperature + accuracy-derived weights are more
  interpretable, but the existing cross-validated bank weights are stronger
  on this split.
- VLAD at K=16 is a minimal local-descriptor implementation. Patch-to-patch
  matching or a learned codebook may do better; this round does not show it.

---

## 6. What this changes in the priority list

- **Keep the 128-d interface.** The encoder swap is now a registry argument,
  not a code change.
- **Calibration is task-specific, not universal.** It rescues DINOv2 re-ID
  (0.963 → 0.993) but hurts DINOv2 open-set (0.976 → 0.897). Judge a frozen
  encoder with the head that matches the downstream task, and keep separate
  recognition and re-ID heads when both matter.
- **Separate recognition and re-ID heads if both matter.** Whitening helps
  closed-set, residual helps open-set, and the base head is best for re-ID.
  One head cannot maximize all three.
- **Do not use the current P5b as an encoder test.** Its state is
  position-only; it needs a category/identity-dependent goal first.
- **Next encoder work:** CLIP once cached; patch-level matching beyond VLAD;
  a calibrated open-set score that matches the bank's AUROC.

Reproduce:

```bash
python -m pss.eval.scripts.run_encoder_priority --stage all --device cuda
```
