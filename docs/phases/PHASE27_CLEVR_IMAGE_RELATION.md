# Phase 27 — CLEVR-CoGenT image relation + property composition

> Date: 2026-11-24
> Scope: steps 1–2 of the CLEVR follow-up
> Data: official CLEVR-CoGenT (trainA 8,000 scenes; valA/valB 3,000 each)
> Result artifacts: `eval/results/clevr_image/`

---

## 1. Setup

- Image crops from official `pixel_coords` (GT boxes, detection-oracle), resized
  to 224×224;
- frozen DINOv2-small CLS embeddings (384-d);
- relation head: MLP over `[e_i, e_j, e_i-e_j, e_i*e_j]` (+ optional image-space
  dx/dy/depth);
- property composition: DINOv2 features + (a) joint combination classifier,
  (b) factorized four-head classifier (shape/color/material/size);
- trainA -> valA/valB; official CoGenT split.

Counts:

| split | scenes | objects |
|---|---:|---:|
| trainA | 8,000 | 51,835 |
| valA | 3,000 | 19,803 |
| valB | 3,000 | 19,490 |

---

## 2. Relation results

| method | valA | valB |
|---|---:|---:|
| pixel geometry rule | 0.986 | 0.991 |
| vision only | 0.928 | 0.930 |
| vision + image position | **0.994** | **0.996** |

- DINOv2 crop embeddings alone carry enough signal for relation classification
  (0.93); adding image-space position matches the exact geometry rule.
- This is an **image-space baseline**, not a detector: boxes come from
  `pixel_coords`.
- The relation task does not test the CoGenT property-composition shift.

---

## 3. Property composition (trainA -> valA/valB)

| method | valA exact | valB exact |
|---|---:|---:|
| joint combination | 0.810 | **0.268** |
| factorized (DINOv2 for all) | 0.747 | **0.316** |
| **factorized (color histogram for color)** | **0.801** | **0.795** |

Color head comparison:

| split | DINOv2 CLS | color histogram |
|---|---:|---:|
| valA | 0.774 | **0.845** |
| valB | **0.336** | **0.841** |

Factorized per-attribute accuracy with the color-histogram head:

| split | shape | color | material | size |
|---|---:|---:|---:|---:|
| valA | 0.960 | 0.845 | 0.956 | 0.985 |
| valB | 0.957 | **0.841** | 0.958 | 0.981 |

- The joint classifier memorizes seen combinations: **0.810 -> 0.268** on
  valB.
- DINOv2's color head collapses on valB (0.774 -> **0.336**): it relies on
  shape-color correlations that CoGenT breaks.
- **Color histogram fixes it**: 0.845 / **0.841** on valA/valB. Using the
  histogram only for the color head and DINOv2 for shape/material/size raises
  factorized exact composition from 0.316 to **0.795** on valB.
- The fix is a feature choice at the attribute layer, not a new encoder:
  DINOv2 remains the shape/material/size representation.

---

## 4. Findings

1. **Image-based relation works**: 0.994/0.996 with position, 0.928/0.930
   without; the geometry rule is 0.986/0.991.
2. **Property composition is the real CoGenT test**: joint memorization drops
   54 points on valB; factorized recovers only part of it.
3. **Color was the bottleneck, and a color histogram solves it**: valB color
   0.336 -> **0.841**, factorized exact 0.316 -> **0.795**. Shape/material/size
   were already stable.
4. **GT boxes were an oracle in this phase**; Phase 29 replaces them with a
   trained detector and keeps relation/property within ~0.5 points.
5. **The relation head is a simple MLP**, not the PSS `TypedRelationHead`;
   wiring the PSS typed head is the next integration step.

---

## 5. Next

- GT boxes are replaced in `PHASE29_DETECTOR_REPLACEMENT.md`: CenterNet-lite
  on trainA reaches detection recall 0.997/0.996, relation 0.995/0.994 and
  property exact 0.792/0.790 on valA/valB.
- Wire the PSS `TypedRelationHead` instead of the MLP.
- Add a property-conditioned VQA task (language + attributes) for a fuller
  CoGenT evaluation.
- Step 3 is complete and reported in `PHASE28_TAO_PSS.md`: on 5 official
  Charades videos, PSS tracker IDF1 **0.979** vs IoU **0.653**.

Reproduce:

```bash
python -m pss.eval.scripts.run_clevr_image --device cuda --train 8000 --val 3000
```
