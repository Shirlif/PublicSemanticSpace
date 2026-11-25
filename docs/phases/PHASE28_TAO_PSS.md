# Phase 28 — TAO official frames + PSS tracker

> Date: 2026-11-24
> Scope: step 3 of the CLEVR/TAO follow-up
> Data: TAO official annotations + Charades 480p frames (5 videos, 176 frames)
> Result artifacts: `eval/results/tao_pss/`

---

## 1. Setup

- Official TAO val annotations (`TAO_val/gt/gt.json`), Charades videos from
  `Charades_v1_480.zip`;
- Frames extracted at the official `frame_index` into
  `data/public/tao/frames/val/Charades/<video>/`;
- Detections are **GT boxes** (association benchmark, not a detector);
- `IoU baseline`: greedy IoU matching on GT boxes;
- `PSS tracker`: `pss.memory.Tracker` with appearance (frozen DINOv2 crop
  embedding) + pose; pixel-scale gate/re-ID configuration.

A critical fix: TAO boxes are in 1280×720 coordinates while Charades 480p
frames are 480×270; boxes are scaled to the frame before cropping. Without
this, most boxes were invalid and the PSS tracker looked broken.

---

## 2. Results (5 videos)

| method | IDF1 | ID retention | ID switches |
|---|---:|---:|---:|
| IoU baseline | 0.653 | 0.611 | 5.12 |
| PSS g90 | 0.849 | 0.836 | 1.09 |
| PSS g180 | 0.920 | 0.902 | 0.60 |
| PSS g300 | 0.922 | 0.903 | 0.60 |
| PSS pose only | 0.715 | 0.721 | 1.44 |
| PSS appearance only | 0.909 | 0.890 | 0.84 |
| **PSS g300 + pixel-scale re-ID** | **0.979** | **0.973** | **0.44** |

- PSS tracker improves IDF1 from **0.653** to **0.979** and retention from
  0.611 to 0.973 on this official-frame subset.
- Appearance carries most of the gain; pose-only is 0.715. The pixel-scale
  re-ID gate (the default `0.5` is in metres, wrong for pixels) is necessary
  for the final jump.
- IoU baseline has more switches because it re-creates IDs when overlap drops.

---

## 3. Findings

1. **Official TAO frames are integrated.** TAO annotations + Charades frames +
   frame extraction + PSS tracker run end-to-end.
2. **PSS tracker beats IoU association on this subset** (IDF1 0.979 vs 0.653)
   with GT boxes as detections; appearance is the main signal.
3. **Scale matters.** The first run failed because boxes were not scaled from
   TAO's 1280×720 to Charades' 480×270. This is an adapter bug, not an
   architecture result.
4. **This is an association benchmark, not a detector.** A real detector or
   proposal network is still required; the next step is to replace GT boxes
   with detections.

---

## 4. Boundaries

- 5 Charades videos, 176 frames; small sample.
- GT boxes are oracle detections; no detector.
- DINOv2 crop embeddings; no fine-tuning on TAO.
- PSS tracker configuration was tuned on this small subset; not a held-out
  hyperparameter search.
- No HOTA; IDF1/retention/switches only (the official TAO toolkit can compute
  HOTA when tracker results are exported).

---

## 5. Next

- Run more TAO videos and a held-out config.
- Detector replacement is done in `PHASE29_DETECTOR_REPLACEMENT.md`: COCO
  person detector + PSS reaches IDF1 **0.984** on the same person-only subset
  (GT-box oracle 1.000; IoU 0.653).
- Export TAO-format results and use the official `TaoEval` for HOTA/IDF1.
- Add OWTB as an open-world tracking adapter.

Reproduce:

```bash
python -m pss.eval.scripts.prepare_tao_frames --videos 5
python -m pss.eval.scripts.run_tao_pss --device cuda --videos 5
```
