# Phase 29 — 把 GT boxes 换成检测器

> Date: 2026-11-24
> Scope: CLEVR-CoGenT 与 TAO/Charades 的 GT-box 替换
> Result artifacts: `eval/results/clevr_detector/`, `eval/results/tao_detector/`

---

## 1. CLEVR：CenterNet-lite 检测器

### Setup

- 输入 256×256，输出 64×64 heatmap（stride 4）+ 64×64 depth map；
- 小型 4-block CNN + 两次上采样，heatmap 用 BCE（`pos_weight=50`），
  depth 在 GT 中心处用 L1；
- trainA 8,000 scenes 训练 25 epochs；valA/valB 各 3,000 scenes；
- 测试时只用检测中心：固定 64×64 crop → 冻结 DINOv2 / 颜色直方图；
- GT 中心只用于检测评分和给检测框贴标签，不进入关系/属性输入。

### Results

| split | detection recall | precision | center err (px) | relation | property exact | shape | color | material | size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valA | 0.997 | 0.930 | 4.96 | 0.995 | 0.792 | 0.956 | 0.839 | 0.954 | 0.981 |
| valB | 0.996 | 0.925 | 5.10 | 0.994 | 0.790 | 0.954 | 0.839 | 0.958 | 0.979 |

GT-box baseline (Phase 27) for comparison:

| task | valA | valB |
|---|---:|---:|
| relation (vision + position) | 0.994 | 0.996 |
| property factorized exact (color histogram) | 0.801 | 0.795 |

---

## 2. TAO：COCO person detector

### Setup

- Detector: torchvision Faster R-CNN MobileNetV3-320 FPN, COCO 预训练，
  只取 person 类，score ≥ 0.3；
- 5 个 Charades 视频、176 帧；TAO person category 805；
- GT person boxes 只用于检测 recall/precision 和 IDF1 标签；
- 对比三条路径：检测框 + IoU、检测框 + PSS、GT 框 + PSS（oracle）。

### Results

Person detection: recall **0.970**, precision **0.809** (per-video mean).

| method | IDF1 | ID retention | ID switches |
|---|---:|---:|---:|
| detected boxes + IoU | 0.653 | 0.692 | 1.20 |
| **detected boxes + PSS** | **0.984** | **1.000** | **0.00** |
| GT boxes + PSS (oracle) | 1.000 | 1.000 | 0.00 |

Per video:

| video | frames | GT person instances | det recall | det precision | IoU IDF1 | PSS IDF1 |
|---|---:|---:|---:|---:|---:|---:|
| 21WN7 | 35 | 35 | 1.000 | 0.714 | 0.514 | 1.000 |
| IONV9 | 39 | 39 | 1.000 | 0.684 | 0.462 | 1.000 |
| J7BOV | 32 | 32 | 0.938 | 0.938 | 0.742 | 0.968 |
| OEIR9 | 36 | 72 | 1.000 | 0.973 | 1.000 | 1.000 |
| UR55B | 34 | 46 | 0.913 | 0.737 | 0.545 | 0.955 |

---

## 3. Findings

1. **GT boxes are no longer an oracle in either benchmark.** CLEVR uses a
   trained center/depth detector; TAO uses a COCO person detector.
2. **CLEVR relation survives detection.** Detection recall 0.997/0.996 and
   relation 0.995/0.994, essentially the same as the GT-box baseline
   (0.994/0.996); the detector's depth head replaces `pixel_coords[2]`.
3. **CLEVR property loses 0.5–0.9 points.** Fixed 64×64 crops around detected
   centers (center error 4.96/5.10 px) give factorized exact 0.792/0.790 vs
   0.801/0.795 with GT boxes; the color-histogram head remains the main reason
   valB composition works.
4. **TAO's PSS gain survives detector noise.** PSS IDF1 0.984 vs IoU 0.653;
   oracle is 1.000. False-positive detections do not hurt IDF1 directly
   (only GT-matched boxes are scored), which is why PSS stays high at 0.81
   detection precision.
5. **This is not a general TAO detector.** COCO person only; TAO's LVIS
   categories and class-agnostic proposals are not covered.

---

## 4. Boundaries

- CLEVR detector is trained in-domain on trainA; no official CLEVR detection
  benchmark; center+depth only, no box-size head.
- TAO detector is zero-shot COCO person, 5 Charades videos, small sample;
  no official `TaoEval` HOTA.
- CLEVR relation is dominated by the `left/right` pair selected by
  `_true_pair`; front/behind is not separately stressed in this run.
- The CLEVR relation head is still the lightweight MLP, not the PSS
  `TypedRelationHead`; runtime integration remains future work.

---

## 5. Reproduce

```bash
python -m pss.eval.scripts.train_clevr_detector --device cuda --train 8000 --val 3000 --epochs 25
python -m pss.eval.scripts.run_tao_detector --device cuda --threshold 0.3
```
