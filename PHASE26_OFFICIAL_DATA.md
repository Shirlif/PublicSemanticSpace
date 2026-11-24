# Phase 26 — Official CLEVR-CoGenT + TAO data integration

> Date: 2026-11-24
> Scope: Phase A suite, official data
> Result artifacts: `eval/results/suite_tao/`, `data/public/{clevr_cogent,tao}/`

---

## 1. Data status

| Benchmark | Official source | Status |
|---|---|---|
| TAO annotations | `TAOLabels.zip` from motchallenge.net | **downloaded** (6.3 MB; `TAO_val/gt/gt.json`) |
| TAO frames | TAO source videos (HACS/BDD/Charades/YFCC/AVA/ArgoVerse/LaSOT) | not downloaded; association metric uses oracle boxes |
| CLEVR-CoGenT | `CLEVR_CoGenT_v1.0.zip` from dl.fbaipublicfiles.com | **downloaded + extracted** (24.7 GB) |
| CLEVR adapter | `scenes/*.json` + `images/<split>/` | **official valA/valB run completed** |

Raw data stays under `data/public/` and is gitignored.

---

## 2. TAO official annotation run

Adapter: `pss/eval/suite/adapters/tao.py`.
Predictor: oracle GT boxes + simple IoU tracker (association baseline, not PSS
perception). 50 val videos, official annotations.

| metric | value |
|---|---:|
| IDF1 | 0.478 |
| ID retention | 0.482 |
| ID switches | 7.94 |

This proves the official annotation pipeline works. The number is a baseline
for the association layer; a PSS tracker/perception run replaces the predictor
once frames are available.

---

## 3. CLEVR-CoGenT official run

Official `valA` / `valB` scenes (15,000 each; two episodes per scene: one true
relation, one false relation). Predictor: GT boxes + pixel x / pixel depth
(image-space baseline, not PSS vision).

| split | episodes | relation accuracy |
|---|---:|---:|
| valA | 30,000 | **0.994** |
| valB | 30,000 | **0.994** |
| all | 60,000 | **0.994** |

**Important caveat:** the relation-only task is spatial (left/right/front/
behind) and does not exercise the CoGenT compositional shift (shape/color/
material/size combinations). valA and valB are equal by construction for this
task. Testing CoGenT composition needs a property-conditioned question or an
object-property classifier trained on trainA and tested on valB.

## 4. CLEVR-CoGenT adapter

`pss/eval/suite/adapters/clevr_cogent.py`:

- parses `scenes/*.json` and `images/<split>/`;
- yields left/right/front/behind relation episodes;
- `valA` vs `valB` is the compositional split;
- smoke episodes pass (relation accuracy 1.000 on synthetic scenes).

The current CLEVR predictor uses official `relationships` as GT and pixel
coordinates/depth as the prediction input. It is an image-space baseline, not a
PSS image encoder + relation head result. The next step is an image-based
predictor (PerceptionService/DINOv2 or CLIP + relation head).

---

## 4. Findings

1. **TAO official annotations are integrated.** The suite can enumerate official
   videos/frames/tracks and compute IDF1/retention/switches.
2. **TAO frames are the remaining cost.** Full TAO videos are distributed
   through source datasets (HACS/BDD/Charades/YFCC/AVA/ArgoVerse/LaSOT), often
   large or gated. The association metric is runnable with oracle boxes; a
   perception-based run needs a frame subset.
3. **CLEVR-CoGenT official valA/valB is runnable.** 60,000 episodes, 0.994 /
   0.994 with the pixel baseline; the relation task does not test the CoGenT
   property-composition shift.
4. **Both adapters are honest about oracles.** Boxes/coordinates are explicit
   baselines; they are not presented as PSS capability.

---

## 5. Next

- Replace the CLEVR pixel baseline with an image-based PSS relation predictor
  (crops + frozen encoder + relation head) and add a property-conditioned
  CoGenT split.
- Download a small official TAO frame subset (e.g., HACS/Charades) and run the
  PSS tracker, not only the oracle-box association baseline.
- Add `owtb`, `alfworld`/`calvin` adapters.

Reproduce:

```bash
python -m pss.eval.suite.runner --benchmarks tao --root data/public --out eval/results/suite_tao
python -m pss.eval.suite.runner --benchmarks clevr_cogent --root data/public
```
