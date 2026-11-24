# Phase 15 — End-to-end / bilevel training (frozen encoders)

> Date: 2026-11-24
> Scope: `PSS-认知层架构定位.md` §7, "第五优先：端到端 / bilevel 训练"
> Data: `data/category_pretrain`, `data/tracking_bench_v2` (synthetic)
> Result artifacts: `eval/results/bilevel/`

---

## 1. What was implemented

Encoders stay frozen. Raw expert features are cached once, then a single
trainable graph is optimized:

```
frozen features -> projections -> binder -> typed relations -> matcher
```

| Component | File | Purpose |
|---|---|---|
| `EndToEndPSS` | `pss/learning/bilevel.py` | projections, binder, relation head, pair matcher |
| episodic prototype memory | `pss/learning/bilevel.py` | differentiable support mean (inner memory) |
| multi-task losses | `pss/learning/bilevel.py` | task + re-ID + relation + consistency |
| consistency/audit terms | `pss/learning/bilevel.py` | antisymmetry, exclusivity, permutation, stability |
| experiment runner | `pss/eval/scripts/run_bilevel.py` | feature cache, modes, evaluation |

Two binding views are used:

- **recognition view**: category one-hot is masked, so the classifier cannot
  read the label. Without this, recognition is trivially 1.000 (leak).
- **matching/relation view**: category one-hot is visible, matching the
  existing PSS entity token and the graph matcher prior.

Modes:

| mode | task | notes |
|---|---|---|
| `joint` | category prototypes | all losses together |
| `joint_instance` | instance contrastive | identity-preserving task |
| `reid_heavy` | category, downweighted | re-ID loss x4 |
| `sequential` | category then relation/re-ID | projections frozen after stage 1 |
| `no_consistency` | category | consistency terms removed |
| `no_relation` | category | relation supervision removed |
| `linear_head` | linear classifier | no episodic prototypes |
| `concat` | category | linear binder instead of MLP |

---

## 2. Results (1 seed, 60 epochs, synthetic)

| mode | recognition | re-ID | near-identical | spatial F1 | semantic F1 | tactile F1 |
|---|---:|---:|---:|---:|---:|---:|
| `joint` | 0.979 | 0.889 | 0.600 | 0.371 | 0.779 | 0.779 |
| `joint_instance` | 0.896 | **0.922** | 0.733 | 0.439 | 0.789 | 0.782 |
| `reid_heavy` | 0.948 | 0.828 | 0.733 | 0.276 | 0.676 | 0.772 |
| `sequential` | 0.984 | 0.867 | 0.467 | 0.401 | 0.667 | 0.782 |
| `no_consistency` | 0.969 | 0.889 | 0.600 | **0.663** | **0.856** | 0.779 |
| `no_relation` | 0.943 | 0.889 | 0.600 | 0.025 | 0.667 | 0.782 |
| `linear_head` | **0.995** | 0.900 | 0.733 | 0.285 | 0.695 | 0.782 |
| `concat` | 0.979 | 0.861 | 0.578 | 0.329 | 0.922 | 0.740 |

Consistency / audit constraints (lower is better):

| mode | antisymmetry error | exclusivity violation | permutation MSE | stability MSE |
|---|---:|---:|---:|---:|
| `joint` | 0.005 | 0.000 | 0.0000 | 0.0003 |
| `joint_instance` | 0.008 | 0.001 | 0.0000 | 0.0004 |
| `reid_heavy` | 0.007 | 0.000 | 0.0000 | 0.0003 |
| `sequential` | 0.004 | 0.000 | 0.0000 | 0.0000 |
| `no_consistency` | **0.054** | **0.277** | 0.0000 | 0.0004 |
| `no_relation` | 0.003 | 0.000 | 0.0000 | 0.0004 |
| `linear_head` | 0.005 | 0.000 | 0.0000 | 0.0006 |
| `concat` | 0.006 | 0.000 | 0.0000 | 0.0003 |

---

## 3. Findings

### 3.1 Joint training beats staged training, modestly

`joint` reaches 0.889 re-ID vs `sequential` 0.867 and `concat` 0.861, at
essentially the same recognition (0.979 vs 0.984). The joint graph helps, but
the gain is ~2 points, not a step change.

### 3.2 Task choice trades recognition for re-ID

`joint_instance` is the best re-ID model (**0.922**) but recognition drops to
0.896. `linear_head` has the best recognition (**0.995**) but lower relation
F1. A category task pulls tokens toward category clusters; an instance task
preserves identity. The multi-task weights were not tuned beyond two settings,
so this is a trade-off curve, not an optimum.

### 3.3 Consistency constraints work, and they cost relation accuracy

Removing consistency terms (`no_consistency`) produces the highest spatial F1
(0.663) but also the highest antisymmetry error (0.054) and exclusivity
violation (0.277). The joint model with consistency terms cuts those
violations to 0.005 / 0.000, but spatial F1 falls to 0.371. This is the
expected audit-vs-accuracy trade-off; it is not free.

The permutation metric is zero for all modes after fixing the inverse-
permutation target in the loss/metric. The earlier version compared against
the wrong permutation and reported a spurious 3.3 MSE.

### 3.4 Joint relation F1 is far below the separate relation head

The separately trained typed relation head reaches spatial 0.957 / semantic
0.996 / tactile 0.970. The joint model reaches at best 0.663 / 0.922 / 0.782.
The joint binder is optimized for category/re-ID; it is not a replacement for
the relation-specialized token and supervision. This is the strongest
negative result of the phase.

### 3.5 Absolute re-ID is still below the mature pipeline

The existing graph matcher pipeline is 0.989–0.997 re-ID. The joint model is
0.889–0.922. The gap is architectural (the joint model uses a simple pairwise
MLP, not the graph matcher's edge messages) and task-composition (category
loss, relation/consistency terms). This phase validates the training graph,
not that a simple joint model replaces the mature matcher.

### 3.6 Recognition is comparable

Recognition is 0.979–0.995 across modes, against 0.979 for the existing
separate projection baseline. The bilingual recognition view (category masked)
is essential; without it recognition was trivially 1.000 by label leakage.

---

## 4. Boundaries

- Synthetic data; 1 seed, 60 epochs. The ± values would be large.
- The joint matcher is a simple pairwise MLP, not the edge-message graph
  matcher; absolute re-ID is not an apples-to-apples architecture comparison.
- Multi-task weights are not tuned (no uncertainty weighting / PCGrad).
- Relation labels come from the simulator; no real sensor data.
- The bilevel "inner loop" is a differentiable support mean, not an inner
  gradient adaptation; call it bilevel-style episodic training.

---

## 5. Next steps

- Put the graph matcher's edge-message architecture inside the joint model so
  the architecture is held fixed across joint/sequential.
- Add uncertainty weighting or gradient surgery for the four losses.
- Run 5 seeds and report variance; tune `lambda_task` / `lambda_reid` on a
  validation split.
- Add hard-negative mining for re-ID and a class-balanced relation loss.
- Try an instance-aware task with a slow prototype EMA (write-stability
  constraint) instead of a one-step support mean.
- Evaluate on `tracking_bench_long` (70-frame occlusion) where joint training
  has more room to help.

Reproduce:

```bash
python -m pss.eval.scripts.run_bilevel --device cuda --epochs 60 --seeds 1
```
