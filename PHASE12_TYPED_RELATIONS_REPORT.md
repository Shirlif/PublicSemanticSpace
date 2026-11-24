# Phase 12 — Learnable typed relations

> Date: 2026-11-24
> Scope: `PSS-认知层架构定位.md` §7, "第二优先：把关系层变成可学习"
> Data: `data/tracking_bench_v2`, P5b region grid (synthetic)
> Result artifacts: `eval/results/typed_relations/`

---

## 1. What was implemented

The relation layer now has typed, learnable heads and a persistent write path
into the scene graph. The public PSS interfaces stay unchanged: entity tokens
and cost matrices keep their shape; relation logits are additive records.

| Component | File | Purpose |
|---|---|---|
| Typed relation schema | `pss/memory/typed_relations.py` | spatial (5 predicates), semantic, tactile, temporal, task |
| Pairwise relation head | `pss/memory/typed_relations.py` | `TypedRelationHead` over entity-token pairs |
| Coordinate relation head | `pss/memory/typed_relations.py` | `SpatialRelationHead` for the P5b region grid |
| Relation labels | `pss/memory/typed_relations.py` | `frame_targets`, `temporal_targets`, dead-zone spatial labels |
| Relation-aware matchers | `pss/memory/typed_relations.py` | pure learned edges + hybrid (hand + learned) |
| Scene-graph edges | `pss/memory/scene_graph/__init__.py` | `relations`, `set_relation`, `get_relations`, save/load |
| Scene-graph writer | `pss/memory/typed_relations.py` | `write_relation_edges` on active tracks |
| P5b pluggable predicate | `pss/eval/relational_bench.py` | `goal_ok(..., rel_fn)` / `plan(..., rel_fn)` |
| Experiment runner | `pss/eval/scripts/run_typed_relations.py` | relations / re-ID / graph / P5b |

Types and predicates:

| type | predicates | training signal |
|---|---|---|
| spatial | left, right, above, below, near | frame poses |
| semantic | same_category | category labels |
| tactile | contact, similar_tactile | pose radii + tactile cosine |
| temporal | approaching, receding, stable | distance change across two frames (schema + context; not trained this round) |
| task | goal_satisfied | P5b goal relation (evaluated through the planner) |

---

## 2. Relation prediction

Trained on 200 train pairs, validated on 40, tested on 60; labels come from
the simulator. Corruption zeroes a modality or adds pose noise at test time.

| condition | split | spatial F1 | semantic F1 | tactile F1 | mean F1 |
|---|---|---:|---:|---:|---:|
| clean | test | 0.957 | 0.996 | 0.970 | 0.974 |
| pose noise | test | 0.953 | 0.994 | 0.970 | 0.973 |
| no tactile | test | 0.954 | 0.976 | **0.483** | 0.804 |
| no vision | test | 0.957 | 0.977 | 0.955 | 0.963 |
| mixed | test | 0.952 | 0.982 | 0.911 | 0.949 |

The typed heads learn the frame-local relations. Spatial and semantic are
robust to the corruptions tested; tactile is the only relation that collapses
when its own modality is removed, which is expected and confirms the head uses
the tactile channel rather than a shortcut.

---

## 3. Re-ID: hand edges vs learned typed edges

Same tracking pairs, same Hungarian evaluation, 3 seeds, 200 epochs.

| matcher | clean | near-identical | no tactile | pose noise | no vision | mixed |
|---|---:|---:|---:|---:|---:|---:|
| hand (existing `GraphMatcher`) | **1.000 ± 0.000** | **1.000** | 0.791 | **1.000** | **0.996** | **0.946** |
| pure learned edges | 0.985 ± 0.005 | 0.985 | 0.696 | 0.989 | 0.937 | 0.889 |
| hybrid (hand + learned) | 0.996 ± 0.005 | **1.000** | **0.819** | 0.996 | 0.989 | 0.944 |

- **Pure learned edges do not beat hand-written edges** on clean re-ID; the
  hand distance/category/relative-pose features are a strong prior.
- **The hybrid is the useful configuration**: it keeps the hand edges and adds
  typed relation logits. It improves the missing-tactile condition from 0.791
  to **0.819** and keeps near-identical at **1.000**, with parity elsewhere.
- The result supports "typed relations as a calibrated robustness channel",
  not "replace hand edges everywhere".

---

## 4. Relation logits in the scene graph

`write_relation_edges` runs the typed head on active tracks and stores every
predicate above a threshold in `SceneGraph.relations`. On a 3-object test
frame it wrote **18 edges** (12 spatial, 6 tactile), and a save → load round
trip preserved all 18. The graph now carries both the symbolic track state and
the learned relation logits, so the reasoning layer can query either:

```python
graph.relation_logits("obj_0000", "obj_0001")
# {"spatial:right": {...}, "tactile:contact": {...}, ...}
```

---

## 5. P5b with learned spatial relations

The planner's goal check is now pluggable. A coordinate relation head is
trained on a subset of the five regions and tested on held-out regions:

| relation head | training regions | held-out all-4 exact | per-label |
|---|---|---:|---:|
| in-distribution | all | 1.000 | 1.000 |
| held-out | A/B/C only | 0.571 | 0.893 |

| planner | ID success | OOD success |
|---|---:|---:|
| hand-written `_rel` | 1.000 | **1.000** |
| learned, in-distribution | 1.000 ± 0.000 | **1.000 ± 0.000** |
| learned, held-out regions | 0.842 ± 0.016 | 0.688 ± 0.018 |
| transformer BC (reference) | 1.000 | 0.462 |

- **In-distribution, the learned relation head fully replaces the hand rule**
  without losing P5b (1.000 / 1.000).
- **On held-out region layouts it degrades** (OOD 0.688 vs 1.000). The exact
  geometric predicate remains the stronger prior when the layout is unseen.
- The learned relation layer still transfers better than the monolithic
  transformer policy (0.688 vs 0.462 OOD), which is the useful comparison:
  explicit typed relations generalize further than an end-to-end policy, but
  not as far as an exact geometric rule.

A dead-zone label (`|delta| < 0.1` → no relation) was required for aligned
regions; without it the head is forced to pick a side on e.g. C/D and P5b
drops to 0. This is a calibration detail, not a modelling trick, and is part
of the shipped label function.

---

## 6. Findings

1. **Typed relations are learnable**: spatial 0.957, semantic 0.996, tactile
   0.970 F1 clean.
2. **Pure learned edges are not better than hand edges** for re-ID; the hybrid
   is the better design (missing-tactile 0.791 → 0.819).
3. **Relation logits can be written into and reloaded from the scene graph**;
   the audit path works.
4. **P5b can run on a learned relation head in-distribution (1.000/1.000) but
   loses on held-out layouts (OOD 0.688)**; exact geometry stays the stronger
   prior for unseen configurations.
5. **Temporal and task heads need context**: task is evaluated through P5b;
   temporal needs track history and is left as the next step. The schema and
   the scene-graph write path are already in place for it.

---

## 7. Boundaries

- All data is synthetic; labels are derived from simulator GT.
- The held-out P5b test has 14 region pairs and one training split; it is a
  directional result, not a benchmark suite.
- Re-ID is saturated on the clean benchmark (hand 1.000), so the corruption
  conditions carry the signal.
- The relation head is trained on the training split only; test labels are
  never used.
- Temporal/task relation heads are not trained this round; do not claim them.

---

## 8. Next steps

- Train the temporal head on tracker history (approach/recede/stable) and
  evaluate it on occlusion re-appearance.
- Add task-conditioned relations beyond P5b (reachable, blocked, supports).
- Use relation logits in the active-set / write policy (third priority).
- Test whether relation supervision improves object attention on the
  P1-15 benchmark.

Reproduce:

```bash
python -m pss.eval.scripts.run_typed_relations --stage all --device cuda
```
