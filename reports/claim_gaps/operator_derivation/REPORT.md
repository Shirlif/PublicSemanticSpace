# Operator derivation on the spatial suite

> Generated: 2026-11-26 16:16:56 UTC
> Seed: 0, items per family: 200

## 1. Transitive items (closure table)

| frame | n | accuracy | precision | recall | F1 (yes) |
|---|---:|---:|---:|---:|---:|
| world | 103 | 1.000 | 1.000 | 1.000 | 1.000 |
| viewer | 97 | 1.000 | 1.000 | 1.000 | 1.000 |
| **all** | 200 | 1.000 | 1.000 | 1.000 | 1.000 |

Baseline `direct_only` (answer yes only when the conclusion is observed): accuracy 0.500, F1 —.

## 2. Sparse consecutive chain (derived left_of)

Only consecutive edges are observed; all longer-range left_of facts are derived.

| scenes | precision | recall | F1 | derived/scene |
|---:|---:|---:|---:|---:|
| 200 | 1.000 | 1.000 | 1.000 | 1.960 |

## 3. Holdout on the full ground-truth fact set

A fraction of all true facts (all ten predicates) is hidden; the table must re-derive them without producing false facts.

| scenes | hidden | derived | precision | recall | F1 | tp | fp | fn |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 1026 | 298 | 1.000 | 0.290 | 0.450 | 298 | 0 | 728 |

## 4. Frame scoping

`frame_op` builds the reference frame with `FrameOp.from_axes`; `world_only` ignores the named frame.

| family | n | frame_op acc | world_only acc | frame-sensitive |
|---|---:|---:|---:|---:|
| frame | 200 | 1.000 | 0.495 | 0.505 |
| perspective | 200 | 1.000 | 0.705 | 0.295 |

## 5. Cross-frame transfer

Derive a left_of chain in the world frame, then check the conclusion in another frame.

| target frame | scenes | pairs | agree | contradict | contradiction rate |
|---|---:|---:|---:|---:|---:|
| world_to_viewer | 200 | 412 | 410 | 2 | 0.005 |
| world_to_cam1 | 200 | 400 | 358 | 42 | 0.105 |

## 6. Decision

- Closure derivation is **sound**: holdout precision 1.000 with 0 false facts.
- It is **low-coverage**: recall 0.290 on the full ten-predicate fact set; most relations are not licensed by any composition entry.
- It **earns its place on transitive items**: accuracy 1.000 vs 0.500 for a direct-only baseline.
- Frame scoping is exact with `FrameOp` (1.000), while ignoring the frame drops to 0.495.
- Cross-frame transfer is unsafe: 0.105 of world-frame conclusions contradict cam1-frame truth.

**Recommendation:** do not enable derivation in `PSSRuntime.ingest` by default. Expose it as an opt-in planner query once facts carry a frame tag; without that tag, derivations can mix reference frames.
