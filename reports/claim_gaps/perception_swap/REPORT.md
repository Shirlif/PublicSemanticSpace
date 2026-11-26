# Claim-gap F2: perception-level mid-stream expert swap

> generated: 2026-11-26 07:05:19 UTC
> bundles: 4
> swap at 50% of frames, rollback after 10% of frames
> deadline_ms: 10.0 (guard off/on both reported)

## Result summary

- Positive control on 4/4 bundles: SSR 1.000-1.000, RBE 1.000-1.000, INT_post 1.000-1.000, RCR 1.000-1.000.
- Naive control: RCR below 1.0 on 4/4 bundles; INT_post below 1.0 on 2/4; RCR_assign below 1.0 on 2/4. The negative control requirement is met.
- Pose delta: position mean 0.0017-0.0318 m; match rate 0.870-0.998; quaternion delta 0 deg because every detection and GT instance is axis-aligned.
- Latency: encode p50 2.77-3.74 ms, p99 4.15-11.03 ms; ingest off p99 1.31-1.86 ms; guarded p99 1.68-3.20 ms.

## Protocol

- Incumbent: `smallcnn` (frozen task-trained SmallCNN, 256-d) +
  `checkpoints/projection_v1/vision.pt` (frozen 128-d head).
- Incoming: `dinov2` (the detector's stored `emb`, DINOv2 ViT-S/14 CLS,
  384-d) + a ridge calibration fitted on pre-swap, unlabeled frames
  into the incumbent's public space (`_ridge_map`, the same solver as
  `run_l1_family_calibration`).
- The replay consumes only `obs_detections.parquet` plus frame images for
  the smallcnn crops. `gt.parquet` is opened only by the pose-delta
  scorer, after all PSS metrics.
- Asset availability: `checkpoints/expert_v1/vision.pt` and the bundle
  images were present, so the smallcnn path is the real expert, not a
  fallback. DINOv2 recomputation from crops was tested but not used; the
  stored detector embedding is the representation the bundle carries.
- `positive`: `ModuleRegistry.swap` at frame t, `rollback` at t+k.
- `naive`: direct `registry._modules['vision'] = dinov2` at t (validation
  bypassed) and `registry.rollback()` at t+k (empty stack -> no-op).

## Bundles and calibration

| bundle | frames | detections | swap/rollback frame | DINOv2 source | cal/val samples | held-out cos mean | median | min |
|---|---:|---:|---|---|---:|---:|---:|---:|
| s1_hard/37f0c8b8/seed0 | 100 | 556 | 50/60 | bundle.emb (detector DINOv2 ViT-S/14 CLS) | 24 | 0.881 | 0.926 | 0.218 |
| s1_occlusion_real/16ea9f38/seed0 | 100 | 460 | 50/60 | bundle.emb (detector DINOv2 ViT-S/14 CLS) | 51 | 0.998 | 0.999 | 0.991 |
| s2_twins/3d49ce1b/seed0 | 100 | 407 | 50/60 | bundle.emb (detector DINOv2 ViT-S/14 CLS) | 48 | 0.996 | 0.998 | 0.987 |
| s1_camera_real/b4bb5f0e/seed0 | 100 | 285 | 50/60 | bundle.emb (detector DINOv2 ViT-S/14 CLS) | 20 | 0.955 | 0.979 | 0.728 |

The calibration is label-free and uses only frames before the swap.
Held-out cosine is measured on the later pre-swap frames, not on the
swapped segment.

## Positive vs negative control

| bundle | control | SSR | RBE | INT_post | INT_all | RCR | RCR_assign | INT_interference | stock retention |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s1_hard/37f0c8b8/seed0 | positive | 1.000 | 1.000 | 1.000 | 0.984 | 1.000 | 0.896 | 0.000 | 1.000 |
| s1_hard/37f0c8b8/seed0 | naive | 0.000 | 0.000 | 0.930 | 0.771 | 0.000 | 0.324 | 0.000 | 1.000 |
| s1_occlusion_real/16ea9f38/seed0 | positive | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| s1_occlusion_real/16ea9f38/seed0 | naive | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| s2_twins/3d49ce1b/seed0 | positive | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| s2_twins/3d49ce1b/seed0 | naive | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| s1_camera_real/b4bb5f0e/seed0 | positive | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| s1_camera_real/b4bb5f0e/seed0 | naive | 0.000 | 0.000 | 0.943 | 0.887 | 0.000 | 0.595 | 0.000 | 1.000 |

| control | SSR mean | RBE mean | INT_post mean | INT_all mean | RCR mean | RCR_assign mean |
|---|---:|---:|---:|---:|---:|---:|
| positive | 1.000 | 1.000 | 1.000 | 0.996 | 1.000 | 0.974 |
| naive | 0.000 | 0.000 | 0.968 | 0.914 | 0.000 | 0.730 |

**Negative control requirement:** the naive run must fail on INT or RCR.
RCR is the deterministic discriminator: after a direct dict mutation the
rollback stack is empty, so the incoming module stays active and the
post-rollback embeddings are not the baseline embeddings. INT is the
capability-side check; it is reported honestly even when it does not move.

`stock retention` is the repository's `id_retention` on a single trace.
It is 1.000 by construction here because `emb_ref` is a per-detection id,
so it is not evidence for or against the swap.

## Pose delta vs GT (GT read only in this step)

| bundle | matched/total | match rate | pos mean m | pos median m | pos p95 m | quat mean deg | quat p95 deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| s1_hard/37f0c8b8/seed0 | 555/556 | 0.998 | 0.0318 | 0.0095 | 0.2054 | 0.000 | 0.000 |
| s1_occlusion_real/16ea9f38/seed0 | 400/460 | 0.870 | 0.0029 | 0.0021 | 0.0142 | 0.000 | 0.000 |
| s2_twins/3d49ce1b/seed0 | 400/407 | 0.983 | 0.0017 | 0.0016 | 0.0026 | 0.000 | 0.000 |
| s1_camera_real/b4bb5f0e/seed0 | 271/285 | 0.951 | 0.0239 | 0.0067 | 0.0664 | 0.000 | 0.000 |

Detections are matched greedily to the nearest same-frame GT instance
within 0.5 m; the match rate is the fraction of detections that found a
GT partner. These deltas describe the detector, not the swap.

## Latency (F3 deadline interface)

| bundle | encode p50 | encode p99 | ingest off p50 | off p99 | off max | guard on p50 | on p99 | on max | timeouts | degraded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s1_hard/37f0c8b8/seed0 | 3.74 | 11.03 | 0.93 | 1.86 | 1.93 | 1.44 | 3.20 | 3.49 | 0 | 0 |
| s1_occlusion_real/16ea9f38/seed0 | 3.05 | 4.15 | 0.77 | 1.35 | 1.59 | 1.19 | 1.68 | 2.81 | 0 | 0 |
| s2_twins/3d49ce1b/seed0 | 2.92 | 4.69 | 0.77 | 1.34 | 2.77 | 1.21 | 3.10 | 57.66 | 1 | 1 |
| s1_camera_real/b4bb5f0e/seed0 | 2.77 | 4.50 | 0.71 | 1.31 | 1.93 | 1.15 | 1.94 | 2.07 | 0 | 0 |

`encode` is per-frame perception work (image crop + smallcnn +
projection + dinov2 projection) measured during precompute. The guarded
number is the caller wait around runtime ingest only; it is not an
end-to-end perception budget. The guard snapshot below is the actual
F3 output for the positive run.

| bundle | guard budget ms | frames | timeout rate | max consecutive | worker late max ms | elapsed p99 ms | errors | relations skipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s1_hard/37f0c8b8/seed0 | 10.0 | 100 | 0.000 | 0 | 0.00 | 3.20 | 0 | 0 |
| s1_occlusion_real/16ea9f38/seed0 | 10.0 | 100 | 0.000 | 0 | 0.00 | 1.68 | 0 | 0 |
| s2_twins/3d49ce1b/seed0 | 10.0 | 100 | 0.010 | 1 | 58.23 | 3.10 | 0 | 2 |
| s1_camera_real/b4bb5f0e/seed0 | 10.0 | 100 | 0.000 | 0 | 0.00 | 1.94 | 0 | 0 |

Guard timeouts occurred on: s2_twins/3d49ce1b/seed0. A timed-out
frame returns empty events and enters a short work-shedding cooldown, so the
guarded run's INT_post can dip; the unguarded positive run is the correctness
reference. A residual caller spike above the budget is guard-thread
preemption, not an undetected worker overrun (E21).

## Guard-run correctness

| bundle | control | SSR | RBE | INT_post | RCR |
|---|---|---:|---:|---:|---:|
| s1_hard/37f0c8b8/seed0 | positive_guard | 1.000 | 1.000 | 1.000 | 1.000 |
| s1_occlusion_real/16ea9f38/seed0 | positive_guard | 1.000 | 1.000 | 1.000 | 1.000 |
| s2_twins/3d49ce1b/seed0 | positive_guard | 1.000 | 1.000 | 0.980 | 1.000 |
| s1_camera_real/b4bb5f0e/seed0 | positive_guard | 1.000 | 1.000 | 1.000 | 1.000 |

The guard does not enter any correctness metric; this table only checks
that enabling it did not change the swap contract on these traces.

## Registry audit

```json
{
  "positive": [
    {
      "action": "register",
      "modality": "vision",
      "version": "vision-smallcnn-256-v1"
    },
    {
      "action": "register",
      "modality": "geometry",
      "version": "geometry-fixed-stub-v1"
    },
    {
      "action": "swap",
      "success": true,
      "modality": "vision",
      "old_version": "vision-smallcnn-256-v1",
      "new_version": "vision-dinov2-384-ridgecal-30f-v1",
      "latency_ms": 0.19020000036107376,
      "error": ""
    },
    {
      "action": "rollback",
      "success": true,
      "modality": "vision",
      "old_version": "vision-dinov2-384-ridgecal-30f-v1",
      "new_version": "vision-smallcnn-256-v1",
      "latency_ms": 0.0022999993234407157,
      "error": ""
    }
  ],
  "naive": [
    {
      "action": "register",
      "modality": "vision",
      "version": "vision-smallcnn-256-v1"
    },
    {
      "action": "register",
      "modality": "geometry",
      "version": "geometry-fixed-stub-v1"
    }
  ]
}
```

The naive audit contains only registrations: direct dict mutation bypasses
the registry's audit path, and an empty-stack rollback returns without an
entry. The attempted rollback and its error are recorded in `metrics.json`
under `bundles[].naive.rollback`.

## How to read this

- The positive control passes SSR/RBE/RCR by construction; that is the
  contract under test, not a capability claim.
- The naive control fails RCR deterministically. It may also fail INT if
  the post-rollback expert changes track assignments; INT is reported
  per bundle rather than assumed.
- `RCR_assign` is capability-side, not the contract metric: the positive
  run can score below 1.0 because the runtime state legitimately diverges
  during the swapped segment (e.g. `s1_hard`). RCR (active version plus
  identical post-rollback embeddings) is the contract check.
- `INT_all` includes the swapped segment, where a different expert is
  supposed to change the trace; `INT_post` is the post-rollback retention.
- INT_interference stays 0 for both controls: the geometry stub is not
  touched, so the swap is local to the vision modality.
- The smallcnn expert is task-trained on `data/expert_v1` categories
  (block/bottle/box/cup/plate/toy_fruit), while these bundles contain
  MuJoCo cubes. This domain shift is part of the experiment: it is the
  incumbent expert, not a tuned cube detector.
- DINOv2 native features are the detector's stored `emb`, so the
  incoming expert is the representation the bundle actually carries.
  Recomputing DINOv2 from crops was tested and differs (cosine ~0.63-0.90
  on sampled crops) because the detector's processor is not replicated
  exactly; the stored embedding avoids that drift.
- The ridge calibration is transductive: it uses unlabeled pre-swap frames
  from the same stream. That is the few-shot calibration protocol, not a
  claim that the incoming expert is calibrated without any data.

## Reproduce

```bash
python -m pss.eval.scripts.run_gap_perception_swap \
  --out reports/claim_gaps/perception_swap
```
