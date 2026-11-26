# ObjectFolder 2.0 real-data replay (F10 spike)

## Data inventory

- render root: `D:\PublicSemanticSpace\eval\results\objectfolder\renders`
- objects: 100
- license: CC-BY-4.0
- citation: Gao et al., ObjectFolder 2.0, CVPR 2022
- vision images/object: [2]
- touch images/object: [12]

## Object-level cross-modal binding

| protocol | direction | R@1 | R@5 | chance |
|---|---|---:|---:|---:|
| zero_shot | vision_to_touch | 0.010 | 0.040 | 0.010 |
| zero_shot | touch_to_vision | 0.020 | 0.060 | 0.010 |
| ridge_half_split | vision_to_touch | 0.040 | 0.280 | 0.020 |
| ridge_half_split | touch_to_vision | 0.040 | 0.180 | 0.020 |

- R@1 chance = 1/n = 0.010
- R@5 chance = min(5, n)/n = 0.050
- same-object cosine: 0.0365
- cross-object cosine: 0.0325
- cosine gap: 0.0040
- permutation test: observed R@1 0.010, null mean 0.010, p=0.6593 (1000 perms)

## pss_eval path

| system | property | status | metrics |
|---|---|---|---|
| pss-runtime | determinism | pass | agreement=1.000, final_state_match=1.000, observation_id_match=1.000 |
| pss-runtime | reset_exactness | pass | replay_agreement=1.000, reset_exact=1.000 |
| pss-runtime | state_hygiene | pass | digest_ok=1.000, finite_arrays=1.000, state_bytes=616.000, within_size_bound=1.000 |
| pss-runtime | contract | pass | audit_present=1.000, invalid_keeps_state=1.000, invalid_rejected=1.000, rollback_exact=1.000, swap_accepted=1.000 |
| pss-runtime | swap_rollback | pass | post_rollback_swap=1.000, rollback_exact=1.000, swap_changed_state=1.000, swap_success_rate=1.000 |
| pss-runtime | label_accuracy | skip |  |
| pss-runtime | budget | pass | budget_ms=50.000, elapsed_p50_ms=1.413, elapsed_p99_ms=3.068, errors=0.000, frames=200.000, timeout_rate=0.005 |
| pss-runtime | latency | pass | max_ms=3.708, n=200.000, p50_ms=1.598, p99_ms=3.192 |
| pss-runtime | cross_modal_binding | fail | chance=0.010, chance_r5=0.050, n=100.000, permutation_p=0.627, permutations=200.000, r1=0.010, r5=0.030, touch_to_vision_r1=0.010, touch_to_vision_r5=0.040 |
| objectfolder-passthrough | determinism | pass | agreement=1.000, final_state_match=1.000, observation_id_match=1.000 |
| objectfolder-passthrough | reset_exactness | pass | replay_agreement=1.000, reset_exact=1.000 |
| objectfolder-passthrough | state_hygiene | pass | digest_ok=1.000, finite_arrays=1.000, state_bytes=59.000, within_size_bound=1.000 |
| objectfolder-passthrough | contract | skip |  |
| objectfolder-passthrough | swap_rollback | skip |  |
| objectfolder-passthrough | label_accuracy | skip |  |
| objectfolder-passthrough | budget | skip |  |
| objectfolder-passthrough | latency | pass | max_ms=0.005, n=200.000, p50_ms=0.002, p99_ms=0.003 |
| objectfolder-passthrough | cross_modal_binding | fail | chance=0.010, chance_r5=0.050, n=100.000, permutation_p=0.657, permutations=200.000, r1=0.010, r5=0.040, touch_to_vision_r1=0.020, touch_to_vision_r5=0.060 |
| objectfolder-nn-binding | determinism | pass | agreement=1.000, final_state_match=1.000, observation_id_match=1.000 |
| objectfolder-nn-binding | reset_exactness | pass | replay_agreement=1.000, reset_exact=1.000 |
| objectfolder-nn-binding | state_hygiene | pass | digest_ok=1.000, finite_arrays=1.000, state_bytes=4762.000, within_size_bound=1.000 |
| objectfolder-nn-binding | contract | skip |  |
| objectfolder-nn-binding | swap_rollback | skip |  |
| objectfolder-nn-binding | label_accuracy | skip |  |
| objectfolder-nn-binding | budget | skip |  |
| objectfolder-nn-binding | latency | pass | max_ms=0.190, n=200.000, p50_ms=0.076, p99_ms=0.172 |
| objectfolder-nn-binding | cross_modal_binding | fail | chance=0.010, chance_r5=0.050, n=100.000, permutation_p=0.657, permutations=200.000, r1=0.010, r5=0.040, touch_to_vision_r1=0.020, touch_to_vision_r5=0.060 |

## Caveats

- ObjectFolder is a public dataset of implicit object representations; rendered vision/tactile images are not raw robot sensor captures.
- No poses, depth, segmentation, scenes, or synchronized multi-frame stream: L2 identity/tracking and sim2real claims are out of scope.
- The standard identity_retention property is not applicable: with no poses every detection collapses to one track (degenerate pass). It was removed; cross_modal_binding is used instead.
- The PSS runtime's projection here is a random adapter projection, not a projection head trained on real ObjectFolder pairs; its binding R@1 is a path-connectivity measurement, not a PSS capability result.
- Historical 100-object ObjectFolder results (zero-shot R@1 ~chance; trained tactile encoder held-out R@1 ~0.22) are the stronger evidence and should be cited instead of this 20-object spike for capability.
