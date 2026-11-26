# ObjectFolder 2.0 real-data replay (F10 spike)

## Data inventory

- render root: `D:\PublicSemanticSpace\eval\results\objectfolder\renders`
- objects: 20
- license: CC-BY-4.0
- citation: Gao et al., ObjectFolder 2.0, CVPR 2022
- vision images/object: [2]
- touch images/object: [12]

## Object-level cross-modal binding

| protocol | direction | R@1 | R@5 | chance |
|---|---|---:|---:|---:|
| zero_shot | vision_to_touch | 0.050 | 0.300 | 0.050 |
| zero_shot | touch_to_vision | 0.050 | 0.300 | 0.050 |
| ridge_half_split | vision_to_touch | 0.000 | 0.700 | 0.100 |
| ridge_half_split | touch_to_vision | 0.100 | 0.500 | 0.100 |

- R@1 chance = 1/n = 0.050
- R@5 chance = min(5, n)/n = 0.250
- same-object cosine: 0.0493
- cross-object cosine: 0.0437
- cosine gap: 0.0057
- permutation test: observed R@1 0.050, null mean 0.050, p=0.6623 (1000 perms)

## pss_eval path

| system | property | status | metrics |
|---|---|---|---|
| pss-runtime | determinism | pass | agreement=1.000, final_state_match=1.000, observation_id_match=1.000 |
| pss-runtime | reset_exactness | pass | replay_agreement=1.000, reset_exact=1.000 |
| pss-runtime | state_hygiene | pass | digest_ok=1.000, finite_arrays=1.000, state_bytes=612.000, within_size_bound=1.000 |
| pss-runtime | contract | pass | audit_present=1.000, invalid_keeps_state=1.000, invalid_rejected=1.000, rollback_exact=1.000, swap_accepted=1.000 |
| pss-runtime | swap_rollback | pass | post_rollback_swap=1.000, rollback_exact=1.000, swap_changed_state=1.000, swap_success_rate=1.000 |
| pss-runtime | label_accuracy | skip |  |
| pss-runtime | budget | pass | budget_ms=50.000, elapsed_p50_ms=1.030, elapsed_p99_ms=1.637, errors=0.000, frames=40.000, timeout_rate=0.000 |
| pss-runtime | latency | pass | max_ms=2.008, n=40.000, p50_ms=1.122, p99_ms=1.984 |
| pss-runtime | cross_modal_binding | fail | chance=0.050, chance_r5=0.250, n=20.000, permutation_p=0.289, permutations=200.000, r1=0.100, r5=0.300, touch_to_vision_r1=0.050, touch_to_vision_r5=0.200 |
| objectfolder-passthrough | determinism | pass | agreement=1.000, final_state_match=1.000, observation_id_match=1.000 |
| objectfolder-passthrough | reset_exactness | pass | replay_agreement=1.000, reset_exact=1.000 |
| objectfolder-passthrough | state_hygiene | pass | digest_ok=1.000, finite_arrays=1.000, state_bytes=58.000, within_size_bound=1.000 |
| objectfolder-passthrough | contract | skip |  |
| objectfolder-passthrough | swap_rollback | skip |  |
| objectfolder-passthrough | label_accuracy | skip |  |
| objectfolder-passthrough | budget | skip |  |
| objectfolder-passthrough | latency | pass | max_ms=0.005, n=40.000, p50_ms=0.002, p99_ms=0.005 |
| objectfolder-passthrough | cross_modal_binding | fail | chance=0.050, chance_r5=0.250, n=20.000, permutation_p=0.672, permutations=200.000, r1=0.050, r5=0.300, touch_to_vision_r1=0.050, touch_to_vision_r5=0.300 |
| objectfolder-nn-binding | determinism | pass | agreement=1.000, final_state_match=1.000, observation_id_match=1.000 |
| objectfolder-nn-binding | reset_exactness | pass | replay_agreement=1.000, reset_exact=1.000 |
| objectfolder-nn-binding | state_hygiene | pass | digest_ok=1.000, finite_arrays=1.000, state_bytes=998.000, within_size_bound=1.000 |
| objectfolder-nn-binding | contract | skip |  |
| objectfolder-nn-binding | swap_rollback | skip |  |
| objectfolder-nn-binding | label_accuracy | skip |  |
| objectfolder-nn-binding | budget | skip |  |
| objectfolder-nn-binding | latency | pass | max_ms=0.060, n=40.000, p50_ms=0.025, p99_ms=0.055 |
| objectfolder-nn-binding | cross_modal_binding | fail | chance=0.050, chance_r5=0.250, n=20.000, permutation_p=0.672, permutations=200.000, r1=0.050, r5=0.300, touch_to_vision_r1=0.050, touch_to_vision_r5=0.300 |

## Caveats

- ObjectFolder is a public dataset of implicit object representations; rendered vision/tactile images are not raw robot sensor captures.
- No poses, depth, segmentation, scenes, or synchronized multi-frame stream: L2 identity/tracking and sim2real claims are out of scope.
- The standard identity_retention property is not applicable: with no poses every detection collapses to one track (degenerate pass). It was removed; cross_modal_binding is used instead.
- The PSS runtime's projection here is a random adapter projection, not a projection head trained on real ObjectFolder pairs; its binding R@1 is a path-connectivity measurement, not a PSS capability result.
- Historical 100-object ObjectFolder results (zero-shot R@1 ~chance; trained tactile encoder held-out R@1 ~0.22) are the stronger evidence and should be cited instead of this 20-object spike for capability.
