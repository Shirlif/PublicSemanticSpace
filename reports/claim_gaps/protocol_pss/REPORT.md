# pss_eval: property report

> seeds: 0, 1, 2; stream: n=400, dim=384; bundles: s1_hard/37f0c8b8/seed0, s1_occlusion_real/16ea9f38/seed0, s2_twins/3d49ce1b/seed0, s1_camera_real/b4bb5f0e/seed0

## Aggregate

| system | property | smoke | status | metrics |
|---|---|---|---|---|
| e5-flat | determinism | True | pass | agreement=1.000, final_state_match=1.000, observation_id_match=1.000 |
| e5-flat | reset_exactness | True | pass | replay_agreement=1.000, reset_exact=1.000 |
| e5-flat | state_hygiene | True | pass | digest_ok=1.000, finite_arrays=1.000, state_bytes=5191.500, within_size_bound=1.000 |
| e5-flat | contract | False | fail | audit_present=0.000, invalid_keeps_state=0.000, invalid_rejected=0.000, rollback_exact=0.000, swap_accepted=1.000 |
| e5-flat | swap_rollback | False | fail | post_rollback_swap=1.000, rollback_exact=0.000, swap_changed_state=1.000, swap_success_rate=1.000 |
| e5-flat | identity_retention | False | skip |  |
| e5-flat | label_accuracy | False | skip |  |
| e5-flat | budget | False | skip |  |
| e5-flat | latency | False | pass | max_ms=0.050, n=100.000, p50_ms=0.016, p99_ms=0.033 |
| transformer-state | determinism | True | pass | agreement=1.000, final_state_match=1.000, observation_id_match=1.000 |
| transformer-state | reset_exactness | True | pass | replay_agreement=1.000, reset_exact=1.000 |
| transformer-state | state_hygiene | True | pass | digest_ok=1.000, finite_arrays=1.000, state_bytes=622.000, within_size_bound=1.000 |
| transformer-state | contract | False | skip |  |
| transformer-state | swap_rollback | False | skip |  |
| transformer-state | identity_retention | False | skip |  |
| transformer-state | label_accuracy | False | skip |  |
| transformer-state | budget | False | skip |  |
| transformer-state | latency | False | pass | max_ms=0.090, n=100.000, p50_ms=0.035, p99_ms=0.070 |
| pss-runtime | determinism | True | pass | agreement=1.000, final_state_match=1.000, observation_id_match=1.000 |
| pss-runtime | reset_exactness | True | pass | replay_agreement=1.000, reset_exact=1.000 |
| pss-runtime | state_hygiene | True | pass | digest_ok=1.000, finite_arrays=1.000, state_bytes=3486.917, within_size_bound=1.000 |
| pss-runtime | contract | False | pass | audit_present=1.000, invalid_keeps_state=1.000, invalid_rejected=1.000, rollback_exact=1.000, swap_accepted=1.000 |
| pss-runtime | swap_rollback | False | pass | post_rollback_swap=1.000, rollback_exact=1.000, swap_changed_state=1.000, swap_success_rate=1.000 |
| pss-runtime | identity_retention | False | pass | identity_retention=0.776, instance_purity=0.784, n_detections=427.000, n_instances=6.000, threshold=0.500 |
| pss-runtime | label_accuracy | False | skip |  |
| pss-runtime | budget | False | pass | budget_ms=50.000, elapsed_p50_ms=1.198, elapsed_p99_ms=2.467, errors=0.000, frames=100.000, timeout_rate=0.003 |
| pss-runtime | latency | False | pass | max_ms=2.846, n=100.000, p50_ms=1.360, p99_ms=2.278 |

## Per-bundle statuses

| system | property | s1_hard/37f0c8b8/seed0 | s1_occlusion_real/16ea9f38/seed0 | s2_twins/3d49ce1b/seed0 | s1_camera_real/b4bb5f0e/seed0 |
|---|---|---|---|---|---|
| e5-flat | determinism | pass | pass | pass | pass |
| e5-flat | reset_exactness | pass | pass | pass | pass |
| e5-flat | state_hygiene | pass | pass | pass | pass |
| e5-flat | contract | fail | fail | fail | fail |
| e5-flat | swap_rollback | fail | fail | fail | fail |
| e5-flat | identity_retention | skip | skip | skip | skip |
| e5-flat | label_accuracy | skip | skip | skip | skip |
| e5-flat | budget | skip | skip | skip | skip |
| e5-flat | latency | pass | pass | pass | pass |
| transformer-state | determinism | pass | pass | pass | pass |
| transformer-state | reset_exactness | pass | pass | pass | pass |
| transformer-state | state_hygiene | pass | pass | pass | pass |
| transformer-state | contract | skip | skip | skip | skip |
| transformer-state | swap_rollback | skip | skip | skip | skip |
| transformer-state | identity_retention | skip | skip | skip | skip |
| transformer-state | label_accuracy | skip | skip | skip | skip |
| transformer-state | budget | skip | skip | skip | skip |
| transformer-state | latency | pass | pass | pass | pass |
| pss-runtime | determinism | pass | pass | pass | pass |
| pss-runtime | reset_exactness | pass | pass | pass | pass |
| pss-runtime | state_hygiene | pass | pass | pass | pass |
| pss-runtime | contract | pass | pass | pass | pass |
| pss-runtime | swap_rollback | pass | pass | pass | pass |
| pss-runtime | identity_retention | pass | pass | pass | pass |
| pss-runtime | label_accuracy | skip | skip | skip | skip |
| pss-runtime | budget | pass | pass | pass | pass |
| pss-runtime | latency | pass | pass | pass | pass |

## Pre-registration

ok: **True**

Smoke properties are protocol hygiene checks.  `contract` and
`swap_rollback` are expected to fail or be skipped for contract-free
external systems; `budget` / `identity_retention` skip when the system
or stream lacks the required capability.
