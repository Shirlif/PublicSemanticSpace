# Claim-gap F8: learned relation gating + OOD degradation

> generated: 2026-11-26 09:53:02 UTC
> train regions: ['A', 'B', 'C']; test regions: ['D', 'G']; held-out pairs: 14
> noise levels: ['0.0', '0.03', '0.08', '0.15']; tau=0.25, strict eps=0.1; seeds=3, n_train/pair=200, n_test/pair=250

## TL;DR

- Acceptance bar (`deployed gate > geometry(0.25)` at some noise): **True**.
- No-fallback binary gate beats `geometry(0.25)`: **False**; beats `hybrid_diag`: **False**.
- No-fallback multi-expert gate beats `geometry(0.25)`: **False**; beats `hybrid_diag`: **False**.
- Any no-fallback learned gate beats the tuned fixed geometric dead zone: **False**.
- OOD fallback active on unseen D/G: **True** (fallback rate at noise 0.15: 0.943).
- Noise-feature ablation removes the binary gate's gain: **False**.
- Noise 0.15 headline: geo0.25 0.829, hybrid_diag 0.859, tuned fixed geo 0.972; binary gate 0.614 (+OOD 0.830), multi gate 0.644 (+OOD 0.969).
- **Verdict**: Acceptance is met numerically only by the deployed multi gate: at noise 0.15 it reaches 0.969 vs geometry(0.25) 0.829 and hybrid_diag 0.859, but this is within 0.003 of the tuned fixed dead zone (0.972) and does not exceed it. The deployed binary gate is 0.830 (+0.001 over geo0.25), essentially geometry(0.25). No no-fallback learned gate beats geo0.25 (binary 0.614, multi 0.644 vs 0.829). The gate fits A/B/C validation almost perfectly but does not transfer to D/G because the learned head is OOD-broken on vertical-aligned / unseen-coordinate pairs. The OOD fallback is necessary for safety, but the only safe performance gain is a wider/adaptive geometric dead zone. E3's hybrid_diag advantage at tau=0.25 is an under-tuned-dead-zone artifact.

## 1. Main results (A/B/C -> D/G)

| method | noise=0.0 | noise=0.03 | noise=0.08 | noise=0.15 |
|---|---:|---:|---:|---:|
| geometry(0.10) | 1.000 ± 0.000 | 0.987 ± 0.003 | 0.733 ± 0.003 | 0.541 ± 0.003 |
| geometry(0.25) [E3 strict] | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.981 ± 0.002 | 0.829 ± 0.001 |
| learned head | 0.571 ± 0.000 | 0.585 ± 0.019 | 0.573 ± 0.005 | 0.536 ± 0.003 |
| hybrid [E3] | 0.571 ± 0.000 | 0.585 ± 0.019 | 0.568 ± 0.005 | 0.506 ± 0.004 |
| hybrid_diag [E3] | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.986 ± 0.002 | 0.859 ± 0.002 |
| gate (geo0.25 vs learned) | 0.738 ± 0.121 | 0.757 ± 0.102 | 0.694 ± 0.081 | 0.614 ± 0.054 |
| gate binary + OOD fallback | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.981 ± 0.002 | 0.830 ± 0.001 |
| gate (eps grid + learned) | 0.810 ± 0.178 | 0.825 ± 0.158 | 0.740 ± 0.113 | 0.644 ± 0.095 |
| gate multi + OOD fallback | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.998 ± 0.001 | 0.969 ± 0.002 |
| gate, no noise features | 0.619 ± 0.067 | 0.626 ± 0.052 | 0.604 ± 0.041 | 0.558 ± 0.029 |
| gate no-noise + OOD fallback | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.981 ± 0.002 | 0.830 ± 0.001 |
| tuned fixed geo (val-selected) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.972 ± 0.001 |
| best fixed geo (test oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.976 ± 0.003 |
| oracle(geo0.25, learned) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.986 ± 0.002 | 0.866 ± 0.005 |
| oracle(eps grid, learned) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.000 |

`best_fixed_geo_val` is the fixed `geometry(eps)` with the best A/B/C
validation accuracy per noise; it is selected without touching D/G.
`best_fixed_geo_test` is the best fixed eps on the test set itself and is
only an upper bound for any fixed dead zone.

### Paired comparisons (pooled samples, 95% CI)

| noise | comparison | diff | 95% CI |
|---:|---|---:|---|
| 0.0 | gate_binary_vs_geometry_eps0.25 | -0.2619 | [-0.2703, -0.2535] |
| 0.0 | gate_binary_vs_hybrid_diag | -0.2619 | [-0.2703, -0.2535] |
| 0.0 | gate_binary_vs_best_fixed_geo_val | -0.2619 | [-0.2703, -0.2535] |
| 0.0 | gate_binary_vs_best_fixed_geo_test | -0.2619 | [-0.2703, -0.2535] |
| 0.0 | gate_binary_vs_gate_ablation | +0.1190 | [+0.1116, +0.1265] |
| 0.0 | gate_multi_vs_geometry_eps0.25 | -0.1905 | [-0.1980, -0.1830] |
| 0.0 | gate_multi_vs_hybrid_diag | -0.1905 | [-0.1980, -0.1830] |
| 0.0 | gate_multi_vs_best_fixed_geo_val | -0.1905 | [-0.1980, -0.1830] |
| 0.0 | gate_multi_vs_best_fixed_geo_test | -0.1905 | [-0.1980, -0.1830] |
| 0.0 | gate_ablation_vs_geometry_eps0.25 | -0.3810 | [-0.3902, -0.3717] |
| 0.0 | gate_ablation_vs_best_fixed_geo_val | -0.3810 | [-0.3902, -0.3717] |
| 0.03 | gate_binary_vs_geometry_eps0.25 | -0.2431 | [-0.2513, -0.2349] |
| 0.03 | gate_binary_vs_hybrid_diag | -0.2431 | [-0.2513, -0.2349] |
| 0.03 | gate_binary_vs_best_fixed_geo_val | -0.2430 | [-0.2513, -0.2348] |
| 0.03 | gate_binary_vs_best_fixed_geo_test | -0.2431 | [-0.2513, -0.2349] |
| 0.03 | gate_binary_vs_gate_ablation | +0.1304 | [+0.1229, +0.1379] |
| 0.03 | gate_multi_vs_geometry_eps0.25 | -0.1751 | [-0.1824, -0.1679] |
| 0.03 | gate_multi_vs_hybrid_diag | -0.1751 | [-0.1824, -0.1679] |
| 0.03 | gate_multi_vs_best_fixed_geo_val | -0.1750 | [-0.1823, -0.1678] |
| 0.03 | gate_multi_vs_best_fixed_geo_test | -0.1751 | [-0.1824, -0.1679] |
| 0.03 | gate_ablation_vs_geometry_eps0.25 | -0.3735 | [-0.3828, -0.3643] |
| 0.03 | gate_ablation_vs_best_fixed_geo_val | -0.3734 | [-0.3827, -0.3642] |
| 0.08 | gate_binary_vs_geometry_eps0.25 | -0.2875 | [-0.2964, -0.2787] |
| 0.08 | gate_binary_vs_hybrid_diag | -0.2923 | [-0.3010, -0.2836] |
| 0.08 | gate_binary_vs_best_fixed_geo_val | -0.3054 | [-0.3142, -0.2966] |
| 0.08 | gate_binary_vs_best_fixed_geo_test | -0.3061 | [-0.3149, -0.2973] |
| 0.08 | gate_binary_vs_gate_ablation | +0.0902 | [+0.0836, +0.0968] |
| 0.08 | gate_multi_vs_geometry_eps0.25 | -0.2410 | [-0.2495, -0.2326] |
| 0.08 | gate_multi_vs_hybrid_diag | -0.2458 | [-0.2542, -0.2375] |
| 0.08 | gate_multi_vs_best_fixed_geo_val | -0.2590 | [-0.2673, -0.2506] |
| 0.08 | gate_multi_vs_best_fixed_geo_test | -0.2596 | [-0.2680, -0.2512] |
| 0.08 | gate_ablation_vs_geometry_eps0.25 | -0.3777 | [-0.3872, -0.3682] |
| 0.08 | gate_ablation_vs_best_fixed_geo_val | -0.3956 | [-0.4050, -0.3863] |
| 0.15 | gate_binary_vs_geometry_eps0.25 | -0.2148 | [-0.2241, -0.2055] |
| 0.15 | gate_binary_vs_hybrid_diag | -0.2451 | [-0.2535, -0.2368] |
| 0.15 | gate_binary_vs_best_fixed_geo_val | -0.3579 | [-0.3673, -0.3485] |
| 0.15 | gate_binary_vs_best_fixed_geo_test | -0.3619 | [-0.3714, -0.3524] |
| 0.15 | gate_binary_vs_gate_ablation | +0.0561 | [+0.0505, +0.0617] |
| 0.15 | gate_multi_vs_geometry_eps0.25 | -0.1844 | [-0.1936, -0.1752] |
| 0.15 | gate_multi_vs_hybrid_diag | -0.2148 | [-0.2235, -0.2061] |
| 0.15 | gate_multi_vs_best_fixed_geo_val | -0.3275 | [-0.3367, -0.3183] |
| 0.15 | gate_multi_vs_best_fixed_geo_test | -0.3315 | [-0.3408, -0.3222] |
| 0.15 | gate_ablation_vs_geometry_eps0.25 | -0.2709 | [-0.2808, -0.2609] |
| 0.15 | gate_ablation_vs_best_fixed_geo_val | -0.4140 | [-0.4236, -0.4044] |

## 2. Key control: is it just a wider dead zone?

Geometry-only accuracy for a sweep of fixed dead zones:

| eps | noise=0.0 | noise=0.03 | noise=0.08 | noise=0.15 |
|---:|---:|---:|---:|---:|
| 0.10 | 1.000 ± 0.000 | 0.987 ± 0.003 | 0.733 ± 0.003 | 0.541 ± 0.003 |
| 0.15 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.863 ± 0.003 | 0.654 ± 0.003 |
| 0.20 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.944 ± 0.000 | 0.751 ± 0.005 |
| 0.25 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.981 ± 0.002 | 0.829 ± 0.001 |
| 0.30 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.995 ± 0.002 | 0.887 ± 0.002 |
| 0.35 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.998 ± 0.000 | 0.930 ± 0.004 |
| 0.40 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.954 ± 0.003 |
| 0.45 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.969 ± 0.003 |
| 0.50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.976 ± 0.003 |
| 0.60 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.965 ± 0.001 |
| 0.75 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.983 ± 0.002 | 0.863 ± 0.001 |

| noise | val-selected eps | val acc | test acc of that eps | test-oracle eps |
|---:|---:|---:|---:|---:|
| 0.0 | 0.10 | 1.000 | 1.000 | 0.10 (1.000) |
| 0.03 | 0.15 | 1.000 | 1.000 | 0.20 (1.000) |
| 0.08 | 0.35 | 1.000 | 0.999 | 0.45 (1.000) |
| 0.15 | 0.45 | 0.981 | 0.972 | 0.50 (0.976) |

If a fixed `geometry(eps)` tuned on A/B/C matches or beats the gate,
the learned gate is not adding calibration beyond a wider dead zone.

## 3. OOD degradation and fallback

| noise | fallback rate | feature-OOD rate | region-OOD rate | fallback acc | non-fallback acc | gate no-fallback acc |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.000 | 0.714 | 0.571 | 1.000 | nan | 0.738 |
| 0.03 | 0.969 | 0.687 | 0.571 | 1.000 | 1.000 | 0.757 |
| 0.08 | 0.943 | 0.689 | 0.571 | 0.980 | 1.000 | 0.694 |
| 0.15 | 0.943 | 0.739 | 0.577 | 0.823 | 0.947 | 0.614 |

Per novel-region breakdown at noise 0.15:

| region | n | fallback rate | gate+OOD acc | gate no-fallback acc | geo0.25 acc |
|---|---:|---:|---:|---:|---:|
| D | 4500 | 1.000 | 0.918 | 0.713 | 0.918 |
| D+G | 1500 | 1.000 | 0.754 | 0.200 | 0.754 |
| G | 4500 | 0.866 | 0.768 | 0.653 | 0.764 |

### Expert selection (multi-expert gate, deployed choice)

| expert | noise=0.0 | noise=0.03 | noise=0.08 | noise=0.15 |
|---|---:|---:|---:|---:|
| geometry_eps0.1 | 0.000 | 0.004 | 0.005 | 0.003 |
| geometry_eps0.15 | 0.000 | 0.005 | 0.006 | 0.004 |
| geometry_eps0.2 | 0.000 | 0.005 | 0.008 | 0.002 |
| geometry_eps0.25 | 0.000 | 0.010 | 0.011 | 0.005 |
| geometry_eps0.3 | 0.000 | 0.000 | 0.000 | 0.000 |
| geometry_eps0.35 | 0.000 | 0.000 | 0.000 | 0.003 |
| geometry_eps0.4 | 0.000 | 0.006 | 0.010 | 0.008 |
| geometry_eps0.45 | 0.667 | 0.650 | 0.632 | 0.632 |
| geometry_eps0.5 | 0.333 | 0.319 | 0.313 | 0.316 |
| geometry_eps0.6 | 0.000 | 0.000 | 0.000 | 0.000 |
| geometry_eps0.75 | 0.000 | 0.000 | 0.000 | 0.000 |
| learned | 0.000 | 0.000 | 0.015 | 0.025 |

### Expert selection (multi-expert gate, no fallback)

| expert | noise=0.0 | noise=0.03 | noise=0.08 | noise=0.15 |
|---|---:|---:|---:|---:|
| geometry_eps0.1 | 0.167 | 0.177 | 0.196 | 0.223 |
| geometry_eps0.15 | 0.095 | 0.098 | 0.094 | 0.089 |
| geometry_eps0.2 | 0.095 | 0.143 | 0.110 | 0.079 |
| geometry_eps0.25 | 0.190 | 0.158 | 0.118 | 0.087 |
| geometry_eps0.3 | 0.119 | 0.105 | 0.108 | 0.104 |
| geometry_eps0.35 | 0.000 | 0.000 | 0.001 | 0.007 |
| geometry_eps0.4 | 0.071 | 0.086 | 0.093 | 0.082 |
| geometry_eps0.45 | 0.048 | 0.000 | 0.000 | 0.001 |
| geometry_eps0.5 | 0.000 | 0.005 | 0.009 | 0.013 |
| geometry_eps0.6 | 0.024 | 0.022 | 0.007 | 0.001 |
| geometry_eps0.75 | 0.000 | 0.000 | 0.000 | 0.000 |
| learned | 0.190 | 0.208 | 0.263 | 0.315 |

The learned expert's selection rate and the dominant geometry eps are the
direct evidence for (or against) a wider-dead-zone explanation.  The
deployed table is dominated by the OOD fallback; the no-fallback table
shows what the learned gate itself routes to.

## 4. Ablation: remove local-noise features

| noise | full gate | no-noise gate | delta (full - no-noise) | 95% CI |
|---:|---:|---:|---:|---|
| 0.0 | 0.738 | 0.619 | +0.1190 | [+0.1116, +0.1265] |
| 0.03 | 0.757 | 0.626 | +0.1304 | [+0.1229, +0.1379] |
| 0.08 | 0.694 | 0.604 | +0.0902 | [+0.0836, +0.0968] |
| 0.15 | 0.614 | 0.558 | +0.0561 | [+0.0505, +0.0617] |

If the full gate's advantage over `geometry(0.25)` disappears when the
noise features are removed, the gain comes from noise-adaptive routing;
if it persists, it comes from the margin/learned-score features.

## 5. Anti-overfitting: leave-one-region-out CV

Gate validation accuracy on held-out A/B/C pairs (3-fold pair CV): binary 1.000, multi 0.998, ablation 0.998. This is in-distribution accuracy; it does not transfer to D/G (see the main table).

(CV disabled in this run.)

## 6. Verdict and limitations

**Verdict**: Acceptance is met numerically only by the deployed multi gate: at noise 0.15 it reaches 0.969 vs geometry(0.25) 0.829 and hybrid_diag 0.859, but this is within 0.003 of the tuned fixed dead zone (0.972) and does not exceed it. The deployed binary gate is 0.830 (+0.001 over geo0.25), essentially geometry(0.25). No no-fallback learned gate beats geo0.25 (binary 0.614, multi 0.644 vs 0.829). The gate fits A/B/C validation almost perfectly but does not transfer to D/G because the learned head is OOD-broken on vertical-aligned / unseen-coordinate pairs. The OOD fallback is necessary for safety, but the only safe performance gain is a wider/adaptive geometric dead zone. E3's hybrid_diag advantage at tau=0.25 is an under-tuned-dead-zone artifact.

Limitations:
- Synthetic five-region grid; only 5 regions and 14 primary held-out pairs.
- The learned head is E3's architecture (absolute-coordinate features), so
  it is OOD-broken on vertical-aligned pairs such as C<->G and D<->C; this
  is a property of the head, not of the gate.
- The grid bench has no category or tactile modality, so the contact/near
  gate features are constant and cannot contribute.
- The OOD detector combines a feature-space novelty score with the
  distance outside the convex hull of the training regions.  D is an
  extrapolation (outside the A/B/C hull) and falls back to geometry; G is
  an interpolation (inside the hull), so the gate may still route there.
  The feature detector catches some relation-level OOD cases (e.g. C<->G).

Recommendations:
- Tune the geometric dead zone to the local noise level (a fixed eps ~0.5
  at sigma ~0.15 already dominates the E3 hybrid at tau=0.25).
- If a learned head is kept, train it on vertical-aligned pairs and mixed
  noise, and use a translation-invariant feature set.
- Treat learned gating as a safety layer only after the head and the
  geometric dead zone are properly calibrated; otherwise it is a wider
  dead zone in learned clothing.
