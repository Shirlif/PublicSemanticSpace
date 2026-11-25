# §10 follow-up experiments

> generated: 2026-11-25 17:18:21 UTC
> design: `notes/CLAIM_GAP_EXPERIMENTS.md`; first batch: `reports/claim_gaps/REPORT.md`

## §10 coverage

| §10 item | experiment | key result | status |
|---|---|---|---|
| 0 non-PSS baseline | E5 flat runtime | PSS SRR/RBE/RCR/PCR/AUD/BAR=1.0; flat 0.000/0.000/0.000/0.000/0.000/0.000 | completed |
| 1 A5 retrain-all | E6 retrain proxy | frozen+alt prototypes 0.964; retrained 0.979 @ 13.5s; 1-shot 0.922 | completed (proxy) |
| 1b full retrain (5ep experts) | E6b retrain with experts | frozen+alt prototypes 0.922; retrained 0.943; total 29.0s | completed (cheap retrain) |
| 1c 30ep full retrain | E6c retrain with 30ep experts | frozen+alt prototypes 0.953; retrained 0.969; total 38.0s | completed |
| 2 L2 cognitive layer | E12 + E18 + E19 + E20 | occlusion full 0.889 / no_memory 0.615 / no_entity 0.213; valid all-bundle full 1.000 / no_memory 0.827 / no_entity 0.270 (19 bundles); S4 no_relations 0/3 | completed (runtime-level; perception-level partial) |
| 2b L2 S4 variants | E18 S4-PSS | full 100% (3 seeds); no_relations 0%; no_memory 100%; no_entity 100% | completed (demo) |
| 2c L2 runtime mid-task | E19 KB update + planner swap | plan changed 100%/100%; baseline 0%/0%; updated 100%/100% | completed |
| 2d L2 runtime cross-layer | E20 occlusion + conflict + audit | no_entity plan=False; no_uncertainty conflict=False; no_audit rollback=False | completed |
| 3 non-oracle detector | E7 detection noise | oracle 0.989 → miss20 0.871 → miss30 0.697 | completed |
| 6 more expert families | E11 three-family swap | zero-shot 0.186; k=1 0.936 | completed |
| 7 real robot | — | not available in this environment | N/A |
| 8 tactile degradation | E15 modality conflict | tactile_zero fixed 0.979; vision_shuffle abstain 0.083/1.000 | completed |
| 9 scale | E8 dim/fact/scene sweep | dim 128/256/512 invariant; 1000 facts update 52.9ms; 500/2000 bounded True | completed |
| 10 TMS production store | E10 TMSFactStore | K3/K4/K5/AUD/source-retract min=1.00 | completed |
| 11 more seeds | E9 L0/L1 | L0 n=10; L1 n=5 | completed |
| C3 RAG baseline | E13 exact-RAG | exact RAG acc 1.000 @ 1.38ms vs PSS 0.0003ms | completed |
| semantic space | E14 space ablation | closed-set invariant 0.969/0.969/0.969; cross-modal R@1 0.510 vs random 0.115 | completed |
| open-set | E16 P2 rerun | PSS AUROC 0.993 / reject 0.958; energy 0.937/0.753 | completed |

## E5: contract-free baseline

| metric | PSS | flat |
|---|---:|---:|
| SSR | 1.000 | 1.000 |
| SRR | 1.000 | 0.000 |
| INT | 0.000 | 0.667 |
| RBE | 1.000 | 0.000 |
| RCR | 1.000 | 0.000 |
| USR | 1.000 | 1.000 |
| LOC | 0.000 | 0.000 |
| PCR | 1.000 | 0.000 |
| AUD | 1.000 | 0.000 |
| RBE-KB | 1.000 | 0.000 |
| BAR | 1.000 | 0.000 |
| MON | 1.000 | 0.000 |

## E6: retrain-all proxy

| method | vision ZCR | tactile ZCR | cost |
|---|---:|---:|---:|
| frozen + alt prototypes | 0.964 | 0.234 | 3.18 ms |
| 1-shot ridge | 0.922 | 0.208 | 107.87 ms |
| retrained projection + prototypes | 0.979 | 0.234 | 13.47 s |

## E6b: retrain with 5-epoch experts

| method | vision ZCR | tactile ZCR | cost |
|---|---:|---:|---:|
| frozen + alt prototypes | 0.922 | 0.161 | 4.37 ms |
| 1-shot ridge | 0.766 | 0.141 | 106.97 ms |
| retrained projection + prototypes | 0.943 | 0.182 | 13.50 s (+ expert 15.50 s) |

The 5-epoch expert retrain is weaker than the pretrained expert, so this is a
cheap-retrain lower bound. Its point is cost: tens of seconds for experts plus
projections, versus milliseconds for rebuilding prototypes on the frozen expert.

## E6c: retrain with 30-epoch experts

| method | vision ZCR | tactile ZCR | cost |
|---|---:|---:|---:|
| frozen + alt prototypes | 0.953 | 0.234 | 2.87 ms |
| 1-shot ridge | 0.927 | 0.208 | 109.51 ms |
| retrained projection + prototypes | 0.969 | 0.224 | 13.41 s (+ expert 24.60 s) |

Even at 30 epochs, retraining the expert does not beat keeping the pretrained
expert and retraining only the projection: expert retraining adds ~25 s and
loses the quality edge. Prototype rebuild remains the 3 ms baseline.

## E7: detection noise

| condition | miss | sigma | ID retention | switches |
|---|---:|---:|---:|---:|
| oracle | 0.00 | 0.000 | 0.989 | 0.02 |
| miss10_sigma2cm | 0.10 | 0.020 | 0.975 | 0.06 |
| miss20_sigma5cm | 0.20 | 0.050 | 0.871 | 0.58 |
| miss30_sigma8cm | 0.30 | 0.080 | 0.697 | 2.06 |

## E8: scale

| dim | SSR | RBE | INT |
|---:|---:|---:|---:|
| 128 | 1.000 | 1.000 | 0.000 |
| 256 | 1.000 | 1.000 | 0.000 |
| 512 | 1.000 | 1.000 | 0.000 |

| facts | update ms | rollback |
|---:|---:|---:|
| 100 | 5.1 | 1 |
| 500 | 30.5 | 1 |
| 1000 | 52.9 | 1 |

| entities/relations | active | bounded | latency ms |
|---|---:|---|---:|
| 50/100 | 50/100 | True | 0.004 |
| 200/1000 | 50/100 | True | 0.025 |
| 500/2000 | 50/100 | True | 0.036 |

## E10: TMS on the production fact store

| property | mean | min |
|---|---:|---:|
| K3_propagation | 1.000 | 1.000 |
| K4_conflict | 1.000 | 1.000 |
| K4_resolution | 1.000 | 1.000 |
| K5_rollback | 1.000 | 1.000 |
| K5_label_restored | 1.000 | 1.000 |
| AUD | 1.000 | 1.000 |
| source_retract | 1.000 | 1.000 |

## E11: expert families

| family | vision ZCR | tactile ZCR | vision k=1 |
|---|---:|---:|---:|
| category_pretrain | 0.200 | 0.067 | 1.000 |
| category_pretrain_instance | 0.167 | 0.067 | 0.900 |
| category_pretrain_instance8 | 0.192 | 0.100 | 0.908 |

## E12: L2 bundle replay

| variant | ID retention | switches | p95 ms |
|---|---:|---:|---:|
| full | 0.889 | 0.00 | 5.002 |
| no_memory | 0.615 | 0.96 | 5.431 |
| no_entity | 0.213 | 60.19 | 2.293 |
| no_relations | 0.889 | 0.00 | 5.554 |

## E13: RAG with oracle retrieval

| history | full | learned RAG | exact RAG | exact p50 ms | PSS p50 ms |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.234 | 0.492 | 1.000 | 1.30 | 0.0003 |
| 100 | 0.086 | 0.398 | 1.000 | 1.36 | 0.0003 |
| 500 | 0.039 | 0.359 | 1.000 | 1.44 | 0.0003 |
| 1000 | 0.078 | 0.375 | 1.000 | 1.38 | 0.0003 |

## E14: semantic-space ablation

| variant | accuracy | cross-modal R@1 |
|---|---:|---:|
| full per-modality | 0.969 | 0.510 |
| cross-modal sum | 0.969 | - |
| random projection | 0.969 | 0.115 |

Closed-set accuracy is invariant because the bank is re-fit in each space;
the learned alignment shows up in cross-modal R@1. Claim alignment, not
closed-set accuracy.

## E15: modality conflict

| condition | PSS fixed | PSS abstain cov/acc | transformer |
|---|---:|---:|---:|
| clean | 0.979 | 0.854/0.982 | 0.976 |
| tactile_zero | 0.979 | 0.854/0.982 | 0.964 |
| vision_shuffle | 0.474 | 0.083/1.000 | 0.153 |
| geometry_shuffle | 0.458 | 0.068/0.923 | 0.734 |

## E16: open-set

| method | AUROC | unseen reject | seen top-1 |
|---|---:|---:|---:|
| PSS prototypes | 0.993 | 0.958 | 0.979 |
| transformer softmax | 0.885 | 0.514 | 0.976 |
| transformer energy | 0.937 | 0.753 | 0.976 |

## E9: more seeds

| suite | metric | n | mean | std | min |
|---|---|---:|---:|---:|---:|
| L0 | SSR | 10 | 1.000 | 0.000 | 1.000 |
| L0 | INT | 10 | 0.000 | 0.000 | 0.000 |
| L0 | RBE | 10 | 1.000 | 0.000 | 1.000 |
| L0 | RCR | 10 | 1.000 | 0.000 | 1.000 |
| L0 | BAR | 10 | 1.000 | 0.000 | 1.000 |
| L0 | MON | 10 | 1.000 | 0.000 | 1.000 |
| L0 | ZCR | 10 | 0.668 | 0.004 | 0.663 |
| L1 | SSR_L1 | 5 | 1.000 | 0.000 | 1.000 |
| L1 | INT_L1 | 5 | 0.000 | 0.000 | 0.000 |
| L1 | RBE_L1 | 5 | 1.000 | 0.000 | 1.000 |
| L1 | RCR_L1 | 5 | 1.000 | 0.000 | 1.000 |
| L1 | ZCR_L1 | 5 | 0.671 | 0.014 | 0.650 |
| L1 | ZCR_ratio_mean | 5 | 1.060 | 0.034 | 1.012 |
| L1 | swap_latency_ms_L1 | 5 | 0.115 | 0.008 | 0.101 |

## E17: runtime permissions and audit

| profile | allowed actions | denied (audited) |
|---|---:|---:|
| read_only | 1/6 | 5 |
| service | 2/6 | 4 |
| operator | 5/6 | 1 |
| admin | 6/6 | 0 |

## E18: L2 S4-PSS variants

| variant | seeds | success rate | wall mean s |
|---|---:|---:|---:|
| full | 3 | 100% | 2.04 |
| no_entity | 3 | 100% | 1.92 |
| no_memory | 3 | 100% | 2.39 |
| no_relations | 3 | 0% | 1.19 |

`no_relations` fails because the pick-place goal is a relation; `no_memory` and
`no_entity` succeed because this demo has no occlusion. The demo tests relations,
not memory. E12 supplies the memory ablation on bundles.

## E19: L2 runtime mid-task updates

| scenario | plan changed | baseline goal | updated goal |
|---|---:|---:|---:|
| kb_update | 100% | 0% | 100% |
| planner_swap | 100% | 0% | 100% |

`kb_update`: goal changes G -> B mid-task. `planner_swap`: direct route is
blocked and `runtime.planner` is swapped to a detour library. In both cases
the old plan fails and the replanned one succeeds without restarting.

## E20: L2 runtime cross-layer

| variant | entity persistence | plan | task | conflict | rollback |
|---|---:|---:|---:|---:|---:|
| full | True | True | True | True | True |
| no_entity | False | False | False | False | True |
| no_relations | True | True | True | True | True |
| no_uncertainty | True | True | True | False | True |
| no_audit | True | True | True | True | False |
| no_language | True | True | True | True | True |

`no_entity` drops the occluded bottle and cannot plan the goal;
`no_uncertainty` does not flag the conflicting fact; `no_audit` cannot
rollback. This is the runtime-level occlusion/conflict/audit evidence.

## E12b: all-bundle replay

| variant | ID retention | switches |
|---|---:|---:|
| full | 0.576 | 0.00 |
| no_memory | 0.476 | 0.32 |
| no_entity | 0.156 | 27.33 |
| no_relations | 0.576 | 0.00 |

## Files

- `reports/claim_gaps/nonpss/`
- `reports/claim_gaps/scale/`
- `reports/claim_gaps/detect_noise/`
- `reports/claim_gaps/families/`
- `reports/claim_gaps/retrain_all/`
- `reports/claim_gaps/tms/`
- `reports/claim_gaps/l2_memory/`
- `reports/claim_gaps/rag_exact/`
- `reports/claim_gaps/modality_conflict/`
- `reports/claim_gaps/openset/`
- `reports/claim_gaps/space_ablation/`
- `reports/claim_gaps/runtime_audit/`
- `reports/claim_gaps/runtime_cross_layer/`
- `reports/claim_gaps/l2_runtime_updates/`
