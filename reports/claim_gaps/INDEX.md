# Claim-gap experiment index

> Generated 2026-11-26. Design and verdicts: `notes/CLAIM_GAP_EXPERIMENTS.md`.
> First batch report: `reports/claim_gaps/REPORT.md`; §10 follow-ups: `reports/claim_gaps/FOLLOWUPS.md`.

| id | question | key result | report |
|---|---|---|---|
| E1 | Is entity state auditable? | provenance/retraction/rollback/rebind/tamper all 1.00; 3 negative controls 0.00 | `entity_audit/REPORT.md` |
| E2 | How much of binding is tactile? | 70-frame retention 0.989; tactile +0.117, graph +0.072, state +0.489 | `tactile_binding/REPORT.md` |
| E3 | When does a learned relation hybrid help? | hybrid_diag 0.857 vs geo_tau 0.828 at noise 0.15; naive hybrid 0.503 | `hybrid_relations/REPORT.md` |
| E4 | Is long-history fact selection really unsolved? | query-conditioned k=4 0.241 → 1.000; simple recency rule ties learned | `fact_active/REPORT.md` |
| E5 | Does the suite separate a contract-free architecture? | PSS passes 9 contract metrics; flat fails 8; INT 0.667 | `nonpss/REPORT.md` |
| E6 | What does retrain-all cost? | prototype rebuild 3.2 ms/0.964; projection retrain 13.5 s/0.979 | `retrain_all/REPORT.md` |
| E6b | Retrain experts (5 epochs) + projections? | 29 s, 0.943 | `retrain_all_experts/REPORT.md` |
| E6c | Retrain experts (30 epochs) + projections? | 38 s, 0.969; still below frozen expert + projection retrain 0.979 | `retrain_all_experts_30ep/REPORT.md` |
| E7 | Does binding survive non-oracle detection? | 0.989 → 0.975 (10% miss) → 0.871 (20%) → 0.697 (30%) | `detect_noise/REPORT.md` |
| E8 | Do contract metrics survive scale? | dim 128/256/512 invariant; 1000 facts update 52.9 ms, rollback 1; 500/2000 bounded | `scale/REPORT.md` |
| E9 | Are results seed-stable? | L0 10 seeds invariant; L1 5 seeds SSR/RBE/INT/RCR=1.0, ZCR 0.671±0.014 | `scorecards/property/` |
| E10 | Is TMS on the production fact store? | K3/K4/K5/AUD/source-retract all 1.00 | `tms/REPORT.md` |
| E11 | Does the contract hold across expert families? | 3 families zero-shot 0.186; k=1 0.936; tactile 0.067–0.100 | `families/REPORT.md` |
| E12 | Does the L2 runtime depend on its memory layer? | valid bundles: full 1.000 / no_memory 0.827 / no_entity 0.270 | `l2_memory/REPORT.md`, `l2_memory_all/REPORT.md` |
| E13 | Is the active-set cost advantage robust to a strong RAG? | exact-RAG 1.000 at 1.38 ms vs PSS 0.0003 ms; learned-RAG 0.375 is a lower bound | `rag_exact/REPORT.md` |
| E14 | What does per-modality projection buy? | closed-set invariant 0.969; cross-modal R@1 0.510 vs random 0.115 | `space_ablation/REPORT.md` |
| E15 | Is tactile degradation handled explicitly? | tactile_zero fixed 0.979; vision_shuffle abstain 0.083/1.000 | `modality_conflict/modality_conflict.md` |
| E16 | Does open-set rejection still hold? | PSS AUROC 0.993 / reject 0.958 vs energy 0.937/0.753 | `openset/p2_openset.md` |
| E17 | Are runtime permissions audited? | read_only denies 5 actions; admin 0; denials audited | `runtime_audit/` |
| E18 | Which cognitive layer does the L2 demo exercise? | S4 `full`/`no_memory`/`no_entity` 3/3, `no_relations` 0/3 | `FOLLOWUPS.md` |
| E19 | Can the runtime update knowledge / swap a module mid-task? | KB update and planner swap both replan 100%, baseline 0% | `l2_runtime_updates/REPORT.md` |
| E20 | Does the runtime cross-layer depend on its components? | `no_entity` cannot plan; `no_uncertainty` misses conflict; `no_audit` cannot rollback | `runtime_cross_layer/runtime.md` |

## Reading guide

- **Strongest new evidence for C1**: E1 (audit) + E2 (signal decomposition) + E12 (L2 memory ablation) + E7 (detection noise) + E20 (occlusion at runtime).
- **Strongest new evidence for C2**: E3 (distribution-aware hybrid; the naive hybrid is a negative control).
- **Strongest new evidence for C3**: E4 (query conditioning) + E13 (exact-RAG cost) + E8 (1000 facts).
- **Strongest new evidence for C4**: E5 (contract-free separation) + E8 (scale) + E10 (production TMS) + E17 (permissions) + E19 (mid-task update/swap).
- **Honest negatives**: E6c (expert retrain does not beat projection retrain), E14 (closed-set invariant), E18 (S4 does not test memory), E6/E11 (tactile remains weak).
