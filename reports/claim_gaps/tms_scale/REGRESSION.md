# F7 regression: runtime TMS migration + E12/E19/E20

> generated: 2026-11-26
> worktree: `D:\PublicSemanticSpace\.claude\worktrees\agent-a3f049ed86b9c818d`
> base: `origin/master` = `3826594`

## What changed at the call site

The runtime's fact store lives on `RelationFactLayer`, not `EntityLayer`
(`EntityLayer` owns object slots / IDs). `pss/runtime/layers.py` now defaults
to:

```python
self.facts = facts or TMSFactStore()   # snapshot_mode="delta"
```

Injection is unchanged: passing `facts=MultiSourceFactStore()` or
`facts=TMSFactStore(...)` still works, and the legacy snapshot/restore payload
shape is kept for a `MultiSourceFactStore`. `components()["fact_store"]` reports
`TMSFactStore` for a default runtime.

`TMSFactStore` now constructs its wrapped `MultiSourceFactStore` with
`record_history=False`; the TMS wrapper owns versioning. Direct users of
`MultiSourceFactStore` keep the old default (`record_history=True`).

## Commands

```powershell
python -m pytest tests/test_tms_incremental.py tests/test_tms_store.py `
  tests/property/test_tms.py tests/property/test_hypothesis_sequences.py `
  tests/test_runtime.py tests/test_runtime_rules.py `
  tests/test_multisource_rules.py tests/test_fact_store.py `
  tests/test_pss_eval_adapters.py -q

python -m pss.eval.scripts.run_gap_tms --seeds 5 --out <tmp>
python -m pss.eval.scripts.run_gap_l2_runtime_updates --seeds 3 --out <tmp>
python -m pss.eval.scripts.run_gap_l2_memory `
  --root D:/PublicSemanticSpace/bundles/dev --out <tmp>
python -m pss.eval.scripts.run_runtime
```

Result: all targeted tests pass. `tests/test_tms_incremental.py` covers
incremental-vs-full equivalence, the reverse-dependency index, all three
snapshot modes, the runtime default, and the injectable legacy store.

## E10 (TMS on the production store)

| property | mean | min |
|---|---:|---:|
| K3_propagation | 1.000 | 1.000 |
| K4_conflict | 1.000 | 1.000 |
| K4_resolution | 1.000 | 1.000 |
| K5_rollback | 1.000 | 1.000 |
| K5_label_restored | 1.000 | 1.000 |
| AUD | 1.000 | 1.000 |
| source_retract | 1.000 | 1.000 |

`all_pass: True`. Unchanged from `reports/claim_gaps/tms/REPORT.md`.

## E12 (L2 bundle replay, 33 bundles)

Overall (deterministic identity columns):

| variant | n | ID retention | switches | reappearance |
|---|---:|---:|---:|---:|
| full | 33 | 0.576 | 0.00 | 0.576 |
| no_memory | 33 | 0.476 | 0.32 | 0.571 |
| no_entity | 33 | 0.156 | 27.33 | 0.121 |
| no_relations | 33 | 0.576 | 0.00 | 0.576 |

Valid bundles only (19 of 33):

| variant | n | ID retention | switches |
|---|---:|---:|---:|
| full | 19 | 1.000 | 0.00 |
| no_memory | 19 | 0.827 | 0.56 |
| no_entity | 19 | 0.270 | 47.47 |
| no_relations | 19 | 1.000 | 0.00 |

These match `reports/claim_gaps/l2_memory_all/REPORT.md` exactly
(1.000 / 0.827 / 0.270 on the 19 valid bundles). The reported p95 latency is a
wall-clock measurement and varied between runs (sub-millisecond this run vs
2.6 ms in the archived report); the identity metrics are deterministic.

## E19 (L2 runtime mid-task updates)

| scenario | plan changed | baseline goal | updated goal | baseline exec | updated exec |
|---|---:|---:|---:|---:|---:|
| kb_update | 100% | 0% | 100% | 100% | 100% |
| planner_swap | 100% | 0% | 100% | 0% | 100% |

Unchanged from `reports/claim_gaps/l2_runtime_updates/REPORT.md`.

## E20 (cross-layer runtime)

| variant | entity persistence | plan | task success | conflict flagged | recommendation | rollback | citations |
|---|---:|---:|---:|---:|---|---:|---:|
| full | yes | yes (2) | yes | yes | review | yes | 1 |
| no_entity | no | no (0) | no | no | accept | yes | 0 |
| no_relations | yes | yes (2) | yes | yes | review | yes | 1 |
| no_uncertainty | yes | yes (2) | yes | no | disabled | yes | 1 |
| no_audit | yes | yes (2) | yes | yes | review | no | 1 |
| no_language | yes | yes (2) | yes | yes | review | yes | 0 |

State size also matches `reports/claim_gaps/runtime_cross_layer/runtime.md`
(full: 3 entities / 12 relations / 3 facts / audit 10 / version 9; no_entity:
2 / 2 / 2 / 5 / 4; no_audit: audit 0, version 0).

## Scale and snapshot cost

See `REPORT.md` and `metrics.json` in this directory. At 10k facts the delta
update is 0.014 ms median and rollback is 6.2 ms; the full-copy mode is 581 ms
per update and 161.8 ms per rollback at the same size. The incremental
propagation path is 0.013 ms at 10k facts (30k graph nodes) versus 78.5 ms for
the full recomputation, with identical labels on every sample.
