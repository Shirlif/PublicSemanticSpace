# Claim-gap E10: TMS on MultiSourceFactStore

> seeds: 5

| property | mean | min | pass |
|---|---:|---:|---|
| K3_propagation | 1.000 | 1.000 | True |
| K4_conflict | 1.000 | 1.000 | True |
| K4_resolution | 1.000 | 1.000 | True |
| K5_rollback | 1.000 | 1.000 | True |
| K5_label_restored | 1.000 | 1.000 | True |
| AUD | 1.000 | 1.000 | True |
| source_retract | 1.000 | 1.000 | True |

**All pass: True**

## What this closes

K3/K4/K5 were previously demonstrated on the standalone
``VersionedKnowledgeBase``. This suite runs them on ``TMSFactStore``, which
keeps ``MultiSourceFactStore`` as the production assertion table. The same
propagation, conflict, resolution, rollback and source-retraction properties
hold, so the TMS is no longer a parallel module.
