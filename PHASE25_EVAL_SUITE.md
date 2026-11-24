# Phase 25 — Evaluation suite Phase A skeleton

> Date: 2026-11-24
> Scope: public-benchmark evaluation infrastructure for PSS
> Result artifacts: `eval/results/suite/`, `pss/eval/suite/`

---

## 1. What was built

A unified evaluation skeleton, not a data merge:

| Component | File | Purpose |
|---|---|---|
| `EvalEpisode` schema | `pss/eval/suite/schema.py` | frames / entities / relations / facts / goal / query / tools / expected |
| `BenchmarkAdapter` | `pss/eval/suite/adapter.py` | prepare / episodes / evaluate / sample_episodes |
| registry | `pss/eval/suite/registry.py` | name → adapter |
| metrics | `pss/eval/suite/metrics.py` | accuracy / exact match / AUROC / FPR@95TPR |
| predictors | `pss/eval/suite/predictors.py` | BFCL keyword, OpenOOD prototype, ZsRE FactStore |
| runner | `pss/eval/suite/runner.py` | suite execution + `metrics.json` + `report.md` |
| adapters | `pss/eval/suite/adapters/{bfcl,openood,zsre}.py` | Phase A benchmarks |
| docs | `pss/eval/suite/README.md` | formats, contract, next adapters |

Raw data stays under `data/public/<bench>/` (gitignored). The suite holds only
adapters, manifests and results.

---

## 2. Smoke results

`python -m pss.eval.suite.runner --smoke`:

| benchmark | layer | metric | value | n |
|---|---|---|---:|---:|
| bfcl | language | tool_call_accuracy | 1.000 | 3 |
| bfcl | language | argument_accuracy | 0.667 | 3 |
| openood | open_set | auroc | 1.000 | 9 |
| openood | open_set | fpr_at_95tpr | 0.000 | 9 |
| openood | open_set | accuracy | 0.778 | 9 |
| zsre | memory | edit_success | 1.000 | 2 |
| zsre | memory | locality | 1.000 | 2 |
| zsre | memory | rollback_ok | 1.000 | 2 |

These numbers validate the pipeline, not benchmark performance: the episodes
are tiny built-in samples. The BFCL predictor is a keyword baseline; the
OpenOOD predictor is a prototype baseline; the ZsRE predictor uses the PSS
`FactStore` directly.

---

## 3. Design decisions

1. **Adapter, not merge.** Datasets keep their native storage and licenses;
   only the episode schema is unified.
2. **Layer-tagged results.** Each adapter declares `layer`, so the report can
   say which PSS layer a metric belongs to instead of producing a single score.
3. **Smoke path is mandatory.** Every adapter ships `sample_episodes`, so the
   runner and report can be tested without downloading data.
4. **No silent downloads.** `prepare()` raises `BenchmarkUnavailable` with
   instructions; the runner records skipped benchmarks by default.
5. **Deterministic predictors first.** The skeleton proves the plumbing; PSS-
   backed predictors replace them later without touching adapters.

---

## 4. Expected data formats

- **BFCL**: `data/public/bfcl/*.json`; question + function schema + expected call.
- **OpenOOD**: `data/public/openood/features.npz` with `train_x/train_y/
  test_x/test_y/test_ood`.
- **ZsRE**: `data/public/zsre/zsre.jsonl` with subject/property/old/new/locality.

See `pss/eval/suite/README.md` for the exact keys.

---

## 5. Limitations

- No official data is bundled or downloaded yet; real runs require a manual
  data drop.
- The BFCL parser is tolerant but not a complete implementation of every BFCL
  release.
- OpenOOD is feature-level (`features.npz`), not an end-to-end image loader;
  the encoder/feature export is a separate step.
- The smoke predictors are baselines, not PSS capability claims.
- No `clevr_cogent`, `tao`, `alfworld` adapters yet.

---

## 6. Next

1. Add `clevr_cogent` (relation OOD) and wire the typed relation head.
2. Add `tao`/`owtb` (entity persistence) and the tracker/HOTA adapter.
3. Add `alfworld`/`calvin` (planning) and the runtime planner adapter.
4. Replace the smoke predictors with PSS-backed ones:
   - BFCL → `LLMLanguageHead` + `ToolRegistry`;
   - OpenOOD → `PerceptionService` + calibrated projection;
   - ZsRE → `PSSRuntime` fact/version/audit.
5. Write the preregistered suite protocol before real runs.

Reproduce:

```bash
python -m pss.eval.suite.runner --smoke
```
