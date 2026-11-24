# PSS evaluation suite — Phase A skeleton

Unified episode schema + per-benchmark adapters + metrics + runner. The suite
does **not** merge raw datasets; it converts each benchmark into `EvalEpisode`
and scores `Prediction` with benchmark-specific metrics.

## Quick start

```bash
python -m pss.eval.suite.runner --smoke
python -m pss.eval.suite.runner --benchmarks bfcl,openood,zsre
```

`--smoke` uses tiny built-in episodes and validates plumbing only. Real runs
require the official data under `data/public/<bench>/`; raw data is
gitignored and never committed.

## Layout

```
pss/eval/suite/
  schema.py          EvalEpisode / Frame / EntitySpec / RelationSpec / FactSpec / GoalSpec
  adapter.py         BenchmarkAdapter protocol + BenchmarkUnavailable
  registry.py        name -> adapter
  metrics.py         accuracy / exact_match / AUROC / FPR@95TPR
  predictors.py      deterministic smoke baselines (BFCL keyword, OpenOOD prototype, ZsRE FactStore)
  runner.py          suite runner + report
  adapters/
    bfcl.py
    openood.py
    zsre.py
    clevr_cogent.py
    tao.py
data/public/<bench>/  raw data, gitignored
eval/results/suite/   metrics.json + report.md
```

## Adapter contract

```python
class BenchmarkAdapter(ABC):
    name: str
    layer: str          # language | open_set | memory | entity | relation | planning | audit
    license: str
    def prepare(self) -> None: ...          # raise BenchmarkUnavailable with instructions
    def episodes(self, split="test") -> Iterator[EvalEpisode]: ...
    def evaluate(self, predictions) -> list[EvalResult]: ...
    def sample_episodes(self) -> list[EvalEpisode]: ...  # --smoke
```

The runner never imports benchmark-specific code; add a new adapter to
`registry.ADAPTERS`.

## Official data status

- **TAO annotations**: downloaded to `data/public/tao/annotations/`
  (`TAO_val/gt/gt.json`). Charades frames are extracted to
  `data/public/tao/frames/`; the PSS tracker beats the IoU baseline
  (IDF1 0.979 vs 0.653 on 5 videos, GT boxes as detections).
- **CLEVR-CoGenT**: official `CLEVR_CoGenT_v1.0.zip` is downloading to
  `data/public/clevr_cogent/`. Once extracted, the adapter parses
  `scenes/*.json` and `images/<split>/`.
- Raw data is gitignored; adapters only read it.

## Expected data formats (Phase A)

### BFCL

`data/public/bfcl/*.json`, a list (or `{"questions": [...]}`) of:

```json
{"question": "...", "function": [{"name": "...", "parameters": {...}}],
 "ground_truth": {"name": "...", "arguments": {...}}}
```

The adapter is tolerant of `prompt`/`query`, `functions`/`tools`,
`expected`/`answer`.

### OpenOOD

`data/public/openood/features.npz` with:

- `train_x`, `train_y`: in-distribution features/labels;
- `test_x`, `test_y`: test features/labels;
- `test_ood`: boolean OOD flags.

### CLEVR-CoGenT

After extracting `CLEVR_CoGenT_v1.0.zip` under `data/public/clevr_cogent/`:

```
scenes/CLEVR_CoGenT_TrainA_R1_scenes.json
scenes/CLEVR_CoGenT_TrainB_R1_scenes.json
scenes/CLEVR_CoGenT_ValA_scenes.json
scenes/CLEVR_CoGenT_ValB_scenes.json
images/{trainA,trainB,valA,valB}/
```

The adapter yields left/right/front/behind relation episodes from the 3D
coordinates; `valA` vs `valB` is the compositional split.

### TAO

`data/public/tao/annotations/TAO_val/gt/gt.json` (official `TAOLabels.zip`).
Frames are optional for the association metric; place official frames under
`data/public/tao/frames/` to run a perception-based tracker later.

### ZsRE

`data/public/zsre/zsre.jsonl` (or `edits.jsonl`), one edit per line:

```json
{"subject": "Q1", "property": "capital", "old_value": "A", "new_value": "B",
 "locality": [{"subject": "Q2", "property": "capital", "value": "C"}]}
```

## Metrics

- language: tool-call accuracy, argument accuracy
- open_set: AUROC, FPR@95TPR, accuracy
- memory: edit success, locality, rollback_ok

## Licensing and leakage

- Only official splits; no training on test; use validation for calibration.
- Raw data is not committed; check each adapter's `license` before use.
- Smoke numbers are not benchmark performance.

## Next adapters (Phase B/C)

- `owtb` (open-world tracking; TAO is the first entity adapter)
- `alfworld` / `calvin` (planning + execution)
- `objectfolder_real` / `touch_and_go` (real tactile)
