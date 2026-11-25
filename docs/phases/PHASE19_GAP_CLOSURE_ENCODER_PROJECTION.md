# Phase 19 — Gap closure 1–2: encoder and projection defaults

> Date: 2026-11-24
> Scope: `PSS-认知层架构定位.md` §5, rows 1–2
> Data: `tracking_bench_v2` (synthetic)
> Result artifacts: `eval/results/runtime_encoder/`

---

## 1. Updated gap table

| 层 | 目标 | 当前（Phase 19 后） | 状态 |
|---|---|---|---|
| 编码器 | CLIP/DINOv2/VLM | runtime 默认 DINOv2（冻结）；SmallCNN 为 fallback；CLIP/VLM 权重不在本地 | 部分关闭 |
| 投影 | 对齐 + 校准 | DINOv2 默认路径：PCA-whiten 96（vision+geometry）+ residual，reliability 进 uncertainty；共享头、开放集、SmallCNN fallback、CLIP/VLM 投影未解决 | 部分 |
| 实体绑定 | object attention / persistent slots | runtime `EntityLayer` 已有 persistent slots；object attention 仍实验 | 部分 |
| 关系 | typed object attention + 学习规则 | typed heads 已是默认；规则仍是 symbolic | 部分 |
| 规划 | 学习 + 搜索 | A*/BFS + versioned operators；无 learned heuristic | 未关闭 |
| 语言 | LLM decoder | adapter + tools/audit；无真实 decoder | 阻塞（无权重） |
| 审计 | 版本/回滚/权限 | runtime version/rollback/audit；权限仍粗 | 部分 |

---

## 2. Row 1 — encoder default

Implemented `pss/runtime/perception.py`:

- `PerceptionService` defaults to frozen **DINOv2** vision + tactile MLP;
- matching projection is `checkpoints/encoder_priority/dinov2/projection`;
- SmallCNN + `phase45_projection` is the fallback, selectable by config;
- `PSSRuntime.ingest_sample` uses the service; `summary()` reports the active
  encoder/fallback.

Runtime comparison (5 tracking sequences, graph matchers trained per variant):

| variant | vision | image size | fallback | p50 | p95 | ID retention |
|---|---|---|---:|---:|---:|---:|
| **dinov2** | DINOv2 | 224 | no | 16.7 ms | — | 1.000 |
| smallcnn | SmallCNN | 64 | yes | 4.7 ms | — | 1.000 |

Both retain identity on this short set. The cost is **~3.6x observation
latency** for DINOv2 at 224×224. The accuracy trade-off is the Phase 11 one:
DINOv2 re-ID 0.993 vs base 0.997, but closed-set 0.969 vs 0.979 and open-set
0.897 vs 0.993. Row 1 closes the architecture gap (pretrained default), not the
accuracy gap.

CLIP is implemented but not default: the local cache has only its config, not
the 1.7 GB weights. VLM is not attempted.

---

## 3. Row 2 — projection alignment/calibration

The DINOv2 default projection is the calibrated one from Phase 11:

```json
{"whiten_dim": 96, "whiten_modalities": ["vision", "geometry"], "residual": true}
```

`PerceptionService.describe()` reports this calibration record. The service also
loads per-modality reliability from the prototype metrics:

| variant | reliability (vision / tactile / geometry) |
|---|---|
| dinov2 | 1.000 / 0.000 / 0.500 |
| smallcnn | 1.000 / 0.500 / 1.000 |

`extract_frame` now attaches `modality_reliability` to each `Observation`, and
`PSSRuntime.ingest` averages it into `UncertaintyService`. A modality below 0.5
sets `low_modality_reliability` and recommends `review`.

This is the architectural requirement from §3.2: the projection returns a
calibrated vector **plus reliability**, not a bare vector.

---

## 4. Findings

1. **The runtime now uses a pretrained encoder by default**, with a one-flag
   fallback. DINOv2 costs ~3.6x observation latency on CPU/GPU at 224px.
2. **The projection is calibrated by default** for the DINOv2 path
   (whiten 96 + residual), and the calibration record is visible in
   `summary()["perception"]`.
3. **Reliability is propagated to the runtime**, not lost at the projection.
   The low tactile reliability (0.000 for DINOv2 prototypes) makes the runtime
   recommend review when tactile is the only evidence; this is the intended
   behavior.
4. **This does not make accuracy better.** The synthetic benchmark still favors
   SmallCNN on closed-set/open-set; DINOv2 is at parity on short re-ID and
   slower. The row closes the architecture/default gap, not the metric gap.

---

## 5. Boundaries

- 5 sequences; both variants retain 1.000, so the runtime comparison is a
  latency/architecture check, not a re-ID accuracy claim.
- DINOv2 projection is in-distribution; CLIP/VLM untested.
- Reliability values come from the prototype metrics of the corresponding
  projection; they are not per-frame calibrated.
- No real data, no sim2real.

---

## 6. Next rows

- **Row 3 (entity binding)**: make object attention a first-class binding
  option in `EntityLayer` and compare against the tracker graph matcher.
- **Row 4 (relations)**: replace the remaining symbolic rule thresholds with a
  learned/versioned rule layer; keep exact geometry as a fallback.
- **Row 5 (planning)**: add a learned cost-to-go heuristic to A* and report
  expansions/latency.
- **Row 6 (language)**: blocked on LLM weights; the adapter is ready.
- **Row 7 (audit)**: add permission profiles (read-only/operator/admin) and
  enforce them in `assert_fact` / `plan` / `act` / `rollback`.

Reproduce:

```bash
python -m pss.eval.scripts.run_runtime_encoder --device cuda --sequences 5
```
