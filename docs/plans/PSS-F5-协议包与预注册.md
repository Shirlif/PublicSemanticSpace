# F5：`pss_eval` 协议包与外部 baseline 预注册

> 状态：**前半段 + 后半段均已实现**（2026-11-26）。
> 前半段：协议包 + 两个外部 adapter + runner + smoke property。
> 后半段：PSS runtime adapter、真实 bundle 流、5 个新 property、三方 runner 对照。
> 本文同时是设计与**预注册**：外部 baseline 与 PSS runtime 应在哪些项通过 / 失败，先写死，再跑。

## 1. 范围与边界

目标（对应 `docs/plans/PSS-后续研究方向评估.md` §3.3 / §4 F5）：

- 抽出一个不依赖 PSS 内部类型的 `SystemUnderTest` 协议；
- 让 E5 的 contract-free flat runtime 通过同一套 property 接口运行；
- 再接一个外部风格的 transformer-state baseline；
- 把 PSS 自身 runtime 包成 adapter，证明同一 suite 能把 PSS 与 contract-free 系统分开；
- 用真实 bundle 检测流替换合成 Gaussian 流；
- 预先登记各系统应在哪些 property 上通过 / 失败 / skip。

严格边界：

- `pss_eval/protocol.py` 仍然只 import stdlib + numpy，**零 `pss` 依赖**；由 AST 扫描与子进程导入两条测试保证。
- adapter 可以 import `pss`：`adapters/flat.py`、`adapters/pss_runtime.py`。
- 真实数据只读：`D:/PublicSemanticSpace/bundles/dev`。
- **PSS 路径不打开 `gt.parquet`**：`adapters/pss_runtime.py` 不做任何 parquet 读取；只有 `pss_eval/bundles.py` 在显式请求 harness 身份标签时才读 GT，且标签只落在 `Stream.identity_labels`，绝不进入 `Observation`。
- 本次未修改 `pss/property/*`、`pss/detect/replay.py`、`pss/runtime/*`、`pss/projection/*`、`pss/prototypes/*`、`pss/encoders/*`，也未 commit。

## 2. 包结构

```
pss_eval/
  protocol.py        # 零 PSS 依赖：核心协议 + 5 个可选能力协议 + 规范化摘要
  fixtures.py        # 纯 numpy 合成流；Stream.identity_labels 供 harness 使用
  bundles.py         # 真实 bundle -> Stream；仅此处按需读取 gt.parquet
  properties.py      # 4 个原 property + 5 个新 property（default / extended）
  runner.py          # 同一 runner 跑任意 SystemSpec；三方系统工厂；bundle 聚合
  preregistration.py # 可执行预注册（外部 + protocol + pss-runtime contract 指标）
  __main__.py        # python -m pss_eval --suite {external,protocol} [--bundles]
  adapters/
    flat.py          # E5 contract-free flat runtime
    transformer.py   # 单层 causal self-attention + KV cache
    pss_runtime.py   # PSSRuntime（entity layer / memory / versioned state）适配
tests/
  test_pss_eval_protocol.py
  test_pss_eval_adapters.py
  test_pss_eval_runner.py
  test_pss_eval_pss_runtime.py
  test_pss_eval_bundles.py
```

`import pss_eval` 不会加载 `pss`；只有 `pss_eval.adapters.flat` 与
`pss_eval.adapters.pss_runtime` 才 import PSS 内部模块。

## 3. 协议语义

核心接口（`pss_eval.protocol.SystemUnderTest`）：

| 方法 | 语义 |
|---|---|
| `reset(seed=...)` | 回到初始状态，同一 seed 必须可复现 |
| `ingest(observation)` | 追加一条观测（真实 bundle 流中一条 = 一帧） |
| `step()` | 对最近一条观测给出 `Prediction` |
| `state()` | 返回持久状态快照；值只用标准 Python 容器/标量 + numpy 数组 |

`Observation` 有 `observation_id / modality / features / metadata?`。`metadata` 是
adapter 私有的不透明旁路（真实 bundle 帧的逐检测 pose / score / class / embedding），
协议本身不检查它，且**不得携带标签**：标签只在 harness 的 `LabeledSample` 与
`Stream.identity_labels` 里。

`Prediction` 为 `observation_id / embedding / label? / scores?`。

可选能力用独立协议表达：

- `SnapshotCapable`：`snapshot() -> token`、`rollback(token) -> bool`；
- `SwapCapable`：`swap(module_descriptor: Mapping) -> bool`，合规系统必须先校验再改状态；
- `AuditCapable`：`audit() -> list[dict]`；
- `IdentityCapable`：`identity_assignments() -> Mapping[detection_ref, entity_id]`；
- `BudgetCapable`：`budget() -> {budget_ms, reserve_ms}`、`budget_stats() -> {...}`。

未实现某项能力时，对应 property 报 `skip`，绝不算隐式通过。

## 4. 三个系统

### 4.1 `e5-flat`（`adapters/flat.py`）

包装 `pss.property.nonpss` 的 `FlatRegistry` / `FlatKB`，刻意保留负架构特征：

- 任何模块（wrong-dim / non-normalized / NaN）都被接受；
- 无 rollback、无版本路由；
- 无审计；
- KB 是 last-writer-wins 的扁平字典。

它是**负架构**，不是竞争性模型；用途是证明 suite 能把 contract-free 系统与 PSS 分开。

### 4.2 `transformer-state`（`adapters/transformer.py`）

单层 causal self-attention，状态 = 有界 KV cache + 一个 running hidden vector。
纯 numpy、无训练、无分类头，因此 `label=None`，也不具备任何可选能力。

### 4.3 `pss-runtime`（`adapters/pss_runtime.py`）

包装 `pss.runtime.runtime.PSSRuntime`（`build_runtime("full", tier="frozen_real")`），
保留 entity layer、tracker/memory、版本化 `CognitiveState` 与 audit：

- `ingest` 把一帧的逐检测 embedding 经 `ModuleRegistry` 当前 vision 模块投影到固定
  `public_dim`，再构造 `pss.memory.perception.Observation` 交给 runtime；
- `state()` 只暴露认知状态（entities / relations / facts / version）+ registry 的
  active versions / swap depth；**不包含 append-only audit**，因此 PSS 的
  “rollback 追加一条审计”不会破坏 rollback 精确性；
- `snapshot/rollback` 同时恢复 runtime state version 与 registry swap 栈；
- `swap` 走 `ModuleRegistry` 的真实准入契约：wrong-dim / 非单位范数 / NaN 被拒且状态不变；
- `identity_assignments` 来自 tracker 的 matched/new/reid 事件；
- `budget` 包一层 `DeadlineGuard`（默认 50 ms），`budget_stats` 暴露 p50/p99 与 timeout 计数。

adapter 不读任何 parquet；真实 bundle 的检测数据由 `Observation.metadata` 传入。

## 5. 真实 bundle 流（`bundles.py`）

`DEFAULT_BUNDLE_NAMES` 是 F2 使用的 4 个带 `emb` 的开发 bundle：

| bundle | frames | detections |
|---|---:|---:|
| `s1_hard/37f0c8b8/seed0` | 100 | 556 |
| `s1_occlusion_real/16ea9f38/seed0` | 100 | 460 |
| `s2_twins/3d49ce1b/seed0` | 100 | 407 |
| `s1_camera_real/b4bb5f0e/seed0` | 100 | 285 |

每帧一个 `Observation`：

- `features` = 该帧检测 embedding 的 L2 归一化均值（384 维），供外部 baseline 使用；
- `metadata["detections"]` = 逐检测 `ref / pos / quat / score / category / embedding`；
- `sample.label = None`（真实检测流没有标量标签）。

`identity_labels` 只在 `with_gt_identities=True` 时由 `gt.parquet` 构建：按同
`physics_step` 的最近 GT 位姿（≤ 0.5 m）把检测映射到 `instance_id`。这是 harness
侧的评测标签；系统侧观测里没有任何 GT 字段（测试会检查）。

## 6. Property 与判定

| property | smoke | 判定 | 缺失能力 |
|---|---|---|---|
| `determinism` | 是 | 同 seed + 同流 → 预测逐条相同、最终 state digest 相同、id 对齐 | — |
| `reset_exactness` | 是 | reset 后 state 等于新建实例；重放前缀预测一致 | — |
| `state_hygiene` | 是 | state 是 dict；数组有限；可规范化；≤ 256 KiB | — |
| `contract` | 否 | snapshot/rollback 精确、非法模块被拒且不改状态、审计非空 | 缺 snapshot/swap/audit → skip |
| `swap_rollback` | 否 | 连续 3 次合法 swap 成功且改状态；一次 rollback 精确；rollback 后仍可 swap | 缺 snapshot/swap → skip |
| `identity_retention` | 否 | 同一 GT instance 的检测保持同一 track id；未分配计入惩罚；≥ 0.5 | 缺 identity 或流无身份标签 → skip |
| `label_accuracy` | 否 | 预测标签准确率 ≥ 均匀 chance | 流无标量标签或系统不输出标签 → skip |
| `budget` | 否 | budget 配置有效；caller p99 ≤ budget + 2 ms；timeout 计数自洽；无错误 | 缺 budget 或未启用 → skip |
| `latency` | 否 | 逐观测 ingest+step 墙钟 p99 ≤ 1000 ms，并报告 p50/p99/max | — |

`identity_retention` 的 0.5 阈值表示“多数检测保持同一实例的 modal track”，
远高于无记忆系统逐帧新建 ID（约 1/n_frames）；未分配检测按不一致计。

## 7. 预注册

### 7.1 原始外部 suite（合成流，4 个 property）

| system | determinism | reset_exactness | state_hygiene | contract |
|---|---|---|---|---|
| `e5-flat` | pass | pass | pass | **fail** |
| `transformer-state` | pass | pass | pass | **skip** |

`e5-flat` 的 contract 指标预期精确为：

| metric | 预期 | 理由 |
|---|---:|---|
| `swap_accepted` | 1 | E5 对合法模块总是返回成功 |
| `invalid_rejected` | 0 | 无准入契约，wrong-dim 模块也被接受（对应 E5 `SRR=0`） |
| `invalid_keeps_state` | 0 | 非法模块替换后状态已改变 |
| `rollback_exact` | 0 | `FlatRegistry.rollback()` 永远失败（对应 E5 `RBE=0`） |
| `audit_present` | 0 | 无审计（对应 E5 `AUD=0`） |

### 7.2 扩展 protocol suite（合成流，9 个 property）

| system | det | reset | hygiene | contract | swap_rollback | identity | label_acc | budget | latency |
|---|---|---|---|---|---|---|---|---|---|
| `e5-flat` | pass | pass | pass | fail | fail | skip | (测量) | skip | pass |
| `transformer-state` | pass | pass | pass | skip | skip | skip | skip | skip | pass |
| `pss-runtime` | pass | pass | pass | **pass** | **pass** | skip | skip | **pass** | pass |

`pss-runtime` 的 contract 指标预期全为 1.0：
`swap_accepted / invalid_rejected / invalid_keeps_state / rollback_exact / audit_present = 1,1,1,1,1`。

`e5-flat` 的 `label_accuracy` 在合成流上不预注册：它依赖合成类的可分性与每 seed
codebook，作为测量值报告（本次为 fail，即低于 chance），不作为放宽预期的对象。

### 7.3 扩展 protocol suite（真实 bundle 流）

与 7.2 相同，另外：

| system | identity_retention | label_accuracy |
|---|---|---|
| `e5-flat` | skip | skip |
| `transformer-state` | skip | skip |
| `pss-runtime` | **pass**（阈值 0.5） | skip（检测流无标量标签） |

## 8. 可证伪条件

以下任一情况出现，说明 harness 或预注册有问题，必须先解释而不是放宽预期：

1. `e5-flat` 的 `contract` 变成 `pass`，或五个 contract 指标不等于 7.1 表；
2. `e5-flat` 的 `swap_rollback` 变成 `pass` / `skip`，或 `pss-runtime` 的
   `contract` / `swap_rollback` 不是 `pass`；
3. `transformer-state` 的 `contract` 不是 `skip`（例如能力探测把 `state()` 误判成
   `snapshot()`）；
4. 任一系统在 `determinism` / `reset_exactness` / `state_hygiene` 上失败；
5. `pss_eval.protocol` 在导入时把 `pss` 带进 `sys.modules`；
6. runner 在同一 seed 下给出不同状态 digest；
7. `pss-runtime` 的 `state()` 中出现 NaN/Inf 或超过 256 KiB；
8. `pss_eval.adapters.pss_runtime` 出现任何 parquet 读取调用（GT 边界）。

## 9. 本次实际观察（2026-11-26）

### 9.1 测试

```
python -m pytest tests/test_pss_eval_protocol.py tests/test_pss_eval_adapters.py \
  tests/test_pss_eval_runner.py tests/test_pss_eval_pss_runtime.py \
  tests/test_pss_eval_bundles.py
```

结果：**36 passed**（原有 19 项无回归；新增 17 项覆盖 adapter、bundle、能力与 GT 边界）。

### 9.2 合成流 protocol suite

`python -m pss_eval --suite protocol --seeds 1 --n 8 --dim 16 --public-dim 16`：

- `e5-flat`: determinism/reset/hygiene/latency pass；contract/swap_rollback fail；
  identity/budget skip；label_accuracy fail（低于 chance，未预注册）。
- `transformer-state`: 三项 smoke + latency pass；其余 skip。
- `pss-runtime`: 除 identity/label_accuracy（合成流无标签）skip 外全部 pass，
  包括 contract 五项 = 1。
- `preregistration: OK`。

### 9.3 真实 bundle 三方对照（3 seeds × 4 bundles）

```
python -m pss_eval --suite protocol --seeds 3 --bundles \
  --out reports/claim_gaps/protocol_pss
```

产物：`reports/claim_gaps/protocol_pss/report.json` 与 `REPORT.md`。
聚合结果（4 bundle × seeds 0/1/2）：

| system | contract | swap_rollback | identity_retention | budget | latency p50/p99 (ms) |
|---|---|---|---|---|---:|
| `e5-flat` | fail（1,0,0,0,0） | fail | skip | skip | 0.016 / 0.031 |
| `transformer-state` | skip | skip | skip | skip | 0.036 / 0.073 |
| `pss-runtime` | **pass**（1,1,1,1,1） | **pass** | **pass**（0.7765） | **pass**（p99 2.47） | 1.36 / 2.28 |

`pss-runtime` 逐 bundle 的 identity retention：

| bundle | identity_retention | instance_purity |
|---|---:|---:|
| `s1_hard` | 0.5881 | 0.6168 |
| `s1_occlusion_real` | 0.8696 | 0.9062 |
| `s2_twins` | 0.8600 | 0.8592 |
| `s1_camera_real` | 0.7883 | 0.7544 |

`pss-runtime` 的 budget 属性：50 ms 预算，p50 1.20 ms、p99 2.47 ms、
timeout rate 0.0025、errors 0；state 约 3.5 KiB（bundle 平均），远低于 256 KiB。
`preregistration: OK`。

## 10. 复现

```bash
# 原有外部 suite（默认，行为不变）
python -m pss_eval --seeds 3 --dim 32 --n 32 --out reports/pss_eval

# 三方 protocol suite（合成流）
python -m pss_eval --suite protocol --seeds 3 --n 32 --dim 32 --out reports/pss_eval_protocol

# 三方 protocol suite（真实 4 bundle）
python -m pss_eval --suite protocol --seeds 3 --bundles \
  --out reports/claim_gaps/protocol_pss

# 测试
python -m pytest tests/test_pss_eval_protocol.py tests/test_pss_eval_adapters.py \
  tests/test_pss_eval_runner.py tests/test_pss_eval_pss_runtime.py \
  tests/test_pss_eval_bundles.py
```

## 11. 仍未完成 / 已知限制

- **分类准确率**：真实 bundle 的检测表没有 GT 类别，`label_accuracy` 在 bundle 流上
  只能 skip；PSS 也没有分类头。要测真实类别准确率，需要在 harness 侧把检测匹配到
  `gt.class` 并定义 per-frame 聚合，目前未做。
- **identity retention 的 GT 来源**：用同帧最近位姿（≤ 0.5 m）匹配 GT instance；
  `s1_hard` 只有 0.59，说明该 bundle 的检测/位姿关联本身较难，不是 PSS 单独的问题。
  未做多匹配半径敏感性分析。
- **swap 的真实编码器族**：protocol suite 的 swap 用可复现的投影模块检验契约；
  真实的 `smallcnn → dinov2` 感知级换入仍由 F2 报告覆盖，两者不重复。
- **anytime 预算**：property 使用 50 ms 以保持 determinism 可复现；F3 的 10 ms
  硬界与降级曲线仍是独立结果，本 suite 不替代它。
- **contract 深度**：当前检查 snapshot/rollback 精确、非法模块拒绝、审计非空；
  尚未做完整审计链校验、跨层传播与 tamper detection（entity_audit 已有独立实现）。
- **ZCR / 多 swap 点**：本 suite 未接 ZCR；swap 只在 prefix 后单点执行，未做
  流中多 swap 点的对照。
