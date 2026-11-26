# F5：`pss_eval` 协议包与外部 baseline 预注册

> 状态：前半段实现完成（协议包 + 两个 adapter + runner + smoke property）。
> 本文同时是设计与**预注册**：外部 baseline 应在哪些项失败、为什么，先写死，再跑。

## 1. 范围与边界

目标（对应 `docs/plans/PSS-后续研究方向评估.md` §3.3 / §4 F5）：

- 抽出一个不依赖 PSS 内部类型的 `SystemUnderTest` 协议；
- 让 E5 的 contract-free flat runtime 通过同一套 property 接口运行；
- 再接一个外部风格的 transformer-state baseline；
- 预先登记它们应在哪些 property 上失败。

本次**不做**：PSS runtime 自身的 adapter、分类准确率 / ZCR 打分、swap 延迟、完整
budget / anytime 曲线。这些留给 F5 后半段，避免与 F3/F4 的文件边界冲突。

严格增量：`pss_eval/`、`tests/test_pss_eval_*.py`、本文。未修改
`pss/detect/replay.py`、`pss/property/*`、`scripts/run_property_l2.py`、
`scripts/reproduce_l2.py`、`pss/eval/scripts/run_gap_families*.py`，也未改
`pyproject.toml`（现有 `include = ["pss*"]` 已覆盖 `pss_eval*`）。

## 2. 包结构

```
pss_eval/
  protocol.py       # 只 import stdlib + numpy；核心协议 + 可选能力协议 + 规范化摘要
  fixtures.py       # 纯 numpy 的合成流（标签只给 harness，不进入协议）
  properties.py     # 3 个 smoke property + 1 个 contract property
  runner.py         # 同一套 property 跑任意 SystemSpec；JSON / Markdown 报告
  preregistration.py# 可执行的预注册检查
  __main__.py       # python -m pss_eval
  adapters/
    flat.py         # E5 contract-free flat runtime 的 adapter
    transformer.py  # 单层 causal self-attention + KV cache 的 state baseline
tests/
  test_pss_eval_protocol.py
  test_pss_eval_adapters.py
  test_pss_eval_runner.py
```

`import pss_eval` 不会加载 `pss`；只有 `pss_eval.adapters.flat` 才 import PSS 内部模块。
协议层的零依赖由两条测试共同保证：AST 扫描 + 子进程导入后检查 `sys.modules` 中没有
`pss*`。

## 3. 协议语义

核心接口（`pss_eval.protocol.SystemUnderTest`）：

| 方法 | 语义 |
|---|---|
| `reset(seed=...)` | 回到初始状态，同一 seed 必须可复现 |
| `ingest(observation)` | 追加一条观测 |
| `step()` | 对最近一条观测给出 `Prediction` |
| `state()` | 返回持久状态快照；值只用标准 Python 容器/标量 + numpy 数组 |

`Observation` 只有 `observation_id / modality / features`，**不带标签**。标签保存在
harness 的 `LabeledSample` 里，runner 只把 `observation` 传给系统。

`Prediction` 为 `observation_id / embedding / label? / scores?`。无监督 state 模型
可以只给 embedding。

可选能力用独立协议表达，避免把 swap/rollback 硬塞进核心接口：

- `SnapshotCapable`：`snapshot() -> token`、`rollback(token) -> bool`；
- `SwapCapable`：`swap(module_descriptor: Mapping) -> bool`，合规系统必须先校验再改状态；
- `AuditCapable`：`audit() -> list[dict]`。

未实现某项能力时，对应 property 报 `skip`，绝不算隐式通过。

## 4. 两个外部 baseline

### 4.1 `e5-flat`（`adapters/flat.py`）

包装 `pss.property.nonpss` 的 `FlatRegistry` / `FlatKB`，刻意保留负架构特征：

- 任何模块（wrong-dim / non-normalized / NaN）都被接受；
- 无 rollback、无版本路由；
- 无审计；
- KB 是 last-writer-wins 的扁平字典。

它是**负架构**，不是竞争性模型；用途是证明 suite 能把 contract-free 系统与 PSS 分开。

### 4.2 `transformer-state`（`adapters/transformer.py`）

单层 causal self-attention，状态 = 有界 KV cache + 一个 running hidden vector。
纯 numpy、无训练、无分类头，因此 `label=None`。它代表“只有学习到的隐状态、没有
类型化实体 / registry / rollback / audit / budget”的外部系统。

## 5. Property 与判定

| property | smoke | 判定 |
|---|---|---|
| `determinism` | 是 | 同 seed + 同流 → 预测逐条相同、最终 state digest 相同、id 对齐 |
| `reset_exactness` | 是 | reset 后 state 等于新建实例；重放前缀预测一致 |
| `state_hygiene` | 是 | state 是 dict；数组有限；可规范化；≤ 256 KiB |
| `contract` | 否 | snapshot/rollback 精确、非法模块被拒且不改状态、审计非空 |

`contract` 只在系统同时具备三项可选能力时运行；否则 `skip`。

## 6. 预注册：预期失败项与理由

先写死以下预期（`pss_eval/preregistration.py` 是可执行版本）：

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

`transformer-state` 的 contract 预期为 `skip`，因为它没有 `snapshot/swap/audit`
任何一项能力；它没有类型化状态，也就不存在“回滚到某个契约版本”的语义。

三个 smoke property 对两者都应通过：它们只检查协议卫生（确定性、reset、state 可序列化），
不检查架构能力。这正是“同一套 property 接口能跑外部系统”的最小证据。

## 7. 可证伪条件

以下任一情况出现，说明 harness 或预注册有问题，必须先解释而不是放宽预期：

1. `e5-flat` 的 `contract` 变成 `pass`，或五个 contract 指标不等于上表；
2. `transformer-state` 的 `contract` 不是 `skip`（例如能力探测把 `state()` 误判成
   `snapshot()`）；
3. 任一系统在 `determinism` / `reset_exactness` / `state_hygiene` 上失败；
4. `pss_eval.protocol` 在导入时把 `pss` 带进 `sys.modules`；
5. runner 在同一 seed 下给出不同状态 digest。

## 8. 本次实际观察（2026-11-26）

在本 worktree 上运行 `python -m pytest tests/test_pss_eval_protocol.py
tests/test_pss_eval_adapters.py tests/test_pss_eval_runner.py -q`：19 passed。
`python -m pss_eval --seeds 1` 输出 `preregistration: OK`，即：

- `e5-flat`: determinism / reset_exactness / state_hygiene = pass，contract = fail；
  contract 五项指标为 `1, 0, 0, 0, 0`（顺序同 §6 表）。
- `transformer-state`: 三项 smoke = pass，contract = skip。

## 9. 复现

```bash
python -m pss_eval --seeds 3 --dim 32 --n 32 --out reports/pss_eval
python -m pytest tests/test_pss_eval_protocol.py tests/test_pss_eval_adapters.py tests/test_pss_eval_runner.py -q
```

## 10. 仍未完成（F5 后半段）

- PSS runtime 自身的 `SystemUnderTest` adapter，以及它在 contract property 上的通过；
- 用真实 bundle / 检测流替换合成流；
- 准确率、ZCR、swap 延迟、budget / anytime 的 property 化；
- 把 `contract` 从“能力存在 + 基本精确性”扩展为完整审计/传播检查。
