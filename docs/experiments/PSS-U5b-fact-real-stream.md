# PSS U5b：回放长历史流上的 query-conditioned fact selection 与 coverage loss

> 日期：2026-11-28
> 上游：`docs/experiments/PSS-U5-runtime-completeness.md` §2
> 脚本：`pss/eval/scripts/run_upper_bound_u5b.py`
> 结果：`eval/results/upper_bound/u5b/metrics.json`、`eval/results/upper_bound/u5b/REPORT.md`
> 状态：已实际运行；本报告数字全部来自本轮 JSON 与终端输出。

---

## 0. 一句话结论

仓库里**没有真实 fact 流**；本轮把 query-conditioned fact selection 搬到目前最长的单条回放轨迹 `s1_occlusion_20`（仿真 MuJoCo/Isaac occlusion bundle，20 实体 × 1200 步，抽出 **21,168** 条 visibility fact），并在标题与边界中明确标为 **replay，不是 real**。主结果是：**当 `query_recency` 已经达到 coverage 1.000 时，子模 greedy 的 coverage-aware 选择也恰好是 1.000，增量为 0.000，没有可测空间**；只有在 k 小于 query 数、预算不够一 query 一槽时，coverage-aware 才比 query_recency 高 **+0.125 ~ +0.271**，且增益来自共享预算分配与未用完的槽位，不是 value resolution。`learned_query` 在全部 12 个单元格上与 `query_recency` 完全相等；historical 查询下忽略 `as_of` 的 recency 直接掉到 **0.000**，说明 `as_of` 是必要条件。

---

## 1. 数据来源与真实性判断

### 1.1 实际检查过的候选数据

| 路径 | 实际内容 | 能否直接做 fact 流 |
|---|---|---|
| `data/open_stream/tier_a/b/c/index.json` | 物体类别/实例的渲染帧（`rgb/depth/segmentation/tactile/meta_json`），4000 条 category 样本 | 不能：只有观测帧，没有 `(entity, attribute, value, timestamp, reliability)` |
| `bundles/repro/pss-bundles-v0.1.1/` | 打包发布的 bundle；`include_derived=false`，`entity_trace/pred_states/pss_trace` 等派生轨迹被明确排除 | 不能：公开包不含派生 fact 轨迹 |
| `data/tracking_bench_long/` | 60 条 80 帧 occlusion 序列，字段同样是帧、poses、segmentation | 不能：是跟踪基准，不是 fact 记录 |
| `bundles/dev/` | 完整开发 bundle，含 `gt.parquet` 与 `entity_trace/pred_states/pss_trace` 派生轨迹 | **可以**：从 `gt` 抽 per-entity 状态 |

结论：**仓库内没有真实 fact store 轨迹**。`data/open_stream` 与 `data/tracking_bench_long` 都是渲染数据；`bundles/repro` 明确排除了派生轨迹；只有 `bundles/dev` 的 `gt.parquet` 提供了逐帧的实体状态，可以作为“回放长历史流”的原料。

### 1.2 本轮使用的流

`bundles/dev/s1_occlusion_20/18c18c5e/seed0/gt.parquet` 是单条 replay 里最长的一条：

| 属性 | 值 |
|---|---:|
| 源 GT 行数 | 24,000 |
| 写入 fact 数 | **21,168** |
| 因完全遮挡跳过 | 2,832 |
| keys（实体） | 20 |
| 步数 | 1,200 |
| visibility 状态变化 | 169 |
| 连续重复写 | 0 |

fact schema：

| 字段 | 取值 |
|---|---|
| entity | `instance_id`，`cube_00` … `cube_19` |
| attribute | visibility state |
| value | `round(visibility_ratio × 9)`，取 1..9 |
| timestamp | `physics_step` |
| reliability | `0.5 + 0.5 × visibility_ratio` |
| 写入规则 | 仅在 `visibility_ratio > 0` 时写一条 fact |

也就是说，每步每个可见实体写一条 fact；完全遮挡时不写，key 上保留上一次可见的值。训练/测试按 replay seed 划分：测试固定 `seed0`，三个训练 run 分别用 `seed1/2/3`。

### 1.3 真实性边界

这是 **replay**，不是 real：

- 数据由 MuJoCo/Isaac 仿真生成（`manifest.json` 中 `isaac_sim=5.1.0.0`、`detector=oracle`），不是真实传感器记录；
- 只有 visibility 一个动态属性；20 个实体的 pose 在整条轨迹上几乎不变（逐实体 `dx/dy/dz` 变化约 `1e-8`），因此不存在连续的 pose drift；
- 没有矛盾写、没有对抗更新、没有多属性 key；
- 因此**不能**把本报告的数字外推为“真实 fact store 上的 selection 上界”。

---

## 2. 设置

- **Workload**：`Q ∈ {8, 16}` 个 query，每个 query 是 `(key, as_of)`；答案是 `key` 在 `as_of` 之前（含）最新事实的 value。
- **两种时间 regime**：
  - `current`：所有 `as_of = 1199`，即 U5/E4 的当前时刻查询；
  - `historical`：`as_of` 在 `[first_ts(key), 1199]` 内均匀采样，对应带漂移/遮挡的历史查询。
- **预算**：`k ∈ {4, 8, 16}`；3 seeds（训练 replay seed 与查询采样 seed 同时变），40 epochs，CPU。
- **支持关系**：`support[i,j]=True` 当 fact `i` 满足 `key_i = key_j`、`value_i = target_j`、`ts_i ≤ as_of_j`。support 由 replay 的答案规则诱导，是 oracle 信号，不是学到的 relevance 模型。
- **相关关系**：`relevant[i,j]=True` 只要求同 key 且 `ts_i ≤ as_of_j`（不看 value），用于 `recall@k`。
- **三种指标**：
  - `accuracy`：从选中集合中按“同 key 且 `ts ≤ as_of` 的最新 fact”解析出的 value 等于 target；
  - `recall@k`：选中集合至少有一条同 key、`ts ≤ as_of` 的 fact（value-blind）；
  - `coverage`：选中集合至少有一条 value-correct 的 supporting fact；`coverage_loss = 1 − coverage` 是 greedy 最小化的目标。

方法：

| 方法 | 说明 |
|---|---|
| `full` | 全部 21,168 条 fact，上界参考 |
| `random` | 随机 k 条 |
| `hot` | 频率最高 20% key 的最近 fact，query-independent |
| `recency` | 全局最近 k 条 fact |
| `query_recency (as_of)` | 按 key 频率分配预算，每 query 取同 key 且 `≤ as_of` 的最近 b 条，并集 |
| `query_recency (no as_of)` | 同上但忽略 `as_of`（U5 原始规则） |
| `learned_query` | `ActiveSetPolicy` 在训练 replay 上做 listwise 训练，per-query 打分后取 top-b 并集；候选限制在 `ts ≤ as_of` |
| `coverage_aware` | 在 `support` 上做子模 greedy max-coverage，预算 k，tie-break = reliability + recency |
| `coverage_aware + learned` | 同样 greedy，tie-break 换成 learned 分数的聚合 |

复现命令：

```bash
python -u -m pss.eval.scripts.run_upper_bound_u5b \
  --seeds 3 --qs 8,16 --ks 4,8,16 \
  --regimes current,historical --epochs 40 \
  --out eval/results/upper_bound/u5b
```

---

## 3. 主表

以下全部是 3 seeds 的 **mean ± std**。

### 3.1 current regime：accuracy

| method | Q=8 k=4 | Q=8 k=8 | Q=8 k=16 | Q=16 k=4 | Q=16 k=8 | Q=16 k=16 |
|---|---:|---:|---:|---:|---:|---:|
| full | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| random | 0.167 ± 0.156 | 0.292 ± 0.118 | 0.417 ± 0.118 | 0.167 ± 0.059 | 0.333 ± 0.106 | 0.458 ± 0.029 |
| hot | 0.167 ± 0.118 | 0.167 ± 0.118 | 0.167 ± 0.118 | 0.188 ± 0.000 | 0.188 ± 0.000 | 0.188 ± 0.000 |
| recency | 0.125 ± 0.102 | 0.417 ± 0.118 | 0.625 ± 0.102 | 0.146 ± 0.118 | 0.354 ± 0.078 | 0.771 ± 0.029 |
| query_recency (as_of) | 0.542 ± 0.059 | **1.000 ± 0.000** | **1.000 ± 0.000** | 0.292 ± 0.029 | 0.583 ± 0.078 | **1.000 ± 0.000** |
| query_recency (no as_of) | 0.542 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.292 ± 0.029 | 0.583 ± 0.078 | 1.000 ± 0.000 |
| learned_query | 0.542 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.292 ± 0.029 | 0.583 ± 0.078 | 1.000 ± 0.000 |
| coverage_aware | 0.667 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.562 ± 0.051 | 0.854 ± 0.059 | 1.000 ± 0.000 |
| coverage_aware + learned | 0.667 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.562 ± 0.051 | 0.854 ± 0.059 | 1.000 ± 0.000 |

### 3.2 current regime：coverage

| method | Q=8 k=4 | Q=8 k=8 | Q=8 k=16 | Q=16 k=4 | Q=16 k=8 | Q=16 k=16 |
|---|---:|---:|---:|---:|---:|---:|
| full | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| random | 0.167 ± 0.156 | 0.292 ± 0.118 | 0.417 ± 0.118 | 0.167 ± 0.059 | 0.333 ± 0.106 | 0.458 ± 0.029 |
| hot | 0.167 ± 0.118 | 0.167 ± 0.118 | 0.167 ± 0.118 | 0.188 ± 0.000 | 0.188 ± 0.000 | 0.188 ± 0.000 |
| recency | 0.125 ± 0.102 | 0.417 ± 0.118 | 0.625 ± 0.102 | 0.146 ± 0.118 | 0.354 ± 0.078 | 0.771 ± 0.029 |
| query_recency (as_of) | 0.542 ± 0.059 | **1.000 ± 0.000** | **1.000 ± 0.000** | 0.292 ± 0.029 | 0.583 ± 0.078 | **1.000 ± 0.000** |
| query_recency (no as_of) | 0.542 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.292 ± 0.029 | 0.583 ± 0.078 | 1.000 ± 0.000 |
| learned_query | 0.542 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.292 ± 0.029 | 0.583 ± 0.078 | 1.000 ± 0.000 |
| coverage_aware | 0.667 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.562 ± 0.051 | 0.854 ± 0.059 | 1.000 ± 0.000 |
| coverage_aware + learned | 0.667 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.562 ± 0.051 | 0.854 ± 0.059 | 1.000 ± 0.000 |

### 3.3 current regime：recall@k

| method | Q=8 k=4 | Q=8 k=8 | Q=8 k=16 | Q=16 k=4 | Q=16 k=8 | Q=16 k=16 |
|---|---:|---:|---:|---:|---:|---:|
| full | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| random | 0.167 ± 0.156 | 0.375 ± 0.204 | 0.417 ± 0.118 | 0.167 ± 0.059 | 0.333 ± 0.106 | 0.458 ± 0.029 |
| hot | 0.167 ± 0.118 | 0.167 ± 0.118 | 0.167 ± 0.118 | 0.188 ± 0.000 | 0.188 ± 0.000 | 0.188 ± 0.000 |
| recency | 0.125 ± 0.102 | 0.417 ± 0.118 | 0.625 ± 0.102 | 0.146 ± 0.118 | 0.354 ± 0.078 | 0.771 ± 0.029 |
| query_recency (as_of) | 0.542 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.292 ± 0.029 | 0.583 ± 0.078 | 1.000 ± 0.000 |
| query_recency (no as_of) | 0.542 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.292 ± 0.029 | 0.583 ± 0.078 | 1.000 ± 0.000 |
| learned_query | 0.542 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.292 ± 0.029 | 0.583 ± 0.078 | 1.000 ± 0.000 |
| coverage_aware | 0.667 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.562 ± 0.051 | 0.854 ± 0.059 | 1.000 ± 0.000 |
| coverage_aware + learned | 0.667 ± 0.118 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.562 ± 0.051 | 0.854 ± 0.059 | 1.000 ± 0.000 |

### 3.4 historical regime：accuracy

| method | Q=8 k=4 | Q=8 k=8 | Q=8 k=16 | Q=16 k=4 | Q=16 k=8 | Q=16 k=16 |
|---|---:|---:|---:|---:|---:|---:|
| full | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| random | 0.042 ± 0.059 | 0.083 ± 0.059 | 0.292 ± 0.156 | 0.042 ± 0.029 | 0.083 ± 0.059 | 0.167 ± 0.078 |
| hot | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| recency | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| query_recency (as_of) | 0.542 ± 0.059 | **1.000 ± 0.000** | **1.000 ± 0.000** | 0.250 ± 0.000 | 0.521 ± 0.029 | **1.000 ± 0.000** |
| query_recency (no as_of) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| learned_query | 0.542 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.250 ± 0.000 | 0.521 ± 0.029 | 1.000 ± 0.000 |
| coverage_aware | 0.667 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.458 ± 0.078 | 0.708 ± 0.078 | 1.000 ± 0.000 |
| coverage_aware + learned | 0.667 ± 0.059 | 0.875 ± 0.102 | 0.833 ± 0.118 | 0.458 ± 0.078 | 0.667 ± 0.128 | 0.938 ± 0.088 |

### 3.5 historical regime：coverage

| method | Q=8 k=4 | Q=8 k=8 | Q=8 k=16 | Q=16 k=4 | Q=16 k=8 | Q=16 k=16 |
|---|---:|---:|---:|---:|---:|---:|
| full | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| random | 0.042 ± 0.059 | 0.083 ± 0.059 | 0.292 ± 0.156 | 0.042 ± 0.029 | 0.083 ± 0.059 | 0.167 ± 0.078 |
| hot | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| recency | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| query_recency (as_of) | 0.542 ± 0.059 | **1.000 ± 0.000** | **1.000 ± 0.000** | 0.250 ± 0.000 | 0.521 ± 0.029 | **1.000 ± 0.000** |
| query_recency (no as_of) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| learned_query | 0.542 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.250 ± 0.000 | 0.521 ± 0.029 | 1.000 ± 0.000 |
| coverage_aware | 0.667 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.458 ± 0.078 | 0.708 ± 0.078 | 1.000 ± 0.000 |
| coverage_aware + learned | 0.667 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.458 ± 0.078 | 0.708 ± 0.078 | 1.000 ± 0.000 |

### 3.6 historical regime：recall@k

| method | Q=8 k=4 | Q=8 k=8 | Q=8 k=16 | Q=16 k=4 | Q=16 k=8 | Q=16 k=16 |
|---|---:|---:|---:|---:|---:|---:|
| full | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| random | 0.042 ± 0.059 | 0.125 ± 0.000 | 0.292 ± 0.156 | 0.042 ± 0.029 | 0.146 ± 0.029 | 0.229 ± 0.078 |
| hot | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| recency | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| query_recency (as_of) | 0.542 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.250 ± 0.000 | 0.521 ± 0.029 | 1.000 ± 0.000 |
| query_recency (no as_of) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| learned_query | 0.542 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.250 ± 0.000 | 0.521 ± 0.029 | 1.000 ± 0.000 |
| coverage_aware | 0.667 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.479 ± 0.078 | 0.729 ± 0.078 | 1.000 ± 0.000 |
| coverage_aware + learned | 0.667 ± 0.059 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.521 ± 0.059 | 0.750 ± 0.051 | 1.000 ± 0.000 |

---

## 4. coverage loss 是否有增量

### 4.1 增量表

`coverage_aware − query_recency`，正数表示同等预算下多覆盖的 query 比例：

| regime | Q | k | Δ vs query_recency | Δ vs learned_query | query_recency 已饱和 |
|---|---:|---:|---:|---:|---|
| current | 8 | 4 | **+0.125** | +0.125 | 否 |
| current | 8 | 8 | **+0.000** | +0.000 | **是（1.000）** |
| current | 8 | 16 | **+0.000** | +0.000 | **是（1.000）** |
| current | 16 | 4 | **+0.271** | +0.271 | 否 |
| current | 16 | 8 | **+0.271** | +0.271 | 否 |
| current | 16 | 16 | **+0.000** | +0.000 | **是（1.000）** |
| historical | 8 | 4 | **+0.125** | +0.125 | 否 |
| historical | 8 | 8 | **+0.000** | +0.000 | **是（1.000）** |
| historical | 8 | 16 | **+0.000** | +0.000 | **是（1.000）** |
| historical | 16 | 4 | **+0.208** | +0.208 | 否 |
| historical | 16 | 8 | **+0.188** | +0.188 | 否 |
| historical | 16 | 16 | **+0.000** | +0.000 | **是（1.000）** |

12 个单元格中有 **6 个** `query_recency` 已经 1.000（`k ≥ Q` 或 `k ≥ 不同 query key 数`）；在这 6 个单元格里 coverage-aware 也都是 1.000，增量为 **0.000**。这是覆盖率指标的上界性质：coverage 最大就是 1.0，query_recency 已经覆盖全部 query 时，任何 coverage 目标都不可能再增加。

### 4.2 为什么 k<Q 时 coverage-aware 更高

原因不是 value resolution，而是**共享预算没有被 query_recency 用完**：

| cell | query_recency 实际选中数 | coverage_aware 选中数 | query_recency coverage | coverage_aware coverage |
|---|---:|---:|---:|---:|
| current Q=8 k=4 | 3.3 | 4.0 | 0.542 | 0.667 |
| current Q=16 k=4 | 2.7 | 4.0 | 0.292 | 0.562 |
| current Q=16 k=8 | 5.3 | 8.0 | 0.583 | 0.854 |
| historical Q=16 k=8 | 8.0 | 8.0 | 0.521 | 0.708 |

这里有两个相关的预算分配效应。第一，`query_recency` 对每个 query 各取最近 b 条后做并集，当多个 query 落在同一个 key 上时并集会小于 `k`（current Q=8,k=4 平均只选出 3.3 条；Q=16,k=4 为 2.7；Q=16,k=8 为 5.3），剩下的槽位没有被重新分配。第二，即使并集刚好用满 `k`（historical Q=16,k=8 为 8.0 条），`query_recency` 也只是按 key 频率决定“哪些 query 拿到槽位”，而不是按“一个 fact 能同时覆盖多少 query”来选；多个 query 共享同一个 key 时，greedy 可以用一条 fact 覆盖它们，从而用同样的 8 条 fact 覆盖更多 query。两种效应都属于 **budget allocation / query multiplicity**，不是“找到了更正确的 value”。

### 4.3 learned_query 没有超过 query_recency

在全部 12 个单元格中，`learned_query` 的 accuracy / recall / coverage 与 `query_recency (as_of)` **逐位相等**（包括 k=4/8 的欠预算情形）。这与 U5 合成流的结论一致：query-conditioned 的收益来自“按 query 选集合”，而不是“学到一个更强的模型”；这里连“学习”都被一个 as_of-aware 的最近规则完全复现。训练 replay seed 1/2/3、测试 replay seed 0 的划分没有改变这一点。

### 4.4 as_of 是必要条件

historical regime 下，`query_recency (no as_of)` 的 accuracy/coverage/recall 全部是 **0.000**：它取的是 key 的全局最近 fact，而这些 fact 往往写在 `as_of` 之后，value 已经因 visibility 变化而不同。`query_recency (as_of)` 与 `learned_query` 在 k 足够时回到 1.000。因此，本流上真正的“必要条件”是 **as_of-aware query conditioning**；coverage loss 只是在预算不足时改善多 query 的分配。

### 4.5 coverage 与 accuracy 不完全等价

`coverage_aware + learned` 在 historical 的 6 个单元格中 coverage 全是 1.000，但 accuracy 有 4 个低于 1.000（0.875 / 0.833 / 0.667 / 0.938）。原因是 greedy 的 tie-break 选了更早的同 value fact；同一 key 上另一个被选中的 fact 可能更晚但 value 不同，于是“取最新选中 fact”的解析规则给出错误答案。coverage 只保证“集合里至少有一条正确证据”，不保证“按最新规则解析仍正确”。这是 coverage loss 与 answer accuracy 之间的真实边界，不能把 coverage 1.000 直接当成 accuracy 1.000。

---

## 5. 可以 / 不可以声称

### 可以声称

1. 仓库里没有真实 fact 流；本轮使用 `s1_occlusion_20` 回放 bundle 抽出的 **21,168** 条 visibility fact（20 keys、1200 steps、169 次状态变化），并明确标为 replay，不是 real。
2. 在这个回放长历史上，`query_recency` 在 `k ≥ Q` 时 coverage 达到 **1.000**；`coverage_aware` 子模 greedy 在同样单元格也是 **1.000**，增量 **0.000**。
3. 在 `k < Q` 的欠预算单元格，`coverage_aware` 比 `query_recency` 高 **+0.125 ~ +0.271**；机制是 query_recency 的 per-query 并集没有用满 k 个槽位，而 greedy 按 query multiplicity 用满预算。
4. `learned_query` 在全部 12 个单元格与 `query_recency (as_of)` 完全相等；没有证据表明学习模型带来额外增益。
5. historical 查询下忽略 `as_of` 的 recency 为 **0.000**；as_of-aware 是必要条件。

### 不可以声称

1. **不可以**声称 coverage loss 在 `query_recency` 已经 1.000 时还有增量；6/6 个饱和单元格的增量都是 0.000，数学上也没有空间。
2. **不可以**声称 coverage-aware 的增益来自“更好的 value resolution”。它来自预算分配；k≥Q 时它和 query_recency 打平，k<Q 时它赢在“用满槽位 + 按 query 共享选 fact”。
3. **不可以**声称这是真实 fact store 的结果。数据是仿真 MuJoCo/Isaac occlusion replay，只有 visibility 一个动态属性，pose 几乎静止，没有矛盾写、没有多属性 key。
4. **不可以**把 coverage 1.000 外推为 accuracy 1.000；`coverage_aware + learned` 在 historical 有 coverage 1.000 但 accuracy 0.667–0.938 的反例。
5. **不可以**外推到更大的 Q、更长的 key 集合、真实传感器噪声或生产更新频率；本轮只有 20 keys、Q≤16、单条 replay。

---

## 6. 与 U5 合成流的对照

| 维度 | U5 合成流 | U5b 回放流 |
|---|---|---|
| 数据 | `T=1000` 合成 fact，50 keys，16 values | 21,168 条 replay fact，20 keys，9 个 visibility 值 |
| query_recency | k=4/8/16 均 1.000 | k≥Q 时 1.000；k<Q 时 0.250–0.583 |
| learned_query | 与 query_recency 打平 | 与 query_recency 逐位打平 |
| hot | 0.237 / 0.421 / 0.473 | current 0.167–0.188；historical 0.000 |
| coverage-aware | 未测 | k≥Q 时 1.000（无增量）；k<Q 时 +0.125 ~ +0.271 |

U5 的核心结论“缺的是 query conditioning，不是学习模型”在回放流上复现了；U5b 进一步说明：**coverage loss 不是新的必要条件**——在 query conditioning 已经让 coverage 饱和时它没有增量，在预算不足时它只是把共享预算分配得更好。

---

## 7. 产物与复现

| 产物 | 路径 |
|---|---|
| U5b 脚本 | `pss/eval/scripts/run_upper_bound_u5b.py` |
| 指标 JSON | `eval/results/upper_bound/u5b/metrics.json` |
| 自动报告 | `eval/results/upper_bound/u5b/REPORT.md` |
| 运行日志 | `eval/results/upper_bound/u5b.log` |

```bash
python -u -m pss.eval.scripts.run_upper_bound_u5b \
  --seeds 3 --qs 8,16 --ks 4,8,16 \
  --regimes current,historical --epochs 40 \
  --out eval/results/upper_bound/u5b
```

运行环境：CPU、PyTorch、numpy、pandas/pyarrow；依赖 `bundles/dev/s1_occlusion_20`（gitignored 数据）。若该目录不存在，脚本会明确报 `FileNotFoundError`，不会静默回退到合成流。

---

## 8. 边界

1. **没有真实 fact 流。** 本报告使用的是仿真回放，不是真实传感器或生产 fact store；标题与 §1.3 已明确标注 replay。`real_fact_stream_available=false` 写入 `metrics.json`。
2. **单一动态属性。** 只有 visibility 会变化；pose 在整条轨迹上几乎不变，因此没有连续数值 drift。coverage 的“value 变化”来自 visibility 状态切换，不是多属性事实冲突。
3. **oracle support。** greedy 使用由答案规则诱导的 `support` 矩阵（key + target value + `ts ≤ as_of`），不是可部署的 relevance 分类器；真实部署需要先解析 query、再判断 value 是否支持。
4. **没有矛盾写。** replay 的 GT 是唯一真相，没有低可靠度冲突写；因此 `reliability` 只作为 feature / tie-break，不参与答案解析。
5. **单条 replay，20 keys。** 训练用 replay seed 1/2/3，测试用 seed 0；没有跨数据集、跨任务或生产分布测试。Q≤16、k≤16 的结论不能外推到更大规模。
6. **std 是 3 个 run 的总体标准差。** 同一 replay seed 的 test 流固定，变化来自训练 replay 与 query 采样；这是 3-seed 诊断，不是大样本置信区间。
7. **coverage 不等于 accuracy。** coverage 只数“至少一条 supporting fact”；按最新 fact 解析答案时可能被同 key 的更晚错误 value 覆盖，§4.5 有反例。
