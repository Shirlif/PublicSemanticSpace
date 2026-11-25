# Isaac 评测预注册 v1（A4）

> 写于：2026-11-25（先于 official 跑测）
> 对应：`PSS-IsaacSim-评测平台设计.md`、`PSS-IsaacSim-实施计划.md`
> 原则：**指标、阈值、种子、降采样、停止规则先冻结，再跑评测；失败照实报告。**
> 平台：ADR-001 —— 固定桌面 + 脚本化相机轨迹；G1 仅 `base_fixed` 上肢；不做人形腿。

---

## 1. 假设与范围

| 编号 | 假设 | 证据类型 |
|---|---|---|
| H1 | 显式实体状态在遮挡与视角变化下保持身份，优于无持久槽的基线 | S1 IDF1/retention |
| H2 | 实体记忆跨 episode 保留；物理改位后旧事实被覆盖 | S6 edited/old |
| H3 | 关系从感知位姿计算，组合 OOD 不崩 | S3 per-predicate F1 |
| H4 | 记忆回滚逐字段还原 | S6 rollback exactness |
| H5 | 认知层延迟与实体数近似线性、可控 | S8 p50/p95 |

**范围声明**：
- 主结论只来自 `frozen_real` 检测档；`oracle` 只作上界标定，不作能力证据。
- 不声称人形移动/行走能力；`base_mobile` 不存在。
- S7 本期只报**检测器分数校准**；PSS 置信度/弃权（abstain、conflict F1）列为探索性，不进入主结论。

---

## 2. 套件、指标与阈值

| 套件 | 主指标 | 阈值 / 判定 | 备注 |
|---|---|---|---|
| **S1 遮挡 Re-ID** | IDF1 | `frozen_real` 报告均值 + 95% CI；oracle ≥ 0.95 为标定门禁 | 20 物体、40 episodes × 5 seeds |
| | id_retention、switches、duplicate、reappearance | 报告，不作硬阈值 | 与 IDF1 同表 |
| **S2 孪生** | duplicate_rate | 报告；`frozen_real` 为主 | abstain precision 列为探索性（当前 n/a） |
| **S3 关系** | per-predicate F1、consistency violation | macro/micro F1 报告；violation 必须可量化 | 本期为几何关系管线，非 learned-head OOD 证据 |
| **S6 记忆** | rollback_exactness | **必须 = 1.0**（门禁） | edited/old/retention 报告 |
| **S7 校准** | Brier、ECE、selective accuracy | 检测器分数校准报告 | PSS 置信度/conflict F1 为探索性 |
| **S8 成本** | p50/p95 latency、entities/facts | 报告 | footprint/token 若未打点则写 null，不估算 |

### 消融

- 设计 §7 的 9 个变体在 official 上跑；离线套件共享同一批 bundle。
- 每个 delta 用**按 episode 配对的 bootstrap**，报告 95% CI；不做按 seed 配对。
- `flat_context` / `no_tools` 只在 S1/S4 上跑，其余不跑。

---

## 3. 种子与降采样

| 项 | official（3090 Ti） | dev（4070） |
|---|---|---|
| 非闭环 seeds（S1/S2/S3/S6） | 5 | 1 |
| 闭环 seeds（S4/S5） | 3 | 1 |
| S1 物体数 | 20 | 20（冒烟可 4） |
| S4 变体 | 全 9 | 抽样 3 |
| S5 故障类 | 10 | 3 |
| 分辨率 | 640×480 | 480×640 |
| 并行场景 | 8–16 | 2–4 |

- 降采样只允许出现在 **dev**；official 不允许抽样变体/故障类。
- 所有降采样必须在本文件可查；临时变更写进 `reports/<tag>/overrides.jsonl`。

---

## 4. 设备与版本

- 所有 official 数字必须来自**同一台 3090 Ti**，`device_class=official`。
- Isaac Sim / Isaac Lab / PhysX / 驱动版本必须与 `freeze/p7-official.json` 逐项一致；
  任一端升级即视为新 tier，需重跑。
- 4070 只用于 dev 冒烟、离线重放、A9 复算；不得混入 official 统计。
- 图形后端固定 D3D12（`--/app/vulkan=false`），两机一致。

---

## 5. 停止规则（预注册）

| 触发条件 | 动作 |
|---|---|
| oracle IDF1 < 0.95 | 停，查 GT/指标实现，不调 PSS |
| `frozen_real` 与 oracle 差距 < 0.10 | 停，怀疑检测器未真正接入 |
| S6 rollback exactness < 1.0 | 停，修指标/状态引擎 |
| 无故障 S4 execution success < 0.8 | 停，换抓取方案，不硬调 |
| A9 复算逐位不一致 | 停线，修指标引擎 |
| 任何 official episode 混入 dev | 该表作废，重跑 |

---

## 6. 分析与报告

- 每张 scorecard 必须含：`spec hash`、`detector_tier`、`variant`、`seeds`、`device_class`、95% CI、bundle 路径。
- 主表只报 `frozen_real`；oracle/noisy 作为对照，不进入结论句。
- 失败/缺失指标写 null 并注明原因，不用 0 或估计值填充。
- 预注册之后新增的分析一律标为 **exploratory**，不得改主结论。

---

## 7. 归档（A10）

`reports/<tag>/` 必须包含：

```
freeze.json            版本五元组
specs/                 所有 spec + registry
prereg.md              本文件
scorecards/            每套件一表
bundles.index.parquet  episode → 路径 → 指标行号
verify.json            A9 复算结果
overrides.jsonl        所有 --force / 降采样变更
notes.md               人工判断与已知偏差
```

`pss report --tag <tag>` 负责组装；做不到一键重出全部分数，A10 不通过。
