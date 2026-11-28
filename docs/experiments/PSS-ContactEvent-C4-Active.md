# PSS ContactEvent C4：接触事件的主动感知（probe / re-observe）

> 日期：2026-11-29
> 上游：`docs/plans/PSS-ContactEvent-计划.md` §6.5、§8（C4）
> 状态：已完成最小实验；结论为**条件性有效**，不是全面有效。
> 范围：MuJoCo proxy，不是真实机器人；C4 只改变观测，不下发物理 push/tap。

---

## 0. 一句话

**在 live MuJoCo 场景里，passive 固定低速率观测；active 在 force/geometry 证据冲突或几何接近时自适应 re-observe；若事件不在标称时间窗内，则 wait 延长观测。** 三组观测预算下，active 都恢复了 passive 漏掉的事件并把 onset latency 从 14–36 ms 降到约 0.3 ms；代价是每个事件场景多 24–48 帧，wait 条件额外多 60–106 个仿真步。**未使用物理 probe；C2 fusion 未接入，不确定性信号是 force/geometry 的简单合议。**

---

## 1. 方法与口径

### 1.1 Live 场景

- 复用 `pss/sim/scene.py` 的 MuJoCo 桌面与 `pss/sim/sensors.py` 的 `SensorRig`；
- 场景与 C0/C1 的 `pss/events/scenarios.py` 一致：
  - `drop`：球从箱子上方 0.35 m 落下；
  - `collide`：对球施加 +x 2.5 N、60 步的外部推力撞向箱子；
  - `near_miss`：两物体相距较远、无接触（误报对照）。
- `dt = 0.002 s`，仿真逐步推进；每一步都记录 dense 观测（仅用于 GT 与调试，不计入两臂的观测预算）。

### 1.2 Passive 与 Active 观测器

两臂看**同一条物理轨迹**：

| 观测器 | 行为 |
|---|---|
| passive | 固定每 `base_every` 步取一帧；标称时间窗结束后停止 |
| active | 同样的 base 帧；策略触发后按 `dense_every` 取帧 |

主动动作：

- **reobserve**：在标称时间窗内，几何/证据不确定时开启一段密集观测窗口（`reobserve_steps`，最多 3 次、总帧数预算 200）；
- **wait**：到标称时间窗末尾仍不确定（几何接近但尚未出现接触）时，延长仿真最多 `max_wait_steps`，密集观测到接触出现并结束后停止；
- **收束**：一旦接触被确认并在 `end_margin_steps = 30`（60 ms，覆盖 GT 的 `max_gap = 50 ms`）内结束，停止主动观测，避免事件后无谓刷新。

### 1.3 不确定性信号（C2 fusion 未接入）

C4 按要求使用**简单合议**，不是 C2 的校准后验：

- force score：`f = F / (F + 1.0)`；
- geometry score：`g = sigmoid((0.02 - gap) / 0.01)`，`gap = |p_a - p_b| - r_a - r_b`；
- 触发原因按严重度取最早/最强：
  1. `near`：`gap < 0.06 m`（几何预测接近/接触）；
  2. `conflict`：`|f - g| > 0.35`；
  3. `borderline`：`f` 或 `g` 落在 `(0.15, 0.85)`。
- 本次所有触发的原因都是 `near`；`conflict` / `borderline` 分支未成为第一触发原因（见 §6）。

> 说明：工作区里 C2 的 `pss/events/fusion.py` 在本任务并行落地。C4 仍按计划使用 force+geometry 合议，以保持主动策略与 C2 解耦；把 fusion 后验接进主动策略是后续工作。

### 1.4 为什么不用物理 push/tap

物理 probe 会改变轨迹，从而改变 GT；这样 passive/active 比较会混入“probe 造出来的接触”。C4 的最小范围只做**观测型主动感知**（reobserve/wait），保证两臂共享同一条物理轨迹和同一份 GT。物理 probe 留到后续单独实验，并需要单独的 GT 口径。

### 1.5 GT 与事件估计

- GT：**固定参考轨迹**上的 `events_from_contacts`；参考轨迹长度 = `nominal + max_wait_steps`，**与 active 是否 wait 无关**，避免 GT 被策略选择污染；
  - force > 1.0 N，`min_duration = 0.02 s`，`max_gap = 0.05 s`；
- 两臂的事件估计：各自观测帧上的 `run_channel(..., channel="force")`；
  - force score threshold 0.5 恰好对应 1.0 N；
- passive 的观测在 `nominal` 步停止，但仍按固定参考 GT 评分；`short` 条件中发生在标称窗之后的事件对 passive 记为 FN，这正是该条件要测的“固定时间预算”。
- 事件匹配：同 pair，时间 IoU ≥ 0.3 **或** onset 差 ≤ 50 ms（`evaluate_events` 默认）。

---

## 2. 实验设置

三组观测预算，每组 3 seeds（0/1/2）：

| condition | passive 周期 | 标称时间窗 | 说明 |
|---|---:|---:|---|
| `fine` | 10 步 = 20 ms | 400 步 = 0.8 s | 较高频率的被动基线 |
| `coarse` | 30 步 = 60 ms | 400 步 = 0.8 s | 低频率、但时间窗完整 |
| `short` | 10 步 = 20 ms | drop 110 步 / collide 30 步 | 时间窗在事件前结束，考察 wait |

公共参数：`dense_every = 2`（4 ms）、`reobserve_steps = 100`、`max_wait_steps = 150`、`max_episodes = 3`、`max_extra_frames = 200`。

复现：

```bash
python -u -m pss.eval.scripts.run_contact_active \
  --conditions fine coarse short --seeds 0 1 2 \
  --out eval/results/contact_active
```

结果：`eval/results/contact_active/contact_active.json`、`eval/results/contact_active/CONTACT_ACTIVE_REPORT.md`。

---

## 3. 主结果

### 3.1 事件场景总表（drop + collide，6 runs）

每格为跨 run 的 mean ± std；`micro F1` 为 tp/fp/fn 汇总后计算。

| condition | arm | precision | recall | F1 (macro) | micro F1 | onset latency | extra frames | extra steps | probes / run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fine` | passive | 1.000 | 0.833 ± 0.373 | 0.833 ± 0.373 | 0.909 | 14.0 ± 4.0 ms | 0 | 0 | 0 |
| `fine` | **active** | 1.000 | 1.000 | **1.000** | **1.000** | **0.3 ± 0.7 ms** | 31.0 ± 7.1 | 0 | 1.0 |
| `coarse` | passive | 1.000 | 0.167 ± 0.373 | 0.167 ± 0.373 | 0.286 | 36.0 ± 0.0 ms | 0 | 0 | 0 |
| `coarse` | **active** | 1.000 | 1.000 | **1.000** | **1.000** | **0.3 ± 0.7 ms** | 34.2 ± 10.3 | 0 | 1.0 |
| `short` | passive | 1.000 | 0.000 | 0.000 | 0.000 | — | 0 | 0 | 0 |
| `short` | **active** | 1.000 | 1.000 | **1.000** | **1.000** | **0.3 ± 0.7 ms** | 39.2 ± 9.5 | 78.7 ± 19.2 | 1.0 |

### 3.2 分场景

| condition | arm | kind | passive→active F1 (micro) | onset latency (passive → active) | extra frames | extra steps |
|---|---|---|---:|---|---:|---:|
| `fine` | passive→active | drop | 1.000 → 1.000 | 16.0 → 0.0 ms | 24 | 0 |
| `fine` | passive→active | collide | 0.800 → 1.000 | 11.0 ± 5.0 → 0.7 ± 0.9 ms | 38.0 ± 1.6 | 0 |
| `coarse` | passive→active | drop | 0.000 → 1.000 | — → 0.0 ms | 24 | 0 |
| `coarse` | passive→active | collide | 0.500 → 1.000 | 36.0 → 0.7 ± 0.9 ms | 44.3 ± 2.1 | 0 |
| `short` | passive→active | drop | 0.000 → 1.000 | — → 0.0 ms | 30 | 60 |
| `short` | passive→active | collide | 0.000 → 1.000 | — → 0.7 ± 0.9 ms | 48.3 ± 3.4 | 97.3 ± 6.3 |

### 3.3 近失对照（near_miss，GT 为空）

| condition | passive false alarms | active false alarms | active probes |
|---|---:|---:|---:|
| `fine` | 0 | 0 | 0 |
| `coarse` | 0 | 0 | 0 |
| `short` | 0 | 0 | 0 |

两臂在 near_miss 上都没有误报；active 没有触发任何动作，代价为 0。这说明本次主动策略没有靠“到处多看”制造误报。

### 3.4 动作次数与代价

- **probe/action 总次数**：18 次（每个有事件的 run 恰好 1 次）。
  - `fine`：6 次 `reobserve`；
  - `coarse`：6 次 `reobserve`；
  - `short`：6 次 `wait`；
  - `near_miss`：0 次。
- **物理 push/tap 次数**：0。
- **帧代价**（每个事件 run）：
  - `fine`：passive 40 帧 → active 71 帧（+31.0，约 +78%）；
  - `coarse`：passive 13 帧 → active 47.2 帧（+34.2，约 +263%）；
  - `short`：passive 平均 7 帧 → active 46.2 帧（+39.2）；另有 78.7 个额外仿真步（约 157 ms）。
- **时间代价**：`fine` / `coarse` 的 reobserve 在标称窗内，不增加仿真时间；`short` 的 wait 平均多 78.7 步（drop 60 步、collide 97.3 步）。

---

## 4. 结果分析

### 4.1 `fine`：F1 +0.167 / micro +0.091，延迟 14 ms → 0.3 ms

- drop 的 passive 在 20 ms 网格上仍能命中，但 onset 被量化到 16 ms；active 在 step 110（t=0.220）由 `gap < 0.06` 触发 reobserve，在 4 ms 网格上看到 onset 0.244，延迟 0。
- collide seed 2 的 passive **完全漏检**：GT 事件跨 0.088–0.212 s（中间有短暂力跌落，被 `max_gap = 50 ms` 合并），但 20 ms 采样点上的高力点只有 0.10/0.12，0.14/0.16 已跌到阈值下；segmenter 只得到 `[0.10, 0.12]`，浮点上 `0.12 - 0.10 = 0.019999... < 0.02`，又被 `min_duration` 丢弃。active 的密集窗口看到了被合并的完整高力区间，恢复该事件。
- 代价：每个事件 run 约 31 个额外帧，不增加仿真步。

### 4.2 `coarse`：F1 +0.833 / micro +0.714，延迟 36 ms → 0.3 ms

- 60 ms 网格下，drop 的 36 ms 接触完全落在采样点之间（step 120 未接触、step 150 已结束），passive F1=0；active 在 step 120 由 `gap < 0.06` 触发，恢复事件。
- collide seed 0 在 60 ms 网格上恰好采到 0.12 和 0.18 两个高力点，passive 命中但延迟 36 ms；seed 1/2 的高力点只落在一个采样点上，`min_duration` 丢弃，passive 漏检；active 全部恢复。
- 代价：每个事件 run 约 34 个额外帧，不增加仿真步。

### 4.3 `short`：F1 0 → 1，延迟 — → 0.3 ms

- 标称窗在接触前结束：drop 在 0.22 s 停止、collide 在 0.06 s 停止，passive 没有机会看到接触，F1=0。
- active 在标称窗末尾看到 `gap < 0.06`，发起 `wait`；密集观测在 0.244 s / 0.084–0.094 s 捕获 onset，并在接触结束后停止。
- 这一组的主要收益来自**时间预算延长**（wait），不是单纯采样加密；代价是平均 78.7 个额外仿真步，是本组最贵的动作。

### 4.4 触发原因分布

所有主动动作的第一触发原因都是 `near`（几何接近）。`conflict` 和 `borderline` 在接近阶段没有先于 `near` 触发，因此本实验**没有单独验证“证据冲突触发”这一分支**；能说的是：几何接近作为低置信信号足以驱动 reobserve/wait，随后 force 证据完成确认/否证。

---

## 5. 失败项与限制

1. **没有物理 probe**：push/tap 未实现，原因是会改变轨迹与 GT；“主动感知是否值得付出物理动作”未被本实验回答。
2. **C2 fusion 未接入**：不确定性是手写 force/geometry 合议，不是校准后验；概率阈值/可靠性没有校准。
3. **Sim proxy**：MuJoCo 图元、force 近 oracle；没有真实传感器噪声、带宽、延迟、遮挡。不能声称 sim2real。
4. **GT 与 force 通道同源**：GT 由 MuJoCo contact force 生成，事件臂也用 force；这是一个上限/循环性风险。geometry/vision 的独立证据没有用于事件判定。
5. **drop 的 3 个 seed 方差为 0**：本次观测到的 drop 力/事件在 seed 0/1/2 上相同，主要因为初始位姿与几何固定；不能把 drop 的 3 seeds 当作独立重复，3 seeds 只对 collide 提供了轻微变化。
6. **passive 漏检包含采样混叠与浮点边界因素**：collide 的漏检部分是 force 抖动在稀疏采样点上的混叠，以及 `0.12 - 0.10` 略小于 20 ms 的浮点边界；这是真实的稀疏观测失效模式，但对 `base_every`/`min_duration` 取值敏感。
7. **收益依赖 passive 预算**：如果 passive 已经足够密（例如 C1 smoke 的 4 ms 采样，force 在 drop/collide 上 F1=1.0），active 不会带来 F1 增益，只会增加帧/时间代价。C4 的结论是“在欠采样或时间受限时有效”，不是“总是有效”。
8. **动作代价模型很粗**：只用额外帧数和额外仿真步数衡量，没有真实带宽、功耗、运动时间或机会成本。
9. **未做多事件/长时程/遮挡/冲突场景**：C3 的冲突场景没有并入 C4 主表；主动策略在连续多个事件、遮挡、tactile 缺失下的行为未测。

---

## 6. 结论与决策

### 6.1 是否有效

**条件性有效（conditional go）**：

- `fine`：macro F1 0.833 → 1.000，micro F1 0.909 → 1.000，onset latency 14.0 → 0.3 ms；每个事件 run +31 帧、0 额外步。
- `coarse`：macro F1 0.167 → 1.000，micro F1 0.286 → 1.000，onset latency 36.0 → 0.3 ms；每个事件 run +34 帧、0 额外步。
- `short`：macro/micro F1 0 → 1.000；每个事件 run +39 帧、+78.7 步。
- near_miss：两臂 0 误报，active 0 动作、0 代价。
- **没有物理 probe；没有 C2 fusion；没有真实机器人证据。**

### 6.2 建议

- 把 C4 视为“欠采样/固定时间预算下的主动观测”可行性的正面证据，而不是默认开启主动感知的理由。
- 下一步优先做：
  1. 接入 C2 fusion 的校准后验作为不确定性信号，替代手写阈值；
  2. 物理 probe/tap 的独立实验（需要新的 GT 口径与代价模型）；
  3. 在 C3 的遮挡/冲突/近失场景上验证 `conflict`/`borderline` 分支确实能触发；
  4. 把动作代价做成信息增益/帧成本的可调目标，而不是固定 `reobserve_steps`。
- 在 passive 预算足够密（≤4 ms）时，不建议默认开启 active；保持 passive，必要时再按需 reobserve。

---

## 7. 复现与文件

```bash
python -u -m pss.eval.scripts.run_contact_active \
  --conditions fine coarse short --seeds 0 1 2 \
  --out eval/results/contact_active
```

新建/修改：

- `pss/events/active.py`：live 场景 + passive/active 观测策略 + 事件重估计；
- `pss/eval/scripts/run_contact_active.py`：三条件 runner 与聚合；
- `docs/experiments/PSS-ContactEvent-C4-Active.md`：本报告；
- 输出：`eval/results/contact_active/contact_active.json`、`eval/results/contact_active/CONTACT_ACTIVE_REPORT.md`。

未修改 C0–C3 文件。

---

## 8. 边界声明

本实验是 **MuJoCo primitive proxy**，不是真实机器人、真实力/触觉传感器或真实主动感知系统。force/tactile 接近 oracle，audio 未使用，vision 未接入事件判定。所有 F1/延迟/代价数字只应在这个 proxy 与这些观测预算下解读，不应外推为真实机器人性能或 sim2real 结论。
