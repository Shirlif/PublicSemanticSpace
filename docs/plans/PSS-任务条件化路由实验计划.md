# PSS 任务条件化路由实验计划：从固定融合到“谁有话语权谁说话”

> 日期：2026-11-29
> 上游：`docs/plans/PSS-ContactEvent-计划.md`、`docs/experiments/PSS-ContactEvent-Report.md`、`docs/experiments/PSS-ContactEvent-Coverage.md`
> 目的：修正“固定 fusion 在接触任务上被 force 支配”的结论——不是融合错，是**固定权重**错；验证**任务条件化的权威路由**：先按任务决定谁有话语权，再按当前观测调可靠性，权威通道不可用时弃权/降级。
> 状态：方案，未执行。

---

## 0. 一句话

**ContactEvent 的 48-cell 覆盖证明的是“全局固定 fusion 在接触任务上不赢 force”，不是“融合方案错”。** 接触任务里 force 是权威，几何只是预测/定位；距离/关系任务里 force 根本用不上，geometry/vision 才是权威。下一步不是继续拧 fusion 权重，而是做 **task-conditioned authority routing**：

```
w_c = softmax( A(task, c)/T + log R_c(observation) + b_c )
p(y | e) = Σ_c w_c · p_c(y | e_c)
```

`A` 是任务先验权威度，`R` 是当前观测可靠性，`T/b` 在 cal 上校准；权威通道不可用/不可信时弃权或主动感知。

---

## 1. 为什么要重做

### 1.1 固定 fusion 的负结果

- C2 clean force = oracle；noisy force 下 fusion +0.229；
- 48-cell 覆盖：fusion 只在 **1/48** cell 有 +0.05，47/48 `force_noisy` 不输 fusion；
- 原因不是融合算法差，而是**任务本身权威高度不对称**：碰撞/接触由 force 决定，几何测不准碰撞。

### 1.2 任务相对性

| 任务 | 权威通道 | 辅助通道 | 固定 fusion 的问题 |
|---|---|---|---|
| 碰撞/接触 | **force / tactile** | geometry / audio / vision | geometry 噪声被平均进来，稀释 force |
| 距离/相对关系 | **geometry / vision** | force（通常不可用） | force 缺席或无关，固定权重仍给它份额 |
| 材质/冲击 | **audio / tactile** | force | force 只能给强度，不给材质 |
| 遮挡/身份 | **vision** | force / tactile | force 不提供身份/可见性 |

所以正确形态不是“一个大 fusion”，而是**按任务路由**。

---

## 2. 任务与权威表

### 2.1 任务套件

| task | 标签 | 权威通道 | 数据 |
|---|---|---|---|
| `contact` | 是否接触 / onset/offset / 强度 | force, tactile | MuJoCo contact force |
| `distance` | 是否近 / left/right/above/below | geometry, vision | MuJoCo poses + render |
| `material` | 材质类别 / 冲击类型 | audio, tactile | MuJoCo material + 合成音频 |
| `occlusion` | 是否可见 / 身份是否保持 | vision | occluder + segmentation |

### 2.2 权威先验 `A(task, channel)`

初始表（声明，跑前固定；后续可在 cal 上学习）：

| task \ channel | force | tactile | geometry | vision | audio |
|---|---:|---:|---:|---:|---:|
| contact | 0.50 | 0.20 | 0.20 | 0.05 | 0.05 |
| distance | 0.00 | 0.00 | 0.70 | 0.30 | 0.00 |
| material | 0.10 | 0.40 | 0.00 | 0.10 | 0.40 |
| occlusion | 0.00 | 0.00 | 0.10 | 0.90 | 0.00 |

- 这是**先验**，不是最终权重；最终 `w` 还要乘 `R_c`。
- 缺失通道直接 mask 掉（权重 0）。

---

## 3. 路由模型

### 3.1 公式

```
logits_c = A(task, c) / T + log R_c(observation) + b_c
w = softmax(logits over available channels)
p(y | e) = Σ_c w_c · p_c(y | e_c)
```

- `A`：任务先验权威度（表或学习）。
- `R_c`：当前观测可靠性，`R_c ∈ (0,1]`；不可靠时 `log R_c` 拉低权重。
- `T`：温度，cal 上选。
- `b_c`：通道偏置，cal 上选，防止某个通道永久压制。
- 缺失通道：`w_c=0`。
- 弃权：`max(w) < τ_w` 或权威通道不可用且辅助通道 confidence < τ_c 时 abstain。

### 3.2 可靠性 `R_c` 的来源

| 通道 | 可靠性代理 |
|---|---|
| force | 噪声水平（残差方差）、丢帧率、延迟（与几何预测的互相关）、饱和 |
| tactile | 覆盖面积/接触点数、信噪比 |
| geometry | 与 force/tactile 的一致性、模型错配、遮挡/可见性 |
| vision | mask 面积/遮挡比例、帧差/运动模糊 |
| audio | 冲击能量/SNR、背景噪声 |

第一版可以先用**已知的传感器模型参数**（σ/dropout/delay/mismatch）作为 `R`，验证路由逻辑；第二版再换成**在线估计**。

### 3.3 任务来源

- **declared task**：由上层任务指定（推荐，无泄漏）。
- **task classifier**：从观测/上下文估计任务；必须用 held-out 任务验证，避免“任务标签泄漏”。
- 本计划主实验用 declared task；classifier 作为 P2 扩展。

---

## 4. 数据与场景

### 4.1 MuJoCo 多任务场景

复用 `pss/sim` 与 `pss/events/scenarios*`：

| task | 场景 | 标签 |
|---|---|---|
| contact | drop / collide / near_miss / occlusion | contact force ≥1 N；onset/offset |
| distance | 2/4 objects，不同间距与方位 | `near`、left/right/above/below |
| material | 不同 material 的物体碰撞/滑动 | material 类别 |
| occlusion | occluder 遮挡 + 物体移动 | 可见性 / ID retention |

### 4.2 通道可用性

每个任务只给“该任务可用的通道”：

- contact：force/tactile/geometry/audio；
- distance：geometry/vision（force 不可用）；
- material：audio/tactile/force；
- occlusion：vision/force。

这样测试“权威通道缺席时，路由是否降级/弃权”。

### 4.3 数据划分

- train/cal/test 按 scene/seed 划分；
- `A`、`T`、`b`、可靠性校准只在 cal 上选；
- test 只报一次；
- 跨任务：leave-one-task-out。

---

## 5. 基线 / 对照

| arm | 说明 |
|---|---|
| `single_<channel>` | 每个任务上每个通道单独 |
| `fixed_fusion` | C2 的全局 logistic（对照：固定权重） |
| `task_oracle` | 已知任务 + 手写 `A`，无 `R` |
| `task_gate` | 已知任务 + `A` + `R` + 弃权（主候选） |
| `task_gate_learned` | `A` 在 cal 上学习 |
| `oracle_task_channel` | 已知任务 + 已知最佳通道（上限） |
| `task_classifier_gate` | 任务由分类器估计（P2） |

---

## 6. 指标与预注册 bar

### 6.1 主指标

- **每个任务单独**的 F1 / exact-4 / 误差；不报跨任务平均掩盖。
- `gate` vs 该任务最好单通道：`≥ best_single − 0.02`（不回退）；
- 在 fixed fusion 明显差的 task 上：`gate ≥ fixed_fusion + 0.05`。

### 6.2 路由诊断

| 指标 | bar |
|---|---|
| 权威通道平均权重 | ≥0.50（在 contact/distance/material/occlusion 各自） |
| 非权威通道权重 | 显著低于权威（≥0.2 差距） |
| 缺失权威通道时 | 降级到辅助通道或 abstain，不硬用缺失通道 |
| ECE | ≤0.05 |
| abstention precision @ coverage≥0.8 | ≥0.90 |
| 路由开销 | ≤5 ms/帧 |

### 6.3 跨任务

- leave-one-task-out：`gate ≥ best_single − 0.03`；
- 若跨任务失败，说明路由依赖任务先验，不能自动迁移；需要 declared task 或 task classifier。

---

## 7. 实验阶段

### R0：任务套件与单通道 baseline

- 四个任务、各自标签、各自可用通道；
- 每通道单独 F1/误差；
- 出口：权威表与单通道表现一致（例如 contact 上 force 最好、distance 上 geometry 最好）。

### R1：权威先验

- 用声明表 `A`，不做 `R`；
- 对照 `fixed_fusion`；
- 出口：`task_oracle` 是否已经在 fixed fusion 之上。

### R2：可靠性估计

- 先用已知传感器参数（σ/dropout/delay/mismatch）做 `R`；
- 再用在线估计（残差/一致性/遮挡比例）；
- 出口：`R` 与实际通道误差的相关性/校准。

### R3：路由 gate

- `task_gate` vs `best_single` / `fixed_fusion` / `task_oracle`；
- 每任务单独报；
- 出口：H 判定（gate ≥ best_single，且在 fixed fusion 差的 task 上 +0.05）。

### R4：跨任务/跨条件

- leave-one-task-out；leave-one-condition-out（高噪声/高延迟/遮挡）；
- 出口：路由是否泛化，还是只记住任务表。

### R5：冲突/弃权

- 权威通道缺失/不可信/冲突；
- 路由是否降级/abstain；
- 出口：abstention precision/coverage、错误代价。

### R6：PSS 集成与报告

- 路由决策、权重、证据写入 `ContactEvent`/`Relation`/audit；
- 报告 `PSS-TaskRouting-Report.md`；
- 决策：是否把 `task_gate` 设为 opt-in 默认。

---

## 8. 停止规则

| 观测 | 结论 | 动作 |
|---|---|---|
| `task_gate` < `best_single` −0.02 | 路由有代价 | 检查 `A/T/b/R`；不加新通道 |
| `task_gate` 在 fixed fusion 差的 task 上 < +0.05 | 路由没兑现 | 诊断任务标签/可靠性 |
| 权威通道权重 <0.50 | 路由没听先验 | 检查 `A` 尺度/温度/偏置 |
| leave-one-task-out 失败 | 路由记任务，不泛化 | 用 declared task 或 task classifier；不宣称自动迁移 |
| abstention precision <0.90 | 弃权不可靠 | 不宣称 abstention |
| 某任务 fixed fusion 已最好 | 该任务不需要 gate | 保留单通道/fixed，别过度路由 |

**R3 结束先做 H go/no-go；不通过不推进 R4–R6 的默认化。**

---

## 9. 风险与混杂

1. **任务标签泄漏**：task classifier 用同一观测估计任务，再用它路由 → 循环；主实验用 declared task。
2. **权威表在 test 上调**：`A/T/b` 只在 cal 上选；test 一次。
3. **可靠性循环**：用 force 估计 force 可靠性，可能自我确认；用独立线索（残差、延迟、一致性）。
4. **通道缺席**：distance 任务没有 force；路由必须 mask，而不是给 0 权重后仍归一化。
5. **延迟**：`R` 要含时间对齐；4 帧延迟的 force 不能主导 onset。
6. **任务不平衡**：每任务样本量/事件数不同；分层报告。
7. **代理传感器**：force/tactile 近 oracle，audio 合成，vision 仿真；结论限 sim。
8. **多重比较**：每任务一个主指标；Holm。
9. **“任务”定义粒度**：contact vs distance 是粗粒度；同一任务内条件变化由 `R` 处理，不要把所有东西都塞进 `A`。

---

## 10. PSS 集成

- `TaskSpec`：任务名、标签、可用通道、权威先验；
- `ReliabilityEstimator`：每通道 `R_c`；
- `AuthorityRouter`：`A/T/b/R` → `w`；输出 `RouteDecision`（权重、弃权、降级）；
- `ContactEvent`/`Relation`：带 `route`/`weights`/`evidence`/`authority`；
- `CognitiveState`：audit `route_decision`；可回滚；
- `UncertaintyService`：`max(w)<τ` → review/abstain；
- 路由保持 opt-in，先不默认化。

---

## 11. 交付物与时间盒

```
pss/routing/model.py          # TaskSpec / AuthorityPrior / RouteDecision
pss/routing/reliability.py    # R_c 估计
pss/routing/gate.py           # A/T/b/R -> w, abstain
pss/routing/layer.py          # PSS 集成
pss/eval/scripts/run_task_routing_bench.py
tests/test_task_routing.py
docs/experiments/PSS-TaskRouting-Report.md
eval/results/task_routing/
```

| 天 | 内容 | 出口 |
|---|---|---|
| R0 | 任务套件 + 单通道 baseline | 权威表与单通道一致 |
| R1 | 权威先验 vs fixed fusion | task_oracle 是否已赢 |
| R2 | 可靠性估计 | R 与误差相关/校准 |
| R3 | gate | H 判定 |
| R4 | 跨任务/跨条件 | 是否泛化 |
| R5 | 冲突/弃权 | precision/coverage |
| R6 | PSS 集成 + 报告 | 默认化决策 |

---

## 12. 边界

- MuJoCo 图元是 proxy；force/tactile 近 oracle；audio 合成；vision 仿真。
- “任务”是粗粒度声明；真实系统可能需要任务分类器/上下文。
- 不是真实机器人/真实传感器；不声称 sim2real。
- 路由不替代传感器质量；它只决定“该信谁、信多少、什么时候不信”。

---

## 13. 一句话

**不是融合错，是固定权重错。** 先按任务决定谁有话语权（`A`），再按当前观测调可靠性（`R`），权威通道不可用时弃权/降级；用 contact / distance / material / occlusion 四个任务验证 `task_gate` 是否在每个任务上 ≥ 权威单通道，而不是再用一个全局 fusion 去测所有任务。
