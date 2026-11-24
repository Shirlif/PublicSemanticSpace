# Phase 7 报告：关系、动作算子、世界模型与规划

> 日期：2026-11-20（2026-11-22 补 contact 碰撞几何复测）
> 对应计划：`公共语义空间-落地执行计划-v1.md` → Phase 7
> 状态：**规划与异常恢复跑通；contact 碰撞几何 F1 1.000；S1 感知闭环跑通；多实体世界模型跑通；软前置条件弃权跑通**
> 环境：torch 2.10.0+cu126 / scipy 1.18.0

---

## 1. 交付物

| 交付 | 路径 | 状态 |
|---|---|---|
| 关系记录 + 向量编码（三份 part + fact） | `pss/reasoning/relations/` | ✅ |
| 动作算子库（软前置条件 + 效果） | `pss/world_model/actions.py` | ✅ |
| A* 规划器 + 异常重规划 | `pss/reasoning/planner/` | ✅ |
| 观测-预测冲突检测 | `pss/reasoning/conflict/` | ✅ |
| 解析世界模型 + 学习残差 MLP | `pss/world_model/model.py` | ✅ |
| Phase 7 评测 CLI | `pss/reasoning/scripts/run_phase7.py` | ✅ |
| 指标与报告 | `checkpoints/phase7/` | ✅ |
| 测试 | `tests/test_phase7.py` | ✅ 77 passed（全仓） |

---

## 2. 关系计算 vs 仿真 GT

8 条 episode，共 21 帧：

| 范围 | precision | recall | F1 |
|---|---:|---:|---:|
| 总体 | 0.900 | 0.857 | **0.878** |
| `in` | 1.000 | 1.000 | **1.000** |
| `contact` | 1.000 | **0.000** | 0.000 |
| `near` | 0.500 | 1.000 | 0.667 |

- `in` 完全正确。
- `contact` 的 GT 来自 MuJoCo 真实接触，而当前用半径近似，3 个真实接触全部漏掉；precision 1.0 只是因为没预测出任何 contact。
- `near` 召回 1.0，但多报了 2 个（阈值 0.12 与 GT 的判定不完全一致）。

> 结论：几何关系不能只靠"质心距离 + 半径"近似；contact 要用碰撞几何或仿真接触。

### 2.1 contact 碰撞几何复测（2026-11-22）

旧结果暴露了两个问题：contact 全漏，near precision 偏低。后者其实是**评测不对称**：`SensorRig.relations` 只存 `(a, contact/near, b)` 一个方向，关系层为了双向查询存了两个方向，直接比集合会把反向事实算成 FP。

修复：

1. `pss/reasoning/relations/contact.py`：按 MuJoCo geom 类型做真实几何测试。
   - box-box：OBB SAT；
   - capsule-capsule：线段距离；
   - capsule/cylinder-box：线段采样到 OBB 距离；
   - sphere-box / sphere-capsule：点-盒、点-线段距离；
   - ellipsoid：体积等效球。
2. `relations_from_frame()` 使用样本里的 `object_specs`（新生成数据带 `geom_type`/`size`），没有 specs 时回退旧半径法。
3. `evaluate_relations()` 对 `contact` / `near` 的 GT 做对称化，和关系层的双向存储对齐。
4. `in` 的 z 阈值从 0.12 调到 0.14，和 `SensorRig.relations` 的 `BIN_HEIGHT + 0.08` 对齐。

新数据（`data/phase7_v2`、`data/phase7_v2_eval`、`data/phase7_v2_more`）：

| 数据 | episodes | frames | in TP | contact TP | near TP | overall P / R / F1 |
|---|---:|---:|---:|---:|---:|---|
| seed 0 | 8 | 40 | 16 | 18 | 0 | 1.000 / 1.000 / **1.000** |
| seed 1 | 8 | 40 | 16 | 20 | 0 | 1.000 / 1.000 / **1.000** |
| seed 2 | 32 | 192 | 86 | 32 | 14 | 1.000 / 1.000 / **1.000** |

- contact 从 F1 0.000 提到 **1.000**（32 条测试数据共 32 个 contact 事实，无 FP/FN）。
- near 从 precision 0.500 提到 **1.000**；对称化后确认旧 FP 是重复方向，不是阈值问题。
- 32 条 episode 的评测覆盖了 box/cylinder/capsule/cloth/sheet 等形状组合。

局限：

- 仍是 MuJoCo 精确位姿和精确 size，没有真实传感器噪声；真实数据上需要留 1–3 mm 容差并重新标定。
- ellipsoid 用体积等效球，极端细长蛋形可能漏检。
- near 仍用固定 0.12 m 阈值，未按物体尺寸归一化。

复现：

```bash
python -m pss.sim.scripts.generate_dataset --out data/phase7_v2 --scale expert --seed 0 --tactile-mode material
python -m pss.reasoning.scripts.run_phase7 --data data/phase7_v2 --out checkpoints/phase7_v2 --episodes 8 --epochs 100
python -m pss.reasoning.scripts.run_phase7 --data data/phase7_v2_eval --out checkpoints/phase7_v2_eval --episodes 8 --epochs 100
```

---

## 3. 规划器

| 路线 | 成功 | 步数 | 展开节点 | 目标达成 |
|---|---|---:|---:|---|
| 手写 12 步 | ✅ | 12 | — | ✅ |
| A* | ✅ | 12 | **50** | ✅ |
| 无洗衣液（blocked） | ❌ | — | 14 | — |
| 倾倒洒出后重规划 | ✅ | **9** | — | ✅ |

A* 找到的 12 步与手写顺序不同，但同样有效：

```
open_washer → pick_clothes → pick_detergent → pour_detergent
→ put_in_clothes → close_washer → fill_water → press_start
→ wait → open_washer → take_out → hang
```

异常恢复：在 `pour_detergent` 后注入"洗衣液洒在地上"，重规划为：

```
open_washer → pick_detergent_floor → pour_detergent → close_washer
→ press_start → wait → open_washer → take_out → hang
```

这是 6.7 的"感知闭环 + 重规划"第一次在可运行系统里闭环。

### 3.1 S1 闭环：感知 → 规划 → 动作 → 重规划

`pss/reasoning/scripts/run_closed_loop.py` 把 Phase 6 感知、Phase 7 规划和 Phase 8 推动作接到同一个循环：

```text
render → segmentation/tracking → 区域事实 → A* → 一次推进脉冲 → 重新渲染
```

任务：把 block 推到目标区 G。初始 brick 停在 A→G 直线上，规划器必须绕行 B 或 C。block 到达 B 后，外部把 brick 移到 (0.10, 0.105)，挡住 B→G；下一帧感知看到新布局后重新规划。

| 配置 | 成功 | 动作序列 | 推进次数 | replans | 最终距离 |
|---|---|---:|---:|---:|---:|
| seed 0，无异常 | ✅ | `move_A_B → move_B_G` | 17 | 0 | 0.0497 m |
| seed 0，注入障碍移动 | ✅ | `move_A_B → move_B_C → move_C_G` | 32 | 1 | 0.0487 m |
| seed 1，无异常 | ✅ | `move_A_B → move_B_G` | 18 | 0 | 0.0598 m |
| seed 1，注入障碍移动 | ✅ | `move_A_B → move_B_C → move_C_G` | 45 | 2 | 0.0468 m |

关键点：

- 每次外层迭代只执行一个推进脉冲，然后重新渲染和跟踪；实体 ID 在 block 移动过程中保持。
- 规划器只看感知得到的实体位置和关系事实，不使用 GT 位姿；目标 ID 由初始位置选择，之后由 Tracker 保持。
- 异常后旧计划失效：block 在 B，B→G 被挡，规划器改走 B→C→G。
- 全程没有文本中间产物；动作头是 Phase 8 的规则控制器，接口可替换为学习策略。

复现：

```bash
python -m pss.reasoning.scripts.run_closed_loop --out eval/results/closed_loop
python -m pss.reasoning.scripts.run_closed_loop --out eval/results/closed_loop_no_anomaly --no-anomaly
```

局限：

- 单目标、单障碍、离散路点；不是连续空间规划。
- 动作头用仿真位姿做低层闭环，规划层用感知位姿。
- 障碍移动是外部事件，且移动后被保持，避免桌面物理把障碍慢慢推走。
- 仍是代理任务，不是真实操作。

### 3.2 软前置条件与置信度弃权

`OperatorLibrary.applicable()` 会按事实置信度过滤前置条件；`AStarPlanner.plan()` 增加 `confidence_weight`，把低置信度动作的代价抬高，并在所有可行动作都被置信度挡住时返回 `reason="low confidence"`，而不是硬编一条不可靠的路径。

演示：

| 配置 | 结果 |
|---|---|
| 所有前置条件置信度 0.95 | 规划成功，12 步 |
| `detergent on counter` 置信度 0.20，阈值 0.5 | 弃权，`reason=low confidence` |

- 低置信度事实不会让规划器假装确定；它明确输出弃权。
- 置信度加权会优先选择高置信度动作（单测覆盖）。
- 当前置信度值是演示用合成值；还没接 Phase 6 的识别/关系置信度。

---

## 4. 冲突检测

| 项 | 值 |
|---|---|
| 预测 | `detergent in washer`, `door open` |
| 观测 | `detergent on floor`, `door open` |
| missing | `detergent in washer` |
| unexpected | `detergent on floor` |
| 触发重规划 | ✅ |

---

## 5. 世界模型

| 基线 | 1 步位置误差 | 多步 rollout 位置误差 |
|---|---:|---:|
| zero（不变） | 0.0464 | 0.1545 |
| 解析（常速度 + 动作效果） | 0.0400 | 0.1452 |
| **学习 MLP（目标实体）** | **0.0252** | **0.0911** |

- 学习模型在 52 条训练转移 / 12 条验证转移上优于解析与 zero。
- 但这是 **target-only** 模型：只预测动作目标实体，其余实体假设不动；且训练数据是运动学瞬移动作。
- 只能作为"世界模型接口跑通"的证据，不能作为动力学能力结论。

### 5.1 多实体世界模型（2026-11-22）

`pss/world_model/multi.py` 和 `set_model.py` 补上多实体版本。数据：`data/phase7_v2_more`，60 条 episode，240 train / 60 val transitions。

| 模型 | 1 步全实体误差 | 多步 rollout 全实体误差 | 多步 rollout 目标实体误差 |
|---|---:|---:|---:|
| zero | 0.0406 | 0.1515 | 0.1451 |
| 解析 | 0.0359 | 0.1539 | 0.1442 |
| target-only MLP | **0.0234** | 0.1019 | — |
| 全局多实体 MLP | 0.0388 | 0.1385 | — |
| **per-entity set MLP** | **0.0243** | **0.0965** | **0.0943** |

- **set 模型**：每个实体走共享 MLP，输入自己的 pos/quat/vel、动作 one-hot、是否 target、其他实体均值、target 相对位置/距离/接触标志。rollout 时用模型自己的预测状态重建特征。
- set 模型在全实体 rollout 上优于 target-only（0.0965 vs 0.1019）和全局多实体（0.1385）。
- 但仍未达到计划建议的 **5 cm** 门槛。原因是数据里非目标实体大多不动，target-only 的"其余静止"先验很强；要再降误差，需要碰撞/接触更丰富的 episode，或把关系图作为显式输入。
- 全局多实体 MLP 最差：一个全局向量同时回归所有实体，容量和归纳偏置都不合适。

复现：

```bash
python -m pss.reasoning.scripts.run_phase7 \
  --data data/phase7_v2_more --out checkpoints/phase7_v2_more \
  --episodes 32 --epochs 300
```

---

## 6. 冻结的决策

| # | 决策 |
|---|---|
| 1 | 关系记录驱动规划；向量编码只做检索/接地 |
| 2 | 动作算子 v1 手写，前置条件为合取；软置信度已接：低于阈值弃权，置信度加权代价 |
| 3 | A* 状态 = 关系事实集合；启发式为缺失目标数 |
| 4 | 异常恢复 = 观测事实覆盖预测 + 从当前状态重规划 |
| 5 | 世界模型 v1 为 target-only；v2 增加 per-entity set 模型，全实体 rollout 0.0965 m，仍未达 5 cm |
| 6 | `contact` 已改用碰撞几何（OBB SAT / 线段距离）；半径近似只作为无 specs 时的回退 |

---

## 7. 已知边界

| 边界 | 说明 |
|---|---|
| contact 代理误差 | 已用碰撞几何复测，解除了该项；真实数据仍需容差标定 |
| 离散关系状态 | 未使用连续置信度做软规划 |
| 启发式不可采纳 | 一个动作可满足多个目标时高估 |
| 世界模型小 | 52 条训练转移；动作是瞬移 |
| 无真实执行器 | 规划结果由事实模拟执行，不是机器人 |

---

## 8. 下一步：Phase 8（多读出头）

| 工作项 | 说明 |
|---|---|
| 动作头 | 实体状态 → 原语参数 / 末端位姿增量 |
| 文本头 | 冻结小 LLM + 适配器，低频调用；禁止反传核心表征 |
| 目标/价值头 | 内部动机打分，供 Phase 9 使用 |
| 消融 | 文本头开关不改变核心任务指标 |
| 验收 | 文本头不绑架表征；动作头端到端成功率 |

Phase 7 的关系/规划/世界模型会成为 Phase 8 输出头的下游消费者。
