# 算子层设计（P0–P3）

> 日期：2026-11-26
> 范围：关系算子的统一接口、四类算子族、组合闭包与推理路径。
> 状态：P0–P3 并行开发中。

---

## 1. 目标与边界

目标不是"用矩阵乘法实现逻辑"，而是让**四类算子共享同一个接口**，使推理引擎不关心背后是 SE(3)、组合表还是嵌入变换：

```
apply / inverse / compose / residual
```

向量空间负责它真正擅长的事：**在噪声里挑候选、算残差、给置信度**。可靠的定真、版本和回滚仍然交给 `TMSFactStore`。

明确不做的：

| 原语 | 原因 | 去向 |
|---|---|---|
| 全称量词 / 计数 | VSA 容量约 `d/log d`，d=96 可靠上限 8–10 | `pss/reasoning/queries.py` 符号集合 |
| 否定即失败 | 向量无法可靠表达 | 规划器约束 / 符号层 |
| 时间推理 | 本质是历史查询 | `TemporalHistory` |

## 2. 四类算子

| 族 | 数学结构 | 实现 | 场景 |
|---|---|---|---|
| frame | SE(3) 群 | `pss/binding/frames.py` | 参照系、相对位置 |
| semantic | 有符号置换 + 低秩 | `pss/binding/operators.py` | 语义关系组合 |
| action | 半群（可逆性不定） | `pss/world_model/conditional.py` | 状态变换、前置条件 |
| constraint | 类型格 / 区间 | `pss/reasoning/constraints.py` | 兼容性、成员度 |

空间关系不进入 96 维语义空间：它们来自位姿，算子是刚体变换，解析精确。语义关系才用学习算子。

## 3. 文件地图

| 文件 | 内容 | 优先级 |
|---|---|---|
| `pss/binding/protocol.py` | `Operator` ABC、`OperatorChain` | 地基 |
| `pss/reasoning/relations/closure.py` | 逆关系表、组合闭包表、版本/回滚 | P0 |
| `pss/binding/frames.py` | `Pose`、`FrameOp`、`relation_in_frame` | P0 |
| `pss/reasoning/relations/inference.py` | 前向链、后向链、派生事实 | P0 |
| `pss/world_model/conditional.py` | 条件效果、资源约束、动作库 | P1 |
| `pss/reasoning/constraints.py` | 类型子树成员度、属性区间、兼容性 | P1 |
| `pss/binding/operators.py` | 有符号置换、低秩、算子库、公理残差 | P2 |
| `pss/reasoning/queries.py` | 集合/量词/时序查询 | P3 |

## 4. 两个必须统一的约定

**组合顺序。** `a.compose(b)` 表示 `a ∘ b`：`b` 先作用，`a` 后作用，与矩阵乘法 `A @ B` 一致。闭包表的 `compose(r1, r2)` 读的是一条路径 `a -r1-> b -r2-> c`，对应算子 `T_r2 ∘ T_r1`。两者方向一致，但写法不同，容易搞混，代码里必须照此注释。

**残差。** `residual(x, y)` 非负，`0.0` 表示完全一致。不同族的量纲不同（米、`1-cos`、Jaccard），阈值只能按族设置，推理引擎只在族内比较。

## 5. 推理流程

```
1. 前向
   for (a, r1, b) in incoming[b], (b, r2, c) in outgoing[b]:
       类型检查 -> P = closure.compose(r1, r2)
       for (p, prior) in P:
           conf = derived_confidence(c1, c2, prior)
           派生 (a, p, c) with premises=[(a,r1,b),(b,r2,c)]
2. 后向
   目标 (a, ?, c) -> 逆算子反查 / 深度受限 BFS
3. 落库
   TMSFactStore.assert_fact(source="inference", derivation=...)
   冲突 -> UncertaintyService
```

剪枝靠两层：类型检查（`compose` 返回 `None`）和闭包表（空集表示"不授权"，不是"为假"）。

## 6. 公理即代数性质

| 逻辑公理 | 代数形式 | 用途 |
|---|---|---|
| 对称 | `T = T⁻¹` | 训练正则 / 自检 |
| 传递 | `T² ≈ T` | 只能近似，正交算子不可幂等 |
| 互逆 | `T T⁻¹ = I` | 后向查询 |
| 方向性 | `T` 远离 `T⁻¹` | 反对称约束 |

`RelationOperatorBank.axiom_residuals()` 输出这些量，用于发现表与算子不一致。

## 7. 与现有模块的关系

- 不修改 `pss/prototypes/recognize.py`：成员度后续可换成 vMF，但不在本批。
- 不改 `pss/world_model/actions.py`：条件算子是新文件，旧库保持可用。
- `pss/runtime/runtime.py` 的 `ingest` 后续接一步 `derive`，本批只提供纯函数。
- 评估用 `pss/eval/spatial/generator.py` 的 family：`transitive` / `frame` / `perspective` 是现成测试集。

## 8. 验证

每个模块自带 `tests/test_*.py`；集成验证放在 `tests/test_operator_integration.py`，覆盖：

1. 三条 `left_of` 链推出 `left_of`；
2. `left_of ∘ right_of` 不推出任何东西；
3. 参照系旋转 180° 后方向谓词翻转；
4. `TypeConstraint.compose` 对不相交子树返回 `None`；
5. 有符号置换组合非交换；
6. 条件动作的水溢出只在有水时触发；
7. 时序查询返回"昨天"的位置而非最新位置。

## 9. 交付状态（2026-11-27）

| 模块 | 测试 |
|---|---|
| `pss/binding/protocol.py` | 由集成测试间接覆盖 |
| `pss/binding/frames.py` | `tests/test_frames.py` 23 passed |
| `pss/binding/operators.py` | `tests/test_relation_operators.py` 16 passed |
| `pss/reasoning/relations/closure.py` | `validate()` 为空；集成测试覆盖 |
| `pss/reasoning/relations/inference.py` | `tests/test_relational_inference.py` 13 passed |
| `pss/reasoning/constraints.py` | `tests/test_constraints.py` 14 passed |
| `pss/reasoning/queries.py` | `tests/test_queries.py` 14 passed |
| `pss/world_model/conditional.py` | `tests/test_conditional_operators.py` 16 passed |
| `pss/reasoning/composition.py` | `tests/test_composition.py` 9 passed |
| `pss/reasoning/planner/decomposition.py` | `tests/test_plan_decomposition.py` 6 passed |
| 集成 | `tests/test_operator_integration.py` 14 passed |

全量回归：`682 passed, 1 warning`（包含并行会话新增的校准测试；warning 来自既有的 torch nested tensor）。

实现中的两处修正，记录在案：

- 有符号置换的逆必须重新索引符号：`y_i = s_i x[perm_i]` 的逆是 `x = y[inv_perm] * s[inv_perm]`，不能沿用原 signs；否则往返精度不成立。
- 闭包表与算子的组合顺序不同名：`ClosureTable.compose(r1, r2)` 读路径 `a -r1-> b -r2-> c`，对应算子 `T_r2 ∘ T_r1`。

未做（按设计留到下一批）：`PSSRuntime.ingest` 接入 `derive`、`recognize` 的 vMF 成员度、语义算子训练（当前只有代数性质与公理残差，没有学习信号）。

## 10. 空间基准评估（n=200, seed=0）

复现：`make operator-derivation`，产物在 `reports/claim_gaps/operator_derivation/`。

| 项 | 结果 |
|---|---|
| transitive 问答 | 1.000（direct-only 基线 0.500） |
| 稀疏连续链派生 | precision 1.000 / recall 1.000 |
| 全事实集留出 20% | precision 1.000 / recall 0.290（298 tp / 0 fp / 728 fn） |
| frame 帧作用域 | `FrameOp` 1.000；忽略帧 0.495；帧敏感率 0.505 |
| perspective 帧作用域 | `FrameOp` 1.000；忽略帧 0.705；帧敏感率 0.295 |
| 跨帧迁移 | world→viewer 0.5% 矛盾；world→cam1 10.5% |

结论：

- **闭包表是 sound 但低覆盖**：十条谓词的完整事实集上只能恢复 29%，没有产生任何假事实。说明手写表可以信，但不够用，下一步的瓶颈是覆盖率而不是精度。
- **收益集中在链式场景**：transitive 问答从 0.500 提到 1.000，稀疏链 1.000/1.000。这正是它该被用在的地方。
- **帧必须显式作用域**：cam1 视角下，10.5% 的世界系结论与真值矛盾。任何"先在世界系推理、再按视角回答"的流水线都会错这一部分。
- 因此 **`derive` 不接入 `PSSRuntime.ingest` 默认路径**。等 `RelationRecord` 带 `frame` 字段后，作为 planner 的可选查询接入；否则派生事实会混用参照系。

## 11. 最小组合闭环（2026-11-27，更新）

结构：

- `pss/reasoning/composition.py`：`Signature`、`Atom`、best-first 搜索、`Candidate`、`SelectAtom` / `ProjectAtom` / `ConstraintAtom` / `TypeMatchAtom` / `PickAtom`、验证后才写入的 `PlanCache`。
- `pss/eval/scripts/run_composition_loop.py`：25 洗衣 + 19 空间 + 6 选择，五种策略。
- 空间任务用四个对角候选落点 + 目标帧；非世界系任务被筛选为"世界系首个可行落点在目标帧下不成立"。
- 选择任务的四个诱饵各只违反一个过滤条件；`PickAtom(require_unique=True)` 强制四个过滤都执行。

结果（seed 0）：

| 策略 | 总成功率 | 假声明率 | 平均扩展 | p50 |
|---|---:|---:|---:|---:|
| direct | 0.300 | 0.460 | 0 | 0.009 ms |
| frame_blind | 0.420 | 0.340 | 0 | 0.008 ms |
| typed | **1.000** | **0.000** | 20.4 | 0.257 ms |
| typed + cache | **1.000** | **0.000** | 6.6 | 0.049 ms |
| typed + warm | **1.000** | **0.000** | **3.2** | **0.039 ms** |

分域：洗衣 direct 0.520 / typed 1.000；空间 direct 0.053 / frame_blind 0.368 / typed 1.000；选择 direct 0.167（83.3% 假声明）/ typed 1.000。

帧敏感子集（12）：direct 0.000、frame_blind 0.000、typed 1.000；假声明率分别为 1.000 / 1.000 / 0.000。

缓存：77 命中 / 11 失效 / 12 未命中；验证写入 100 次、拒绝 0 次、12 个条目、覆盖率 1.000；平均扩展 20.4 → 6.6 → 3.2。

结论：

- **类型化组合成立**：成功率 1.000，假声明 0。
- **select / project / constraint 可用**："选红杯子"这类属性筛选能参与组合；`require_unique` 保证每个过滤都必要。
- **缓存只在验证后写入**：失效回放被拒绝（11 次），不产生假事实；第二遍覆盖率 100%，搜索量降到约 16%。

## 12. Planner 目标分解（2026-11-27）

`pss/reasoning/planner/decomposition.py`：

- `decompose(goal, operators, initial)`：后向链式——找能添加目标事实的算子，递归满足前置条件，产出一段有序动作序列。
- `CompositionPlanner.plan(initial, goal)`：同时跑后向分解和 A*，两者都先模拟验证，再按代价取更便宜的；都不成功才返回失败。
- `CompositionPlanner.plan_typed(...)`：把 `best_first` 的结果直接转成 `PlanResult`，组合搜索和动作规划共用一个入口。

测试：`tests/test_plan_decomposition.py` 覆盖完整洗衣目标、不可达目标、自支撑算子终止、双路径取优、typed 入口。

已知边界：后向分解按目标事实的排序顺序处理，每个事实只取第一个生产者；它不是完备的，所以必须和 A* 比较，不能单独使用。

## 13. 验证后写入的缓存

- `PlanCache.record(key, steps, verified)` 是唯一写入口；`verified=False` 只增加 attempts/failures，绝不覆盖已有验证计划。
- `confidence` 用 Laplace 平滑；`coverage(keys)` 给出已覆盖目标的比例。
- `snapshot` / `rollback` 同时保存条目和统计；回滚后失效条目消失。
- 评测中的 `typed_warm` 是同一批任务的第二遍：覆盖率 1.000，平均扩展 3.2。



