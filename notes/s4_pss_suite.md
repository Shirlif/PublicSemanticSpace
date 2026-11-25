# S4-PSS 5-seed 干活能力结果（含记忆消融与双物体任务）

> 日期：2026-11-25
> 平台：Isaac Sim 5.1，G1 fixed-base（只控右臂），桌面 pick-place
> 感知：Isaac GT 框/位姿 + class + 颜色 appearance（不给 GT id）
> 成功判定：所有技能动作 `ok` + GT 物理落位 + 身份保持（记忆/双物体任务）

## 结果

| task | runs | success | rate | 说明 |
|---|---|---|---|---|
| basic pick-place | 5 | 5 | **1.00** | 抓取-搬运-放入 |
| relation (`left_of`) | 5 | 5 | **1.00** | 选中 bowl 左边的 cup |
| memory（2 s 遮挡 + 改位） | 5 | 3 | **0.60** | `same_identity` **5/5**；2 个失败在 place |
| two_object（双杯双箱） | 5 | 5 | **1.00** | 左右关系分配 + 两轮抓放 + 身份跨轮保持 |
| memory with `no_memory` 消融 | 5 | 0 | **0.00** | `same_identity` **0/5**；物理成功仍 3/5 |

表：`scorecards/S4_pss_suite.md` / `.json`。
单次结果：`scorecards/S4_pss_seed*.json`、`S4_pss_relation_seed*.json`、
`S4_pss_memory_seed*.json`、`S4_pss_memory_nomem_seed*.json`、
`S4_pss_two_seed*.json`。

## 关键结论：记忆层是可判别的

memory 任务的两次运行只改变 PSS runtime 变体，场景、种子、技能层完全相同：

| runtime | same_identity | physical_success | task_success |
|---|---|---|---|
| `full` | **5/5** | 3/5 | 3/5 |
| `no_memory` | **0/5** | 3/5 | 0/5 |

`no_memory` 的语义是“目标一旦从观测中消失就删除 track”（`NoMemoryEntityLayer`，
首帧 miss 即丢）。2 s 遮挡后杯子以全新 id（`obj_0001 -> obj_0004`）重新出现，
因此 `same_identity` 全失败；而**物理执行成功率与 full 完全一致（seeds 0/1/3）**，
说明差异来自 PSS 记忆层，而不是技能层或场景难度。

这回答了“架构到底有没有用”的问题：在同一个物理任务里，去掉记忆层，
身份保持从 100% 掉到 0%，执行成功率不变。

## 双物体任务（更难的干活场景）

`specs/s4_pss_two.yaml` + `scripts/run_s4_pss_two.py`：

- 两个杯子、两个箱子、一个参考 bowl；
- PSS 用 `left_of` 把“bowl 左边的杯子”分配到“bowl 左边的箱子”，右边同理；
- 规划器生成 4 步序列（pick/place × 2）；
- 第一轮放完后**重新观测**，检查两个 cup track 的 id 是否保持；
- 5/5：关系分配正确、4 步计划正确、跨轮身份保持、两个杯子都落入正确箱子。

单 seed 结果示例（seed0）：

```
assignments: obj_0001 -> obj_0004 (left), obj_0002 -> obj_0005 (right)
plan: pick|obj_0001, place|obj_0001|obj_0004, pick|obj_0002, place|obj_0002|obj_0005
id_stable_after_first: true
task_success: true
```

## 失败归因（memory 的 2 个失败）

两个失败都发生在 place，不是 PSS 决策：

- **seed2**：`place timeout err=0.2233`，手停在
  `(0.197, -0.196, 0.941)`，是 IK 在跨身位姿下的局部停滞；
- **seed4**：`place timeout err=nan`，PhysX 报 invalid transform，
  是右臂/机器人碰撞体与桌面/箱子的接触数值发散。

两例的 `same_identity` 都是 true，即 PSS 在遮挡+改位后都认出了同一个杯子。
因此不能把 memory 0.60 解读为“PSS 记忆 60%”；正确说法是：
**PSS 记忆 5/5，端到端物理 3/5，瓶颈在技能层。**

## 技能层现状与边界

- 抓取是 suction proxy：`RigidPrim.set_world_poses` + USD transform fallback，
  grasp 后 5 步检查物体是否跟随手，不跟随则显式 `slipped`；
- place 是单段阻尼最小二乘 IK；跨身位姿/接触时会出现停滞或 NaN；
- 尝试过的修复（抬升/越顶 waypoint、joint-limit nullspace、碰撞体开关、
  NaN 恢复、失败重试）在不同 seed 上互相打架，最终回退到简单稳定版本；
  这些尝试没有作为最终结果使用。

这说明当前平台的“干活成功率”主要由低层 IK/接触稳定性决定，而不是 PSS。
在技能层稳定之前，不应把 basic/relation/two_object 的 1.00 外推为
“任意场景都能干活”。

## 下一步（优先级）

1. **技能层**：换更稳的 IK/控制器（joint-limit-aware、碰撞感知、或固定 joint
   的 suction 模型），把 memory 的物理成功率拉到与决策层一致；
2. **关系消融**：对 two_object 跑 `no_relations`，并让目标杯不是固定第一个，
   验证关系层的可判别性；
3. **真实检测器在环**：把 S1-hard 的 GroundingDINO 接进物理任务，
   测“感知噪声 + PSS 记忆/关系”的端到端表现；
4. 以上稳定后，再谈对外声称“干活能力”。
