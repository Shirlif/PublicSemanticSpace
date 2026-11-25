# S4-PSS：PSS 在环的 Isaac 物理任务（dev）

> 日期：2026-11-25
> 目标：不再只看离线指标，直接验证 PSS 在 Isaac 物理空间里“从感知到行动”的闭环能力。

## 场景

- 固定基座 G1、桌面、1 个 cup、1 个 bowl 干扰物、1 个 bin。
- 感知：Isaac GT 框/位姿 + class label（oracle sensor）；**不把 GT 实例 ID 用于关联**，
  appearance 全零。
- PSS：`build_runtime("full")` 连续 ingest 12 帧，形成实体 tracks；
  选 `category=cup` 的 track 与 `category=bin` 的 track。
- 规划：`assert_fact(cup, on, table)` → `Goal(cup in bin)` →
  `RuntimePlanner`（A*，pick/place 两个 operator）→ `Plan.actions = [pick, place]`。
- 执行：`SkillExecutor`（右臂阻尼最小二乘 IK + 吸盘代理）执行 reach/grasp/place。
- 判定：执行后从 Isaac GT 读 cup/bin 位置，cup 落在 bin 足迹内且高度合理。

## 结果

| 项 | 值 |
|---|---|
| plan | `pick -> place` |
| plan confidence | 1.0 |
| goal | `(obj_0001, in, obj_0003)` |
| actions | reach:ok, grasp:ok (dist 0.021), place:ok |
| **task_success** | **true** |
| cup xyz | [0.108, -0.077, 0.85] |
| bin xyz | [0.100, -0.050, 0.80] |
| steps | 284 |
| skill wall | 0.53 s |

表：`scorecards/S4_pss.md` / `.json`。

## 这证明了什么

- PSS 的实体层能在 Isaac 观测流里形成 cup/bin 的 track；
- PSS 能把感知结果写成事实、设目标、用符号规划器生成 `pick -> place`；
- 技能层能在物理仿真里执行该计划，并**真实移动物体**；
- 判定来自物理 GT，不是脚本里的硬编码 success。

## 这还没证明什么

1. 感知是 **oracle 框/位姿**（虽然 class/pose 来自 Isaac，appearance 为零），
   不是 GroundingDINO 图像检测；
2. 只有 1 个 seed / 1 条 episode，没有稳定性统计；
3. 任务简单（一个 cup、一个 bin），没有关系/记忆/干扰/故障；
4. 吸盘代理，不是真实平行夹爪；
5. 目标选择只按 `category=cup`，没有利用 PSS 的关系/记忆层。

## 下一步（继续加“干活”难度）

1. 把感知换成渲染图像 + GroundingDINO（或者 Isaac instance seg + 框），
   验证 PSS 在真实视觉输入下仍能选中正确 cup；
2. 加两个 cup + 一个 bowl，目标改成关系描述（“bowl 左边的 cup”），
   逼 PSS 用关系层选目标；
3. 加长时记忆：中途把 cup 移到别处/遮挡，PSS 仍能找到并放入 bin；
4. 加故障：grasp_slip / object_relocate / camera_dropout，测恢复；
5. 跑 3–5 seeds，报成功率，而不是单条 true。
