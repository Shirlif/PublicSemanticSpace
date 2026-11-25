# P5 / W8–W10 进度 — 人形执行层（S4/S5）

> 开始：2026-11-25
> 状态：T29/T30/T31/T32/T33 完成（dev 脚本基线）；T28/T34 受阻，如实记录。

## T28 G1 全身 + 行走策略：**部分完成**

- G1 全身（200 prims）可加载、可关节控制；右臂 7 DoF IK 到桌面目标误差 ~1.3 cm。
- **行走策略未安装**：Isaac Lab 2.3.2 的官方 Policies 列表里没有 G1 locomotion
  checkpoint；`freeze.json` 的 `locomotion_policy` 记 `none`。
- 因此只跑 `base_fixed`；`base_mobile` 留空。

## T29 技能库（`pss/isaac/skills.py`）

- 技能：`home / reach / grasp / place / release / walk_to / turn`。
- 状态：`ok / timeout / slipped / collision / unreachable / unsupported`。
- 低层：阻尼最小二乘位置 IK（G1 右臂 Jacobian；列偏移 +6 处理 root DOF）。
- 每步把基座 pin 到固定位姿（base_fixed 近似）。

## T30 平行夹爪：**代理实现**

- G1 资产没有平行夹爪（只有 ThreeFinger / Inspire）。
- 用**吸盘代理**：grasp 时把物体吸到 hand link 并逐步随动；release 松开。
  这是设计止损点允许的简化（吸盘/简化夹爪），但必须在表里标成代理。

## T31 S4 闭环：脚本基线

- `scripts/run_s4.py` + `specs/s4_g1_pick_place.yaml`：home → reach → grasp →
  place → release，最多重试一次。
- 结果：**execution_success = 1.000**（1 episode，dev；`scorecards/S4.md`）。
- **不是 PSS 在环**：这是执行层/技能序列的基线，PSS planner 尚未接入。
- bin 是实心方块（pad），`in(cup,bin)` 用“落在 bin 顶面足迹内”近似。

## T32 故障引擎 10 类（`pss/isaac/faults.py`）

全部 10 类可注入并写 events：

| kind | dev 行为 |
|---|---|
| occlusion_insert | 插入静态遮挡板 |
| object_relocate | 瞬移杯子 |
| object_remove | 把杯子移出工作区 |
| grasp_slip | 释放吸盘携带 |
| camera_dropout | 记录事件（脚本策略无相机） |
| lighting_shock | 改灯光强度 |
| twin_swap | 交换两个实例位姿 |
| joint_limit | 越界关节指令 |
| base_disturb | 记录事件（基座固定） |
| path_blocked | 记录事件（无行走） |

## T33 S5 表

见 `scorecards/S5.md`：10 类故障各 1 episode，**mean recovery = 0.900**；
唯一失败是 `object_remove`（物体被移出工作区，脚本策略无法找回）。
数值见文件；注意这不是 PSS recovery，camera_dropout/base_disturb/path_blocked
在 base_fixed 脚本策略下只是事件注入。

## T34 base_mobile 附表：**阻塞**

无 locomotion policy，`walk_to` 返回 `unsupported`；`scorecards/S4_mobile.md`
只写阻塞原因，不编数字。

## 必须如实说明

1. T28 的行走、T34 的 mobile 附表没做——缺 G1 locomotion checkpoint。
2. T30 没有平行夹爪，用吸盘代理；S4 成功不能当作夹爪能力证据。
3. T31/T33 是**脚本基线**，PSS 不在执行回路；不能当作 PSS 执行/恢复能力。
4. S4 的 bin 是实心 pad，`in` 是足迹近似；不是真正的容器。
5. 下一步：接 PSS planner → 技能序列；装 Isaac Lab + G1 locomotion；再做真实
   parallel jaw 或明确把吸盘代理写进设计变更。
