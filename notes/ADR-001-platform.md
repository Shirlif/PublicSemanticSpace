# ADR-001：评测平台从人形腿改为“固定桌面 + 脚本化相机轨迹”

> 日期：2026-11-25
> 状态：已采纳
> 影响：`PSS-IsaacSim-评测平台设计.md`、`PSS-IsaacSim-实施计划.md`

## 背景

本项目的目标是验证 **PSS 架构在空间感知与空间记忆上的优势**，不是人形机器人的运动控制能力。

原设计选择 Unitree G1 作为执行平台，并预留 `base_mobile`（冻结行走策略）。实际推进时发现：

1. NVIDIA Isaac Lab 5.1 的资产桶没有 G1 locomotion checkpoint；
2. Unitree 官方 `unitree_rl_gym` 的 `motion.pt` 是 12-DoF 腿、Isaac Gym 训练，不能直接控制
   29-DoF G1；`unitree_rl_lab` 有 29-DoF 训练链但不带权重；
3. 社区 29-DoF checkpoint 的引擎、观测布局、DoF、许可证各不相同，接入成本高；
4. 即使接上，行走/平衡控制器的方差会成为 PSS 结论的混杂变量。

## 决策

1. **感知/记忆主平台 = 固定桌面 + 脚本化相机轨迹**。
   - 相机运动写进 `spec.cameras[].motion`，随 spec hash 冻结；
   - 视角变化由时间插值产生，不重新编译 USD；
   - S1/S2/S3/S6 全部使用这一平台。
2. **操作平台 = G1 `base_fixed` 上肢**。
   - 保留技能层 `reach/grasp/place/release`；
   - 不使用人形腿，不接 locomotion checkpoint；
   - `freeze.json` 记 `locomotion_policy: not_applicable`。
3. **人形腿 `base_mobile` 取消**。
   - 需要房间级移动时，只加 `wheeled_differential` 附表（Nova Carter/Jetbot + 差速控制器）；
   - 轮式附表不参与主表结论。
4. **`base_disturb` / `path_blocked`** 在主表只记录事件；只有在轮式附表上才做物理注入。

## 后果

### 正面

- 去掉行走策略这一硬阻塞，T28/T34 不再卡工期；
- 视角变化可复现、可 hash，直接服务于空间感知/记忆结论；
- 底层执行保持恒定，符合 capability-controlled 原则；
- 资源和时间集中到 S1/S2/S3/S6。

### 负面 / 限制

- 不再声称“移动人形 loco-manipulation”能力；
- 若未来要做房间级 SLAM/导航，仍需补轮式底座或独立的移动平台；
- S4/S5 的 `base_mobile` 附表不存在，S5 的 10 类故障中 `base_disturb`/`path_blocked`
  在主表降级为事件级。

## 对任务清单的影响

| 任务 | 原内容 | 变更 |
|---|---|---|
| T28 | G1 全身 + 行走策略冻结 | 改为固定基座 G1 上肢；`locomotion_policy: not_applicable` |
| T34 | `base_mobile` 附表 | 改为可选 `wheeled_differential` 附表，不再依赖 RL |
| T44（新增） | — | `cameras[].motion` spec + runner 插值 |
| T45 | — | 本 ADR 记录 |

## 实施顺序

1. 更新设计/计划文档（本 ADR + 两处正文）；
2. 给 `pss/spec` 增加 `cameras[].motion` 校验；
3. runner 在每物理帧按轨迹更新相机位姿（与 occluder motion 相同的时间插值）；
4. 用 W1/S1 重跑一条带相机轨迹的 episode，确认图像视角变化；
5. 把 S1/S2/S3/S6 的 spec 默认切到相机轨迹；
6. 关闭 T28/T34 的腿策略依赖，轮式附表按需再开。
