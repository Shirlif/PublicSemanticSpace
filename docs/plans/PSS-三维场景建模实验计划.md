# PSS 三维场景建模实验计划：从一维谓词回到三维场景

> 日期：2026-11-28
> 上游：`docs/plans/PSS-问题重构.md`、`docs/experiments/PSS-Boundary-Decision-Memo.md`
> 目的：验证一个架构级判断——**不要把三维空间问题压成一维/二维标量谓词；先用所有感官观测优化一个三维场景模型，再在三维里回答关系/碰撞/支撑问题。**
> 状态：D1–D6 已执行（2026-11-28）。**注意：本计划使用虚拟针孔投影 + 单点观测，不是真实 3D 环境；其结论只对“点观测三角化”成立。** v2 已改用 MuJoCo 桌面环境 + 真实 RGB-D/分割/表面点云 + 形状拟合，见 `docs/plans/PSS-三维场景建模实验计划-v2.md`。

---

## 0. 一句话

**一维单帧谓词的上限约 0.63–0.71，类平衡/多帧补丁只能到 0.85–0.87；本计划检验：把观测还原成三维场景（位姿 + 形状 + 物理约束 + 协方差），用 2–3 个视角是否就能超过一维多帧的 0.846。** 如果是，架构应从“逐对谓词”转向“场景模型 + 三维查询”；如果不是，问题不在维度，而在对应/约束/噪声模型，按停止规则诊断，不再加谓词 arm。

---

## 1. 原则：为什么一维不够

### 1.1 观测是投影，三维状态是潜变量

- 一维 `dx/dy` 是三维相对位姿的一个投影；多对一，不可逆。
- near-aligned 是投影的奇异区：`|dx|≈eps` 时，一维标量对三维变化最不敏感，噪声最容易推过阈值。
- 一维上的温度、先验、多帧平均，都是在补投影丢掉的信息；补不回来。

### 1.2 正确形态：analysis-by-synthesis

```
多模态观测（RGB / 深度 / 分割 / 触觉 / 位姿）
   → 三维场景优化（位姿 + 形状 + 物理约束 + 时序）
   → 场景模型（带协方差）
   → PSS 语义标注（实体 / 关系 / affordance）
   → 三维查询与分析（对齐 / 碰撞 / 支撑 / 规划）
```

- 所有观测共同约束**同一个三维状态**，噪声在多视角/多模态之间平均。
- 物理约束（支撑、不穿透、接触、时间平滑）排除一维无法区分的配置。
- 关系查询在三维后验上做，返回 belief，而不是 bool。
- 不确定时主动选下一个视角/动作，而不是硬判。

---

## 2. 假设

> **H：在一个受控的 CLEVR 多视角观测任务上，联合三维场景优化（3 个视角 + 深度 + 支撑/不穿透约束）在 near-aligned 关系上显著超过一维多帧（M=64），并且位姿误差与关系置信度可校准。**

子假设：

| 子假设 | 内容 | 证伪预测 | 若成立 |
|---|---|---|---|
| **H1 维度收益** | 3D 优化 3 视角 > 1D 多帧 64 帧 | 3D ≤ 0.846（1D M=64） | 维度不是关键；转诊断对应/约束 |
| **H2 约束收益** | 支撑/不穿透约束提升 near-aligned | 去掉约束后精度不变 | 约束无效；检查是否过强/冲突 |
| **H3 不确定度可校准** | 关系 belief 的 ECE ≤0.05 | ECE >0.10 | 协方差/模型误设；不宣称 uncertainty |
| **H4 观测预算** | 3 视角 ≥0.90；5 视角 ≥0.95 | 3 视角 <0.85 | 视角数/基线不足；检查相机布局 |

---

## 3. 数据与观测模拟

### 3.1 数据

- CLEVR CoGenT：`traina_8000.pt` / `vala_3000.pt`，含 `3d_coords`、`pixel`、`scene`、`obj`、`scenes`。
- 场景：约 500 train / 200 test scenes，每场景 3–10 个物体；train/test 按 scene 划分，无泄漏。
- 物体形状/尺寸：CLEVR 的 `shape`/`size`（cube/sphere/cylinder，small/large），用于支撑/碰撞的体积近似。
- 相机：复用 `run_upper_bound_u6._fit_camera` 的 DLT 标定（RMS≈4.84 px，`det(R)=+1`），作为“已标定相机”的起点。

### 3.2 多视角观测模拟

对每个 test scene，围绕场景中心布置 N 个虚拟相机：

- 方位角均匀分布（如 0°/120°/240°），仰角固定，半径固定；
- 对每个物体中心做针孔投影，得到像素观测；
- 深度观测 = 相机系 z + 噪声；
- 像素噪声 σ_px ∈ {1, 2, 4} px；深度噪声 σ_z ∈ {0.5%, 1%, 2%}（相对距离）；
- 视角位姿噪声：yaw/pitch σ ∈ {0°, 1°, 3°}；
- 关联：默认已知 object identity；S2 加 5–10% 错误关联/离群点，测鲁棒损失。

> 这是**受控的感知前端模拟**，不是完整渲染。它检验估计器与三维假设，不检验渲染/检测器。

### 3.3 划分

| 集合 | 用途 |
|---|---|
| train scenes | 拟合先验、调超参（鲁棒损失、约束权重） |
| cal scenes | 选视角数/阈值/校准 |
| test scenes | 只报一次 |

---

## 4. 场景模型与优化器

### 4.1 状态

每个物体：

- 3D 中心 `p = (x, y, z)`；
- 朝向 `yaw`（CLEVR 物体近似轴对称时可固定为 0）；
- 形状先验：cube / sphere / cylinder + 尺寸；
- 可选：尺度/半径。

场景状态 `X = {p_i, yaw_i}`。

### 4.2 观测因子

| 因子 | 残差 | 作用 |
|---|---|---|
| 重投影 | `project(p_i, cam_k) - uv_ik`，按 σ_px 归一化 | 像素约束 |
| 深度 | `z_cam(p_i, cam_k) - d_ik`，按 σ_z 归一化 | 深度约束 |
| 桌面/支撑 | `bottom_z(p_i) - table_z` | 物体在桌面上 |
| 不穿透 | `max(0, r_i + r_j - ||p_i - p_j||)` | 物体不重叠 |
| 接触（可选） | `|surface_distance(i,j)|` 软约束 | 接触/堆叠 |
| 先验 | `p_i - p_i^prior`，按场景先验协方差 | 房间布局先验 |
| 时序 | `p_i^(t) - p_i^(t-1)` | 多帧平滑 |

### 4.3 求解

- `scipy.optimize.least_squares`，`loss="soft_l1"` 或 Huber，`ftol/xtol` 固定；
- 变量维度小（N 物体 × 3–4），CPU 可解；
- 初始化：单视角三角化/反投影 + 桌面先验；
- 输出：`X*`、残差、Jacobian。

### 4.4 不确定度

- 线性化协方差：`Σ ≈ σ² (JᵀJ)^{-1}`；
- 关系 belief：从 `N(X*, Σ)` 采样 M=200 个场景，统计关系成立比例；
- 或对关系函数做线性化 probit；
- 校准：在 cal 上做 temperature scaling / isotonic，test 只报一次。

---

## 5. 关系查询（在三维里，不在 1D 里）

| 查询 | 三维定义 | 对应旧指标 |
|---|---|---|
| `axis_aligned_3d` | `|x_i-x_j| ≤ eps_xy` 或 `|y_i-y_j| ≤ eps_xy`，且同支撑面 | near-aligned（旧 horizontal/vertical） |
| `same_level` | `|bottom_z_i - bottom_z_j| ≤ eps_z` | 新增，纯三维 |
| `contact` | 表面距离 ≤ eps_c | 新增 |
| `collision` | 穿透深度 > 0 | 新增 |
| `support` | 物体底部接近桌面/另一物体顶部 | 新增 |

- 主指标仍用 `axis_aligned_3d`，与一维结果可直接对照；
- `same_level` / `contact` / `collision` / `support` 是三维独有查询，用来证明“回到三维”不只是换个算法，而是能回答一维回答不了的问题。

---

## 6. 基线 / 对照

| arm | 说明 |
|---|---|
| `1d_hard` | 旧 hard dead zone（0.685） |
| `1d_balanced` | 类平衡 Bayes（0.848） |
| `1d_multiframe_M64` | 一维多帧 64（0.846） |
| `3d_oracle` | 真值位姿，上限参照 |
| `3d_1view` | 单视角最小二乘 |
| `3d_3view` | 3 视角联合优化（主候选） |
| `3d_5view` | 5 视角 |
| `3d_3view_noconstraint` | 去掉支撑/不穿透（H2 对照） |
| `3d_3view_nodepth` | 去掉深度（H1 对照） |
| `3d_3view_outliers` | 5–10% 错误关联/离群（鲁棒性） |

---

## 7. 指标与预注册 bar

### 7.1 主指标

- `axis_aligned_3d` accuracy，在 near-boundary 子集（真实 `|Δx|` 或 `|Δy|` 在 `[0.5eps, 1.5eps]`）；
- 3 视角读数。

### 7.2 守卫 / 次要

| 指标 | bar |
|---|---|
| 3D 位置误差 median | ≤1 cm |
| 3D 位置误差 p90 | ≤3 cm |
| overall relation accuracy | ≥0.95 |
| `same_level` / `contact` / `collision` accuracy | ≥0.95 |
| 关系 belief ECE | ≤0.05 |
| 单场景更新延迟（CPU） | ≤50 ms |
| clean 性能 | 不低于 1D 基线 −2 pt |

### 7.3 预注册对照

| ID | 对照 | 主指标 | 方向 |
|---|---|---|---|
| S1 | `3d_3view` vs `1d_multiframe_M64` | near-aligned acc | 3D ≥ 1D +0.05 |
| S2 | `3d_3view` vs `3d_3view_noconstraint` | near-aligned acc | 有约束 ≥ 无约束 +0.03 |
| S3 | `3d_3view` vs `3d_1view` | near-aligned acc | 3 视角 ≥ 1 视角 +0.10 |
| S4 | `3d_3view` vs `3d_oracle` | near-aligned acc | 差距 ≤0.05 |
| S5 | belief ECE | ECE | ≤0.05 |

---

## 8. 实验阶段

### S0：模拟器 + 指标 + 基线复现

- 新建 `pss/scene/`（model / observation / optimizer / relations）与 `pss/eval/scripts/run_scene_bench.py`。
- 复现 1D 基线：hard 0.685、balanced 0.848、M=64 0.846。
- 出口：观测模拟、指标、基线在同一 split 上可复现。

### S1：三维位置估计（无约束）

- 1 / 3 / 5 视角的最小二乘/三角化；只做位置，不做形状/约束。
- 指标：位置误差、`axis_aligned_3d`。
- 出口：3 视角是否已超过 1D M=64。

### S2：物理约束 + 消融

- 加桌面/支撑、不穿透；做 `noconstraint`、`nodepth`、`outliers` 对照。
- 出口：H2 判定；约束是否伤害 clean 性能。

### S3：不确定度与主动视角

- 协方差 + Monte Carlo belief；cal 上校准；test 报 ECE/risk-coverage。
- 主动视角：按信息增益选第 2/3 个视角，对比固定视角。
- 出口：H3 判定；belief 是否可校准。

### S4：PSS 集成

- `SceneModel` → `Entity` 的 pose/covariance/provenance；
- `RelationFactLayer` 从场景模型查询 `axis_aligned_3d` / `contact` / `collision` / `support`；
- `UncertaintyService` 消费 belief；`CognitiveState` 记录 audit；
- 1D `axis_mode="bayes"` 保留为 fallback。
- 出口：三维关系写入 runtime，审计可查；旧测试不回归。

### S5：报告与 go/no-go

- 主表、消融、校准、延迟；
- 决策：3D 场景层是否成为默认几何层；1D 是否只做 fallback；
- 失败项照实保留，不调 bar。

---

## 9. 停止规则

| 观测 | 结论 | 动作 |
|---|---|---|
| `3d_3view` < `1d_multiframe_M64` | H1 假 | 停止加视角；诊断对应/离群/相机/约束 |
| `3d_3view` ≥0.90 且 > 1D +0.05 | H1 成立 | 推进 S2/S4 |
| 加约束反而变差 | H2 假 | 去约束；检查约束权重/冲突 |
| `3d_3view` 与 `3d_oracle` 差距 >0.10 | 估计器问题 | 查初始化/关联/鲁棒损失，不加新臂 |
| ECE >0.10 | H3 假 | 不宣称 uncertainty；先修协方差/模型 |
| 5 视角 <0.90 | 视角/几何不足 | 检查相机布局/基线；不再调谓词 |
| 任何 guard 回退 >2 pt | 有代价 | 回滚；记录 trade-off |

**S1 结束必须做一次 go/no-go：如果 3D 3 视角没有超过 1D M=64，就不再推进 S2–S4，转诊断。**

---

## 10. 与 PSS 的关系

- **场景模型是几何基底**；PSS 是语义/关系层，不是几何估计器。
- `Entity` 已支持 3D pose；新增 covariance/provenance 即可承载场景后验。
- `RelationFactLayer` 从“逐对谓词函数”改成“场景模型上的查询”；1D 谓词只做 fallback。
- `UncertaintyService` 消费关系 belief；不确定时 review/abstain/主动观测。
- `CognitiveState` 的记录/版本/审计不变；新增 `scene_version` 与 `pose_covariance` 字段（兼容性变更需评估）。

---

## 11. 交付物与时间盒

```
pss/scene/model.py                  # SceneModel / ObjectState / Observation
pss/scene/optimizer.py              # 因子 + least_squares + 协方差
pss/scene/relations.py              # axis_aligned_3d / same_level / contact / collision / support
pss/eval/scripts/run_scene_bench.py # 一个基准，10 个臂
tests/test_scene_model.py
tests/test_scene_optimizer.py
eval/results/scene_bench/
docs/experiments/PSS-Scene-Model-Report.md
```

| 天 | 内容 | 出口 |
|---|---|---|
| D1 | S0 模拟器 + 指标 + 1D 基线 | 同一 split 可复现 |
| D2 | S1 1/3/5 视角位置估计 | H1 初判 |
| D3 | S2 约束 + 消融 | H2 判定 |
| D4 | S3 不确定度/主动视角 | H3 判定 |
| D5 | S4 PSS 集成 | runtime 写入 + 回归 |
| D6 | S5 报告 + go/no-go | 决策 |

---

## 12. 边界

- CLEVR 多视角是**模拟观测**，不是真实 RGB-D；相机标定、检测器、分割器都是受控的。
- 物体形状用简单几何近似；可变形物体、透明/反光、严重遮挡未测。
- 关联默认已知；S2 才加错误关联，真实检测器关联仍是开放项。
- 关系定义（eps_xy/eps_z/eps_c）是任务契约的一部分；换任务要重新预注册。
- 本计划不替代真实机器人/真实传感器验证；它验证的是“三维建模是否比一维谓词更有效”这一架构判断。

---

## 13. 一句话

**先建三维场景，再在三维里判断：用 3 个视角 + 深度 + 物理约束，检验是否超过一维多帧的 0.846。** 如果 3 视角都赢不了一维多帧，问题就不在维度，而在对应/约束/噪声模型；如果赢了，PSS 的几何层应该从“逐对谓词”改成“场景模型 + 三维查询”。
