# PSS 三维场景建模实验计划 v2：真实 MuJoCo 环境 + 形状拟合

> 日期：2026-11-28
> 上游：`docs/plans/PSS-三维场景建模实验计划.md`、`docs/experiments/PSS-Scene-Model-Report.md`
> 目的：修正 v1 的两个根本问题——**没有真实 3D 仿真环境**、**3D 观测被压成单点**；在 MuJoCo 桌面环境里，用真实 RGB-D/分割/表面点云做 3D 形状拟合，与 1D 在**等观测预算**下对照。
> 状态：方案，v2 环境桥接已完成并 smoke 通过；形状拟合与对照未执行。

---

## 0. 一句话

**v1 的“3D”只是把已知中心点投影加噪；真实传感器给的是可见表面，表面质心相对物体中心偏约 5 cm。** v2 用仓库已有的 MuJoCo 桌面环境（`pss/sim`）渲染 RGB-D/实例分割，提取每个物体的表面点云，做**已知形状的位姿拟合**，再与 1D 在相同视角数下比较。核心假设：**等预算下，3D 形状拟合应当显著优于 1D 质心谓词；3 视角能否赢 64 帧，是第二个、更强的命题。**

---

## 1. 为什么 v1 的结论要撤回/限定

v1 的 `PSS-Scene-Model-Report.md` 只能写成：

> “在**每物体每视角一个中心点**的虚拟投影观测下，3D 点三角化 3 视角不如 1D 64 视角。”

它不能写成“3D 建模不如 1D”，因为：

1. **没有 3D 环境**：没有物体、渲染、遮挡、物理；只是已知 `3d_coords` 的针孔投影。
2. **观测被压成单点**：丢掉了投影范围、形状、朝向、表面几何、可见性。
3. **求解器是点三角化**：逐物体 `(x,y,z)`，不是形状/位姿拟合。
4. **比较不等预算**：3 视角 vs 64 视角。
5. **1D 基线不是最强 1D**：`backproject_mean`，不是类平衡 Bayes。

v2 逐条修掉。

---

## 2. 3D 仿真环境（必要条件，已完成桥接）

### 2.1 现有环境

- 模拟器：**MuJoCo 4.0.1**（本机已装）。
- 场景：`pss/sim/scene.py` 的桌面 + 图元物体（box/sphere/cylinder/ellipsoid/capsule），真实物理、支撑、接触、遮挡。
- 传感器：`pss/sim/sensors.py` 的 `SensorRig`，输出 RGB、depth、instance segmentation、物体位姿、接触。
- 桥接：新增 `pss/scene/sim3d.py` 的 `Sim3DScene`：
  - 从 renderer 提取真实相机内外参；
  - 每个视角渲染 RGB/depth/seg；
  - 按 geom ID 提取每个物体的 mask、bbox、可见像素数；
  - 用深度反投影得到**世界坐标表面点云**；
  - 从物理状态读真实 3D 位姿（标签）。
- smoke 已验证：5 物体、160×120、5 个物体全部渲染；可见表面点云质心相对物体中心偏 **~5 cm**（物体半径 3–7 cm）。

### 2.2 观测阶梯

| 层级 | 观测 | 信息 |
|---|---|---|
| **O1** | mask 质心 + 中值深度 | 单点（旧 v1） |
| **O2** | O1 + bbox 宽高 + 可见像素数 | 范围/尺度，可约束深度 |
| **O3** | mask 表面点云（32–64 点）+ 深度 | 表面几何、朝向、部分遮挡 |
| **O4** | O3 + 多视角 + 可见性/遮挡 | 完整多视角约束 |

### 2.3 噪声

- 像素 σ_px ∈ {0, 1, 2, 3}；
- 深度 σ_depth ∈ {0, 0.5%, 1%, 2%}（相对深度）；
- 相机位姿 σ_yaw/pitch ∈ {0°, 1°, 2°}；
- 遮挡：通过场景物体布局与可选 occluder；mask 可见像素数作为可见性；
- 关联：默认按 geom ID 已知；S3 加 10% 错误关联/离群。

### 2.4 感知栈依赖（不在 v2 主实验里造）

扫描之后还有四个工程问题；行业已有成熟栈，调研见 `docs/research/PSS-扫描到语义到标定到变化检测.md`：

| 子问题 | 成熟栈 | v2 主实验 |
|---|---|---|
| 点云分割 | PCL/Open3D：体素→RANSAC 平面→欧氏聚类/LCCP；Mask3D | MuJoCo oracle instance segmentation |
| 语义/标签 | 3D 语义分割（PointNet++/RandLA-Net/MinkowskiNet/KPConv）；2D 检测+SAM；OpenScene | 图元类别已知 |
| 扫描↔机器人标定 | AX=XB（OpenCV `calibrateHandEye`、ROS `easy_handeye`）+ ICP/NDT/TEASER++ + ArUco + tf2 | MuJoCo 已知相机内外参 |
| 场景变化检测 | OctoMap 占用差分、TSDF/ESDF 差分、DynaSLAM/Kimera、物体级 scene graph diff | 静态场景，不做 |

**原则**：v2 主实验先用 oracle 分割/标定，把“已知模型 + 部分观测下的 3D 位姿 vs 1D 谓词”隔离出来；分割/语义/标定/变化检测各自是独立变量，后续逐个加噪声/误差做消融，不要在一个实验里同时变四个因素。

---

## 3. 3D 模型：已知模型 + 位姿估计，不是形状拟合

### 3.1 如果已有完整模型

如果物体已经有完整 3D 模型（激光扫描、离线重建、CAD），那么：

- **不需要估形状/尺寸**；形状是已知的；
- 需要的是**已知模型在 live 观测中的 6DoF 位姿**（中心 + yaw/朝向）；
- 这等价于“游戏引擎里有一个已知 asset，但它的 transform 需要从部分观测中配准出来”。

所以 T2 应改名为 **已知模型的位姿估计（known-model pose estimation）**，不是 shape fitting。5 cm 偏差的消除不是靠“拟合形状”，而是靠**用完整模型的几何去配准部分观测**。

### 3.2 但“有模型” ≠ “有 live transform”

| 场景 | 是否已有 transform | 还需要什么 |
|---|---|---|
| 静态场景 + 原地全扫描 | 是，transform 在扫描里 | 分割/语义/坐标系对齐/变化检测 |
| 离线扫描模型 + live 部分视图 | 否 | **6DoF 配准/跟踪**（T2 的主线） |
| 动态/移动物体 | 否 | 在线配准/跟踪 + 更新 |
| 透明/反光/遮挡 | 模型可能不完整 | 多视角/多传感器/补扫 |

激光扫描本身也有问题：单次扫描只看到一面；透明/反光物体扫不好；配准有误差；动态场景需要重新定位；扫描给的是几何，不是物体标签/中心/语义。**“扫描出模型”解决的是“asset 从哪来”，不自动解决“live pose 从哪来”。**

### 3.3 拟合对象

对每个物体，已知类别/图元（box/cylinder/sphere/ellipsoid）或网格：

- 变量：**只估位姿** `(x,y,z)`、yaw（形状/尺寸已知，不估）；
- 残差：表面点到已知模型表面的距离；mask/轮廓 reprojection；depth 残差；
- 多视角联合：同一物体的所有视角点云共享一个位姿；
- 物理约束：底部接近桌面（软）、物体之间不穿透（软）；
- 鲁棒损失：Huber/soft_l1，处理遮挡边缘与离群点。

> **资料依据**：`docs/research/PSS-局部观测物体中心估计.md`。可见表面质心不是中心；已知模型时用配准/render-and-compare（ICP、PWP3D、Krull）；模型未知时才需要图元拟合/类别级网络（Schnabel、ParSeNet、NOCS、CenterSnap、FoundationPose）。本实验已知图元，走“已知模型 + 位姿估计”。

### 3.4 T2 gate：先证明位姿误差被消除

在跑 1D/3D 对照之前，必须先在 MuJoCo 观测上验证：

| 指标 | bar |
|---|---|
| 已知模型位姿的中心误差 median | **≤1 cm** |
| 已知模型位姿的中心误差 p90 | **≤2 cm** |
| 点云质心误差（对照） | 记录（当前 ~5 cm） |
| 单视角 vs 多视角 | 记录 |

**T2 gate 不过，不进入 T3。** 不允许把带 5 cm 质心偏差的结果拿去做 1D/3D 对照。

### 3.3 求解

- 初始化：点云质心 + 类别尺寸先验；
- `scipy.optimize.least_squares`，变量维度小；
- 协方差：Jacobian 线性化；
- 输出：位姿 + 协方差；关系查询在拟合后的 3D 位姿上做。

---

## 4. 等预算对照

### 4.1 视角预算

N ∈ {1, 2, 4, 8, 16}；**每个 N 下 1D 和 3D 用同一批渲染**。

- 1D：从同一 mask 取质心/bbox，套类平衡 Bayes；
- 3D：从同一 mask 取点云，做形状拟合；
- 另报 `1d_centroid_64` 作为 64 帧预算点（不与 N=3 直接比优劣，只作为强命题的参照）。

### 4.2 臂

| arm | 观测 | 模型 |
|---|---|---|
| `1d_centroid_N` | O1 | 类平衡 Bayes |
| `1d_extent_N` | O2 | 类平衡 Bayes + 范围特征 |
| `3d_cloud_N` | O3 | 点云质心（偏差对照） |
| `3d_known_model_pose_N` | O3 | **已知模型 + 单视角位姿估计** |
| `3d_known_model_pose_joint_N` | O4 | **已知模型 + 多视角联合位姿估计** |
| `3d_oracle_transform` | 真值 | 已知模型 + 真值位姿（游戏引擎/全扫描上界） |
| `3d_shape_fit_N` | O3 | 形状也未知时的对照（可选，不是主路径） |
| `oracle` | 真值 | 上限 |
| `1d_centroid_64` | O1 | 64 视角类平衡 Bayes |

### 4.3 指标

- **主指标**：`axis_aligned` 在 near-boundary 子集的准确率（标签来自真实 3D 位姿）；
- pose center error median/p90；
- 3D 独有查询：`same_level`、`contact`、`collision`、`support`；
- belief ECE / risk-coverage（若启用协方差）；
- 延迟。

---

## 5. 假设与 bar

| 假设 | 内容 | bar | 证伪 |
|---|---|---|---|
| **H1' 等预算 3D 优势** | `3d_known_model_pose_N ≥ 1d_centroid_N + 0.05`（N=3/5） | 3D 3 视角 ≥0.90 | 等预算不赢 → 不是维度问题 |
| **H2' 已知模型收益** | `3d_known_model_pose_N ≥ 3d_cloud_N + 0.05` | 已知模型位姿 ≥ 点云质心 +0.05 | 配准没消偏 → 查实现 |
| **H3' 强命题** | `3d_known_model_pose_3 ≥ 1d_centroid_64` | 3 视角 ≥ 64 帧 | 强命题假，照实写 |
| **H4' 校准** | near 子集 ECE ≤0.05 | ECE ≤0.05 | 不过 → 不宣称 uncertainty |
| pose 守卫 | median ≤1 cm、p90 ≤3 cm | — | 不达 → 观测/配准上限 |

所有 bar 在跑之前写入；不调。

---

## 6. 实验阶段

### T1：环境桥接（已完成）

- `pss/scene/sim3d.py`：`Sim3DScene` / `SimFrame` / `SimObjectObservation`。
- smoke：5 物体、5 视角、RGB/depth/seg、mask/bbox/点云/真值。
- 诊断：可见表面质心偏 ~5 cm → 形状拟合必要。

### T2：已知模型位姿估计（先过位姿误差 gate）

- `pss/scene/pose_fit.py`：已知 box/cylinder/sphere/ellipsoid 模型，只估位姿（中心 + yaw），不估形状/尺寸；
- 点到已知模型表面 + 多视角联合 + 桌面/不穿透软约束；
- 单物体 smoke：位姿中心误差 vs 点云质心误差；
- **gate：中心 median ≤1 cm、p90 ≤2 cm**；不过不进入 T3。

### T3：等预算对照

- `pss/scene/arms_sim3d.py` 注册 1D/3D 臂；
- `run_scene_sim_bench.py` 用 `Sim3DScene` 生成等预算观测；
- N=1/2/4/8/16，3 seeds；
- 出口：H1'/H2' 判定。

### T4：强命题与校准

- `1d_centroid_64` vs `3d_shape_3`；
- belief/ECE；
- 3D 独有查询；
- 出口：H3'/H4' 判定。

### T5：报告与 go/no-go

- `docs/experiments/PSS-Scene-Sim-Report.md`；
- 决策：3D 形状拟合是否成为默认几何层；1D 是否 fallback；
- 失败项照实保留。

---

## 7. 停止规则

| 观测 | 结论 | 动作 |
|---|---|---|
| `3d_known_model_pose_3` < `1d_centroid_3` | H1' 假 | 停止；诊断配准/观测 |
| `3d_known_model_pose_3` ≈ `3d_cloud_3` | H2' 假 | 配准没消偏；查残差/初始化 |
| `3d_known_model_pose_3` < `1d_centroid_64` | H3' 假 | 照实写；不强推 3 帧赢 64 帧 |
| pose median >1 cm | 守卫失败 | 记录观测上限；不调 bar |
| ECE >0.10 | H4' 假 | 不宣称 uncertainty |

**T3 结束先做 H1'/H2' go/no-go；不通过就不推进 T4/T5 的默认化。**

---

## 8. 与 PSS 的关系

- 场景模型仍是几何基底；PSS 是语义层。
- `Sim3DScene` 提供真实渲染观测；`pose_fit` 提供带协方差的 3D 位姿。
- `pss/scene/pss_bridge.py` 可复用：Entity 的 pos/covariance/provenance、Relation 的 `aligned_x/y`、`same_level`、`contact`、`collision`、`support`。
- 1D 类平衡 Bayes 仍是 fallback；只有 T3/T4 通过才讨论默认化。

---

## 9. 交付物

```
pss/scene/sim3d.py                 # MuJoCo -> scene benchmark 桥（已完成）
pss/scene/pose_fit.py              # 已知模型位姿估计（不是形状拟合）
pss/scene/arms_sim3d.py            # 1D/3D 臂
pss/eval/scripts/run_scene_sim_bench.py
tests/test_sim3d.py
tests/test_pose_fit.py
docs/experiments/PSS-Scene-Sim-Report.md
eval/results/scene_sim/
```

---

## 10. 边界

- MuJoCo 图元物体是 proxy，不是真实资产；但它是**真实渲染 + 物理 + 遮挡**的 3D 环境。
- 物体形状已知（类别先验）；未知形状/可变形物体未测。
- 关联默认按 geom ID（oracle 分割）；真实检测器/分割器是后续阶段。
- 相机是 MuJoCo free camera；真实标定噪声/滚动快门未测。
- 不把 MuJoCo 结果冒充真实机器人；它是 3D 仿真环境，不是 sim2real。

---

## 11. 混杂因素与预注册（跑之前）

### 11.1 风险表

| 风险 | 为什么会改变结论 | 控制 / 消融 | 状态 |
|---|---|---|---|
| **实例尺寸/形状与 known model 不一致** | 3D 拟合有系统偏差；或 3D 用真实尺寸、1D 用类别默认，信息不对称 | 用**扫描/真实实例尺寸**作为 known model；category-prior 单独消融；1D 也给 bbox 尺寸 | 必须固定 |
| **观测信息不对称** | 3D 拿点云+形状，1D 只拿质心，比较的是“观测模型”不是“估计器” | 加 `1d_extent_N`（bbox）；明确因果问题是“3D 观测+已知模型 vs 1D 观测+谓词” | 必须固定 |
| **标定/分割 oracle** | 3D 需要内外参/mask，1D 只需 2D；3D 拿 oracle 会占便宜 | 主实验 oracle，明确标注；单独做 mask/标定扰动消融 | 主实验固定 |
| **深度噪声尺度** | MuJoCo 桌面 ~1.1 m，1% = 1.1 cm；旧虚拟 sim 10 m = 11 cm，难度差 10× | 用 RealSense-like 噪声（轴向 σ 随 z²）；σ sweep 只作敏感性 | 必须固定 |
| **桌面/支撑先验** | 钉住 z，给 3D 免费信息；1D 没有 | 1D 也给同样 z 先验；`no_table_prior` 消融 | 必须固定 |
| **eps/窗口** | `eps_xy=0.12`、near `[0.06,0.18]` 直接决定难度 | 预注册；sweep 只作敏感性，不作主判定 | 必须固定 |
| **near support / 统计功效** | 有些场景没有 near pair；100 场景 ×3 seeds 可能测不出 +0.05 | 跑前统计 near pair 数；不够就加 scenes/seeds；paired bootstrap | 必须固定 |
| **1D 基线太弱** | 只用质心是 strawman | `1d_extent` + 类平衡 Bayes；同 split/同 cal | 必须固定 |
| **初始化/局部最优** | 点云质心 + 支撑先验可能落局部极小 | 多起点；报拟合失败率；失败样本单独统计 | 消融 |
| **对称/轴欠约束** | 圆柱/球中心沿轴可能不定 | 支撑/多视角；报中心误差，不报 yaw | 消融 |
| **遮挡/分割粘连** | 接触物体会被合并成一个 mask，3D 点云脏 | oracle geom ID；mask erosion/dilation 消融 | 消融 |
| **渲染/物理非确定** | 不同 backend/seed 结果不可复现 | 固定 seed/backend；投影/反投影单测 | 必须固定 |
| **测试集调参** | 在 test 上调 τ/权重/eps | train/cal/test 分离；test 只报一次 | 必须固定 |
| **多重比较** | 多臂里挑一个赢的 | 一个主指标；Holm 校正；失败照实留 | 必须固定 |

### 11.2 跑之前必须固定的六件事

1. **T2 gate**：已知模型位姿中心误差 median ≤1 cm、p90 ≤2 cm；不过不比较。
2. **信息对称**：1D 也拿 bbox/extent；3D 用扫描/真实实例尺寸；双方用同一批 mask。
3. **oracle 标注**：主实验的标定/分割是 oracle，必须在报告里写明；扰动消融单独做。
4. **桌面尺度噪声**：用 1.1 m 距离下的 RealSense-like 深度噪声，不沿用旧 10 m 虚拟 sim 的 1% 绝对假设。
5. **功效预检**：先统计 near pair 数与 CI 宽度；不够就加 scenes/seeds，不跑完再说“不显著”。
6. **预注册**：eps、near 窗口、bar、seeds、Holm 校正，跑前写入；不调。

---

## 12. 一句话

**v1 的结论只对“已知中心点投影”成立；v2 在 MuJoCo 桌面环境里用真实 RGB-D/分割/表面点云做已知模型的位姿估计，等预算对照 1D。** 先过 T2 的 ≤1 cm 位姿 gate，再证明 `3d_known_model_pose_3` 赢 `1d_centroid_3`；不通过就停，不再用虚拟投影冒充 3D。
