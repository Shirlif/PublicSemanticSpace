# 局部观测下的物体中心/位姿估计：资料与方案

> 日期：2026-11-28
> 上游：`docs/plans/PSS-三维场景建模实验计划-v2.md` §3
> 目的：解释 MuJoCo smoke 里“可见表面点云质心相对物体中心偏 ~5 cm”不是 bug，而是部分观测的几何事实；给出成熟解法，并把它变成 v2 的 T2 gate。
> 状态：资料整理 + 方案建议；未执行代码。

---

## 0. 一句话

**你看到的是半个表面，表面质心不是物体中心。** 这不是“估计器不准”，是**部分观测的逆问题**；标准解法不是取质心，而是**用已知形状/完整模型去拟合**：已知图元拟合、render-and-compare、点云配准、或类别级位姿网络。v2 必须先把中心误差压到 ≤1–2 cm，才能做 1D/3D 对照。

---

## 1. 问题定义

给定：

- 一个物体的部分表面观测（mask + depth + 多视角点云）；
- 已知或可推断的形状/尺寸（本实验：MuJoCo 图元，类别先验）；
- 相机内外参、支撑面/物理约束。

求：

- 物体完整 3D 位姿/中心 `T`（以及协方差）。

为什么质心偏：

- 可见表面只是完整表面的一个子集；对圆柱/球/盒，可见 patch 的质心沿视线方向偏；
- 偏多少取决于视角、遮挡、物体尺寸/形状；
- 对 near-aligned 判据（`eps_xy=0.12 m`），5 cm 偏差是 ~40% 的判据，必须消除。

---

## 2. 成熟解法（按与本实验的匹配度排序）

### 2.0 如果已有完整扫描模型

激光扫描/离线重建确实能给你**物体的完整模型**，但“有模型”和“有 live transform”是两件事：

| 场景 | 是否已有 transform | 还需要什么 |
|---|---|---|
| 静态场景 + 原地全扫描 | 是，transform 在扫描坐标系里 | 分割/语义/与机器人坐标系对齐/变化检测 |
| 离线模型 + live 部分视图 | 否 | **6DoF 配准/跟踪**（已知模型位姿估计） |
| 动态/移动物体 | 否 | 在线配准/跟踪 + 更新 |
| 透明/反光/遮挡 | 模型可能不完整 | 多视角/多传感器/补扫 |

所以：

- **形状拟合可以省略**（如果模型已知）；
- **位姿估计/配准不能省略**（如果 live 观测是部分的、或物体可能移动）；
- 5 cm 偏差的消除不是靠“拟合形状”，而是靠**用完整模型的几何去配准部分观测**。

这也是 v2 把 T2 改名为“已知模型位姿估计”的原因。

### 2.1 已知形状的图元拟合（模型未知时）

**思路**：物体是 box/cylinder/sphere/ellipsoid；把**完整图元**拟合到部分点云，中心是图元的中心，不是可见 patch 的质心。

- **RANSAC + 图元模型**：Schnabel et al., *Efficient RANSAC for Point-Cloud Shape Detection*, Computer Graphics Forum 26(2), 2007. DOI: [10.1111/j.1467-8659.2007.01016.x](https://doi.org/10.1111/j.1467-8659.2007.01016.x)。一次检测平面/球/圆柱/圆锥/环面。
- **可微参数曲面拟合**：Sharma et al., *ParSeNet: A Parametric Solution for Point Cloud Segmentation and Surface Fitting*, CVPR 2020. [arXiv:2004.06799](https://arxiv.org/abs/2004.06799)。平面/球/柱/锥/环面的分割 + 拟合。
- **鲁棒最小二乘**：点到图元表面的正交距离 + Huber/Cauchy 损失；比代数拟合（Taubin 等）在 partial scan 下偏差小得多。
- **对称/尺寸先验**：圆柱/球的中心沿轴可能欠约束；用支撑面、已知尺寸、多视角联合补足。

**为什么能消偏**：拟合的是**完整图元参数**（中心、轴、半径），可见 patch 只提供约束；未看见的一侧由形状先验外推。

**本实验的最小实现**：

- box：点到 6 个面的距离 + 已知尺寸；或平面提取 + 角点；
- cylinder：法向协方差最小特征向量给轴，投影到垂直平面做 2D 圆拟合；
- sphere：几何（正交距离）拟合，已知半径时只估中心；
- ellipsoid：已知轴长时估中心/朝向；
- 多视角共享同一 `T`；支撑/不穿透作为软约束。

### 2.2 Render-and-compare / analysis-by-synthesis（次选，和 2.1 互补）

**思路**：在候选位姿渲染已知图元，比较 **mask/轮廓 + depth**，优化位姿。

- PWP3D: Prisacariu & Reid, *PWP3D: Real-Time Segmentation and Tracking of 3D Objects*, ECCV 2012 / IJCV 2012（逐像素前景/背景后验，天然处理部分遮挡）。
- Krull et al., *Learning Analysis-by-Synthesis for 6D Pose Estimation in RGB-D Images*, ICCV 2015. [arXiv:1508.04546](https://arxiv.org/abs/1508.04546)。CNN 比较观测与渲染，学习位姿后验；Occlusion LineMOD 基准。
- 优点：用**完整轮廓**，不只是可见点；对遮挡/纹理缺失稳。
- 缺点：需要可微/可渲染；优化慢；本实验可用 MuJoCo 自己渲染候选位姿，做小规模搜索/局部优化。

### 2.3 点云配准到完整模型（有 CAD/网格时）

- ICP: Besl & McKay, *A Method for Registration of 3-D Shapes*, PAMI 1992；point-to-plane ICP。
- FPFH + SAC-IA: Rusu et al., 2009；PPF: Drost et al., CVPR 2010；Go-ICP、TEASER++ 等全局配准。
- 适用：有完整网格/点云模型；本实验是图元，2.1 更直接。

### 2.4 类别级/学习式位姿（形状未知时）

- NOCS: Wang et al., *Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation*, CVPR 2019. [arXiv:1901.02970](https://arxiv.org/abs/1901.02970)。
- CenterSnap: Irshad et al., *CenterSnap: Single-Shot Multi-Object 3D Shape Reconstruction and Categorical 6D Pose and Size Estimation*, 2022. [arXiv:2203.01929](https://arxiv.org/abs/2203.01929)。单阶段、中心表示、面向部分观测。
- FoundationPose: Wen et al., *FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects*, 2024. [arXiv:2312.08344](https://arxiv.org/abs/2312.08344)。CAD 或少量参考图，无需逐物体微调。
- 适用：真实资产、形状未知；本实验有图元先验，不需要上大模型。

### 2.5 游戏引擎里的对应物

游戏引擎运行时**不做**“从部分表面推中心”：它拥有精确 transform/碰撞体。

- 碰撞代理：凸包/凸分解（如 V-HACD）、AABB/OBB、support mapping；
- 距离/接触：GJK / EPA（Gilbert-Johnson-Keerthi；Ericson, *Real-Time Collision Detection*, 2004）；
- 资产管线：摄影测量/扫描用 ICP + bundle adjustment + 网格重建。

所以“游戏引擎的成熟方案”准确说是：**运行时用精确几何；离线资产/配准管线用 ICP/图元拟合/凸分解**。我们缺的是后者，不是前者。

---

## 3. 对 v2 的建议

### 3.1 方法

**已知模型 + 多视角联合位姿估计**：

- 变量：**只估位姿** `(x,y,z)`、yaw；形状/尺寸已知，不估；
- 残差：
  - 点到已知模型表面的距离；
  - mask/轮廓 reprojection（render-and-compare，可选）；
  - depth 残差；
  - 支撑（底部≈桌面）、不穿透（软）；
- 多视角共享同一 `T`；
- 鲁棒损失（Huber/soft_l1）；
- 初始化：点云质心 + 已知模型先验；必要时多起点。

### 3.2 T2 gate（必须先过）

在跑 1D/3D 对照之前，先验证拟合中心误差：

| 指标 | bar |
|---|---|
| 拟合中心误差 median | **≤1 cm** |
| 拟合中心误差 p90 | **≤2 cm** |
| 对比：点云质心误差 median | 记录（当前 ~5 cm） |
| 多视角 vs 单视角 | 记录 |

**如果 T2 gate 不过，不进入 T3**；先修拟合（残差/初始化/形状先验），不要用带 5 cm 偏差的中心去比较 1D/3D。

### 3.3 消融

- `cloud_centroid`：点云质心（偏差对照）；
- `shape_points`：点到图元表面，单视角；
- `shape_points_multiview`：多视角联合；
- `shape_silhouette`：+ mask/轮廓；
- `shape_full`：+ 支撑/不穿透。

主对照：`shape_full` vs `1d_centroid`，**等视角预算**。

---

## 4. 不能做什么

1. **不能取可见表面质心当物体中心**；5 cm 偏差会直接污染 near-aligned。
2. **不能用球近似所有形状**；圆柱/盒的中心/轴需要形状拟合。
3. **不能单视角拟合圆柱轴**；轴沿视线时欠约束，需要多视角/支撑先验。
4. **不能把 render-and-compare 的结果当“真值”**；它仍是估计，要和 MuJoCo 真值比。
5. **不能跳过 T2 gate**；中心误差没压下去，后面的 1D/3D 对照没有意义。

---

## 5. 参考（已验证链接）

- Schnabel, Wahl, Klein. *Efficient RANSAC for Point-Cloud Shape Detection*. CGF 2007. [DOI](https://doi.org/10.1111/j.1467-8659.2007.01016.x)
- Sharma et al. *ParSeNet*. CVPR 2020. [arXiv:2004.06799](https://arxiv.org/abs/2004.06799)
- Krull et al. *Learning Analysis-by-Synthesis for 6D Pose Estimation in RGB-D Images*. ICCV 2015. [arXiv:1508.04546](https://arxiv.org/abs/1508.04546)
- Wang et al. *NOCS*. CVPR 2019. [arXiv:1901.02970](https://arxiv.org/abs/1901.02970)
- Irshad et al. *CenterSnap*. 2022. [arXiv:2203.01929](https://arxiv.org/abs/2203.01929)
- Wen et al. *FoundationPose*. 2024. [arXiv:2312.08344](https://arxiv.org/abs/2312.08344)

未逐条附链接的经典工作（按标题/ venue 引用）：

- Besl & McKay, *A Method for Registration of 3-D Shapes*, PAMI 1992（ICP）。
- Rusu et al., *Fast Point Feature Histograms (FPFH)*, ICRA 2009。
- Drost et al., *Model Globally, Match Locally: Efficient and Robust 3D Object Recognition*, CVPR 2010（PPF）。
- Prisacariu & Reid, *PWP3D: Real-Time Segmentation and Tracking of 3D Objects*, ECCV 2012。
- Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2004（GJK/EPA/碰撞代理）。

---

## 6. 一句话

**不要用可见表面的质心当中心；用已知形状去拟合完整图元，多视角联合，先过“中心误差 ≤1–2 cm”的 T2 gate，再做 1D/3D 对照。** 游戏引擎运行时靠精确 transform，不解决这个问题；它对应的成熟技术是资产/配准管线的 ICP、图元拟合、render-and-compare。
