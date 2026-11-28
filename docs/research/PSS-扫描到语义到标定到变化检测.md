# 扫描点云 → 物体分割 → 语义 → 标定 → 变化检测：行业成熟方案调研

> 日期：2026-11-28
> 上游：`docs/research/PSS-局部观测物体中心估计.md`、`docs/plans/PSS-三维场景建模实验计划-v2.md`
> 目的：四个“扫描之后的工程问题”不要自己造；先确认行业成熟栈，再决定 v2 用哪条最小路径。
> 状态：资料调研；未执行代码。

---

## 0. 一句话

**这四个问题都不是新问题，行业里有成熟库/管线：分割用 PCL/Open3D 的“体素→RANSAC 平面→欧氏聚类/LCCP”，语义用 3D 语义分割或 2D 检测+SAM 提升到 3D，标定用 AX=XB 手眼标定 + ICP/NDT 配准，变化检测用 OctoMap 占用差分/TSDF 差分/动态 SLAM。** v2 主实验不需要重建这套栈：MuJoCo 已经给 oracle 分割/相机，先用它隔离“3D 位姿 vs 1D 谓词”；真实感知栈是后续集成阶段。

---

## 1. 四个子问题

| # | 子问题 | 输入 | 输出 |
|---|---|---|---|
| 1 | 点云分割成物体 | 场景点云 | 每个物体的点簇/实例 mask |
| 2 | 语义/标签 | 物体点簇 + RGB/图像 | 类别、属性、实例 id |
| 3 | 扫描坐标系 ↔ 机器人坐标系 | 相机/扫描仪、机器人 FK、标定目标 | 刚体变换 `T_robot_scan` |
| 4 | 场景变化检测 | 历史模型 + 新扫描 | 移动/新增/删除的物体 |

---

## 2. 点云分割成物体

### 2.1 经典/行业管线（首选，无需训练）

PCL/Open3D 的标准桌面管线：

1. **体素下采样**（VoxelGrid，1 cm 量级）；
2. **RANSAC 平面移除**（桌面/地面，距离阈值 2 cm 量级）；
3. **欧氏聚类**（`EuclideanClusterExtraction`，cluster tolerance 2 cm、min/max cluster size）；
4. 或 **区域生长**（法向/曲率相似）；
5. 或 **LCCP**（supervoxel + 凸性/平滑性合并，适合 RGB-D 杂乱场景）；
6. 每个 cluster 输出一个实例 mask。

参考：[PCL Euclidean cluster extraction](https://pcl.readthedocs.io/projects/tutorials/en/latest/cluster_extraction.html)；PCL region growing / LCCP 教程；Open3D 的 `segment_plane` + `cluster_dbscan`。

**优点**：无训练、可解释、参数少、实时。
**缺点**：接触/遮挡物体会粘连；参数对密度/噪声敏感；透明/反光差。

### 2.2 学习式实例分割

- **Mask3D**：Transformer + object queries，直接预测 3D 实例 mask；[arXiv:2210.03105](https://arxiv.org/abs/2210.03105)。
- PointGroup / SoftGroup / 3D-BoNet：经典 3D 实例分割基线（按标题/venue 引用）。
- 2D 检测 + SAM + 2D-to-3D 提升：YOLO/Grounding DINO + [Segment Anything](https://arxiv.org/abs/2304.02643)（按标题引用），把 2D mask 反投影/融合到点云。

**适用**：杂乱、类别多、需要泛化；需要数据/权重。

### 2.3 本实验建议

- **v2 主实验**：MuJoCo 的 instance segmentation 是 oracle（geom ID），直接用；这是为了隔离“3D 位姿 vs 1D 谓词”，不是感知前端。
- **真实栈集成**：桌面场景先用 PCL/Open3D 经典管线；杂乱场景再上 Mask3D。
- 不要把“分割”混进 1D/3D 对照；分割误差是另一个变量，单独做消融。

---

## 3. 语义/标签

### 3.1 3D 语义分割

- **PointNet++**（Qi et al., NeurIPS 2017）：层次点云特征，室内小场景基线。
- **RandLA-Net**（Hu et al., CVPR 2020）：随机采样 + 局部特征聚合，大场景高效。
- **MinkowskiNet**（Choy et al., CVPR 2019）：稀疏体素卷积，ScanNet/SemanticKITTI 强基线。
- **KPConv**（Thomas et al., ICCV 2019）：连续点卷积，rigid/deformable 变体。
- 数据集：ScanNet、S3DIS、SemanticKITTI。

### 3.2 Open-vocabulary / 2D-to-3D

- **OpenScene**：把 CLIP 特征提升到 3D，零样本开放词表分割/检索；[arXiv:2211.15654](https://arxiv.org/abs/2211.15654)。
- **OpenMask3D / ConceptFusion / LERF**：实例级/多模态开放词表（按标题引用）。
- **2D 检测 + SAM + 提升**：Grounding DINO / OWL-ViT + SAM，按 mask 反投影投票到 3D 点；适合已有 2D 栈、类别开放。

### 3.3 本实验建议

- v2 的类别是 MuJoCo 图元（box/cylinder/sphere/ellipsoid），**类别先验已知**；语义不是瓶颈。
- 真实栈：先用 2D 检测 + SAM + 3D 投票（工程量小、可解释）；需要开放词表再上 OpenScene。
- 语义标签写入 PSS 的 `Entity.category/attributes`，来源/置信度进 provenance。

---

## 4. 扫描坐标系 ↔ 机器人坐标系

### 4.1 手眼标定（AX = XB）

- 形式：`A X = X B`；eye-in-hand（相机在末端）或 eye-on-base（相机固定、marker 在末端）。
- 经典解：Tsai–Lenz、Park–Martin、Horaud、Andreff、Daniilidis（OpenCV `cv::calibrateHandEye` 全部支持）。
- 输入：机器人 FK 的末端位姿 + 标定板/ArUco 在相机中的位姿（`solvePnP`）；需要 ≥3（实践 10–20）个姿态，旋转要多样。
- 工具：OpenCV `calibrateHandEye`；ROS [easy_handeye](https://github.com/marcoesposito1988/easy_handeye) / `easy_handeye2`（rqt GUI、ArUco、发布 `/tf`）；MoveIt hand-eye 插件；ViSP `vpHandEye`。

### 4.2 扫描到地图/模型的配准

- **ICP**（Besl & McKay, PAMI 1992）：点到点/点到面；Open3D/PCL 都有。
- **NDT**（Biber & Strasser, IROS 2003）：正态分布变换，激光扫描常用。
- **全局配准**：FPFH + RANSAC、PPF（Drost et al., CVPR 2010）、Go-ICP、TEASER++（Yang et al., RSS 2020）、KISS-ICP（Vizzo et al., ICRA 2023）。
- **fiducial**：ArUco/AprilTag/ChArUco，快速给粗变换，再用 ICP 精配准。
- **ROS**：`tf2` 维护坐标系树；`robot_state_publisher` + `static_transform_publisher`。

### 4.3 本实验建议

- MuJoCo 的相机内外参是已知的，v2 不需要手眼标定；**oracle 标定**用于隔离位姿问题。
- 真实栈：ArUco/ChArUco 做粗标定 → OpenCV `calibrateHandEye` → ICP/NDT 精配准 → `tf2`。
- 标定误差要单独评估；不要把它混进 1D/3D 对照。

---

## 5. 场景变化检测

### 5.1 占据栅格/TSDF 差分

- **OctoMap**：八叉树概率占据地图，支持传感器噪声下的更新和动态变化；[octomap.github.io](https://octomap.github.io/)。比较历史/当前占据状态即可检测变化。
- **TSDF/ESDF**：Voxblox、VoxelHashing、KinectFusion 类；比较历史/当前 TSDF 或 ESDF 的符号/距离变化。
- **动态 SLAM**：DynaSLAM（[arXiv:1806.05620](https://arxiv.org/abs/1806.05620)）检测/剔除移动物体；Kimera（[arXiv:1910.02490](https://arxiv.org/abs/1910.02490)）实时度量-语义建图；MaskFusion、DS-SLAM 等。

### 5.2 物体级变化

- 维护**物体级 scene graph**（物体 id、pose、类别、时间戳）；新扫描做分割+配准后，和旧图做匹配：
  - 匹配上但 pose 变化 → moved；
  - 旧图有、新图无 → removed；
  - 新图有、旧图无 → added；
  - 属性/关系变化 → updated。
- PSS 的 `CognitiveState`/`Entity`/`RelationFactLayer` 已经适合承载物体级图；变化检测可以写在 `EntityLayer`/`RelationFactLayer` 的更新路径里。

### 5.3 本实验建议

- v2 静态场景不需要变化检测；真实栈用 OctoMap/TSDF 做低层变化，物体级 scene graph diff 做高层变化。
- 变化检测的评测指标：precision/recall of moved/added/removed、延迟、误报率。

---

## 6. v2 主实验用什么

| 子问题 | v2 主实验 | 真实栈（后续） |
|---|---|---|
| 分割 | MuJoCo oracle instance segmentation | PCL/Open3D 经典管线；Mask3D |
| 语义 | 图元类别已知 | 2D 检测 + SAM；OpenScene |
| 标定 | MuJoCo 已知相机内外参 | AX=XB + ArUco + ICP/NDT + tf2 |
| 变化检测 | 静态场景，不做 | OctoMap/TSDF + scene graph diff |

**理由**：v2 要回答的是“已知模型 + 部分观测下的 3D 位姿，是否比 1D 谓词更好”。分割/语义/标定/变化检测是四个独立变量；把它们混进来，会再次变成“多因素同时变”的不可归因实验。先用 oracle 把位姿问题隔离，再逐个加噪声/误差做消融。

---

## 7. 参考

**已验证链接**

- [PCL Euclidean cluster extraction](https://pcl.readthedocs.io/projects/tutorials/en/latest/cluster_extraction.html)
- [easy_handeye](https://github.com/marcoesposito1988/easy_handeye)
- [OctoMap](https://octomap.github.io/)
- [Mask3D](https://arxiv.org/abs/2210.03105)
- [OpenScene](https://arxiv.org/abs/2211.15654)
- [DynaSLAM](https://arxiv.org/abs/1806.05620)
- [Kimera](https://arxiv.org/abs/1910.02490)
- [Schnabel RANSAC](https://doi.org/10.1111/j.1467-8659.2007.01016.x)
- [ParSeNet](https://arxiv.org/abs/2004.06799)
- [NOCS](https://arxiv.org/abs/1901.02970)
- [CenterSnap](https://arxiv.org/abs/2203.01929)
- [FoundationPose](https://arxiv.org/abs/2312.08344)
- [Krull analysis-by-synthesis](https://arxiv.org/abs/1508.04546)

**经典工作（按标题/venue 引用，未逐条附链接）**

- Qi et al., *PointNet++*, NeurIPS 2017.
- Hu et al., *RandLA-Net*, CVPR 2020.
- Choy et al., *MinkowskiNet*, CVPR 2019.
- Thomas et al., *KPConv*, ICCV 2019.
- Besl & McKay, *A Method for Registration of 3-D Shapes*, PAMI 1992.
- Biber & Strasser, *The Normal Distributions Transform*, IROS 2003.
- Yang et al., *TEASER++*, RSS 2020.
- Vizzo et al., *KISS-ICP*, ICRA 2023.
- Oleynikova et al., *Voxblox*, IROS 2017.
- Kirillov et al., *Segment Anything*, ICCV 2023.
- Liu et al., *Grounding DINO*, ECCV 2024.
- Garrido-Jurado et al., *ArUco*, Pattern Recognition 2014.
- Rosinol et al., *Kimera*, 2020.

---

## 8. 一句话

**这四个问题都有成熟栈：PCL/Open3D 分割、3D/2D 语义、AX=XB+ICP/NDT 标定、OctoMap/TSDF+scene graph 变化检测。** v2 先用 MuJoCo oracle 把“3D 位姿 vs 1D 谓词”隔离出来；真实栈是后续集成，不要在一个实验里同时变四个因素。
