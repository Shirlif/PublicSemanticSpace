# F10 可行性 spike：ObjectFolder 2.0 作为真实数据回放源

> 日期：2026-11-26
> 工作线：F10 spike（不是完整实验；不提交、不改 runtime/eval/projection 核心）
> 结论先行：**ObjectFolder 2.0 可以作为 F10 的 object-level 跨模态回放源（GO, limited），但不能作为 L2 identity/tracking、sim2real 或真机验证的回放源（NO-GO）。R-Gate 与真实 RGB-D 两条路在当前环境无硬件/无同步数据，均 NO-GO。**

---

## 0. 一句话回答

**能作为什么**：一个公开、可复现、CC-BY-4.0 的 **object-level 视-触绑定** 数据集；其渲染特征可以进入 `pss_eval` 的 Stream / property / PSS runtime adapter 路径，用于验证“真实公开数据能走通评测管线”，并给出 object-level R@1 的外部参照。

**不能作为什么**：ObjectFolder 没有场景、6-D 位姿、深度、分割、时间同步或多帧交互流，也没有真实传感器噪声；因此不能支撑 L2 的 identity retention、SSR/RBE/INT/RCR 等属性，也不能支撑 sim2real / 真机主张。

**需要多少工作量**：本次 spike 约 1 天；把它做成一份可引用的生产级 addendum 约 2–4 天。真正强的 F10 主张只能来自 R-Gate 同步采集，需 4–8 周 + 硬件，当前阻塞。

---

## 1. 数据清单

### 1.1 本地资产

| 项 | 值 |
|---|---|
| 根目录 | `D:/PublicSemanticSpace/_external/ObjectFolder` |
| 总大小 | 9.4 GB |
| 对象数 | 100（`ObjectFolder1-100/1..100`）+ 1 个 demo = 101 个 `ObjectFile.pth` |
| 渲染器仓库 | `OF_render.py`、`VisionNet_*`、`TouchNet_*`、`AudioNet_*`、`taxim_render.py`、`calibs/`、`demo/` |
| 元数据 | `objects.csv`，1000 行；本地只有前 100 个对象 |
| 许可 | **CC BY 4.0**（`LICENSE`） |
| 引用 | Gao et al., *ObjectFolder 2.0: A Multisensory Object Dataset for Sim2Real Transfer*, CVPR 2022 |
| 原始网格来源 | ABO、3D Model Haven、YCB、Google Scanned Objects；README 要求同时遵守原始许可 |

`objects.csv` 字段：`id, name, size(m), material, source_url`。前 100 个对象的材质分布：Polycarbonate 24、Wood 21、Ceramic 19、Steel 17、Plastic 11、Iron 6、Glass 2。

### 1.2 每个对象的内容

```
ObjectFolder1-100/<id>/
  ObjectFile.pth      # 隐式表示：VisionNet + AudioNet + TouchNet
  model.obj           # 网格
  model.mtl
  textures/           # Diffuse / Normal / OcclusionRoughnessMetallic PNG
```

`ObjectFile.pth` 约 40–55 MB；纹理是主要体积来源。它是**隐式神经表示**，不是原始传感器采集。

### 1.3 已渲染产物（本次复用，未重新渲染）

| 项 | 值 |
|---|---|
| 渲染根 | `D:/PublicSemanticSpace/eval/results/objectfolder/renders` |
| 对象数 | 100 |
| vision | 2 张/对象，256×256 RGB |
| touch | 12 张/对象，160×120 RGB（Taxim 渲染） |
| 历史渲染 | 2026-11-22 20:21–20:56 完成 100 对象（2 vision + 4 touch，约 35 分钟）；12-touch 集随后补齐 |

注意：当前 12-touch 渲染集与历史 4-touch 集**不是同一批触觉查询**（t12 重采样并覆盖了 4-touch 图像），因此历史 4-touch 零样本数字不能从当前目录精确复现；本报告同时给出当前 12-touch 的零样本基线，并以 12-touch 上训练的触觉编码器结果作为最强证据。

### 1.4 渲染可行性（当前环境实测）

环境：Python 3.13.7 / PyTorch 2.10.0+cu126 / CUDA 可用。渲染器 `environment.yml` 原本要求 Python 3.8 / PyTorch 1.8.1。

- demo vision：4 帧约 20 秒（≈5 s/帧，含模型加载）。
- demo touch：4 帧约 1.5 秒（Taxim 预计算 76 步约 1.5 秒）。
- 结论：**当前环境可以跑渲染器**（至少 demo 单对象 vision + touch 均成功）。但尚未在 100 对象规模上重跑；本次 spike 完全复用已有渲染，未触发该风险。
- 粗估：100 对象 ×（2 vision + 12 touch）约 25–40 分钟；1000 对象约 4–7 小时（GPU）。

---

## 2. 字段映射：哪些能进 PSS，哪些不能

| ObjectFolder 字段 | PSS 目标 | 能否映射 | 说明 |
|---|---|---|---|
| 渲染 vision RGB（256×256，2 视角） | `pss_eval.Observation.features`（经 DINOv2）；bundle 的 `obs_detections.emb` | **部分** | 只能作为 embedding；没有检测框、分割、位姿、分数、类别预测。 |
| 渲染 touch RGB（160×120，12 触觉点） | `pss_eval.Observation.features`（经 DINOv2） | **部分** | 不是 PSS 触觉专家期望的 8×8 或 12×32 接触图；不能直接喂 `pss/encoders/tactile/*`。 |
| object id（1..100） | `Stream.identity_labels`、instance id | **是** | 静态 object-level identity；不是时序 track。 |
| `objects.csv` name / material / size / url | 类别/属性元数据 | **部分** | material 可作为弱属性；不是任务标签。 |
| — | `pss/memory/perception.Observation`（pos, quat, appearance, raw） | **否** | 无 pose/quat；可填 appearance，但其余缺失。 |
| — | `RDEpisode` / `RDFrame`（rgb, depth, mask, tactile 8×8, sensor_pose, object_pose, timestamps, sync_offset） | **否** | 无 depth、mask、pose、时间戳、同步误差。 |
| — | bundle `obs_detections.parquet`（逐帧检测、pose、score、class_pred、emb） | **否** | 无场景、多物体帧、检测器输出。 |
| — | `gt.parquet` 实例轨迹 | **否** | 无时序 track。 |
| AudioNet 冲击声 | 音频模态 | **未使用** | PSS 当前没有 audio expert；可留作未来扩展。 |

**结论**：ObjectFolder 能填充 `pss_eval` 的 `Observation.features` + `Stream.identity_labels`，因此可以走 `pss_eval` 的协议路径；但它无法填充 `pss/memory/perception.Observation` 的位姿字段、`RDEpisode` 的传感器字段或 bundle 的检测表。**它不是 RD2 的替代品。**

---

## 3. 最小端到端结果

### 3.1 特征提取

- 编码器：`facebook/dinov2-small`（`AutoModel`，pooler/mean），384 维。
- 视觉：每个对象 2 张图的 DINOv2 特征取均值并 L2 归一化。
- 触觉：每个对象 12 张触觉 RGB 的 DINOv2 特征取均值并 L2 归一化。
- 复用现有 `pss/rd2/objectfolder.py` 的 `_load_encoder` / `_encode_paths` / `_retrieval` / `_ridge_retrieval`，只加了特征缓存。
- 20 对象（280 张图）与 100 对象（1400 张图）在单 GPU 上均为分钟级；缓存后重跑秒级。

### 3.2 20 对象（主 spike）

对象：按数字 id 取前 20 个（1–20）；2 vision + 12 touch/对象；chance(R@1)=1/20=0.05，chance(R@5)=5/20=0.25。

| 协议 | 方向 | R@1 | R@5 | chance R@1 | chance R@5 |
|---|---|---:|---:|---:|---:|
| zero-shot | vision→touch | **0.050** | 0.300 | 0.050 | 0.250 |
| zero-shot | touch→vision | **0.050** | 0.300 | 0.050 | 0.250 |
| Ridge half-split (10 train / 10 test) | vision→touch | 0.000 | **0.700** | 0.100 | 0.250 |
| Ridge half-split | touch→vision | 0.100 | 0.500 | 0.100 | 0.250 |

- 同物余弦 0.0493 vs 异物 0.0437，gap **+0.0057**（信号极弱）。
- 置换检验（1000 次）：观测 R@1=0.050，null 均值 0.050，**p=0.662** → 零样本 R@1 不高于随机。
- 解读：20 对象上零样本 top-1 完全在 chance；R@5 仅略高于 5/n chance（0.30 vs 0.25）。Ridge 的 R@5 明显高于 chance（0.70 vs 0.25），但 R@1 仍在 chance 附近。

### 3.3 100 对象（一致性检查）

对象：全部 100 个；2 vision + 12 touch；chance(R@1)=0.01，chance(R@5)=0.05。

| 协议 | 方向 | R@1 | R@5 | chance R@1 | chance R@5 |
|---|---|---:|---:|---:|---:|
| zero-shot | vision→touch | **0.010** | 0.040 | 0.010 | 0.050 |
| zero-shot | touch→vision | **0.020** | 0.060 | 0.010 | 0.050 |
| Ridge half-split (50/50) | vision→touch | 0.040 | **0.280** | 0.020 | 0.050 |
| Ridge half-split | touch→vision | 0.040 | **0.180** | 0.020 | 0.050 |

- 同物余弦 0.0365 vs 异物 0.0325，gap **+0.0040**。
- 置换检验：观测 R@1=0.010，null 均值 0.010，**p=0.659**。
- 解读：100 对象上零样本 R@1 在 chance，R@5 甚至略低于 5/n chance（0.04 vs 0.05）；Ridge R@5 高于 chance（0.28 vs 0.05），但 R@1 仍很低。**DINOv2 零样本不足以做 object-level top-1 绑定。**

### 3.4 历史最强证据（同一 12-touch 渲染集，100 对象，5 折 held-out）

| 方法 | 方向 | held-out R@1 | held-out R@5 |
|---|---|---:|---:|
| 训练触觉编码器（tactile-only） | vision→touch | **0.220 ± 0.051** | **0.730 ± 0.117** |
| 训练触觉编码器（tactile-only） | touch→vision | 0.100 ± 0.045 | 0.520 ± 0.112 |
| 对称投影（symmetric） | vision→touch | **0.240 ± 0.058** | 0.620 ± 0.117 |
| 对称投影（symmetric） | touch→vision | 0.100 ± 0.032 | 0.430 ± 0.051 |

来源：`eval/results/objectfolder_tactile_t12/TACTILE_ENCODER_REPORT.md`、`eval/results/objectfolder_symmetric_t12/SYMMETRIC_REPORT.md`。

- 训练触觉编码器把 vision→touch held-out R@1 从零样本的 0.01 拉到 **0.22**，R@5 拉到 **0.73**；这证明 ObjectFolder 的共享实例 latent 是真实的、可学的。
- 但 R@1 仍远低于 R-Gate 的 **0.7** 门槛（`pss/rd2/protocol.md` §7：`≥ max(0.6, 仿真值×0.7)`）。对称投影把 R@1 提到 0.24，仍未过线。

### 3.5 进入 `pss_eval` 路径

Stream：每个对象 1 条 vision + 1 条 touch observation，`identity_labels` 映射到 `object:<id>`；共 40 条（20 对象）/ 200 条（100 对象）。系统：

- `pss-runtime`：现有 `PSSRuntimeAdapter`（`input_dim=384`，`public_dim=128`，`budget_ms=50`）。
- `objectfolder-passthrough`：直接输出观测特征，作为 binding 的 control。
- `objectfolder-nn-binding`：最近邻跨模态绑定 replay adapter（实现 `identity_assignments`）。

属性：`extended_properties()` 去掉 `identity_retention`，加入自定义 `cross_modal_binding`（object-level R@1，带 200 次置换检验，只有 R@1>chance 且 p<0.05 才 pass）。

| 系统 | determinism | reset | state hygiene | contract | swap/rollback | budget | latency | cross_modal_binding |
|---|---|---|---|---|---|---|---|---|
| pss-runtime | pass | pass | pass | pass | pass | pass | pass | **fail** |
| objectfolder-passthrough | pass | pass | pass | skip | skip | skip | pass | **fail** |
| objectfolder-nn-binding | pass | pass | pass | skip | skip | skip | pass | **fail** |

具体数值：

- 20 对象：`pss-runtime` R@1=0.100 / R@5=0.300 / chance=0.050 / 置换 p=0.289；passthrough R@1=0.050 / p=0.672。`pss-runtime` 的 0.100 只是 20 个里对 2 个（chance 1 个），不显著。
- 100 对象：`pss-runtime` R@1=0.010 / R@5=0.030 / chance=0.010 / p=0.627；passthrough R@1=0.010 / p=0.657。
- 延迟：`pss-runtime` 在 40 条观测上 p50≈1.1–1.6 ms、p99≈2.0–3.2 ms；budget 50 ms 下 pass（100 对象出现 1/200 timeout，p99 仍远低于 budget）。
- 结论：**真实 ObjectFolder 特征确实进入了 PSS 的 `pss_eval` 路径**；PSS runtime 的 contract 属性（determinism/reset/state hygiene/swap/rollback/budget/latency）全部成立。但 **binding 能力不成立**：所有系统的 object-level R@1 都不显著高于 chance。

### 3.6 为什么没有用标准 `identity_retention`

`IdentityRetentionProperty` 需要时序 track 和 harness 侧 instance labels。ObjectFolder 没有位姿/时序；实测把 4 个对象的 8 条检测喂给 `PSSRuntimeAdapter`，全部并到 **1 条 track**，`identity_retention=1.0` 是退化值，不是绑定证据。因此本次显式移除该属性，改用 `cross_modal_binding`；报告中也不把 `identity_retention` 作为任何主张。

---

## 4. Sim vs Real 可比性

### 4.1 任务语义差异

| 维度 | 合成 bundle / PSS L2 | ObjectFolder 2.0 |
|---|---|---|
| 场景 | 多物体、逐帧检测 | 单物体、静态渲染 |
| 时序 | 多帧 episode | 无时间维 |
| 位姿 | 每检测 6-D pose + GT | 无 |
| 深度/分割 | 有（bundle/仿真） | 无 |
| 触觉 | 8×8 或 12×32 接触图 | 160×120 渲染 RGB |
| 编码器 | SmallCNN / 触觉 MLP/conv | DINOv2（视觉和触觉都用） |
| 标签 | 实例 track、类别、GT | object id、material |
| 属性 | SSR/RBE/INT/RCR/PCR/AUD/BAR 等 | 只能做 object-level R@1 |
| 真实传感器 | 否（仿真/渲染） | 否（隐式渲染） |

### 4.2 结论：**不能直接对比**

- **不能**把 ObjectFolder 的 R@1 与 bundle 的 L2 指标（identity retention、SSR、RBE、INT、RCR、BAR、延迟）放在同一张表里比较；两者不是同一任务。
- **也不能**把 ObjectFolder 的跨模态 R@1 与 PSS 合成数据的跨模态 R@1 直接比较：触觉模态不同（8×8 接触图 vs 160×120 渲染 RGB）、编码器不同（PSS tactile MLP/conv vs DINOv2）、对象集不同。
- ObjectFolder 与 PSS 的唯一共享语义是“同一 object 的 vision 与 touch 应该在 embedding 空间互相检索”。这可以作为 **L2 的组件级外部参照**，但不能替代 L2 属性。

### 4.3 可以做的有限对比

- **方向一致性**：ObjectFolder 上零样本 DINOv2 binding 在 chance，训练触觉编码器后 held-out R@1 0.22 / R@5 0.73；这与 PSS 合成数据上“跨模态绑定需要共享 latent / 训练，零样本不够”的结论方向一致。
- **数量级参照**：ObjectFolder 的 0.22 仍低于 R-Gate 的 0.7 门槛；说明“公开 object-level 数据 + 训练编码器”也不足以宣称强跨模态绑定，更不能宣称 sim2real。
- **协议路径**：ObjectFolder 特征能跑通 `pss_eval` 的 Stream/property/PSS runtime adapter，说明协议层不依赖合成数据的内部类型；但 identity/tracking 属性需要真实时序数据，ObjectFolder 无法提供。

---

## 5. 成本估算与 go/no-go

### 5.1 ObjectFolder 回放（object-level）

| 项 | 估算 |
|---|---|
| 数据获取 | 已本地 100 对象（9.4 GB）；更多对象约 3.8 GB/100 对象（压缩），1000 对象约 38 GB 压缩 / ~94 GB 解压 |
| 渲染 | 2 vision + 12 touch/对象：100 对象约 25–40 分钟；1000 对象约 4–7 小时（GPU） |
| 特征提取 | DINOv2-small，100 对象分钟级；缓存后秒级 |
| 工程 | 本次 spike 约 1 天（特征提取 + binding 指标 + pss_eval 接入 + 报告）；生产级 addendum 约 2–4 天（更多对象、重跑训练编码器、误差棒、消融） |
| 能支持的主张 | 公开数据上的 object-level 跨模态绑定；零样本 vs 训练编码器的对比；协议路径的外部有效性 |
| 不能支持的主张 | L2 identity/tracking、sim2real、真机、R-Gate |

### 5.2 R-Gate 获取（真实同步数据）

| 项 | 要求 |
|---|---|
| 硬件 | 6/7 轴桌面机械臂（重复定位 ≤1 mm）、RGB-D（RealSense D4xx 或同级，≥30 FPS）、DIGIT/GelSight 触觉（≥30 FPS）、硬件触发/PTP 同步、AprilTag/动捕/SLAM 位姿 |
| 规模 | 20–50 实例、500–2000 episodes、5–10 s/episode、≤5 ms 同步误差、≥3 种光照/背景 |
| 工作量 | 采购 + 标定 + 采集 + 校验约 **4–8 周**（乐观），另加硬件成本 |
| 当前状态 | **阻塞**：本环境无硬件、无同步数据 |
| 能支持的主张 | R-Gate 三项：触觉可分性、跨模态 R@1、绑定误差；L2/sim2real 的外部有效性 |

### 5.3 真实 RGB-D（视觉-only 或部分）

| 项 | 要求 |
|---|---|
| 硬件 | RGB-D 相机 + 标定 + 采集台 |
| 工作量 | 2–4 周（若只做视觉冒烟）；完整 R-Gate 仍需触觉 + 位姿 |
| 当前状态 | **阻塞**：无硬件 |
| 公开替代 | YCB-Video / ScanNet 等可做视觉-only，但没有同实例同步触觉/6-D 位姿，**不能**满足 R-Gate 三项 |
| 能支持的主张 | 视觉域差/检测器冒烟；不能支持触觉/跨模态/L2 |

### 5.4 go/no-go 表

| 路径 | 数据门槛 | 工作量 | 结论 | 能支持的主张 |
|---|---|---|---|---|
| **ObjectFolder 回放** | 本地 100 对象 / 9.4 GB；2 vision + 12 touch | spike 1 天；生产 2–4 天 | **GO (limited)** | object-level 跨模态绑定；协议路径；零样本 vs 训练编码器 |
| **R-Gate 获取** | 20–50 实例、500–2000 episodes、同步 RGB-D + 触觉 + 6-D 位姿 | 4–8 周 + 硬件 | **NO-GO（当前阻塞）** | L2/sim2real/真机外部有效性 |
| **真实 RGB-D** | 相机 + 标定 + 采集；完整 R-Gate 还需触觉/位姿 | 2–4 周（视觉-only） | **NO-GO（当前阻塞）** | 视觉-only 冒烟；不足以支撑 F10 |

**建议**：如果继续 F10，投入 2–4 天把 ObjectFolder 回放做成一份**有明确边界的公开数据 addendum**（引用历史训练编码器 held-out R@1 0.22 / R@5 0.73，明确写“不是 sim2real、不是 L2”），不要投入数周扩大 ObjectFolder 对象数——对象数不能补上缺失的位姿/时序。真正能改变论文结论的只有 R-Gate 同步采集，而它当前被硬件阻塞。

---

## 6. 阻塞点

1. **无位姿/深度/分割/时序/场景**：ObjectFolder 是静态单物体渲染，无法构造 `pss/memory/perception.Observation` 或 `RDEpisode`，无法测 L2 identity/tracking。零位姿下 PSS tracker 把所有检测并到 1 条 track，`identity_retention` 退化。
2. **触觉模态不匹配**：ObjectFolder touch 是 160×120 渲染 RGB；PSS 触觉专家期望 8×8 或 12×32 接触图。不能直接复用 PSS 触觉专家/投影头；本次只能用 DINOv2 对触觉图编码。
3. **零样本绑定在 chance**：DINOv2 零样本 object-level R@1 在 20/100 对象上均不显著高于 chance（p≈0.66）；R@5 在 12-touch 集上也不高于 5/n chance。训练触觉编码器可到 held-out R@1 0.22 / R@5 0.73，但仍低于 0.7 门槛。
4. **无硬件**：R-Gate 与真实 RGB-D 均被硬件阻塞；WT-MM 等公开数据是 B 类（有同步视频+触觉，但无物体 6-D 位姿/实例 ID），不能替代 R-Gate。
5. **渲染集版本差异**：当前 12-touch 渲染集与历史 4-touch 集不是同一批查询；历史 4-touch 零样本 R@5 0.120 不能从当前目录精确复现。本报告用当前 12-touch 零样本基线（R@5 0.04）和 12-touch 训练编码器结果（R@5 0.73）分别说明。
6. **渲染环境**：渲染器原本要求 Python 3.8 / PyTorch 1.8.1；当前 Python 3.13 / PyTorch 2.10 的 demo vision + touch 均成功，但 100 对象规模重渲染尚未验证。本次 spike 复用已有渲染，未受影响。
7. **许可/引用**：ObjectFolder 为 CC BY 4.0，需引用 Gao et al. CVPR 2022；原始网格来自 ABO / 3D Model Haven / YCB / Google Scanned Objects，分发前需核对其原始许可。不要把数据提交进 git。

---

## 7. 能支持 / 不能支持的主张

### 能支持

- 在公开多感官物体数据集（ObjectFolder 2.0，CC BY 4.0）上，渲染的 vision/touch 特征可以提取，并进入 PSS 的 `pss_eval` Stream / property / PSS runtime adapter 路径。
- DINOv2 零样本 object-level 跨模态绑定在 20 和 100 对象上均不显著高于 chance（R@1 0.05/0.01，置换 p≈0.66）；训练触觉编码器把 100 对象 held-out vision→touch R@1 提到 **0.22**、R@5 提到 **0.73**，但仍低于 R-Gate 的 0.7 门槛。
- PSS runtime 的 contract 属性（determinism、reset exactness、state hygiene、swap/rollback、budget、latency）在真实 ObjectFolder 特征流上成立。
- ObjectFolder 缺少位姿/场景/时序，标准 `identity_retention` 不适用；用 object-level `cross_modal_binding` 替代是必要的，且该属性在本次数据上 fail。

### 不能支持

- “PSS/L2 在真实机器人数据上有效”——没有真实机器人数据。
- “sim2real 迁移成功”——ObjectFolder 是渲染/隐式表示，不是真实传感器。
- “ObjectFolder 可以替代 R-Gate”——不能；它没有同步视-触-位姿 episode。
- “PSS runtime 提升了 ObjectFolder 上的跨模态绑定”——没有；其 R@1 在 chance，且本次 projection 是随机 adapter projection，不是真实训练头。
- “协议的身份保持属性可以推广到真实数据”——没有测；ObjectFolder 无法测时序 identity。

---

## 8. 复现命令

```bash
# 20-object spike（主结果）
python -m pss.eval.scripts.run_gap_real_replay \
  --renders D:/PublicSemanticSpace/eval/results/objectfolder/renders \
  --objects-csv D:/PublicSemanticSpace/_external/ObjectFolder/objects.csv \
  --limit 20 --seed 0 --permutations 1000 \
  --cache reports/claim_gaps/real_replay/features_20.npz \
  --out reports/claim_gaps/real_replay

# 100-object consistency check
python -m pss.eval.scripts.run_gap_real_replay \
  --renders D:/PublicSemanticSpace/eval/results/objectfolder/renders \
  --objects-csv D:/PublicSemanticSpace/_external/ObjectFolder/objects.csv \
  --limit 100 --seed 0 --permutations 1000 \
  --cache reports/claim_gaps/real_replay/features_100.npz \
  --out reports/claim_gaps/real_replay/scale100

# tests
python -m pytest tests/test_objectfolder_replay.py tests/test_objectfolder.py \
  tests/test_objectfolder_tactile.py tests/test_objectfolder_symmetric.py tests/test_rd2.py -q
```

结果文件：

- `reports/claim_gaps/real_replay/real_replay.json`（20 对象，含 binding + pss_eval 明细）
- `reports/claim_gaps/real_replay/REPLAY.md`（20 对象原始表）
- `reports/claim_gaps/real_replay/features_20.npz`（20 对象特征缓存）
- `reports/claim_gaps/real_replay/scale100/real_replay.json`（100 对象）
- `reports/claim_gaps/real_replay/scale100/REPLAY.md`（100 对象原始表）
- `reports/claim_gaps/real_replay/features_100.npz`（100 对象特征缓存）

---

## 9. 改动文件（worktree 绝对路径）

Worktree：`D:/PublicSemanticSpace/.claude/worktrees/agent-abbf49601442d3d79`

| 文件 | 类型 | 说明 |
|---|---|---|
| `pss/rd2/objectfolder.py` | 修改 | `run_objectfolder` 新增 `renders_root`，可复用只读渲染树，不复制数据 |
| `pss/rd2/objectfolder_replay.py` | 新增 | 特征提取、binding 指标、pss_eval Stream、`cross_modal_binding` 属性、passthrough/NN adapter、编排与报告 |
| `pss/eval/scripts/run_gap_real_replay.py` | 新增 | CLI |
| `tests/test_objectfolder_replay.py` | 新增 | 不依赖真实数据的单元测试（绑定指标、Stream、属性、adapter、渲染目录发现） |
| `reports/claim_gaps/real_replay/SPIKE.md` | 新增 | 本报告 |
| `reports/claim_gaps/real_replay/real_replay.json` | 新增 | 20 对象结果 |
| `reports/claim_gaps/real_replay/REPLAY.md` | 新增 | 20 对象原始表 |
| `reports/claim_gaps/real_replay/features_20.npz` | 新增 | 20 对象特征缓存 |
| `reports/claim_gaps/real_replay/scale100/real_replay.json` | 新增 | 100 对象结果 |
| `reports/claim_gaps/real_replay/scale100/REPLAY.md` | 新增 | 100 对象原始表 |
| `reports/claim_gaps/real_replay/features_100.npz` | 新增 | 100 对象特征缓存 |

未改动：`pss/knowledge/*`、`pss/memory/active_policy.py`、`pss/memory/typed_relations.py`、`pss/runtime/*`、`pss_eval/*`、`pss/projection/*`。未 git commit。
