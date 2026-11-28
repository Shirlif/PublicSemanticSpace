# Phase 3 专家选型报告（bake-off）

> 状态：第一轮已跑（2026-11-22）；视觉/几何已定稿，触觉 C/D 已跑；D 量化最优但无 LICENSE，按研究用途保留，主线 R0
> 写于：2026-11-22
> 最后更新：2026-11-22
> 对应计划：`公共语义空间-落地执行计划-v1.md` → Phase 3 / Phase RD / D.6
> 对应契约：`contracts/experts.yaml`
> 原则：**先写判据与指标，再跑对比；类别准确率不是唯一判据，失败也照实报告。**

---

## 1. 目的

回答一个问题：每个模态的专家，应该用当前自训基线（R0）、现成模型（R1）、蒸馏/适配器（R1.5），还是公开真实数据自训（R2）？

本报告只做专家选型。投影头、原型库、绑定算子保持冻结；换专家只通过 `ExpertSpec` 契约，不改 Phase 4/5 的张量形状与归一化假设。

---

## 2. 预注册判据

按顺序过四关。任一关不通过，就不能进主线。

| 关 | 问题 | 通过条件 | 不通过时 |
|---|---|---|---|
| **G1 表征** | 是否有稠密表征，而不是只有 logits？ | 能取到 penultimate 或中间层 | 该模型不进 R1 候选 |
| **G2 域** | 预训练域与目标域是否重叠？ | 仿真/RD2 冒烟无特征崩塌 | 升级到 R1.5 或 R2 |
| **G3 属性** | 属性线性探针是否不低于基线？ | 相对 A 组下降 ≤ 10%（初值，待标定） | 转 R1.5 蒸馏或 R2 |
| **G4 许可** | 许可是否允许当前交付形态？ | 可商用或可开源 | 只做评测，不进主线 |

### 2.1 杀死标准

| 编号 | 条件 | 触发后的动作 |
|---|---|---|
| **K-E1** | 任一候选许可不通过 | 从主线候选移除，保留评测记录 |
| **K-E2** | 属性探针相对 A 组下降 > 10% | 不采用 R1；转 R1.5 或 R2 |
| **K-E3** | 闭集 top-1 相对 A 组下降 > 5 个百分点 | 不采用 |
| **K-E4** | 单样本延迟超出预算 | 不采用，或转 R1.5 小模型 |
| **K-E5** | 真实冒烟特征分布明显崩塌 | 不采用，先做域适配 |

> 阈值是初值，Phase 0 后写死；写死之后不得为了通过而回改。

---

## 3. 候选矩阵

| 组 | ExpertSpec | 模态 | 路线 | 特征层 | 输出维度 | 许可 | 状态 |
|---|---|---|---|---|---|---|---|
| A | `vision_smallcnn_v0` | vision | R0 | penultimate | 256 | project-internal | 已跑 |
| A | `vision_handcrafted_v0` | vision | R0 | fixed | 440 | project-internal | 已跑（2026-11-28） |
| A | `tactile_mlp_v0` | tactile | R0 | penultimate | 64 | project-internal | 已跑 |
| A | `tactile_handcrafted_v0` | tactile | R0 | fixed | 118 (8x8) | project-internal | 已跑（真实触觉，2026-11-28） |
| A | `geometry_handcrafted_v0` | geometry | R0 | fixed | 16 | project-internal | 已跑 |
| B | `vision_dinov2_vit_s14` | vision | R1 | `blocks.11.norm1` | 384 | Apache-2.0 | 已跑 |
| B2 | `vision_squeezenet_expert_v0` | vision | R1.5 | penultimate | 256 | BSD-3-Clause | 已跑 |
| C | `tactile_octopi_v0` | tactile | R1 | pooler_output | 1024 | unknown | 已跑 |
| D | `tactile_real_smallcnn_v0` | tactile | R2 | penultimate | 128 | unknown | 已跑 |

---

## 4. 评测协议

### 4.1 数据

| 项 | 内容 |
|---|---|
| 闭集 | `data/category_pretrain`：20 类，16 已见 + 4 未见 |
| 划分 | 变化 index `v % 8 < 6` → train；`== 6` → val；`== 7` → test |
| 开放集 | 4 个未见类别：`cloth`、`tool`、`sponge`、`sheet` |
| 真实冒烟 | RD2 种子集，同实例同步；到位前只跑仿真 |
| 原则 | 按实例/变化分组；测试集不参与选型；不跨划分调阈值 |

### 4.2 指标

| 指标 | 定义 | 角色 |
|---|---|---|
| 闭集 top-1 | 留出实例上的类别准确率 | 主 |
| 开放集 AUROC | 已见 vs 未见 max-score 的可分性 | 主 |
| 未见拒绝率 | val max score 5 分位阈值下的拒绝比例 | 主 |
| 已见误拒率 | 同一阈值下已见类别被拒比例 | 主 |
| 属性探针 | 从专家特征线性预测 hardness / roughness / thermal / size_x / size_y；连续值报 R² 与 Spearman ρ | 主 |
| 跨模态 R@1 / R@5 | 同实例不同模态检索 | 主 |
| 单样本延迟 | 参考硬件，batch=1，p50 / p95 | 辅 |
| 峰值显存 | 参考硬件 | 辅 |
| 许可 | 可商用 / 可开源 / 仅研究 | 门槛 |

### 4.3 运行配置

| 项 | 值 |
|---|---|
| 随机种子 | `[0, 1, 2]`，报告 mean ± std |
| 预处理 | 以 `ExpertSpec.preprocess` 为准，候选之间不得混用 |
| 投影头 | 固定同一套，不参与选型 |
| 融合权重 | 选型阶段固定，不重新 CV |
| 参考硬件 | NVIDIA GeForce RTX 4070 Laptop GPU（CUDA），batch=32 |
| 代码版本 | `bench_expert_zoo.py` SHA256 前缀 `348c4728177e8240` |
| 数据版本 | `data/category_pretrain/index.json` SHA256 前缀 `0e3b837207968f2b` |

---

## 5. 结果

### 5.1 闭集 top-1

| ExpertSpec | seed 0 | seed 1 | seed 2 | mean ± std | 相对 A |
|---|---:|---:|---:|---:|---:|
| `vision_smallcnn_v0` | 0.9948 | 0.9896 | 0.9844 | 0.990 ± 0.004 | — |
| `vision_dinov2_vit_s14` | 0.9740 | 0.9583 | 0.9635 | 0.965 ± 0.006 | -2.5 pp |
| `vision_squeezenet_expert_v0` | 0.9583 | 0.9531 | 0.9531 | 0.955 ± 0.002 | -3.5 pp |
| `geometry_handcrafted_v0` | 0.9115 | 0.9167 | 0.9219 | 0.917 ± 0.004 | — |
| `tactile_mlp_v0` | 0.2188 | 0.2240 | 0.2188 | 0.220 ± 0.002 | — |
| `tactile_octopi_v0` | 0.2188 | 0.2344 | 0.2240 | 0.226 ± 0.006 | +0.6 pp |
| `tactile_real_smallcnn_v0` | 0.2552 | 0.2552 | 0.2448 | 0.252 ± 0.005 | +3.2 pp |

> `相对 A` 按同模态基线计算：vision → SmallCNN，tactile → MLP，geometry → 手工特征；跨模态行不作判决。

### 5.2 开放集

| ExpertSpec | AUROC | 未见拒绝率 | 已见误拒率 |
|---|---:|---:|---:|
| `vision_smallcnn_v0` | 0.752 ± 0.035 | 0.109 ± 0.015 | 0.024 ± 0.011 |
| `vision_dinov2_vit_s14` | 0.605 ± 0.042 | 0.102 ± 0.015 | 0.035 ± 0.006 |
| `vision_squeezenet_expert_v0` | 0.525 ± 0.031 | 0.107 ± 0.037 | 0.061 ± 0.019 |
| `tactile_mlp_v0` | 0.242 ± 0.005 | 0.030 ± 0.003 | 0.076 ± 0.009 |
| `tactile_octopi_v0` | 0.332 ± 0.030 | 0.012 ± 0.007 | 0.038 ± 0.018 |
| `tactile_real_smallcnn_v0` | 0.178 ± 0.005 | 0.008 ± 0.000 | 0.057 ± 0.009 |
| `geometry_handcrafted_v0` | 0.238 ± 0.012 | 0.006 ± 0.001 | 0.082 ± 0.012 |

> 这是单专家 max-prob 探针，不是 Phase 5 的原型融合；不能与 `CATEGORY_PRETRAIN_REPORT.md` 的融合 AUROC 0.960 直接比较。

### 5.3 属性探针

| ExpertSpec | hardness | roughness | thermal | size_x | size_y | 平均 | 相对 A |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vision_smallcnn_v0` | 0.593 | 0.694 | 0.292 | 0.936 | 0.924 | 0.688 | — |
| `vision_dinov2_vit_s14` | 0.382 | 0.438 | -0.170 | 0.905 | 0.920 | 0.495 | -0.193 |
| `vision_squeezenet_expert_v0` | 0.410 | 0.575 | 0.027 | 0.831 | 0.893 | 0.547 | -0.141 |
| `geometry_handcrafted_v0` | 0.121 | -0.018 | 0.121 | 0.897 | 0.686 | 0.361 | — |
| `tactile_mlp_v0` | 0.877 | 0.668 | 0.006 | 0.000 | -0.006 | 0.309 | — |
| `tactile_octopi_v0` | 0.886 | 0.894 | -0.501 | -0.319 | -0.428 | 0.106 | -0.203 |
| `tactile_real_smallcnn_v0` | 0.922 | 0.906 | 0.046 | 0.003 | 0.025 | 0.381 | +0.072 |

> R² 为负表示不如直接预测均值。属性探针失败一律视为不合格，即使类别 top-1 更高。

### 5.4 跨模态检索

| 模态对 | R@1 | R@5 |
|---|---:|---:|
| V-G（SmallCNN → geometry） | 0.177 | 0.531 |
| V-T（SmallCNN → tactile） | 0.089 | 0.266 |
| G-T（geometry → tactile） | 0.026 | 0.130 |
| DINOv2-G（DINOv2 → geometry） | 0.130 | 0.469 |
| DINOv2-T（DINOv2 → tactile） | 0.052 | 0.135 |
| V-Octopi（SmallCNN → Octopi） | 0.120 | 0.302 |
| Octopi-V（Octopi → SmallCNN） | 0.021 | 0.094 |
| G-Octopi（geometry → Octopi） | 0.047 | 0.177 |
| V-D（SmallCNN → PhysiCLeAR） | 0.062 | 0.229 |
| D-V（PhysiCLeAR → SmallCNN） | 0.047 | 0.120 |
| G-D（geometry → PhysiCLeAR） | 0.010 | 0.104 |

> 方向性 Ridge 映射 A→B，只在 train split 上拟合；反向值和完整矩阵见 `eval/results/expert_zoo/expert_zoo.md`。这是专家级探针，不是 Phase 4 的 InfoNCE 投影。

### 5.5 成本与许可

| ExpertSpec | 延迟 p50 (ms) | 延迟 p95 (ms) | 峰值显存 (MB) | 许可 | 结论 |
|---|---:|---:|---:|---|---|
| `vision_smallcnn_v0` | 0.976 | 1.134 | 45.802 | project-internal | 采用（视觉主专家） |
| `tactile_mlp_v0` | 0.372 | 0.536 | 9.783 | project-internal | 主线 R0；D 仅研究用途 |
| `geometry_handcrafted_v0` | 0.154 | 0.169 | 9.568 | project-internal | 采用（几何主专家） |
| `vision_dinov2_vit_s14` | 6.172 | 6.447 | 294.173 | Apache-2.0 | 仿真主线不采用；R-Gate 复测 |
| `vision_squeezenet_expert_v0` | 2.438 | 2.601 | 27.964 | BSD-3-Clause | 不采用 |
| `tactile_octopi_v0` | 60.076 | 60.922 | 1656.151 | 无 LICENSE | 仿真主线不采用；转 R2 |
| `tactile_real_smallcnn_v0` | 0.790 | 0.903 | 80.863 | 无 LICENSE | 研究用途保留；主线 R0 |

> Octopi 为 303.4M 参数、1.66GB CUDA 峰值；延迟含 3 帧重复输入。

> 延迟为 batch=1、RTX 4070 Laptop；峰值显存为前 64 个 view 的 CUDA 峰值，仅作量级参考。

### 5.6 非神经网络视觉对照与真实图像复核（2026-11-28）

新增 0 参数手工视觉专家 `vision_handcrafted_v0`（HSV + HOG + LBP，440 维），并在真实图像文件上复核三条路线。完整报告：`docs/experiments/PSS-手工视觉专家与真实图像复核.md`。

合成基准（`data/category_pretrain`，1920 视图，3 seeds）：

| 专家 | 闭集 top-1 | 开集 AUROC | 属性 R² | 延迟 p50 | 参数量 |
|---|---:|---:|---:|---:|---:|
| `vision_smallcnn_v0` | **0.990** | **0.752** | **0.688** | 1.43 ms | 274,560 |
| `vision_handcrafted_v0` | 0.964 | 0.519 | 0.592 | 1.13 ms | **0** |
| `vision_dinov2_vit_s14` | 0.965 | 0.605 | 0.495 | 6.27 ms | 22,056,576 |

真实图像文件（CLEVR-CoGenT，组合偏移 valB；以及 TAO/Charades 真实照片 12 类）：

| 数据 | 任务 | SmallCNN | DINOv2 | 手工 |
|---|---|---:|---:|---:|
| CLEVR valB | `front_shape` | 0.350 | **0.998** | 0.884 |
| CLEVR valB | `front_material` | 0.914 | **0.986** | 0.985 |
| CLEVR valB | `front_size` | 0.979 | 0.998 | **1.000** |
| TAO test | `category`（12 类） | 0.244 | **0.456** | 0.285 |

结论：**神经网络不是唯一途径，但也不是可无条件替代**。封闭合成域上手工特征闭集≈DINOv2；开放集和真实照片上冻结预训练骨干明显更强；材质/尺寸等物理属性上手工特征以 0 参数打平。SmallCNN 在 CLEVR 组合偏移下 0.882→0.350，是“特征几何决定 PSS 下游”的直接证据。

真实触觉复核（PhysiCLeAR 真实 GelSight：333/36/39 clips；WT-MM 真实 12×32：2853/613/607 窗口）：

| 数据 / 任务 | 手工 | 合成 MLP | 真实预训练 CNN | 多数类 |
|---|---:|---:|---:|---:|
| PhysiCLeAR roughness | **0.692** | 0.385 | 0.641（PhysiCLeAR）/ 0.667（WT-MM） | 0.402 |
| PhysiCLeAR texture | **0.744** | 0.282 | 0.641（PhysiCLeAR）/ 0.667（WT-MM） | 0.468 |
| PhysiCLeAR hardness | 0.462 | 0.538（F1 0.233） | 0.385（PhysiCLeAR）/ 0.462（WT-MM） | 0.438 |
| WT-MM gripper 开/合 | 0.364 | **0.605** | 0.328（WT-MM） | 0.537 |

粗糙度/纹理上 0 参数手工特征最强；硬度没有专家真正学会（单帧静态信号不足）；WT-MM 夹爪状态由监督过的小 MLP 最好。合成 8×8 对照：MLP 闭集 0.220 vs 手工 0.184，但属性 R² 0.309 vs 0.369（手工的 hardness 0.921 / roughness 0.870 均更高）。详见 `docs/experiments/PSS-真实触觉专家复核.md`。

---

## 6. 结论

| 模态 | 采用 | 依据（G1–G4） | 回滚方案 |
|---|---|---|---|
| vision | `vision_smallcnn_v0`（R0） | G1–G4 通过；闭集 0.990 vs DINOv2 0.965 / SqueezeNet 0.955；属性 0.688 vs 0.495 / 0.547；延迟 0.98 vs 6.17 / 2.44 ms。DINOv2/SqueezeNet 触发 K-E2（属性探针下降 >10%）；闭集下降未到 K-E3 的 5 pp 阈值，但结合属性和延迟，仍不采用。 | 已是当前基线；若 R-Gate 翻转，换 DINOv2 并只重训投影头 |
| tactile | `tactile_mlp_v0`（R0，主线）；D 仅研究用途 | C 组 Octopi 触发 K-E2，已淘汰。D 组 PhysiCLeAR：闭集 0.252、属性 0.381，均优于 R0（0.220 / 0.309），G1–G3 通过；但 G4 无 LICENSE（默认 all rights reserved），按研究用途保留，不进主线。D 的 roughness 0.906 / hardness 0.922 明显优于 R0。 | 主线固定 R0；D 只在研究轨道复现，不发布权重/数据/衍生 checkpoint |
| geometry | `geometry_handcrafted_v0`（R0） | 闭集 0.917，属性 0.361；无 R1 候选参与。开放集单专家 AUROC 低，但最终开放集在 Phase 5 用原型融合评。 | 若 R1 深度特征在 R-Gate 更好，作为并行特征加入，不替换 |

### 6.1 决策规则回顾

- B/C 在类别和属性探针上都不低于 A → 采用 R1
- C 类别可以但属性探针明显掉 → 转 D（R2）
- D 也救不回来 → 问题在配对监督或本体，不在编码器
- 任何一组许可不通过 → 只做评测，不进主线

**实际执行**

- 视觉：R1（DINOv2）和 R1.5（SqueezeNet）在 G3（属性）不过；闭集下降未到 K-E3 的 5 pp 阈值，但综合 G3 与延迟，采用 R0。
- 触觉：C（Octopi）触发 K-E2，已淘汰。
- 触觉单项：Octopi 的 hardness/roughness 优于 R0，但预注册按五项平均判 K-E2，不能事后改判据。
- D（R2 PhysiCLeAR）：G1–G3 通过，G4 无 LICENSE；量化最优，但按研究用途保留，不进主线；主线 R0。
- 几何：无 R1 候选，保留 R0。
- 许可：视觉/几何通过（project-internal / Apache-2.0 / BSD-3-Clause）；Octopi 与 PhysiCLeAR 均无 LICENSE，按默认版权为 all rights reserved，未进主线。

### 6.2 本次结论的边界

- 数据是 MuJoCo 低保真渲染 + 合成触觉，R0 的胜利有“同域过拟合”成分；DINOv2 的 ImageNet 先验在这种数据上不占优，不代表真实图像上也输。
- DINOv2 保留为 R-Gate 候选：RD2 种子集到位后，用真实 RGB-D 复测闭集和属性探针，再决定是否替换。
- 触觉 C/D 都已跑；D 量化最优但许可未清，R0 暂用。R0 只是过渡，D 清权后替换。
- 单专家开放集 AUROC 只作辅助；最终开放集结论在 Phase 5 的原型融合上做。
- Octopi 的 C 组结果是域桥结果：8×8 合成接触图上采样到 224×224 CLIP 输入。它的 hardness/roughness 信号更好，说明真实 GelSight 数据上值得复测；但本次不能作为采用依据。
- 触觉的 size/thermal 本来就不是触觉该负责的属性；五项平均会惩罚 Octopi。即便按单项看，也不能改变预注册的整体判据。
- D 组也是域桥：真实 GelSight 64×64 预训练，评测时把 8×8 合成网格上采样到 64×64。它赢 R0 说明真实触觉预训练有用，但不等于真实台面已解决。
- D 的许可问题已核实：PhysiCLeAR/Octopi 仓库无 LICENSE，数据和权重也没有单独许可声明；见 §6.3。

### 6.3 许可核查（2026-11-22）

| 资产 | 许可状态 | 依据 |
|---|---|---|
| Octopi 1.0 代码 | **无 LICENSE** | GitHub `/license` 端点 404；仓库无 LICENSE/COPYING/NOTICE 文件 |
| Octopi 1.5 代码 | **无 LICENSE** | GitHub `/license` 端点 404；仓库无 LICENSE/COPYING/NOTICE 文件 |
| PhysiCLeAR 数据集 | **无 LICENSE** | 数据在 Octopi 仓库 `dataset/*.mov`；README/仓库无数据许可；论文无 data availability/license 声明 |
| Octopi 权重（7b/13b encoder/project） | **无 LICENSE** | Google Drive 文件夹；README 只写下载方式，无条款 |
| Octopi 1.0 论文（arXiv） | CC BY 4.0（仅论文） | arXiv 2405.02794 页面 |
| Octopi 1.5 论文（arXiv） | arXiv perpetual non-exclusive license（仅论文） | arXiv 2507.09985 页面 |
| RSS 2024 / RSS 2025 论文页 | 未给出代码/数据许可 | RSS 页面只链接 GitHub；RSS 2025 仅称 code 会作为 open-source resources 发布 |
| ViFi-CLIP / multimodal-prompt-learning | MIT（仅第三方代码） | Octopi README 的 Acknowledgements 表 |

结论：

- 论文许可不能推导出代码、数据、权重的许可。Octopi 1.5 的 RSS 页面说“code ... will be released as open-source resources”，但“open-source”是意图描述，不是许可证。
- 两个仓库默认适用版权法，公开可读不等于授权使用、修改或再分发。
- 商用、开源再分发、把权重打进产品，都需要作者书面许可。
- 联系作者：`samson.yu@u.nus.edu`、`harold@comp.nus.edu.sg`（论文首页）；或在 GitHub 开 issue 要求补充 LICENSE。
- 项目决策（2026-11-22）：D 按研究用途保留，主线继续 R0；不发布、不再分发 PhysiCLeAR 数据、Octopi 权重或 D 的衍生 checkpoint。商用替代方案需另找许可明确的数据集（ObjectFolder、Touch and Go 等，需逐一核实）。

---

## 7. 未决问题

- [x] DINOv2 已测：闭集 0.965、属性 0.495，均低于 A；仿真主线不采用，R-Gate 复测
- [x] Octopi C 组已跑：闭集 0.250、属性 0.106，触发 K-E2；已淘汰
- [x] R2（PhysiCLeAR）已跑：闭集 0.252、属性 0.381；量化最优，G4 许可待清
- [x] D 决策：研究用途保留；主线 R0；不发布权重/数据/衍生 checkpoint
- [x] PhysiCLeAR/Octopi 许可已查：两仓库无 LICENSE；论文许可不覆盖代码/数据/权重；需作者书面许可
- [ ] D 清权后在 RD2 真实数据上复测
- [ ] Octopi 真实 GelSight 数据复测（R-Gate 参考）
- [ ] 属性探针阈值待 Phase 0 后写死
- [ ] RD2 种子集到位前，本报告只能给仿真结论
- [ ] 参考硬件的延迟预算待定
- [ ] R-Gate：用真实 RGB-D 复测 DINOv2 vs SmallCNN

---

## 8. 复现

```bash
# 全量（Makefile target）
make expert-zoo

# 等价命令
python -m pss.encoders.scripts.bench_expert_zoo \
  --data data/category_pretrain \
  --experts contracts/experts.yaml \
  --out eval/results/expert_zoo \
  --device cuda

# 只跑 DINOv2（首次需要下载权重）
python -m pss.encoders.scripts.bench_expert_zoo \
  --only vision_dinov2_vit_s14 --allow-download \
  --out eval/results/expert_zoo_dinov2 \
  --device cuda

# 只跑 C 组 Octopi（需先下载 7b/encoder.pt）
python -m pss.encoders.scripts.bench_expert_zoo \
  --only tactile_octopi_v0 \
  --out eval/results/expert_zoo_octopi \
  --device cuda

# R2：PhysiCLeAR 缓存 + 训练
python -m pss.encoders.scripts.train_physiclear \
  --videos _external/octopi/dataset --repo-root _external/octopi \
  --data data/physiclear --out checkpoints/tactile_physiclear --epochs 30

# 只跑 D 组
python -m pss.encoders.scripts.bench_expert_zoo \
  --only tactile_real_smallcnn_v0 \
  --out eval/results/expert_zoo_physiclear \
  --device cuda
```

记录：数据哈希、配置哈希、随机种子、硬件型号、每个候选的 `ExpertSpec.version`。

---

## 9. 边界

1. 本报告只覆盖 S1 主线相关模态；热成像、雷达、运动、气味不在本期。
2. 仿真结论不能直接外推到真机，必须等 R-Gate。
3. 类别 top-1 不是唯一目标，属性探针失败一律视为不合格。
4. 现成模型的输出不是“公共空间”；所有候选仍必须经过各自的投影头。
