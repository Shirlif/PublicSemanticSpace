# PSS 后续优化任务清单

> 日期：2026-11-23
> 范围：**不使用新的真实感官数据**，只做算法、仿真评测、基线、工程化
> 架构定位：`PSS-认知层架构定位.md`
> 依据：`PSS_VS_TRANSFORMER_RESULTS.md`、`公共语义空间-工作总结-v1.md`、`PSS_VS_TRANSFORMER_EXPERIMENT.md`

## 0. 四波实验最终结论（2026-11-23）

> 四波任务已执行完。结论不是“PSS 全面更强”，而是：**PSS 在身份持久性、显式关系组合、开放集拒识、冲突弃权和可审计更新上有稳定结构优势；这些优势的来源不是 GNN 边消息，而是显式关系/目标层 + 实例级触觉 + 集合级匹配。**

### 0.1 总表

| 能力 | 关键结果 | 判定 |
|---|---|---|
| 闭集类别识别 | 0.979 vs 0.978 | 打平 |
| 短时集合级 re-ID | 0.989 vs 0.802（无 tactile 基线） | 系统级更好；公平归因后 PSS 仅 +0.012 |
| 完整 online tracking | 1.000 vs 0.908 | 系统级更好；transformer 无 tactile，架构归因未定 |
| 跨 benchmark v3 | 0.881 vs 0.672 | 系统级更稳；transformer 无 tactile |
| 70 帧长时遮挡 | 0.983 vs 0.900 | 系统级更好；transformer 无 tactile |
| 规划任务 OOD | 1.000 vs 0.924 | PSS 更好，幅度中等 |
| 关系组合 OOD | **1.000 vs 0.434** | **PSS 最强证据** |
| 开放集拒识 | AUROC 0.993 vs 0.912 | **PSS 明确更好** |
| 冲突弃权 | 5–7% coverage，accepted 1.000 | **PSS 独有能力** |
| 增量更新速度 | 43.94 ms vs 3767.7 ms | **PSS 快 24x** |
| 增量零遗忘 | replay -0.001，追平 PSS | **PSS 不再独有** |
| 属性 OOD | +0.011，未过 0.10 | 失败 |
| 学习型 reliability | geometry shuffle 0.59/0.245 | 未达标 |
| 世界模型 rollout | 0.0863 m，目标 <0.05 m | 未达标 |
| 匹配层 scaling N=1000 | 24.35x，retention 0.984 | 通过 |
| 匹配层 scaling N=200 | 3.53x，目标 5x | 未达标 |
| 端到端延迟 | p50 0.5 ms，p95 15 ms | 可用 |

### 0.2 已经站稳的结论

1. **身份持久性在“PSS+tactile”系统上最稳，但公平归因后主要是 tactile 输入。**
   - 短时 0.989 vs 0.802（transformer 无 tactile）；给 transformer 同样 tactile 后 **0.977**，PSS 仅高 **+0.012**。
   - 同 tactile 的 sparse_k1 transformer 到 **0.996**，反超 PSS 0.989；PSS 不是最优 re-ID 架构。
   - 完整 tracking 1.000 vs 无 tactile 0.908；跨 v3 0.881 vs 0.672；70 帧长时 0.983 vs 0.900。
   - 这些系统级结果仍成立，但不能归因给 graph matcher 架构。

2. **显式关系/目标层带来组合泛化。**
   - P5b：PSS **1.000**；transformer ID 0.997、OOD **0.434**、invalid 0.139。
   - 这是目前最干净的架构级证据：不是图边消息，而是显式几何 goal check。

3. **开放集和冲突弃权是结构优势。**
   - 开放集 AUROC **0.993 vs 0.912**，unseen reject **0.958 vs 0.626**。
   - marginal conformal：96.9% accept / 95.8% reject。
   - 冲突下只回答 5–7%，accepted accuracy 1.000；transformer 没有显式弃权。

4. **增量更新的优势是“快、省、可审计”，不是“唯一不忘”。**
   - PSS：old 0.979 不变、new 1.000、更新 43.94 ms、rollback 29.5 ms。
   - Replay：old 0.979、new 1.000、forgetting -0.001，追平零遗忘。
   - PSS 仍快 **24.1x**，且不需要旧数据、可签名回滚。

5. **长时记忆和工程化已经可用。**
   - 70 帧完全遮挡后 PSS 仍 0.983 / 0.967。
   - 端到端 p50 **0.5 ms**、p95 **14–15 ms**、RSS 927 MB。
   - 匹配层 N=1000 快 **24.35x**，true retention 0.984。

### 0.3 被推翻或必须收窄的说法

1. **“PSS 闭集识别更准”不成立。** 0.979 vs 0.978，打平。
2. **“优势来自图结构/边消息”不成立。** 消融：full 0.930、no_edges 0.941、no_assignment **0.969**、greedy 0.967。去掉边消息/assignment loss/改用贪心都更好；full 配置过拟合。
3. **“PSS 能零遗忘”不再独有。** Replay 已追平。
4. **“预训练特征会自然变强”不成立。** DINO raw 0.903（base 0.930），transformer DINO 0.752（base 0.802）；PCA-32 后 DINO-only 0.942/0.804，仍低于 tactile 0.989/0.956。
5. **“学习型 reliability 会超过规则”不成立。** Learned gate clean 0.96/0.935，geometry shuffle 0.59/0.245；规则 reliable 在 zero 下仍更稳。

### 0.4 真正的机制归因

- **成立：显式关系/目标层。** P5b 1.000 vs 0.434。
- **触觉输入解释大部分 re-ID 优势。** 给 transformer 同样的 tactile token + Hungarian 后：PSS 0.989 vs transformer 0.977，delta 仅 **+0.012**。
- **不成立：graph matcher 架构是 re-ID 优势来源。** 公平对比下架构增量很小；GNN 边消息/assignment loss 消融也不支持。

### 0.5 最终可辩护的主张

> **PSS 的 re-ID 既不是 graph matcher 架构优势，也不是最优架构：同 tactile 的 sparse_k1 transformer 0.996 已反超 PSS 0.989。真正剩下的架构级优势是显式符号关系层（P5b 1.000 vs 0.434）。**

不能再说：

- PSS 全面优于 transformer；
- PSS 闭集识别更准；
- 零遗忘是 PSS 独有；
- 图消息传递是主要机制；
- 学习型 reliability 更好；
- 世界模型已经达到 5 cm；
- 合成数据结论可以直接外推到真实机器人。

### 0.6 最大缺口

1. **真实数据 / R-Gate 仍未接。**
2. **架构归因未完全分离**：transformer 还没有拿到同样的 tactile + 集合级匹配。
3. **世界模型未达 5 cm。**
4. **sparse/retrieval attention、EWC、conformal 风险保证仍未补完。**
5. **P5b 仍是符号微基准，不是感知闭环。**

---

## 1. 当前基线

先把已经成立的结论和明显的软肋放在一起看，后面的优先级都从这里来。

### 1.1 已经占优

| 场景 | 结果 | 说明 |
|---|---|---|
| 集合级 re-ID（graph+tactile） | 0.989 vs 0.802（无 tactile 基线） | 系统级更好；公平归因后 PSS 仅 +0.012 |
| 完整 tracking ID retention | 1.000 vs 0.908 | 明确更好 |
| 跨 benchmark v3 | 0.881 vs 0.672 | PSS 掉 4.9 点，transformer 掉 13.0 点 |
| 70 帧长时遮挡 | 0.983 vs 0.900 | reappearance 0.967 vs 0.800 |
| 关系组合 OOD（P5b） | **1.000 vs 0.434** | 最强架构级证据 |
| 规划 OOD 成功率 | 1.000 vs 0.924 | 更好，但幅度约 7–8 点 |
| 开放集 AUROC | 0.993 vs 0.912 | 明确更好 |
| marginal conformal | 0.969 accept / 0.958 reject | 可用 |
| 增量更新速度 | 43.94 ms vs 3767.7 ms | 快 24.1x；但 replay 也零遗忘 |
| 匹配层 scaling N=1000 | 24.35x / retention 0.984 | 通过 |
| 端到端延迟 | p50 0.5 ms / p95 15 ms | 可用 |

### 1.2 没有优势或仍需收窄

| 问题 | 现状 |
|---|---|
| 闭集类别识别 | 0.979 vs 0.978，打平 |
| P5 组合属性 OOD | +0.011，未过 0.10 门槛 |
| 零遗忘 | replay old 0.979 / new 1.000 / forgetting -0.001，追平 PSS |
| graph 边消息 / assignment loss | full 0.930；no_assignment 0.969；no_edges 0.941；不是收益来源 |
| 学习型 reliability | geometry shuffle 0.59/0.245，未达标 |
| 世界模型 rollout | 0.0863 m，未达 5 cm |
| 匹配层 N=200 | 3.53x，未达 5x |
| class-conditional conformal | 最小类 coverage 0.333，校准样本不足 |
| 真实数据 / sim2real | 未接 R-Gate；所有结论仍是合成数据 |
| 架构归因 | transformer 未拿到同样 tactile + 集合级匹配 |

### 1.3 优先级原则

1. **先补最容易被审稿人攻击的软肋**，而不是继续加模块。
2. **先做能改变结论的实验**，再做工程优化。
3. **每个任务必须有可证伪的验收指标**，不能只写“继续优化”。
4. **所有对比统一 seeds、split、参数量/计算量口径**，避免自建集调参质疑。

---

## 2. P0：先补最影响结论可信度的四件事

### P0-1 把 DINOv2 接进 re-ID 主链

**要解决的问题**

当前最强的 re-ID 结果依赖 tactile：graph+tactile 0.989，但去掉 tactile 后 near-identical 只有 0.738。如果审稿人说“你的优势靠额外触觉”，这是最直接的攻击点。

**做法**

- 冻结 DINOv2 ViT-S/14，取 CLS token 或 patch mean；
- 投影到 128 维，和 vision/geometry/category 一起拼进 `entity_token`；
- 在 `pss/memory/learned_metric.py` 增加 `include_dino` 分支；
- `pss/memory/graph_matcher.py` 的 token dim 自动适配；
- 跑四组对照：
  1. base（vision+geometry+category）
  2. +DINOv2
  3. +tactile
  4. +DINOv2+tactile
- 额外做 missing-modality 测试：去掉 tactile、去掉 vision、两者都去掉。

**指标**

- overall / mixed / samecat / near-identical re-ID；
- P1b ID retention、switches、reappearance；
- token 维度、参数量、单帧特征提取延迟。

**验收**

- 无 tactile 时 overall ≥ 0.95，near-identical ≥ 0.90；
- 完整 tracking retention ≥ 0.99；
- 有 tactile 时不回退。

**工作量**：中。DINOv2 已在 expert zoo 验证过，主要是特征缓存和 token 拼接。

**执行结果（2026-11-23）**：未达标。

| variant | overall | near-identical |
|---|---:|---:|
| base | 0.930 | 0.738 |
| dino（384 raw） | 0.903 | 0.658 |
| tactile | **0.989** | **0.956** |
| both（384 raw） | 0.967 | 0.876 |
| dino_pca32 | 0.942 | 0.804 |
| both_pca32 | 0.987 | 0.964 |

- 原始 CLS 高维噪声导致过拟合；PCA-32 后 DINO-only 到 0.942/0.804，仍低于 tactile-only。
- 验收线“无 tactile near ≥0.90”未达到。代码保留为 `dino=True` 可选路径，默认不启用。
- 下一步若要继续 DINO：提高渲染分辨率、用 patch 局部描述子、或把 DINO 余弦作为 pair edge feature；不要继续在 96×96 上调 token 维度。

---

### P0-2 建更难的 tracking benchmark v3，并做跨 benchmark 泛化

**要解决的问题**

v2 只有 60 个序列，容易被说“在自建集上调出来的”。需要证明结论不是 benchmark-specific。

**做法**

新增 `data/tracking_bench_v3`，相对 v2 增加：

- 20–50 个实体，包含更多同类近重复实例；
- 30+ 帧长遮挡，遮挡物和相机都可移动；
- 光照、相机高度、俯仰角变化；
- 随机缺失 tactile / vision / geometry；
- 干扰类别和背景物体；
- 更长的 dormant/reappearance 间隔。

评测分三种：

1. **v2 训练，v2 测试**：复现当前结果；
2. **v2 训练，v3 不重训直接测**：跨 benchmark 泛化；
3. **v3 训练，v3 测试**：在新分布上重新训练的上限。

**指标**

- ID retention、switches、reappearance；
- 按遮挡长度、实体数、缺失模态分桶；
- v2→v3 的掉点幅度。

**验收**

- PSS graph matcher 在 v3 上仍比 transformer 高 ≥5 点；
- 明确报告跨 benchmark 掉点，不隐藏；
- 缺失 modality 下 PSS 的退化曲线可解释。

**工作量**：中到大。生成器可复用 `pss/sim/generator.py` 和 `pss/sim/tracking_bench.py`。

**执行结果（2026-11-23）**：达标。

v3：5 objects、40 frames、30-frame occlusion、160×160、随机运动/机位/光照；train/val/test = 120/30/60。

| train -> test | PSS graph | transformer | delta |
|---|---:|---:|---:|
| v2 -> v2 | 0.930 ± 0.029 | 0.802 ± 0.008 | +0.128 |
| v2 -> v3 | **0.881 ± 0.029** | 0.672 ± 0.022 | **+0.209** |
| v3 -> v3 | 0.847 ± 0.040 | 0.711 ± 0.014 | +0.136 |

- PSS 掉 4.9 点，transformer 掉 13.0 点；跨 benchmark 系统级更稳。注意 transformer 基线无 tactile；公平归因后 re-ID 精度增量只有 +0.012。
- v3→v3 低于 v2→v3 是训练量/难度问题，不是 v3 无效。
- v3 缺 geometry 0.840、缺 vision 0.863；near-identical 缺 geometry 掉到 0.573。

---

### P0-3 补 graph matcher 的 scaling 和延迟

**要解决的问题**

P4 的 23.6x 测的是原型识别，不是 graph matcher。集合级匹配是 O(n²) pair score + Hungarian，实体一多可能吃掉 PSS 的计算优势。当前“计算随激活实体走”不能直接扩展到匹配层。

**做法**

- 先测当前 `GraphMatcherReID.cost_matrix` 在 N=50/200/1000 的 p50/p95；
- 加候选剪枝：
  - pose gate：超出运动门限的 track 不进入匹配；
  - category gate：类别不一致的直接排除；
  - appearance ANN：先检索 top-k，再做图匹配；
- 稀疏边：只保留候选边，避免全连接；
- 分块/批处理 Hungarian；
- 可选 `torch.compile` / ONNX。

**指标**

- 端到端 tracker 每帧延迟；
- 匹配层单独延迟；
- ID retention 是否保持；
- 与 dense transformer 在相同 retention 下的延迟比。

**验收**

- N=200 至少 5x，N=1000 至少 20x；
- retention 不低于 0.98；
- 报告剪枝前后的召回损失。

**工作量**：中。

**执行结果（2026-11-23）**：部分达标。

| N | pruned p50 | transformer p50 | speedup | true retention |
|---:|---:|---:|---:|---:|
| 50 | 2.63 ms | 4.94 ms | 1.88x | 1.000 |
| 200 | 10.37 ms | 36.66 ms | 3.53x | 0.970 |
| 1000 | 34.99 ms | 852.19 ms | **24.35x** | **0.984** |

- N=1000 达标（>20x）；N=200 3.53x，未达 5x。
- v2 上 dense ID retention 0.986，pruned 0.978，相对 retention 0.992 >0.98。
- 实现在 `pss/memory/graph_matcher.py` 的 `PrunedGraphMatcherReID`；单点分量合并成 B×1×1 batch，避免每个分量单独前向。
- 下一步若要追 5x：缓存 track/observation token、把候选生成移到 C 扩展，或对 N<200 直接走 dense。

---

### P0-4 加强 transformer 基线

**要解决的问题**

当前 transformer 是小型、从零训练的。真实审稿人会问：换预训练 VLM、sparse attention、replay/EWC 之后还成立吗？

**做法**

至少补四组：

1. **冻结 DINOv2/CLIP 特征 + transformer head**；
2. **sparse / retrieval / window attention**，对应 PSS 的 active-set；
3. **replay / EWC** 做增量更新，对应 P3 的零遗忘；
4. **energy + conformal** 做开放集，对应 P2 的拒识。

统一报告：

- 参数量、训练时间、推理延迟；
- 相同 seeds、相同 split；
- 相同输入模态和缺失模态条件。

**验收**

- 每个场景都有强基线结果；
- 如果 PSS 在某个强基线下输了，写进结论和边界，不回避。

**工作量**：大。

**执行结果（2026-11-23）**：部分完成。

- DINOv2 CLS + transformer：overall **0.752 ± 0.037**，低于 base transformer **0.802 ± 0.008**；near-identical 0.311 vs 0.387。预训练特征没有自动变成更强基线。
- Replay 增量基线：old **0.979**、new **1.000**、forgetting **-0.001**，追平 PSS 的零遗忘；P3 的 H3 forgetting 由 PASS 变 **FAIL**。
- PSS 仍保留 update time **24.1x**、无需旧数据、可回滚。
- sparse/retrieval attention、EWC、conformal 还没做。

---

### P0-5（新）换编码器，不动 PSS 接口（第一优先）

**要解决的问题**

`PSS-认知层架构定位.md` 把“冻结 CLIP/DINOv2、触觉换更强编码器、投影头做对齐/校准”列为第一优先。此前 DINOv2 直接替换 vision 后 re-ID 反而掉到 0.903，需要判断是编码器不行，还是投影/接入口不行。

**做法**

- 新增可替换编码器注册表：`smallcnn` / `dinov2` / `dinov2_vlad` / `clip`，触觉 `mlp` / `conv` / `wtmm`；
- DINOv2 支持 CLS、patch-mean、patch 局部描述子；
- 新增 VLAD 局部描述子聚合（K=16，仅用 train 拟合码本）；
- 投影头增加 PCA-whitening 前端和线性 residual skip；
- 增加按模态 temperature / reliability 的校准模块；
- 下游接口不变：Observation、128-d 公共空间、`entity_token`、代价矩阵都不改。

**执行结果（2026-11-24）**：部分达标。

| 变体 | 闭集 test | 开放集 AUROC | 校准 AUROC | re-ID overall | re-ID near |
|---|---:|---:|---:|---:|---:|
| base（SmallCNN+MLP） | 0.979 | 0.993 | 0.959 | **0.997 ± 0.007** | 0.987 |
| residual_only（仅加 residual） | 0.984 | **0.998** | 0.979 | 0.989 ± 0.012 | 0.956 |
| whiten_only（仅加 whitening） | 0.990 | 0.969 | 0.981 | 0.948 ± 0.029 | 0.791 |
| calibrated（whiten+residual） | **0.995** | 0.977 | 0.986 | 0.989 ± 0.010 | 0.956 |
| dinov2_plain（未校准） | 0.974 | 0.976 | 0.952 | 0.963 ± 0.035 | 0.880 |
| dinov2（CLS，已校准） | 0.969 | 0.897 | 0.950 | 0.993 ± 0.005 | 0.982 |
| dinov2_vlad（K=16） | 0.964 | 0.841 | 0.966 | **1.000 ± 0.000**（3 seeds） | **1.000** |
| conv_tactile | **0.995** | 0.973 | 0.976 | 0.991 ± 0.013 | **1.000** |

- **投影头是 DINOv2 re-ID 的主因，但不是识别的主因。** 同一个 DINOv2：plain head re-ID 0.963、near 0.880；whiten+residual head re-ID 0.993、near 0.982（+3.0 / +10.2 点；plain 的 seed 方差较大，±0.035，宜看作方向性证据）。但识别方向相反：plain 闭集 0.974 / 开放集 0.976，校准后 0.969 / 0.897。**校准是任务相关的，不是普遍提升**；re-ID 用校准头，原型识别用 plain 头。此前 0.903 是不同 token set（无 tactile）的结果，不能与本表直接比较。
- **编码器替换本身没有提升闭集/开放集。** DINOv2 plain 0.974/0.976、VLAD 0.964/0.841，均低于或接近 base 0.979/0.993；与 expert zoo 结论一致。
- **投影 ablation 是任务相关的 trade-off，没有一个头同时最优：**
  - residual-only 是“安全”的识别改动：闭集 0.984、开放集 **0.998**（本轮最高），但 re-ID 掉到 0.989；
  - whitening-only 闭集 0.990，但开放集 0.969、near-identical 0.791，明显伤 identity；
  - whitening+residual 闭集最高 **0.995**，开放集 0.977，仍低于 base 0.993。
  - 结论：识别用 residual/whiten 头，re-ID 保留 base 头；单一头无法三者兼顾。
- **conv 触觉提升 identity 而非类别识别。** 闭集 0.995（与 calibrated 同源，来自校准后的 vision 投影）；near-identical **1.000**（base 0.987），overall 0.991（base 0.997）。合成触觉的类别信号本来就弱，identity 信号已接近线性可分。
- **P5b 对编码器完全不变**：所有变体都是 PSS 1.000/1.000、transformer 1.000/0.400。但这是**构造性不变**：P5b 的感知路径不加载原型库（GT 类别）、tracker 的 `w_category=0`、规划状态只用位置；编码器根本不进入 P5b 状态。因此本轮不能把 P5b 当作编码器证据；要测“感知→关系”必须让目标依赖感知类别/身份，而不只是位置。
- CLIP 已实现并接入 registry，但本地缓存只有 config、没有 1.7 GB 权重，本轮显式跳过并记录原因。

**产物**

- 代码：`pss/encoders/{registry.py,vision/clip.py,vision/local.py,tactile/conv.py}`、`pss/projection/{heads.py,calibration.py}`、`pss/eval/scripts/run_encoder_priority.py`；
- 结果：`eval/results/encoder_priority/encoder_priority.{json,md}`；
- 报告：`PHASE11_ENCODER_PRIORITY_REPORT.md`；
- 复现：`make encoder-priority`。

**验收判断**

- ✅ “不动 PSS 接口换编码器”已可执行；
- ✅ DINOv2 经校准后不再掉点，re-ID 0.993 接近 base 0.997；
- ✅ residual-only 把开放集从 0.993 提到 **0.998**；
- ⚠️ P5b 全变体不变，但这是位置型 benchmark 的构造性不变，不能当作编码器或“关系层优于感知”的新证据；
- ❌ “更强编码器直接带来更高闭集/开放集”不成立；
- ❌ 没有单一投影头同时最优；校准开放集未超过 base bank（0.959 vs 0.993），不能声称校准提升开放集。

---

### P0-6（新）把关系层变成可学习（第二优先）

**要解决的问题**

`PSS-认知层架构定位.md` 的第二优先要求：object attention + relation labels、typed heads、relation logits 进 scene graph、不再手写边特征。此前的 P1-15 只做了带 typed bias 的 object attention，没有关系监督，也没有把关系写回图。

**做法**

- 新增 `pss/memory/typed_relations.py`：spatial / semantic / tactile 成对关系头，temporal / task 的 schema 和上下文接口；
- 关系标签从仿真 GT 生成，不改 PSS 接口；
- `SceneGraph` 增加 `relations`、`set_relation`、`get_relations`，支持保存/加载；
- `write_relation_edges` 把关系 logits 写到活跃 track 之间；
- 新增 pure learned edge 和 hybrid（手写边 + 学习边）两种 matcher；
- `relational_bench.goal_ok/plan` 支持传入学习到的关系谓词，P5b 可直接用学习关系跑。

**执行结果（2026-11-24）**：部分达标。

**关系预测 F1（test）**

| 条件 | spatial | semantic | tactile | mean |
|---|---:|---:|---:|---:|
| clean | 0.957 | 0.996 | 0.970 | 0.974 |
| pose noise | 0.953 | 0.994 | 0.970 | 0.973 |
| no tactile | 0.954 | 0.976 | **0.483** | 0.804 |
| no vision | 0.957 | 0.977 | 0.955 | 0.963 |
| mixed | 0.952 | 0.982 | 0.911 | 0.949 |

**re-ID（3 seeds）**

| matcher | clean | near | 缺 tactile | pose noise | 缺 vision | mixed |
|---|---:|---:|---:|---:|---:|---:|
| hand（现有） | **1.000 ± 0.000** | **1.000** | 0.791 | **1.000** | **0.996** | **0.946** |
| pure learned | 0.985 ± 0.005 | 0.985 | 0.696 | 0.989 | 0.937 | 0.889 |
| hybrid | 0.996 ± 0.005 | **1.000** | **0.819** | 0.996 | 0.989 | 0.944 |

**P5b（3 seeds）**

| planner | ID | OOD |
|---|---:|---:|
| hand `_rel` | 1.000 | **1.000** |
| learned，in-distribution | 1.000 | **1.000** |
| learned，held-out regions | 0.842 | 0.688 |
| transformer BC（ref） | 1.000 | 0.462 |

- **typed 关系是可学的**：clean 下 spatial 0.957 / semantic 0.996 / tactile 0.970；只有 tactile 在去掉 tactile 时崩（0.483），说明没有走捷径。
- **纯学习边没有超过手写边**：clean re-ID 0.985 vs 1.000；手写的距离/类别/相对位姿仍是强先验。
- **hybrid 是可用形态**：保留手写边 + 学习关系 logits，缺 tactile 时 0.791 → **0.819**，near 保持 **1.000**，其余持平。
- **relation logits 已进 scene graph**：3 物体测试帧写入 18 条边（12 spatial、6 tactile），save/load 完整保留。
- **P5b 在分布内可完全用学习关系**（1.000/1.000）；但 held-out 区域布局掉到 OOD 0.688，仍低于手写几何的 1.000。学习关系比端到端 transformer 泛化得好（0.688 vs 0.462），但不如精确几何谓词。
- **对齐区域需要 dead zone**：`|delta|<0.1` 时四个方向都不触发；否则 C/D、D/G 这类同轴区域会被迫选边，P5b 直接掉到 0。这是标签校准，不是模型 trick。
- **temporal / task**：task 已通过 P5b 评估；temporal 需要 track history，本轮只落 schema 和写回路径，未训练，不声称结果。

**产物**

- 代码：`pss/memory/typed_relations.py`、`pss/memory/scene_graph/__init__.py`、`pss/eval/relational_bench.py`、`pss/eval/scripts/run_typed_relations.py`；
- 结果：`eval/results/typed_relations/typed_relations.{json,md}`、`eval/results/typed_relations/scene_graph.json`；
- 报告：`PHASE12_TYPED_RELATIONS_REPORT.md`；
- 复现：`make typed-relations`。

**验收判断**

- ✅ typed heads 可训练、关系 F1 高；
- ✅ relation logits 可写入/读回 scene graph；
- ✅ hybrid 在缺 tactile 时优于手写边；
- ✅ P5b 分布内可用学习关系替代手写谓词；
- ❌ 纯学习边没有超过手写边，不能声称“去掉手写边更好”；
- ❌ held-out 区域布局下学习关系仍明显弱于精确几何，不能声称“学习关系已具备组合泛化”；
- ⚠️ temporal 未训练，task 只在 P5b 评估。

---

### P0-7（新）学习 active-set / write policy（第三优先）

**要解决的问题**

`PSS-认知层架构定位.md` 的第三优先要求：学“哪些对象/事实值得注意”、学“什么值得写入记忆”、不确定时回退到更大上下文。此前 active-set 是手写调度器/精确查找，write policy 是规则阈值。

**做法**

- 新增 `pss/memory/active_policy.py`：`ActiveSetPolicy`（listwise 训练）和 `WritePolicy`（write/review/skip）；
- active-set 用“每个 observation 在候选 track 上做 softmax”的 listwise loss，而不是 pairwise BCE；
- margin threshold 在 validation 上按目标 fallback rate 校准；
- 写入策略支持 review band：不确定的 fact 不直接写入；
- `pss/eval/scripts/run_active_policy.py` 跑 tracking / write / retrieval 三组实验。

**执行结果（2026-11-24）**：部分达标。

**A. re-ID 候选 active-set（tracking_bench_v2）**

| 设置 | 无 fallback | 有 fallback | 平均 active | fallback |
|---|---:|---:|---:|---:|
| k=1 clean | 0.972 | **1.000** | 1.20 | 0.10 |
| k=1 pose noise | 0.961 | **0.978** | 1.21 | 0.11 |
| k=1 缺 tactile | 0.689 | 0.689 | 1.60 | 0.30 |
| k=2 clean | 1.000 | 1.000 | 2.09 | 0.09 |

- full candidate set + policy score 1.000；tactile top-1 0.917；pose top-1 0.344。
- **listwise 是正确目标**：同一特征的 pairwise BCE 版本 full-score 只有 0.756，listwise 到 0.972。
- **校准 fallback 有效**：k=1 + 10% fallback 追平 full（1.000），平均只查 1.2 个候选；pose noise 下 0.961→0.978。
- **fallback 不能补信息缺失**：缺 tactile 时 full 和 active 都 0.689，说明是信息没了，不是没选中。

**B. fact memory write policy（合成流）**

| policy | accuracy | writes |
|---|---:|---:|
| write all | 0.700 | 1500 |
| rule `rel>0.7` | 0.880 | 716 |
| learned `p>=0.5` | 0.880 | 1083 |
| learned + review band | **0.920** | 1051 |

- 在相同写入量（~1.1k）下，learned 0.88 vs rule 0.84；
- review band（0.4–0.6 不写）再 +4 点，靠弃权避免写入错误事实；
- 低写入预算下 rule 更好，learned 的优势在高写入/高精度区间，不是全面胜出。

**C. 长历史 fact active-set（stress test）**

| policy | accuracy |
|---|---:|
| full | 1.000 |
| hot heuristic | 0.480 |
| learned | 0.372 |
| learned + diversity | 0.428 |
| learned + diversity + fallback | 0.680 |

- 学习策略没有超过手写 hot heuristic；两个原因：(1) 本阶段的 fact 打分**没有 query 条件**，只学了全局 retention prior（recency/hot/reliability），hot 就是同一个先验的手写版；(2) 缺少覆盖率训练，逐条打分会给同一个 hot key 选多条，小 active set 覆盖不到查询 key。
- fallback 到 64 条能恢复到 0.680，说明不确定性门控有效，但仍低于 full。**这是明确负结果**；下一步需要 query-conditioned scoring + set-level coverage/diversity loss。

**产物**

- 代码：`pss/memory/active_policy.py`、`pss/eval/scripts/run_active_policy.py`；
- 结果：`eval/results/active_policy/active_policy.{json,md}`；
- 报告：`PHASE13_ACTIVE_WRITE_POLICY_REPORT.md`；
- 复现：`make active-policy`。

**验收判断**

- ✅ learned active-set 在 re-ID 候选上可用，k=1+fallback 追平 full；
- ✅ 不确定性回退可用，且边界清楚（不能补信息缺失）；
- ✅ write policy 在匹配写入量下优于规则，review band 有效；
- ❌ 长历史稀疏 fact active-set 未超过 hot heuristic，覆盖率训练缺失；
- ⚠️ 尚未接入 `Tracker` / `WorkingMemory` / `FactStore` 的默认路径，目前只通过实验脚本验证。

---

### P0-8（新）输出头升级：结构化状态 + 工具 + LLM decoder（第四优先）

**要解决的问题**

`PSS-认知层架构定位.md` 的第四优先要求：动作/行为头保留；语言输出接 LLM decoder；PSS 提供结构化状态、工具、审计，而不是直接生成语言。此前 `TextHead` 只是模板渲染，没有工具调用、版本引用和审计。

**做法**

- 新增 `PSSState`：objects / typed relations / facts / audit / memory / uncertainty；
- 新增 `ToolRegistry`：`get_objects`、`get_relations`、`get_facts`、`list_audit`、`get_uncertainty`、`update_fact`、`rollback_fact`；
- 写工具默认需要显式 `allow_write=True`，返回 version；
- `LLMLanguageHead` 支持任意 `decoder(prompt)->text`，解析 tool call JSON，循环执行并收集 citations；
- `verify_citations` 校验事实/版本；`verify_answer_entities` 检测生成的虚构实体；
- `PSSState.to_context` 做上限截断，超出部分通过工具查询；
- 动作头 `ActionHead` 不变。

**执行结果（2026-11-24）**：达标（有边界）。

| agent | accuracy | tool calls | citation coverage | unsupported entities |
|---|---:|---:|---:|---:|
| no context | 0.000 | 0 | 0.000 | 0 |
| flat JSON dump，无工具 | 0.714 | 0 | 0.000 | 0 |
| structured state + tools | **1.000** | 8 | **1.000** | 0 |

- 7 条查询：flat 只能答静态 5 条；**update material / rollback + audit 两条只有工具路径能完成**。
- 语言头接的是结构化状态，不是张量→文本的 MLP；动作头保持不变。
- 工具返回 source/version/citation；写工具默认拒绝，显式授权后才写，并返回 version。
- 脚本化 decoder 验证了 tool loop：1 轮工具调用，citation coverage 1.000。
- 虚构实体 `obj_9999` 被 `verify_answer_entities` 抓到（unsupported）。
- 上下文：小 episode structured 564 vs flat 511 tokens（无可压缩空间）；60 物体/200 关系状态 **3439 vs 12666 tokens（0.27x）**。
- 边界：本地没有缓存 LLM；`ToolAgent` 和脚本化 decoder 是 harness 替身，不是语言能力结果。真实 decoder 接口已实现但未接模型。

**产物**

- 代码：`pss/heads/text/structured.py`、`pss/eval/scripts/run_language_head.py`；
- 结果：`eval/results/language_head/language_head.{json,md}`；
- 报告：`PHASE14_LANGUAGE_HEAD_REPORT.md`；
- 复现：`make language-head`。

**验收判断**

- ✅ 语言输出走 LLM decoder 适配器，PSS 提供结构化状态 + 工具 + 审计；
- ✅ 写操作有权限门、版本和回滚；
- ✅ 引用可验证，虚构实体可检测；
- ✅ 动作/行为头保留且未改动；
- ⚠️ 未接真实 LLM，当前准确率是 harness 结果，不是生成质量；
- ⚠️ 基准只有 7 条查询、1 个 episode，需扩展。

---

### P0-9（新）端到端 / bilevel 训练（第五优先）

**要解决的问题**

`PSS-认知层架构定位.md` 的第五优先要求：编码器冻结，训练投影 + 绑定 + 关系；任务损失 + 一致性/审计约束；否则 PSS 永远只是 harness。此前投影、关系头、matcher 都是分开训练的，没有联合目标。

**做法**

- 新增 `pss/learning/bilevel.py`：`EndToEndPSS`（projections + binder + typed relation head + pair matcher）；
- 冻结编码器，缓存 raw expert features，只训练公共空间/绑定/关系/匹配；
- 识别视图屏蔽 category one-hot（防标签泄漏），匹配/关系视图保留 category（与现有 token 一致）；
- 任务损失用 episodic prototype（support mean 是可微 inner memory，query loss 是 outer task）；
- 加 re-ID assignment loss、relation BCE（带正样本权重）、一致性/审计项：spatial 反对称、互斥、matcher 置换不变、输入扰动稳定；
- 对比 joint / sequential / no_consistency / no_relation / linear_head / concat / joint_instance / reid_heavy。

**执行结果（2026-11-24）**：部分达标。

| mode | rec | re-ID | near | spatial F1 | semantic F1 | tactile F1 |
|---|---:|---:|---:|---:|---:|---:|
| joint | 0.979 | 0.889 | 0.600 | 0.371 | 0.779 | 0.779 |
| joint_instance | 0.896 | **0.922** | 0.733 | 0.439 | 0.789 | 0.782 |
| reid_heavy | 0.948 | 0.828 | 0.733 | 0.276 | 0.676 | 0.772 |
| sequential | 0.984 | 0.867 | 0.467 | 0.401 | 0.667 | 0.782 |
| no_consistency | 0.969 | 0.889 | 0.600 | **0.663** | **0.856** | 0.779 |
| no_relation | 0.943 | 0.889 | 0.600 | 0.025 | 0.667 | 0.782 |
| linear_head | **0.995** | 0.900 | 0.733 | 0.285 | 0.695 | 0.782 |
| concat | 0.979 | 0.861 | 0.578 | 0.329 | 0.922 | 0.740 |

一致性/审计指标（越低越好）：

| mode | antisymmetry | exclusivity | permutation | stability |
|---|---:|---:|---:|---:|
| joint | 0.005 | 0.000 | 0.0000 | 0.0003 |
| no_consistency | **0.054** | **0.277** | 0.0000 | 0.0004 |

- **联合训练比分期训练好，但幅度不大**：joint re-ID 0.889 vs sequential 0.867、concat 0.861，rec 基本持平。
- **任务选择是 trade-off**：joint_instance re-ID 最高 **0.922**，但 rec 掉到 0.896；linear_head rec 最高 **0.995**，关系 F1 低。
- **一致性/审计约束有效但有代价**：antisymmetry 0.054→0.005、exclusivity 0.277→0.000，但 spatial F1 0.663→0.371；这是明确的 audit-vs-accuracy trade-off。
- **联合关系 F1 远低于单独关系头**（0.957/0.996/0.970）：联合 binder 被类别/re-ID 目标主导，不能替代专门的关系 token + 监督。这是本阶段最强的负结果。
- **绝对 re-ID 仍低于成熟 pipeline**（0.989–0.997）：joint 用的是简单 pairwise MLP，不是 graph matcher 的边消息；差距是架构 + 多任务组成，不是联合训练本身无效。
- **识别视图必须屏蔽 category one-hot**：不屏蔽时 rec 直接 1.000，是标签泄漏；这是重要方法学修正。
- 置换一致性最初因 inverse-permutation 目标写反，报告了虚假的 3.3 MSE；修正后所有 mode 为 0。

**产物**

- 代码：`pss/learning/bilevel.py`、`pss/eval/scripts/run_bilevel.py`；
- 结果：`eval/results/bilevel/bilevel.{json,md}`、`checkpoints/bilevel/joint_*.pt`；
- 报告：`PHASE15_BILEVEL_REPORT.md`；
- 复现：`make bilevel`。

**验收判断**

- ✅ 编码器冻结，投影/绑定/关系/匹配可联合训练；
- ✅ 任务 + re-ID + 关系 + 一致性/审计约束都在同一图里；
- ✅ joint 优于 sequential/concat（re-ID +2 点左右）；
- ✅ 一致性约束显著降低逻辑冲突率；
- ❌ 联合关系 F1 远低于单独关系头，不能声称联合训练已替代分模块；
- ❌ 绝对 re-ID 仍低于成熟 pipeline，不能声称联合训练带来 SOTA；
- ⚠️ 1 seed、60 epochs、简单 matcher；多任务权重未调优。

---

### P0-10（新）PSS 认知运行时：分层落地

**要解决的问题**

五个优先级各自有了模块，但按 `PSS-认知层架构定位.md` 的分层，它们还没有收敛成一个**拥有状态语义与状态转换**的 runtime：实体层、关系/事实层、不确定性、规划、输出、审计仍是分散的。本阶段把它们合成一个 typed state + services + 跨层闭环。

**做法**

- 新增 `pss/runtime/`：
  - `CognitiveState`：entities / relations / facts / goals / plans / uncertainty / audit，带 version / snapshot / rollback；
  - `EntityLayer`：persistent IDs、object slots、attributes、confidence、provenance、lifecycle；
  - `RelationFactLayer`：symbolic + learned relations、多源 facts、冲突、规则派生；
  - `UncertaintyService`：object confidence、fact/relation 冲突、缺失模态、accept/review/abstain；
  - `RuntimePlanner`：goal + A* + plan validation；
  - `OutputRouter`：语言工具/citations、safety veto；
  - `PSSRuntime`：ingest / assert_fact / set_goal / plan / act / explain / rollback。
- `pss/eval/scripts/run_runtime.py`：桌面任务闭环，含遮挡、冲突事实、坏写入回滚、语言解释，并做逐层消融。

**执行结果（2026-11-24）**：达标（有边界）。

| variant | entity persistence | plan | task success | conflict flagged | recommendation | rollback | citations |
|---|---:|---:|---:|---:|---|---:|---:|
| **full** | **yes** | **yes (2)** | **yes** | **yes** | **review** | **yes** | **1** |
| no_entity | no | no (0) | no | no | accept | yes | 0 |
| no_relations | yes | yes (2) | yes | yes | review | yes | 1 |
| no_uncertainty | yes | yes (2) | yes | **no** | disabled | yes | 1 |
| no_audit | yes | yes (2) | yes | yes | review | **no** | 1 |
| no_language | yes | yes (2) | yes | yes | review | yes | **0** |

- **逐层消融正好打在各自层上**：无实体层→遮挡目标丢失、无法规划；无不确定性→冲突不报；无审计→坏写入不可回滚；无语言→无 grounded explanation。
- **no_relations 仍能完成位置目标**（facts 足够），但 relation 数 12→0；这是边界：当前任务不需要关系，关系型目标才需要。
- 运行时用的是现有 SmallCNN/投影、tracker、A*、工具语言头；变成 runtime 的原因是 **状态是 typed/persistent/versioned/auditable**，不是换了模型。
- 仍然手写：`move_*` 算子、symbolic 关系、工具语言头、active-set/write 默认接线、固定不确定性阈值。

**产物**

- 代码：`pss/runtime/{state,layers,services,runtime}.py`、`pss/eval/scripts/run_runtime.py`；
- 结果：`eval/results/runtime/runtime.{json,md}`；
- 报告：`PHASE16_RUNTIME_REPORT.md`；
- 复现：`python -m pss.eval.scripts.run_runtime`。

**验收判断**

- ✅ 分层架构落成了 typed state + services + 输出路由；
- ✅ 每个消融在对应层失效，边界清楚；
- ✅ version/snapshot/rollback、冲突 review、工具 citations 可用；
- ❌ 规划算子、关系、语言、active-set/write 默认路径仍有手写部分；
- ❌ 单 episode、合成数据、运动学执行，不是真实闭环。

---

### P0-11（新）Phase 17：把学习模块接进运行时

**要解决的问题**

Phase 16 建成了 runtime 骨架，但 learned relations、write policy、decoder 仍是旁路；逻辑层没有版本化；长遮挡没有在 runtime 内验证。

**做法**

- `Goal.relation` + `RuntimePlanner.plan_relational`：符号或学习关系谓词都能驱动规划；
- `PSSRuntime.assert_fact(..., gate=..., gate_features=...)`：learned `WritePolicy` 在 runtime 边界决定 write/review/skip；
- 新增 `VersionedOperatorLibrary`：算子 update/remove/rollback，`Plan.operator_version` 记录版本；
- `OutputRouter(decoder=...)` + `load_local_decoder()`：真实 decoder 接口，缓存不可用时明确记录；
- `run_runtime_phase17.py`：关系目标、写门控、算子版本、12 条长遮挡序列。

**执行结果（2026-11-24）**：达标（有边界）。

**关系目标（4 个谓词）**

| predicate | symbolic plan/success | learned plan/success |
|---|---|---|
| left | 2 / yes | 2 / yes |
| right | 0 / yes | 0 / yes |
| above | 1 / yes | 1 / yes |
| below | 0 / yes | 0 / yes |

- 区域关系头训练分布内 all-4 exact **0.988**；分布内符号/学习计划一致。held-out 泛化仍是 Phase 12 的负结果，本轮不重测。

**写门控**

| policy | accuracy | facts | rejected |
|---|---:|---:|---:|
| write all | 0.750 | 20 | 0 |
| learned gate | **0.850** | 20 | 77 |

**算子版本**

| version | plan | operator version |
|---|---|---:|
| v1 | `[move_cup_A_G]` | 1 |
| v2（移除直连） | `[move_cup_A_B, move_cup_B_G]` | 2 |
| v3（回滚） | `[move_cup_A_G]` | 1 |

**长遮挡（tracking_bench_long，12 条，70 帧）**

- ID retention **1.000**；entity persistence **1.000**；平均 3 实体/3 active（clean pose）。
- 说明 runtime 用的是持久实体层，不是帧内缓存。

**真实 decoder**

- 本地无文本 LLM：`decoder_available=false`；接口已接、脚本化 decoder 已有测试，不声称语言质量。

- **结论**：learned relations 可驱动 planner；write policy 在 runtime 边界提升事实准确率；逻辑层可版本化/回滚；长遮挡在 runtime 内保持身份。仍然手写：`move_*` 算子、关系头的训练分布、active-set 默认接线。

**产物**

- 代码：`pss/runtime/operators.py`、`pss/runtime/{state,services,runtime}.py`、`pss/heads/text/structured.py`、`pss/eval/scripts/run_runtime_phase17.py`；
- 结果：`eval/results/runtime_phase17/runtime_phase17.{json,md}`；
- 报告：`PHASE17_RUNTIME_INTEGRATION_REPORT.md`；
- 复现：`python -m pss.eval.scripts.run_runtime_phase17 --device cpu --sequences 12`。

**验收判断**

- ✅ learned relations 进入 runtime planner；
- ✅ learned write policy 进入 runtime 写路径，带 audit；
- ✅ 算子可版本化/回滚，plan 记录 operator version；
- ✅ 70 帧遮挡在 runtime 内保持身份；
- ❌ decoder 仍不可用；
- ❌ 算子仍手写，active-set 默认接线未完成。

---

### P0-12（新）Phase 18：学习模块设为 runtime 默认

**要解决的问题**

Phase 17 把学习模块接进了 runtime，但默认路径仍可能走手写 fallback；需要把 learned relations、active-set、write policy 设为默认，并保留手写规则作为可切换的 fallback，验证默认确实改变行为。

**做法**

- `EntityLayer` 默认加载 `relation_head.pt` 和 `active_policy.pt`；`RelationFactLayer` 默认加载 `write_policy.pt`；
- 构造函数提供 `use_default_relations` / `use_default_active` / `use_default_write` 开关；
- `components()` 审计当前走的是 learned 还是 fallback；
- 新增 `run_runtime_defaults.py`：defaults + 三个单点消融 + symbolic_only，带低可靠 noisy fact 和高置信 distractor track。

**执行结果（2026-11-24）**：达标（有边界）。

| variant | learned relations | rejected facts | noisy rejected | slots | total relations | task |
|---|---:|---:|---|---:|---:|---:|
| **defaults** | **15** | **1** | **yes** | `obj1, obj2` | 27 | yes |
| no_learned_relations | 0 | 1 | yes | `obj1, obj2` | 12 | yes |
| no_active_policy | 15 | 1 | yes | **`distractor, obj1`** | 27 | yes |
| no_write_policy | 15 | 0 | **no** | `obj1, obj2` | 27 | yes |
| symbolic_only | 0 | 0 | no | `distractor, obj1` | 12 | yes |

- **learned relations 默认生效**：27 vs 12 total；任务/规划/回滚/citation 契约不变。
- **write policy 默认生效**：低可靠 noisy fact 被拒（1 条），关掉后写入。
- **active policy 默认生效**：排除高置信但无观测支持的 distractor；confidence top-k fallback 会选中它。
- **defaults 可逆**：三个开关都可单独关闭，`components()` 记录当前路径；不是死代码 fallback。
- **边界**：write policy 的 query-frequency 未知时传 0.0（保守，可能拒掉稀有但正确的 fact）；active policy 是从 pair matching 适配到 slot 选择，不是专门训练的 slot policy。

**产物**

- 代码：`pss/runtime/layers.py`、`pss/runtime/runtime.py`、`pss/eval/scripts/run_runtime_defaults.py`；
- 结果：`eval/results/runtime_defaults/runtime_defaults.{json,md}`；
- 报告：`PHASE18_LEARNED_DEFAULTS_REPORT.md`；
- 复现：`python -m pss.eval.scripts.run_runtime_defaults`。

**验收判断**

- ✅ 三个 learned 模块默认生效，行为可观测；
- ✅ fallback 可切换、可审计；
- ✅ 任务契约（plan/conflict/rollback/citations）不因默认切换而改变；
- ⚠️ active policy 是适配，不是专用 slot policy；
- ⚠️ write frequency 默认 0.0 需要应用层校准。

---

### P0-13（新）Gap closure 1–2：编码器 / 投影默认

**要解决的问题**

架构表 row 1（编码器）仍写 SmallCNN，row 2（投影）仍写简单 MLP。需要把 pretrained encoder + 校准投影设为 runtime 默认，并保留 fallback。

**做法**

- 新增 `pss/runtime/perception.py`：`PerceptionService` 默认 frozen DINOv2 + tactile MLP，配套 `encoder_priority/dinov2/projection`；
- SmallCNN + `phase45_projection` 作为 fallback，一个 config 开关；
- `PSSRuntime.ingest_sample` 走 perception service；`summary()` 报告 encoder/fallback/calibration/reliability；
- `Observation` 增加 `modality_reliability`；`extract_frame` 从 prototype metrics 填入；
- `UncertaintyService` 对 `<0.5` 的 modality reliability 标记 `low_modality_reliability` 并建议 review。

**执行结果（2026-11-24）**：row 1 部分；row 2 部分（DINOv2 默认路径已校准；共享头、开放集、SmallCNN fallback、CLIP/VLM 投影未解决）。

| variant | vision | image size | fallback | p50 | ID retention |
|---|---|---|---:|---:|---:|
| **dinov2** | DINOv2 | 224 | no | 16.7 ms | 1.000 |
| smallcnn | SmallCNN | 64 | yes | 4.7 ms | 1.000 |

- DINOv2 默认路径生效，代价是 **~3.6x observation latency**；短序列 retention 两者都是 1.000。
- 精度 trade-off 仍是 Phase 11：DINOv2 re-ID 0.993 vs base 0.997，但闭集 0.969 vs 0.979、开放集 0.897 vs 0.993。**row 1 关闭的是架构默认，不是精度差距。**
- DINOv2 投影校准记录：`whiten_dim=96`（vision+geometry）+ `residual=true`；`summary()["perception"]["calibration"]` 可见。
- reliability：DINOv2 `vision 1.0 / tactile 0.0 / geometry 0.5`；tactile 低可靠会触发 runtime review。
- CLIP 代码就绪但本地无权重；VLM 未接。

**产物**

- 代码：`pss/runtime/perception.py`、`pss/runtime/runtime.py`、`pss/memory/perception.py`、`pss/runtime/services.py`；
- 结果：`eval/results/runtime_encoder/runtime_encoder.{json,md}`；
- 报告：`PHASE19_GAP_CLOSURE_ENCODER_PROJECTION.md`；
- 复现：`python -m pss.eval.scripts.run_runtime_encoder --device cuda --sequences 5`。

**验收判断**

- ✅ runtime 默认使用 frozen DINOv2，SmallCNN 可一键 fallback；
- ✅ 投影默认带 whitening+residual，calibration 可审计；
- ✅ modality reliability 进入 runtime uncertainty；
- ⚠️ 精度未提升，且 observation latency ~3.6x；
- ❌ CLIP/VLM 未接（无权重）。

---

### P0-14（新）Gap closure 3：object attention 接成 runtime 绑定选项

**要解决的问题**

架构表 row 3 目标是 object attention / persistent slots；persistent slots 已在 `EntityLayer`，但 object attention 仍是离线实验，没有进入 runtime 的绑定/关联路径。

**做法**

- 新增 `ObjectAttentionReID`：attention 分数 → tracker cost matrix，接口与 GraphMatcherReID 一致；
- `save/load` 保存 token_dim / d_model / heads / layers / include_tactile；
- `EntityLayer(associator=...)` 构造 `TrackerConfig(metric=...)`；`PSSRuntime(associator=...)` 透传；
- `components()` 报告 `associator`；hand / graph / object attention 仍可切换。

**执行结果（2026-11-24）**：部分达标（有边界）。

| method | pair overall | pair near |
|---|---:|---:|
| graph matcher | 1.000 | 1.000 |
| object attention | 1.000 | 1.000 |

| dataset | associator | ID retention | ms/frame |
|---|---|---:|---:|
| v2 | hand | 1.000 | 3.4 |
| v2 | graph matcher | 1.000 | 2.3 |
| v2 | object attention | 1.000 | 2.5 |
| long | hand / graph / object | 1.000 | 0.7 / 0.7 / 0.8 |

- **object attention 已是 runtime 一等绑定选项**，不再是离线实验；接口与 graph matcher 对齐。
- **benchmark 饱和**：one seed 下 graph 和 object attention 都 1.000；P1-15 五 seed 是 0.987 vs 0.989（near 0.956 vs 0.956）——打平，不是胜出。
- **persistent slots 早已具备**：track ID、生命周期、属性、confidence、provenance 都在 `EntityLayer`；本轮补的是 attention 关联路径。
- **边界**：单 seed、饱和 benchmark；runtime 默认仍是 hand，object attention 是 opt-in；没有额外关系监督。

**产物**

- 代码：`pss/memory/object_attention.py`、`pss/runtime/layers.py`、`pss/runtime/runtime.py`、`pss/eval/scripts/run_runtime_binding.py`；
- 结果：`eval/results/runtime_binding/runtime_binding.{json,md}`；
- 报告：`PHASE20_GAP_CLOSURE_BINDING.md`；
- 复现：`python -m pss.eval.scripts.run_runtime_binding --device cuda --sequences 8`。

**验收判断**

- ✅ object attention 可被 EntityLayer/runtime 作为 associator 使用；
- ✅ hand / graph / object attention 可切换，components 可审计；
- ⚠️ benchmark 饱和，不能声称 object attention 更强；
- ⚠️ runtime 默认仍是 hand，未把 object attention 设为默认。

---

### P0-15（新）Gap closure 4：可学习、可版本化的关系规则

**要解决的问题**

架构表 row 4 目标是 typed object attention + 学习规则；typed heads 已是 runtime 默认，但 `near/contact/in` 的几何阈值仍手写（`near_dist=0.12`、`contact_pad=0.01`、`bin_half=0.06`、`bin_z=0.14`）。

**做法**

- 新增 `pss/reasoning/relations/rules.py`：`RelationRuleSet` 保存四个阈值，支持 `fit`（按谓词网格搜索 F1）、`update`（版本+1）、`rollback`；
- `RelationFactLayer.add_geometry(instances)` 用 rule set 生成 records；
- 规则保存到 `checkpoints/relation_rules/rules.json`；`components.relation_rules` 报告 version/source/thresholds；
- 手写阈值保留为默认/fallback。

**执行结果（2026-11-24）**：部分达标（有边界）。

| threshold | hand | learned |
|---|---:|---:|
| near_dist | 0.120 | 0.120 |
| contact_pad | 0.010 | 0.072 |
| bin_half | 0.060 | 0.030 |
| bin_z | 0.140 | 0.140 |

| predicate | hand F1 | learned F1 |
|---|---:|---:|
| in (support 16) | 1.000 | 1.000 |
| contact (support 18) | 0.000 | **0.267** |
| near (support 0) | n/a | n/a |

- 学到的 `contact_pad` 把 contact F1 从 0 提到 **0.267**，`in` 保持 1.000。
- `near` 在该 split 没有 GT，两个规则都不预测，属于 n/a，不是胜负。
- 规则可版本化/回滚：edit 后 version+1，snapshot 恢复；runtime component 记录阈值和版本。
- **边界**：contact 的 GT 来自 collision geometry，规则用 bounding sphere，特征限制了 F1 上限；typed heads 仍是更强的学习关系通道（Phase 12：0.957/0.996/0.970）。

**产物**

- 代码：`pss/reasoning/relations/rules.py`、`pss/runtime/layers.py`、`pss/eval/scripts/run_runtime_rules.py`；
- 结果：`eval/results/runtime_rules/runtime_rules.{json,md}`、`checkpoints/relation_rules/rules.json`；
- 报告：`PHASE21_GAP_CLOSURE_RULES.md`；
- 复现：`python -m pss.eval.scripts.run_runtime_rules`。

**验收判断**

- ✅ 关系阈值可学习、可版本化、可回滚；
- ✅ runtime 通过 `RelationFactLayer` 消费 learned rule set，components 可审计；
- ✅ contact F1 0→0.267，in 保持；
- ⚠️ near 无 GT，无法评估；
- ⚠️ contact 特征（bounding sphere）限制 F1，不能声称规则层已解决接触关系。

---

### P0-16（新）Gap closure 5：学习 heuristic + A* 搜索

**要解决的问题**

架构表 row 5 目标是“学习 + 搜索”；当前是手写 A*/BFS，heuristic 是 `len(goal - state)`，不理解 move cost。

**做法**

- `AStarPlanner` 增加 `heuristic(state, goal)` 和 `heuristic_weight`；
- 新增 `pss/reasoning/planner/learned.py`：`LearnedHeuristic`（MLP），输入每个实体的 current/goal region one-hot；
- 用随机 state/goal + 距离代价算子跑最优 A* 生成 cost-to-go 标签，训练 MLP；
- `RuntimePlanner(heuristic=...)` 接入；`summary()["planning"]["heuristic"]` 可审计；
- 手写 heuristic 保留为默认/fallback；checkpoint `checkpoints/planner/learned_heuristic.pt`。

**执行结果（2026-11-24）**：达标（有边界）。

| planner | success | mean cost | expansions | latency | suboptimality |
|---|---:|---:|---:|---:|---:|
| hand heuristic | 1.000 | 2.114 | 6.29 | 0.13 ms | 0.000 |
| learned heuristic | 1.000 | 2.114 | **3.28** | 3.32 ms | ~0 |
| weighted learned (1.5) | 1.000 | 2.114 | **2.48** | 2.39 ms | ~0 |

- learned heuristic **expansion 减半**（6.29→3.28），weighted 到 2.48，cost/success 不变。
- **小域上 wall-clock 更慢**（0.13→3.3 ms）：MLP heuristic 评估主导；expansion 优势要在大分支/大状态空间才体现。
- runtime 成功使用 learned heuristic，`planning.heuristic=learned`。
- **边界**：3 实体/5 区域（125 状态）、距离代价、单 seed；不声称跨域泛化或速度提升。

**产物**

- 代码：`pss/reasoning/planner/__init__.py`、`pss/reasoning/planner/learned.py`、`pss/runtime/services.py`、`pss/runtime/runtime.py`、`pss/eval/scripts/run_runtime_planning.py`；
- 结果：`eval/results/runtime_planning/runtime_planning.{json,md}`、`checkpoints/planner/learned_heuristic.pt`；
- 报告：`PHASE22_GAP_CLOSURE_PLANNING.md`；
- 复现：`python -m pss.eval.scripts.run_runtime_planning --device cpu --train 4000 --test 400`。

**验收判断**

- ✅ A* 搜索保留，heuristic 可学习；
- ✅ expansions 显著下降，cost/success 不变；
- ✅ runtime 可选用 learned heuristic，可审计；
- ⚠️ 小域 wall-clock 更慢，不能声称整体加速；
- ⚠️ 单 seed、同域标签，泛化未验证。

---

### P0-17（新）Gap closure 6：真实本地 decoder 接入 runtime

**要解决的问题**

架构表 row 6 目标是 LLM decoder；此前只有 adapter，没有真实模型。

**做法**

- 下载并缓存 Qwen2.5-1.5B-Instruct；`load_local_decoder` 优先选它（0.5B/135M 作为 fallback，但太弱）；
- decoder 使用 chat template、greedy generation；
- `LLMLanguageHead` 增加 tool-loop convergence guard：重复调用已执行的工具时直接返回工具结果，避免 budget 耗尽；
- `SYSTEM_PROMPT` 明确“不要重复已执行的 tool call，使用 TOOL_RESULTS”；
- `run_runtime_language.py`：真实 decoder + 7 查询 grounding，对照 deterministic ToolAgent。

**执行结果（2026-11-24）**：部分达标（有边界）。

| agent | accuracy | tool calls | citations | unsupported | latency |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B + tools | **0.857** | 5 | 0 | 0 | 4041 ms |
| deterministic tool agent | **1.000** | 8 | 2 | 0 | <1 ms |

- 真实 decoder 能答静态问题，并能正确发出 `update_fact` / `rollback_fact` 调用。
- 唯一失败是对象枚举：模型列了 3/4 个对象，漏了 `plate_01`（它直接读 context，没有调用 `get_objects`）。
- citations 为 0：模型很少调用读取工具，主要从 prompt 里的 state 回答；ToolAgent 仍是引用/确定性参考。
- 没有 convergence guard 时，1.5B 会重复 `update_fact`/`rollback_fact` 直到 budget 耗尽并返回 unknown；guard 后收敛到工具结果。
- **边界**：单 episode、7 查询、一个 1.5B 模型、4s/query；无多轮对话；未 fine-tune；不是 frontier LLM 结果。

**产物**

- 代码：`pss/heads/text/structured.py`、`pss/eval/scripts/run_runtime_language.py`；
- 结果：`eval/results/runtime_language/runtime_language.{json,md}`；
- 报告：`PHASE23_GAP_CLOSURE_LANGUAGE.md`；
- 复现：`python -m pss.eval.scripts.run_runtime_language --device cuda`。

**验收判断**

- ✅ 真实本地 decoder 接入 runtime，静态问答 + 写工具可用；
- ✅ tool-loop 收敛保护；
- ⚠️ 对象枚举漏项、citations 0，小模型 grounding 不完整；
- ⚠️ 4s/query，单 episode，不能声称生产可用。

---

### P0-18（新）Gap closure 7：runtime 权限配置

**要解决的问题**

架构表 row 7 目标是版本/回滚/权限；版本、快照、回滚、审计已在 `CognitiveState`，但权限仍是粗粒度。

**做法**

- 新增 `pss/runtime/permissions.py`：`PermissionProfile`（read / ingest / write_facts / plan / act / rollback）；
- profiles：`read_only` / `service` / `operator` / `admin`；
- `PSSRuntime._require()` 在 ingest / assert_fact / set_goal / plan / act / rollback / explain / summary 强制执行；
- 拒绝动作先写 `permission_denied` audit，再抛 `PermissionDenied`；
- `summary.permissions` 报告当前 profile；默认 `admin` 保持兼容。

**执行结果（2026-11-24）**：达标（有边界）。

| profile | read | ingest | write facts | plan | act | rollback | denied audit |
|---|---:|---:|---:|---:|---:|---:|---:|
| read_only | yes | no | no | no | no | no | 5 |
| service | yes | yes | no | no | no | no | 4 |
| operator | yes | yes | yes | yes | yes | no | 1 |
| admin | yes | yes | yes | yes | yes | yes | 0 |

- 权限矩阵符合预期；每次拒绝都有 `permission_denied` 审计。
- operator 可写/规划/执行，但 rollback 需要 admin。
- 默认 admin 保持既有实验兼容；生产应使用 operator/service。
- **边界**：profile 是全局粗粒度，不是 per-object ACL；没有认证身份绑定；runtime audit 未做签名链；单进程。

**产物**

- 代码：`pss/runtime/permissions.py`、`pss/runtime/runtime.py`、`pss/eval/scripts/run_runtime_audit.py`；
- 结果：`eval/results/runtime_audit/runtime_audit.{json,md}`；
- 报告：`PHASE24_GAP_CLOSURE_AUDIT.md`；
- 复现：`python -m pss.eval.scripts.run_runtime_audit`。

**验收判断**

- ✅ 四档权限矩阵正确；
- ✅ 拒绝动作有审计；
- ✅ 版本/回滚/权限三者闭环；
- ⚠️ 全局粗粒度、无认证、无签名审计链。

---

### P0-19（新）Phase 25：Phase A 评测套件骨架

**要解决的问题**

之前的评测都是自建合成 benchmark，缺少公开权威评测的接入路径。需要一套不合并原始数据、按层适配、统一指标的评测体系。

**做法**

- 新增 `pss/eval/suite/`：
  - `schema.py`：`EvalEpisode`（frames/entities/relations/facts/goal/query/tools/expected）；
  - `adapter.py`：`BenchmarkAdapter` 协议 + `BenchmarkUnavailable`；
  - `registry.py`：name → adapter；
  - `metrics.py`：accuracy/exact match/AUROC/FPR@95TPR；
  - `predictors.py`：BFCL keyword、OpenOOD prototype、ZsRE FactStore 三个确定性 baseline；
  - `runner.py`：suite 执行 + `metrics.json` + `report.md`；
  - `adapters/{bfcl,openood,zsre}.py`：Phase A 三个 adapter；
  - `README.md`：格式、契约、后续 adapter。
- 原始数据放 `data/public/<bench>/`（gitignored），不提交、不自动下载。

**执行结果（2026-11-24）**：骨架完成，smoke 通过。

| benchmark | layer | metric | smoke value | n |
|---|---|---|---:|---:|
| bfcl | language | tool_call_accuracy | 1.000 | 3 |
| bfcl | language | argument_accuracy | 0.667 | 3 |
| openood | open_set | auroc | 1.000 | 9 |
| openood | open_set | fpr_at_95tpr | 0.000 | 9 |
| openood | open_set | accuracy | 0.778 | 9 |
| zsre | memory | edit_success | 1.000 | 2 |
| zsre | memory | locality | 1.000 | 2 |
| zsre | memory | rollback_ok | 1.000 | 2 |

- smoke 只验证 pipeline，不是 benchmark 性能；episodes 是内置小样本。
- adapter 不合并数据；每个 adapter 声明 `layer`/`license`，结果按层汇总，不造总分。
- `prepare()` 找不到数据时抛 `BenchmarkUnavailable`，runner 默认记录 skipped，不静默下载。
- **边界**：未下载官方数据；BFCL parser 是容错子集；OpenOOD 是 feature-level；smoke predictor 是 baseline，不是 PSS 能力结论。

**产物**

- 代码：`pss/eval/suite/{schema,adapter,registry,metrics,predictors,runner}.py`、`pss/eval/suite/adapters/*`；
- 结果：`eval/results/suite/{metrics.json,report.md}`；
- 报告：`PHASE25_EVAL_SUITE.md`；
- 复现：`python -m pss.eval.suite.runner --smoke`。

**验收判断**

- ✅ 统一 episode schema + adapter + metrics + runner 可运行；
- ✅ bfcl/openood/zsre 三个 adapter smoke 通过；
- ✅ 原始数据不提交、不自动下载；
- ⚠️ 未接官方数据；
- ⚠️ predictor 仍是确定性 baseline，下一步换 PSS-backed。

---

### P0-20（新）Phase 26：接 CLEVR-CoGenT / TAO 官方数据

**要解决的问题**

Phase A 骨架只有内置 smoke 样本；需要把两个公开权威数据集真正接进 suite。

**做法**

- TAO：下载官方 `TAOLabels.zip`（`TAO_val/gt/gt.json`），新增 `adapters/tao.py`：
  - 解析 videos/images/tracks/annotations；
  - oracle GT boxes + IoU tracker 作为关联 baseline；
  - 评估 IDF1 / ID retention / ID switches；
- CLEVR-CoGenT：后台下载官方 `CLEVR_CoGenT_v1.0.zip`（24.7GB），新增 `adapters/clevr_cogent.py`：
  - 解析 `scenes/*.json` + `images/<split>/`；
  - left/right/front/behind 关系 episode；
  - valA vs valB 组合 split；
- registry / runner / README 已接入。

**执行结果（2026-11-24）**：TAO 已可跑；CLEVR 官方 valA/valB 已跑完。

| 项目 | 状态 |
|---|---|
| TAO official annotations | **已下载**（6.3MB） |
| TAO official run（50 val videos，oracle boxes） | IDF1 **0.478** / retention **0.482** / switches 7.94 |
| TAO frames | 未下载（源视频大/需注册）；association metric 不需要 |
| CLEVR-CoGenT zip | 已下载并解压（24.7GB） |
| CLEVR official valA/valB | relation accuracy **0.994 / 0.994**（各 30,000 episodes） |

- TAO 数字是 **oracle GT boxes + IoU tracker** 的关联 baseline，不是 PSS 感知结果。
- CLEVR 当前 predictor 用官方 relationships 作 GT、pixel x/depth 作输入，是 **image-space baseline**；真实 PSS 结果需要图像编码器 + relation head。
- relation-only 任务不检验 CoGenT 的属性组合 shift（shape/color/material/size），valA/valB 相等是预期；属性组合需要另一套 task。
- 原始数据放 `data/public/`（gitignored），不提交、不自动下载。

**产物**

- 代码：`pss/eval/suite/adapters/{tao,clevr_cogent}.py`、registry/runner/README；
- 结果：`eval/results/suite_tao/`；
- 报告：`PHASE26_OFFICIAL_DATA.md`；
- 复现：`python -m pss.eval.suite.runner --benchmarks tao --root data/public`。

**验收判断**

- ✅ TAO 官方 annotations 接入并可评估；
- ✅ CLEVR-CoGenT adapter 就绪、smoke 通过；
- ⚠️ TAO frames 未接，当前是 association baseline；
- ✅ CLEVR 官方 valA/valB 已跑（0.994/0.994，pixel baseline）；
- ⚠️ relation-only 不检验 CoGenT 属性组合 shift；
- ⚠️ 两个 predictor 目前都是 baseline（TAO oracle boxes / CLEVR pixel），不是 PSS 能力结论。

---

### P0-21（新）CLEVR 图像关系 + 属性组合

**要解决的问题**

官方 CLEVR valA/valB 之前只用 pixel/3D 规则 baseline，没有图像编码器；CoGenT 的属性组合 shift 也没有测。

**做法**

- 官方 `pixel_coords` 裁 object crops（GT boxes）→ 冻结 DINOv2-small CLS；
- relation head：MLP over `[e_i,e_j,e_i-e_j,e_i*e_j]`（可选 image-space dx/dy/depth）；
- property：joint combination 分类器 vs factorized 四头（shape/color/material/size）；
- trainA 8,000 scenes；valA/valB 各 3,000。

**执行结果（2026-11-24）**：完成。

**Relation**

| method | valA | valB |
|---|---:|---:|
| pixel geometry rule | 0.986 | 0.991 |
| vision only | 0.928 | 0.930 |
| vision + position | **0.994** | **0.996** |

**Property composition（trainA → valA/valB）**

| method | valA exact | valB exact |
|---|---:|---:|
| joint combination | 0.810 | **0.268** |
| factorized (DINOv2 all) | 0.747 | **0.316** |
| **factorized (color hist for color)** | **0.801** | **0.795** |

| color head | valA | valB |
|---|---:|---:|
| DINOv2 CLS | 0.774 | **0.336** |
| color histogram | **0.845** | **0.841** |

| split | shape | color | material | size |
|---|---:|---:|---:|---:|
| valA | 0.960 | 0.845 | 0.956 | 0.985 |
| valB | 0.957 | **0.841** | 0.958 | 0.981 |

- 图像关系头 + position 追平 geometry rule；vision-only 0.93 说明 crops 有信号。
- joint 组合分类器在 valB 掉 54 点；纯 DINOv2 factorized 只恢复到 0.316；
- **color 是瓶颈**（0.774→0.336），但**颜色直方图修复**：valB color 0.841，factorized exact 提升到 **0.795**；shape/material/size 仍用 DINOv2，稳定。
- 仍是 GT boxes（detection-oracle），relation head 是 MLP，不是 PSS `TypedRelationHead`。

**产物**

- 代码：`pss/eval/scripts/run_clevr_image.py`；
- 结果：`eval/results/clevr_image/clevr_image.{json,md}`、`data/public/clevr_cogent/cache/`；
- 报告：`PHASE27_CLEVR_IMAGE_RELATION.md`；
- 复现：`python -m pss.eval.scripts.run_clevr_image --device cuda --train 8000 --val 3000`。

**验收判断**

- ✅ 图像 relation 可训、可评；
- ✅ CoGenT 属性组合 shift 被量化；factorized 优于 joint；
- ⚠️ GT boxes 是 oracle；color 是瓶颈；
- ⚠️ 未接 PSS typed relation head / 检测器。

---

### P0-22（新）TAO 官方帧 + PSS tracker

**要解决的问题**

TAO 只有 annotations 时只能用 oracle-box IoU baseline；需要用官方 frames 跑真正的 PSS tracker（appearance + pose）。

**做法**

- 下载 Charades 480p 官方视频，按 TAO `frame_index` 抽帧到 `data/public/tao/frames/`；
- 新增 `prepare_tao_frames.py` 和 `run_tao_pss.py`；
- 用 GT boxes 作为 detections（association benchmark），crop → 冻结 DINOv2 → PSS tracker；
- 关键修复：TAO boxes 是 1280×720，Charades 帧是 480×270，必须先缩放；PSS 的 `reid_pose_gate` 默认 0.5（米）在像素尺度下等于禁用 re-ID，需要改成 ~300。

**执行结果（2026-11-24）**：达标（有边界）。

| method | IDF1 | retention | switches |
|---|---:|---:|---:|
| IoU baseline | 0.653 | 0.611 | 5.12 |
| PSS g90 | 0.849 | 0.836 | 1.09 |
| PSS g180 | 0.920 | 0.902 | 0.60 |
| PSS g300 | 0.922 | 0.903 | 0.60 |
| PSS pose only | 0.715 | 0.721 | 1.44 |
| PSS appearance only | 0.909 | 0.890 | 0.84 |
| **PSS g300 + pixel re-ID** | **0.979** | **0.973** | **0.44** |

- 5 个 Charades 视频、176 帧；GT boxes 是 oracle detections，不是检测器。
- appearance 是主要信号（pose-only 0.715）；像素尺度 re-ID gate 是最后提升的关键。
- 第一次低分是 box 未缩放 + re-ID gate 单位错误，不是架构失败。
- **边界**：小样本、GT boxes、DINOv2 未微调、配置在该子集上调过；未用官方 TaoEval HOTA。

**产物**

- 代码：`pss/eval/scripts/{prepare_tao_frames,run_tao_pss}.py`、`pss/eval/suite/predictors.py`；
- 结果：`eval/results/tao_pss/tao_pss.{json,md}`、`data/public/tao/frames/`；
- 报告：`PHASE28_TAO_PSS.md`；
- 复现：`python -m pss.eval.scripts.prepare_tao_frames --videos 5` + `python -m pss.eval.scripts.run_tao_pss --videos 5`。

**验收判断**

- ✅ 官方 TAO frames 接入；
- ✅ PSS tracker 在 association 上明显优于 IoU（0.979 vs 0.653）；
- ✅ 检测器替换完成（P0-23）：COCO person detector + PSS 在同一 person-only 子集 IDF1 **0.984**，GT-box oracle 1.000，IoU 0.653；
- ⚠️ person-only、5 个视频、zero-shot COCO detector；配置在该子集上调过；
- ❌ 未跑官方 HOTA。

---

### P0-23（新）把 GT boxes 换成检测器

**要解决的问题**

Phase 27/28 的 CLEVR 与 TAO 结果都依赖 GT boxes，无法说明检测误差下的表现。

**做法**

- CLEVR：训练轻量 CenterNet-lite（256×256 输入、64×64 heatmap + depth head），
  trainA 8,000 scenes / 25 epochs；测试用检测中心 + 固定 64×64 crop，
  GT 中心只用于评分和贴标签；
- TAO：torchvision Faster R-CNN MobileNetV3-320（COCO person）替代 GT boxes，
  检测框 → 冻结 DINOv2 → PSS tracker；GT person boxes 只用于评分和标签；
- 新增 `train_clevr_detector.py`、`run_tao_detector.py`。

**执行结果（2026-11-24）**：达标（有边界）。

CLEVR（trainA → valA/valB）：

| split | det recall | det precision | center err | relation | property exact | color |
|---|---:|---:|---:|---:|---:|---:|
| valA | 0.997 | 0.930 | 4.96 px | 0.995 | 0.792 | 0.839 |
| valB | 0.996 | 0.925 | 5.10 px | 0.994 | 0.790 | 0.839 |

- 关系与 GT-box baseline 基本持平（0.995/0.994 vs 0.994/0.996）；
- 属性组合只掉 0.5–0.9 点（0.792/0.790 vs 0.801/0.795），检测器不再是大瓶颈。

TAO（5 Charades videos，person-only）：

| method | IDF1 | retention | switches |
|---|---:|---:|---:|
| detected + IoU | 0.653 | 0.692 | 1.20 |
| **detected + PSS** | **0.984** | **1.000** | **0.00** |
| GT + PSS (oracle) | 1.000 | 1.000 | 0.00 |

- person detection recall 0.970 / precision 0.809；PSS 的增益在检测噪声下仍成立；
- **边界**：CLEVR 检测器是 in-domain（trainA）；TAO 只测 person 类、zero-shot COCO、
  5 个视频；未跑官方 HOTA；CLEVR relation head 仍是 MLP。

**产物**

- 代码：`pss/eval/scripts/{train_clevr_detector,run_tao_detector}.py`、`tests/test_clevr_detector.py`；
- 结果：`eval/results/clevr_detector/`、`eval/results/tao_detector/`；
- 报告：`PHASE29_DETECTOR_REPLACEMENT.md`；
- 复现：`python -m pss.eval.scripts.train_clevr_detector --device cuda --train 8000 --val 3000 --epochs 25` + `python -m pss.eval.scripts.run_tao_detector --device cuda --threshold 0.3`。

**验收判断**

- ✅ CLEVR 与 TAO 都不再使用 GT boxes 作为 tracker/relation 输入；
- ✅ 检测器下关系/属性/IDF1 均保持；
- ⚠️ in-domain CLEVR detector；TAO person-only + 小样本；
- ❌ 未接 PSS `TypedRelationHead` / 官方 TaoEval HOTA。

---

## 3. P1：补能力闭环

### P1-5 真正的“物体-关系-动作”组合 OOD（P5b）

**要解决的问题**

当前 P5 只是 high-hardness / 材质 regime，不是关系组合。真正的组合泛化还没测。

**做法**

- 生成 `(物体, 关系, 动作)` 三元组场景；
- 训练集只含见过的组合，测试集含没见过的组合；
- PSS 用 scene graph + 关系 + planner；
- transformer 用端到端策略或实体 token 策略；
- 分 ID / 新组合 / 新物体 / 新关系四档。

**指标**

- 任务成功率；
- 关系 F1；
- 重规划次数；
- 异常恢复率。

**验收**

- 至少在一个组合档位上 PSS 显著更好；
- 如果所有档位都打平，明确写“显式关系结构没有带来组合泛化优势”。

**工作量**：大。

**执行结果（2026-11-23）**：达标。PSS 1.000；transformer ID 0.997、OOD **0.434 ± 0.216**、OOD invalid 0.139。显式几何 goal check 在未见关系对上保持 1.000，是最清楚的组合泛化优势。

---

### P1-6 学习型 reliability / conflict-aware fusion

**要解决的问题**

固定融合在 geometry shuffle 下只有 0.458，transformer 是 0.749。PSS 现在靠弃权规避，coverage 掉到 5–7%。这是 PSS 唯一明显输给 transformer 的场景。

**做法**

- 用模态间一致性、每路 margin、历史可靠性做输入；
- 训练一个小 reliability head，输出每路权重；
- 或用 Bayesian precision weighting；
- 冲突时先降权，再决定是否 abstain；
- 在 clean / shuffle / noise / zero 四类条件下评估。

**指标**

- top-1 accuracy；
- coverage；
- accepted accuracy；
- ECE。

**验收**

- geometry shuffle 准确率从 0.458 提到 0.70+；
- coverage 保持 ≥0.80；
- clean 条件不掉超过 1 点。

**工作量**：中。

**执行结果（2026-11-23）**：未达标。learned gate clean 0.96/0.935，geometry shuffle 0.59/0.245，vision shuffle 0.53/0.792；zero 条件改善（vision 0.98/0.853、geometry 0.82/0.982），但规则 reliable 在 zero 下仍更稳。

---

### P1-7 开放集风险控制

**要解决的问题**

P2 的 0.958 reject 是点估计，没有统计保证；模态损坏时阈值是否可靠也没测。

**做法**

- conformal prediction / per-category threshold calibration；
- 给定 FPR=5% 的拒识保证；
- 在 modality corruption、missing modality、类别不平衡下测 coverage；
- 和 energy / softmax / Mahalanobis 基线比。

**指标**

- AUROC；
- FPR@95% TPR；
- coverage at fixed risk；
- 条件覆盖率。

**验收**

- 在 95% seen accept 下 FPR ≤5%；
- 给出 coverage-risk 曲线，而不是单点。

**工作量**：中。

**执行结果（2026-11-23）**：marginal conformal 通过：0.969 accept / 0.958 reject / AUROC 0.993。class-conditional 平均 0.859，但最小类 coverage 0.333，当前 per-class 校准样本不足。

---

### P1-8 原型质量治理

**要解决的问题**

P3 假设新类样本干净。真实增量更新会遇到坏样本、标注错误、分布漂移。

**做法**

- medoid / trimmed mean 替代简单均值；
- 新样本 outlier rejection；
- 原型方差和漂移监控；
- 污染检测和回滚；
- 和 Phase 9 的审核门、Ed25519 签名包结合。

**指标**

- 10% / 30% 坏样本下的 old/new accuracy；
- 污染检测率；
- 回滚时间；
- 原型方差变化。

**验收**

- 30% 坏样本下旧类掉点 <2 点；
- 回滚 <1 s；
- 污染样本能被显式标记。

**工作量**：中。

**执行结果（2026-11-23）**：旧类始终 0.979；30% 污染下 mean/trimmed/median 新类 1.000，污染标记率 0.97–0.99；medoid 0.728 不适合；回滚 29.5 ms。当前污染远离类中心，均值已够稳，关键是污染检测/标记。

---

### P1-9 把 P1.3 接到 PyBullet 闭环

**要解决的问题**

P1.3 是符号级 5 路点微基准，PSS 的 1.000 来自 A* 完备性，不是感知闭环能力。

**做法**

- 复用 `pss/reasoning/scripts/run_closed_loop.py`；
- 加入感知噪声、动态障碍、执行误差；
- PSS：scene graph → facts → A* → 执行 → 重感知 → 重规划；
- transformer：同样输入，端到端输出动作；
- 测重规划次数、异常恢复时间、最终距离。

**指标**

- 任务成功率；
- 平均动作数；
- 重规划次数；
- 异常恢复时间；
- 最终距离。

**验收**

- PSS 在动态障碍下成功率仍 ≥0.90；
- 重规划后能恢复，不陷入振荡；
- 和 transformer 的差距可解释。

**工作量**：大。

**执行结果（2026-11-23）**：动态障碍 3/3 成功，1–2 次重规划，push 37.3 vs 17.7，最终距离 4.5–6.0 cm。环境是 MuJoCo，不是 PyBullet；场景只有 2 物体、3 seeds。

---

## 4. P2：长期项

### P2-10 世界模型

**目标**：多实体 rollout <0.05 m。

**执行结果（2026-11-23）**：

| variant | 1-step | rollout all | rollout target |
|---|---:|---:|---:|
| one_step | 0.0240 | 0.0932 | 0.1001 |
| rollout_aware | 0.0217 | **0.0863** | 0.1007 |

- rollout loss 有改善（约 7%），但未达 5 cm 目标。
- 误差主要受模型容量/数据覆盖限制。

### P2-11 长时记忆

**目标**：70 帧级长时遮挡后的 re-ID。

**执行结果（2026-11-23）**：80 帧、70 帧完全遮挡。

| method | ID retention | switches | reappearance |
|---|---:|---:|---:|
| PSS graph+tactile | **0.983** | **0.03** | **0.967** |
| Transformer re-ID | 0.900 | 0.20 | 0.800 |

- near-identical：PSS 0.933 vs transformer 0.667。
- 长时记忆没有破坏 PSS 优势。

### P2-12 消融与统计

**执行结果（2026-11-23）**：

| variant | overall |
|---|---:|
| full | 0.930 ± 0.029 |
| no_edges | 0.941 ± 0.011 |
| no_assignment | **0.969 ± 0.033** |
| greedy | 0.967 |
| transformer | 0.802 ± 0.008 |
| + tactile（ref） | **0.989 ± 0.017** |

- paired bootstrap：delta **+0.128**，95% CI **[+0.096, +0.162]**，P(delta≤0)=0.0000；这是**无 tactile transformer** 基线，公平归因后只剩 +0.012。
- graph 169k params vs transformer 337k。
- 结论：图匹配对 transformer 的优势显著；但 full 配置的 assignment loss / edge message 在 v2 上过拟合，去掉 assignment loss 到 0.969，tactile 是最大单项增益。

### P2-13 工程化

**执行结果（2026-11-23）**：

- 端到端 p50 **0.5 ms**、p95 **14–15 ms**；RSS 927 MB。
- graph load 24.9 ms，perception load 23.7 ms。
- dense 0.51/15.33 ms，pruned 0.49/14.25 ms；3 实体场景剪枝收益可忽略。
- p95 长尾主要来自首帧/预热/偶发开销；ONNX/批处理/量化还没做。

## 5. 四波执行回顾

| 波次 | 任务 | 结果 |
|---|---|---|
| 第一波 | P0-1 DINO、P0-3 scaling | DINO 未达标；N=1000 24.35x，N=200 3.53x |
| 第二波 | P0-2 v3、P0-4 强基线、P2-12 消融 | v3 跨 benchmark 达标；DINO transformer 未变强；replay 追平零遗忘；消融显示 full 配置过拟合 |
| 第三波 | P1-6 reliability、P1-7 conformal、P1-8 原型治理、P1-9 闭环、P1-5 P5b | P5b 1.000 vs 0.434；conformal marginal 可用；原型治理可用；闭环 3/3；learned reliability 未达标 |
| 第四波 | P2-10 世界模型、P2-11 长时记忆、P2-13 工程化 | 长时记忆 0.983 vs 0.900；世界模型 0.0863 未达 5 cm；工程化 p50 0.5 ms / p95 15 ms |

### 净结果

- **赢（系统级）**：PSS+tactile 的完整/跨 v3/长时 tracking、关系组合 OOD、规划 OOD、开放集、冲突弃权、更新速度、N=1000 scaling、长时记忆。
- **不能归因给架构**：re-ID 精度在公平归因后只剩 +0.012，主要来自 tactile 输入。
- **平/收窄**：闭集分类、属性 OOD、零遗忘（replay 追平）、N=200 scaling、class-conditional conformal。
- **输/未达标**：DINO 去 tactile 依赖、学习型 reliability、世界模型 5 cm。
- **未做**：真实数据/R-Gate、sparse/retrieval attention、EWC、P5b 感知版。

## 6. 任务总表

| 编号 | 任务 | 状态 | 关键结果 |
|---|---|---|---|
| P0-1 | DINOv2 instance token | 未达标 | DINO raw 0.903；PCA-32 0.942/0.804；仍低于 tactile 0.989/0.956 |
| P0-2 | tracking v3 + 跨 benchmark | 达标 | v2→v3 0.881 vs 0.672；PSS 掉 4.9 点，TR 掉 13.0 点 |
| P0-3 | graph matcher scaling | 部分达标 | N=1000 24.35x；N=200 3.53x，未达 5x；v2 相对 retention 0.992 |
| P0-4 | 强 transformer 基线 | 部分完成 | DINO transformer 0.752 < base 0.802；replay 追平零遗忘；sparse/EWC 未做 |
| P0-4b | 公平归因（同 tactile + Hungarian） | 完成 | re-ID：+0.128 → **+0.012**；tracking：1.000 vs 0.908 → **1.000 vs 0.988**；架构增量不显著 |
| P5b-2 | 碰撞感知执行闭环 | 达标（有边界） | kinematic：PSS 1.000/TR 0.450；physical：PSS 0.700/TR 0.150 |
| P0-4c | EWC 增量基线 | 完成（负结果） | EWC forgetting 0.237；replay 0.002 最好；PSS 快 41.7x |
| P0-4d | sparse attention 基线 | 完成（反超） | sparse_k1 0.996 > PSS 0.989；near 0.991 vs 0.956 |
| P0-5 | 换编码器 + 投影校准 | 部分达标 | DINOv2 re-ID 0.963→0.993（仅换头）；residual-only 开放集 0.998；闭集 0.995；P5b 构造性不变 |
| P0-6 | 关系层可学习（typed heads） | 部分达标 | 关系 F1 0.957/0.996/0.970；hybrid 缺 tactile 0.791→0.819；P5b 分布内 1.000/1.000，held-out OOD 0.688 |
| P0-7 | 学习 active-set / write policy | 部分达标 | re-ID k=1+fallback 0.972→1.000（10% fallback）；write 0.84→0.88 @匹配写入，review 0.92；长历史 fact active-set 未超 hot |
| P0-8 | 输出头：结构化状态 + 工具 + LLM | 达标（有边界） | 7 查询：flat 0.714 / tools 1.000；citation coverage 1.000；虚构实体可检测；上下文 0.27x |
| P0-9 | 端到端 / bilevel 训练 | 部分达标 | joint re-ID 0.889 vs sequential 0.867；joint_instance 0.922；一致性约束降冲突但有精度代价；联合关系 F1 低于单独头 |
| P0-10 | PSS 认知运行时（分层落地） | 达标（有边界） | full：entity/plan/task/conflict/rollback/citations 全通；逐层消融各自失效；relation 12→0 为任务边界 |
| P0-11 | Phase 17：学习模块接进运行时 | 达标（有边界） | 关系头 0.988 驱动 planner；写门控 0.75→0.85；算子 update/rollback；70 帧遮挡 retention 1.000；decoder 不可用 |
| P0-12 | Phase 18：学习模块设为 runtime 默认 | 达标（有边界） | learned relations 15、write 拒绝 noisy fact、active 排除 distractor；fallback 可切换；任务契约不变 |
| P0-13 | Gap closure 1–2：编码器/投影默认 | 部分达标 | runtime 默认 DINOv2（retention 1.000，p50 16.7ms vs 4.7ms）；whiten96+res 投影；reliability 进 uncertainty；精度未提升 |
| P0-14 | Gap closure 3：object attention 绑定选项 | 部分达标 | ObjectAttentionReID 接入 EntityLayer/runtime；pair/tracker 与 graph 打平（1.000）；默认仍 hand |
| P0-15 | Gap closure 4：可学习/可版本化关系规则 | 部分达标 | 阈值 fit/update/rollback；contact F1 0→0.267；in 1.000；near 无 GT；特征限制上限 |
| P0-16 | Gap closure 5：learned heuristic + A* | 达标（有边界） | expansions 6.29→3.28（weighted 2.48），cost/success 不变；小域 latency 0.13→3.3ms；runtime 可审计 |
| P0-17 | Gap closure 6：真实本地 decoder | 部分达标 | Qwen2.5-1.5B：7 查询 0.857；update/rollback 工具可用；citations 0；漏 1 个对象；4s/query |
| P0-18 | Gap closure 7：runtime 权限配置 | 达标（有边界） | read_only/service/operator/admin 矩阵正确；拒绝审计；operator 无 rollback；默认 admin |
| P0-19 | Phase A 评测套件骨架 | 达标（有边界） | schema/adapter/metrics/runner 可运行；bfcl/openood/zsre smoke 通过；未接官方数据 |
| P0-20 | 接 CLEVR-CoGenT / TAO 官方数据 | 部分达标 | TAO 官方 annotations 已接（IDF1 0.478, oracle boxes）；CLEVR 官方 valA/valB 0.994/0.994（pixel baseline） |
| P0-21 | CLEVR 图像关系 + 属性组合 | 达标（有边界） | vision+pos 0.994/0.996；joint 0.810→0.268；color hist 修复后 factorized valB 0.316→0.795 |
| P0-22 | TAO 官方帧 + PSS tracker | 达标（有边界） | 5 Charades 视频：PSS IDF1 0.979 vs IoU 0.653；GT boxes oracle；小样本 |
| P0-23 | 把 GT boxes 换成检测器 | 达标（有边界） | CLEVR CenterNet-lite：recall 0.997/0.996，relation 0.995/0.994，property 0.792/0.790；TAO COCO person + PSS IDF1 0.984 vs IoU 0.653 |
| P1-10 | 事实级知识热更新 | 完成 | PSS 0.171 ms / old 1.000；full 952 ms / old 0.735；replay 914 ms / old 1.000 |
| P1-11 | 长历史 active-set scaling | 完成 | T=1000：PSS 1.000/0.0003 ms；TR 0.076/8.40 ms |
| P1-12 | 多源事实 + 规则热更新 | 完成 | priority 1.000 vs flat 0.800；273 冲突；rule edit 0.029 ms |
| P1-13 | 近似 active-set | 完成 | sigma 0.3：active 0.920 vs full 0.495；延迟 0.0058 ms 平 |
| P1-14 | RAG / external-memory 基线 | 完成 | T=1000：RAG 0.971 / 0.77 ms；full 0.120 / 4.90 ms；PSS 1.000 / 0.0002 ms |
| P1-15 | Object-centric attention 最小实现 | 完成（追平） | 0.987 / near 0.956；graph 0.989；sparse 0.996 |
| P1-5 | P5b 真组合关系 OOD | 达标 | PSS 1.000；transformer ID 0.997 / OOD 0.434 |
| P1-6 | 学习型 reliability | 未达标 | geometry shuffle 0.59/0.245；zero 条件改善但规则更稳 |
| P1-7 | conformal 开放集 | 部分达标 | marginal 0.969/0.958/AUROC 0.993；class-conditional 最小 coverage 0.333 |
| P1-8 | 原型质量治理 | 达标 | 30% 污染下 new 1.000，污染标记 0.988；medoid 0.728；回滚 29.5 ms |
| P1-9 | MuJoCo 闭环 | 达标（有边界） | 动态障碍 3/3 成功，1–2 次重规划；2 物体、3 seeds，不是 PyBullet |
| P2-10 | 世界模型 | 未达标 | rollout 0.0932 → 0.0863；目标 <0.05 |
| P2-11 | 长时记忆 | 达标 | 70 帧遮挡：PSS 0.983/0.967 vs TR 0.900/0.800 |
| P2-12 | 消融与统计 | 达标 | full 0.930；no_assignment 0.969；bootstrap +0.128（无 tactile 基线；公平归因 +0.012） |
| P2-13 | 工程化 | 达标（有边界） | p50 0.5 ms / p95 15 ms；RSS 927 MB；ONNX/批处理未做 |

## 7. 不建议现在做的事

- **继续堆新模块**：当前瓶颈不是模块数量，是证据强度。
- **在 v2 上继续调参**：会加重自建集质疑。
- **把 P5 的负结果藏起来**：P5 已经明确 FAIL，应该保留并解释。
- **声称通用精度优势**：闭集识别是打平，不能外推。
- **用真实数据之外的“伪真实”数据冒充外部验证**：仿真增强只能算鲁棒性测试，不能算 sim2real 证据。
- **再说“零遗忘是 PSS 独有”**：replay 已追平；PSS 的优势是快、省存储、可回滚。
- **再声称“graph 边消息/assignment loss 是主要机制”**：消融不支持；显式关系层才是 P5b 的优势来源。
- **把 P5b 的符号结果说成感知闭环结果**：它仍是无噪声符号微基准。

---

## 8. 归因复跑（2026-11-23）

### 8.1 公平归因：同一 tactile token，同一 Hungarian

| token set | PSS graph | transformer | delta |
|---|---:|---:|---:|
| base | 0.930 ± 0.029 | 0.802 ± 0.008 | **+0.128** |
| tactile | 0.989 ± 0.017 | **0.977 ± 0.017** | **+0.012** |

- 给 transformer 同样 tactile 后，re-ID 差距从 +0.128 缩到 +0.012。
- tracking 同 tactile 后：PSS 1.000 vs transformer **0.988**，差距 +0.012。
- **re-ID 和 tracking 的大部分优势都来自 tactile 输入，不是 graph matcher 架构。**

### 8.2 no_assignment 复跑

| 配置 | overall | near-identical |
|---|---:|---:|
| base + assignment | 0.930 | 0.738 |
| base + no_assignment | **0.969** | **0.902** |
| tactile + assignment | **0.989** | **0.956** |
| tactile + no_assignment | 0.976 | 0.902 |

- no_assignment 在 base 上更好，在 tactile 上更差；assignment loss 与 tactile 有交互。
- 主配置仍保留 tactile + assignment；P1b tracking noassign 0.994，低于 full 1.000。

### 8.3 P5b 感知版

| method | ID success | OOD success | OOD invalid |
|---|---:|---:|---:|
| PSS explicit planner | 1.000 | **1.000** | 0.000 |
| Transformer BC | 0.995 ± 0.010 | 0.453 ± 0.232 | 0.123 |

- 渲染 + tracker 后的 perceived state 上，P5b 结论复现：PSS 1.000 vs transformer 0.453。
- 与纯符号 0.434 一致，说明不是符号状态假象。
- 边界：未执行 push，不是完整物理闭环。

### 8.4 P5b 执行闭环（碰撞感知）

| execution | PSS ID | PSS OOD | TR ID | TR OOD |
|---|---:|---:|---:|---:|
| kinematic placement | 1.000 | **1.000** | 1.000 | 0.450 |
| physical push | 0.800 | **0.700** | 0.750 | 0.150 |

- 规划器加入 segment clearance，直连被挡时经中间区域绕行；物理 push 用 ActionHead，gain 20 / max force 6 / max 40 pulses。
- 物理执行降低绝对成功率（推动漂移/接触/重规划失败），但 PSS 的 OOD 优势仍明显。
- 这是目前最接近“感知→规划→物理执行→重感知”闭环的结果。

### 8.5 EWC 增量基线

| method | old after | new after | forgetting | update time |
|---|---:|---:|---:|---:|
| PSS prototype add | **0.979** | **1.000** | **0.000** | **31.15 ms** |
| Transformer full | 0.809 | 1.000 | 0.169 | 4274 ms |
| Transformer replay | 0.976 | 1.000 | **0.002** | 5066 ms |
| Transformer EWC | 0.741 | 1.000 | 0.237 | 5811 ms |

- EWC 在当前设置下没有降低遗忘，反而更差；replay 仍是最好的零遗忘基线。
- PSS 快 41.7x、无需旧数据、可回滚。

### 8.6 Sparse/retrieval attention

| variant | top-k | overall | near-identical |
|---|---:|---:|---:|
| full attention | - | 0.977 | 0.924 |
| **sparse_k1** | 1 | **0.996** | **0.991** |
| sparse_k2 | 2 | 0.982 | 0.964 |
| sparse_k4 | 4 | 0.974 | 0.978 |
| PSS graph+tactile | - | 0.989 | 0.956 |

- 同 tactile 下 sparse_k1 transformer 反超 PSS；PSS graph matcher 不是最优 re-ID 架构。
- sparse mask 是 dense 实现，不是延迟优化；它是精度基线。

### 8.7 事实级知识热更新

| method | edited acc | old acc | forgetting | edit time |
|---|---:|---:|---:|---:|
| PSS fact update | **1.000** | **1.000** | **0.000** | **0.171 ms** |
| Transformer full | 1.000 | 0.735 | 0.265 | 952.3 ms |
| Transformer replay | 1.000 | 1.000 | 0.000 | 914.1 ms |

- PSS 胜在操作成本、回滚、审计；replay 仍能追平零遗忘。
- 事实级热更新比类别原型更新更接近“知识层热更新”的目标。

### 8.8 长历史 / active-set scaling

| history | full attention acc | RAG acc | full p50 | RAG p50 | PSS p50 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.997 | 0.990 | 0.60 ms | 0.93 ms | 0.0002 ms |
| 100 | 0.633 | 0.982 | 1.03 ms | 0.56 ms | 0.0002 ms |
| 500 | 0.151 | 0.982 | 2.54 ms | 1.21 ms | 0.0002 ms |
| 1000 | 0.120 | **0.971** | 4.90 ms | 0.77 ms | 0.0002 ms |

- Full attention 长历史 accuracy 崩、延迟涨；RAG 保住 accuracy，但仍扫描全 history。
- PSS active-set 延迟不随历史长度变化；优势是检索成本/精确性。

### 8.9 多源事实 + 规则热更新

| store | accuracy vs priority resolver |
|---|---:|
| Multi-source + priority | **1.000** |
| Flat last-write | 0.800 |

- 273 冲突；cascade delete 30 cleared / 270 kept / 30 restored；44.68 ms。
- rule hot update：37 条 derived 改变；direct facts 全保留；rule edit **0.029 ms**；rollback exact。
- 知识层和逻辑层可以分别热更新。

### 8.10 近似 active-set

| noise sigma | active ANN | full-history ANN | active p50 | full p50 |
|---:|---:|---:|---:|---:|
| 0.0 | 1.000 | 1.000 | 0.0056 ms | 0.0118 ms |
| 0.3 | **0.920** | 0.495 | 0.0058 ms | 0.0096 ms |
| 0.5 | **0.700** | 0.150 | 0.0058 ms | 0.0095 ms |

- active-set ANN 对噪声更稳、成本更低；不是只能精确 lookup。

### 8.11 Object-centric attention 最小实现

| method | overall | near-identical | params |
|---|---:|---:|---:|
| Object attention | 0.987 ± 0.011 | 0.956 | 383k |
| PSS graph+tactile（ref） | **0.989** | 0.956 | 169k |
| sparse k=1 transformer（ref） | **0.996** | 0.991 | - |

- typed spatial/semantic/tactile bias + active mask，能工作，基本追平 graph matcher。
- 参数更多、未超 sparse_k1；最小实现没有证明“object attention 更强”。
- 下一步需要关系监督、更多 head 类型、或与外部知识/规划结合。

## 9. 一句话

> **四波之后，PSS 被证明的不是“更强的 transformer”，而是“在身份持久性、显式关系组合、开放集/冲突弃权和快速可审计更新上更强的显式记忆系统”。这比“全面更准”更窄，但也更真实、更难被推翻。**

下一步优先级：**真实数据/R-Gate → 把 P5b 推进到执行闭环 → 用公平归因后的口径重写所有 re-ID 结论。**
