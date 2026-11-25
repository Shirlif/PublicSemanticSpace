# PSS vs Transformer 实验协议（预注册 v1）

> 日期：2026-11-23
> 状态：**预注册，先冻结本文件，再跑实验**
> 目标：证明 PSS 架构本身在特定场景下优于 transformer，而不是证明 PSS 在所有场景更强
> 原则：同输入、同数据、同评测、同预算；先写指标和杀死标准，再跑

---

## 1. 主张

本实验要检验的主张是：

> **在持久实体、开放集更新、多实体关系和组合泛化这几类场景下，PSS 的显式结构比同输入、同数据的 transformer 基线更有优势。**

不主张：

- PSS 在通用视觉、语言生成、端到端 VLA 上全面优于 transformer；
- PSS 在像素级感知上更强；
- PSS 在 near-identical 实例上也能区分。

---

## 2. 实验分层

| 层级 | 目的 | 感知条件 |
|---|---|---|
| L1 架构对照 | 隔离感知，只比架构 | GT 实体、GT 位姿、GT 类别 |
| L2 端到端对照 | 比完整系统 | 两边同一冻结感知 |
| L3 外部验证 | 防自嗨 | RoboTwin / RMBench / AI2-THOR（有 Linux GPU 后） |

本协议先锁定 **L1**。L2/L3 在 L1 有结论后追加，避免一上来被感知误差掩盖。

---

## 3. 假设与主指标

### H1：长遮挡持久实体

**假设**：在完全遮挡 14 帧、遮挡期间物体移动的序列上，PSS 的 ID retention 比 entity-token transformer 高至少 **10 个百分点**。

第一份报告先用 P1a 的 re-ID accuracy 作为 H1 判据；P1b 完整 tracking 的 ID retention 在 P1a 有优势后补。

- 主指标：ID retention
- 次指标：ID switches、reappearance success、wake latency、ID inflation
- 数据：`occlusion_v3` 扩展版，200 train / 40 val / 60 test 序列
- 测试子集：
  - mixed：不同类别
  - same-category：同类多实例
  - near-identical：同尺寸/颜色/材质（边界项，允许 PSS 失败）

### H2：开放集拒绝

**假设**：在 matched seen accuracy 下，PSS 的 unseen rejection rate 比 transformer softmax/energy 基线高至少 **10 个百分点**，或 AUROC 高至少 **0.03**。

- 主指标：AUROC、unseen reject @ seen accept 95%
- 次指标：ECE、seen accuracy
- 数据：`category_pretrain` seen/unseen split；同一冻结特征

### H3：增量类别更新

**假设**：新增类别时，PSS 原型包更新的旧类遗忘比 transformer 微调低至少 **5 个百分点**，更新延迟低至少 **10 倍**。

- 主指标：old-class forgetting、new-class accuracy、update time
- 数据：seen categories + held-out category（cloth / sheet）
- PSS：L2 package add / rollback
- Transformer：最后一层或 LoRA 微调，同数据预算

### H4：多实体计算 scaling

**假设**：实体数 N 从 4 增到 200 时，PSS 的每帧关系计算延迟增长远低于 transformer 全 token attention；N=200 时 PSS 至少快 **10 倍**。

- 主指标：p50/p95 latency、峰值显存
- 次指标：FLOPs、token 数
- 设置：active k=4 固定，N ∈ {4, 20, 50, 200, 1000}

### H5：组合关系泛化

**假设**：训练见过的“物体-关系-动作”组合，测试没见过的组合，PSS 的 OOD 成功率比 transformer 高至少 **10 个百分点**。

- 主指标：OOD task success、relation F1
- 数据：程序化生成组合 split；训练/测试组合不重叠

---

## 4. P1 主实验设计：持久实体

### 4.0 P1 操作化

P1 先拆成两个子实验：

- **P1a 遮挡 re-ID**：取遮挡前最后一帧的 track token 和遮挡后第一帧的 observation token，做匹配。主指标 `re-ID accuracy`。第一份报告用这个。
- **P1b 完整在线 tracking**：把关联器接回完整 `Tracker`，跑 24 帧，主指标 ID retention。P1a 有优势后再做。

本协议把 P1a 作为 H1 的第一判据：如果 P1a 都没有优势，P1b 不继续堆。

### 4.0b P1c：PSS + 学习度量

P1a 暴露的是“手工关联代价”的问题。P1c 保留 scene graph，只把手写 re-ID 代价换成 **Siamese metric embedding**：

- 同一 token：vision + geometry + pose + GT category；
- `MetricEmbedding`：279 → 256 → 128 的 MLP，L2 normalize；
- 训练目标：cosine 相似度的 BCE，正样本同 GT ID，负样本不同；
- 推理：cosine 距离 + Hungarian；
- 对照：PSS hand cost、PSS + learned metric、entity-token transformer；
- 数据/种子/early stop 与 P1a 相同；
- H1c：PSS + learned metric 至少不差于 transformer（delta ≥ 0）；如果能高 5 个百分点，再写成优势。

P1c 不是“纯手工 PSS”，而是“显式 scene graph + 学习关联”。它回答的是：PSS 的结构 + 学习度量，能不能追上 transformer。

### 4.1 任务定义

在线多目标跟踪：

- 输入：每帧可见实体的 token 集合
- 输出：当前观测与历史 track 的关联
- 评估：GT ID retention

每个实体 token 包含：

```text
[pos_x, pos_y, pos_z, quat(4), vision_embedding(d), geometry_embedding(d), category_id]
```

显式存在 mask，遮挡帧没有观测。

### 4.2 公平性约束

两边必须使用完全相同的：

- 实体 token 和特征；
- 训练/验证/测试序列；
- 关联后处理（Hungarian matching，同一 gate 规则）；
- 评测代码和指标；
- 随机种子集合。

PSS 的阈值只在 val 上校准。Transformer 的层数/宽度/学习率只在 val 上选。测试集不参与任何调参。

### 4.3 基线

| 基线 | 说明 |
|---|---|
| PSS full | scene graph + appearance + pose gate + category + time |
| PSS no scene graph | 每帧独立匹配 |
| PSS no category | 去掉类别项 |
| PSS no appearance | 只用位姿和类别 |
| PSS no pose | 只用外观和类别 |
| Transformer A | entity-token transformer，输出关联分数 |
| Transformer B | 把历史 track 序列化成 token，文本状态式 transformer |
| Text-state proxy | 旧的单帧文本状态基线，仅作参考 |

### 4.4 Transformer A 设计

- 输入：当前帧 token + 历史 track token；
- 编码：2–4 层 transformer，d_model=128/256，4/8 heads；
- 输出：当前观测 × 历史 track 的匹配分数矩阵；
- 训练：用 GT 关联生成监督标签；遮挡帧无观测，不产生负样本；
- 损失：binary cross entropy / contrastive；
- 推理：sigmoid 分数 + Hungarian matching；
- 预算：参数量不超过 PSS 中 learnable 参数的 10 倍；报告 FLOPs。

### 4.5 数据生成

新增 `data/tracking_bench_v1`：

- 200 train / 40 val / 60 test sequences；
- 每序列 24 帧：5 可见 + 14 完全遮挡 + 5 可见；
- 遮挡期间物体移动或换位；
- 三个子集：
  - mixed：不同类别
  - samecat：同类多实例
  - near_identical：同尺寸/颜色/材质
- 生成时校验：遮挡帧目标像素为 0，可见帧每目标 ≥20 像素。

### 4.6 P1 判定

- **支持 H1**：PSS full − Transformer A ≥ 10 个百分点，且 5 seeds 的 95% CI 下界 > 0。
- **弱支持**：差值 5–10 个百分点，CI 下界 > 0。
- **不支持**：差值 < 5 个百分点，或 Transformer 更好。
- **边界项**：near-identical 子集允许 PSS 失败，但必须单独报告。

---

## 5. P2 开放集设计

- 特征：同一冻结 vision/tactile/geometry 投影特征；
- PSS：原型库 + 距离阈值 + 开放集校准；
- Transformer：同输入 token 的分类头，softmax / energy / Mahalanobis 三种 OOD 分数；
- 训练：只用 seen classes；
- 测试：seen test + unseen categories；
- 阈值：在 val 上调到 seen accept 95%，再测 unseen reject；
- 判定：PSS unseen reject 高 ≥10 个百分点，或 AUROC 高 ≥0.03。

---

## 6. P3 增量更新设计

- 初始模型：seen categories；
- 新增：cloth 或 sheet；
- PSS：L2 package append + activate + rollback；
- Transformer：最后一层 fine-tune / LoRA，同数据、同 epoch 预算；
- 指标：
  - new-class accuracy；
  - old-class accuracy 变化（forgetting）；
  - update wall-clock；
  - 存储增量；
  - rollback 是否可用。
- 判定：PSS forgetting 低 ≥5 个百分点，update time 低 ≥10 倍。

---

## 7. P4 Scaling 设计

- 固定 active k=4；
- 实体数 N ∈ {4, 20, 50, 200, 1000}；
- PSS：空间索引 + 激活实体关系计算；
- Transformer：全 token attention；
- 指标：p50/p95 latency、峰值显存、FLOPs；
- 判定：N=200 时 PSS 快 ≥10 倍，且延迟增长随 k 而非 N。

---

## 8. P5 组合泛化设计

- 组合空间：物体类别 × 关系 × 动作；
- 训练：随机 70% 组合；
- 测试：held-out 30% 组合；
- PSS：关系事实 + 原型 + 规划；
- Transformer：同输入 token 的端到端策略/分类器；
- 指标：OOD success、relation F1、unseen combination accuracy；
- 判定：PSS OOD success 高 ≥10 个百分点。

---

## 9. 统计与可复现性

- 每个实验至少 **5 个随机种子**；
- 报告均值、标准差、95% bootstrap CI；
- 所有超参数在 val 上选，测试集只跑一次；
- 保存：
  - 数据版本哈希；
  - 训练配置；
  - 模型 checkpoint；
  - 每序列原始指标；
  - 失败案例。
- 代码路径建议：
  - `pss/eval/baselines/entity_transformer.py`
  - `pss/eval/scripts/run_pss_vs_transformer.py`
  - `eval/results/pss_vs_transformer/`

---

## 10. 杀死标准

| ID | 条件 | 结论 |
|---|---|---|
| K1 | P1 PSS − Transformer < 5 个百分点 | H1 不支持，持久实体优势不成立 |
| K2 | P2 PSS 开放集不优于 transformer | H2 不支持 |
| K3 | P3 PSS 遗忘不低于 transformer | H3 不支持 |
| K4 | P4 PSS scaling 不优于 transformer | H4 不支持 |
| K5 | 任何测试集泄漏/预算不公平 | 实验作废 |
| K6 | near-identical 子集 PSS 明显失败 | 记录为能力边界，不掩盖 |

---

## 11. 解释规则

可能出现的结果：

1. **PSS 在 H1/H2/H4 赢，P5 不确定**：说明 PSS 的强项是记忆、开放集、计算结构，不是通用泛化。
2. **PSS 只在 H4 赢**：说明优势主要是工程 scaling，不是能力。
3. **PSS 在 H1 输**：显式实体记忆不如 transformer 时序上下文，需要重新设计 PSS 的关联/记忆机制。
4. **PSS 在 H2 输**：开放集优势不成立，原型阈值/校准需要重做。
5. **PSS 在 H1/H2 赢但 L2 端到端输**：感知域偏移是瓶颈，架构优势被感知误差吃掉。

无论结果如何，都按预注册报告，不做事后改指标。

---

## 12. 执行顺序

1. **P0**：冻结本协议。
2. **P1**：生成 `tracking_bench_v1`，实现 entity-token transformer，跑 5 seeds，出第一份结论。
3. **P2**：开放集对照。
4. **P3**：增量更新对照。
5. **P4**：scaling 对照。
6. **P5**：组合泛化对照。
7. **L2**：同一冻结感知下端到端对照。
8. **L3**：有 Linux GPU 后，在 RoboTwin / RMBench 复现。

---

## 13. 当前不做

- 不用 RoboTwin/AI2-THOR 的运行时作为 P1 前提；
- 不先训练大 VLA；
- 不把 PSS 改成端到端 transformer；
- 不在测试集上调阈值；
- 不把 near-identical 失败藏起来。

---

## 14. 第一份交付物

- `pss/eval/baselines/entity_transformer.py`
- `pss/eval/scripts/run_pss_vs_transformer.py`
- `data/tracking_bench_v1/`
- `eval/results/pss_vs_transformer/p1_tracking.md`

第一份报告只回答一个问题：

> **在 14 帧完全遮挡、物体移动的 tracking 任务上，PSS 的 ID retention 是否稳定优于同输入的 entity-token transformer？**

如果答案是否定的，后面的 H2–H5 不应继续堆，而应先重新设计 PSS 的记忆/关联机制。

---

## 15. P1a 结果（2026-11-23）

数据：`data/tracking_bench_v1`，200 train / 40 val / 60 test。

| 方法 | overall re-ID | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|
| PSS | 0.561 | 0.800 | 0.311 | 0.333 |
| Entity-token transformer | **0.906 ± 0.020** | **0.982 ± 0.017** | **0.938 ± 0.022** | **0.720 ± 0.094** |

H1：delta = **−0.344**，阈值 +0.10，**FAIL**。

### 解释

- 在单步遮挡 re-ID 上，学习到的 transformer 关联分数明显强于 PSS 手工设计的 `pose gate + vision/geometry cosine + category` 代价。
- PSS 在 mixed 上还有 0.800，但 samecat 掉到 0.311，near-identical 0.333，说明当前 PSS 的外观/几何特征没有实例级判别力。
- 这不是“PSS 架构整体失败”，而是**PSS 当前的 re-ID 关联机制失败**。显式 scene graph 的结构本身没有被否定，但“手工关联代价优于学习关联”这个具体主张不成立。

### 结论

- **H1 不支持。**
- 按协议，不把 P1b/H2–H5 当作 H1 已通过的延续。
- 但这不排除 PSS 在开放集、增量更新、scaling、组合泛化上仍有优势；这些是独立假设，需要单独预注册和报告。

### 下一步选择

1. **修 PSS 关联机制**：把手工 re-ID 代价换成学习到的 metric head，但保留 scene graph；再和 transformer 比一次。这是“PSS + 学习关联” vs “transformer”，不是原来的手工 PSS。
2. **换场景**：先测 P2 开放集、P3 增量更新、P4 scaling、P5 组合泛化。这些场景更可能体现显式结构的优势。
3. **P1b 完整 online tracking**：多对象、长时序、新实体出现；比单步 re-ID 更接近真实 tracking，但 transformer 也可能学到。

推荐顺序：先做 **P4 scaling** 和 **P2 开放集**，因为它们最便宜，而且能直接检验“显式结构”的非精度优势。P1a 的负结果保留，不掩盖。

### 15.1 P1c 结果：PSS + 学习度量（2026-11-23）

| 方法 | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|
| PSS hand cost | 0.561 | 0.800 | 0.311 | 0.333 |
| **PSS + learned metric** | **0.787 ± 0.006** | 0.976 | 0.871 | 0.324 |
| Transformer | 0.906 ± 0.020 | 0.982 | 0.938 | 0.720 |

H1c：delta = **−0.119**，阈值 0，**FAIL**。

解释：

- 学习度量把手写 PSS 从 0.561 拉到 0.787；mixed 0.976、samecat 0.871，说明“手工代价”确实是一个瓶颈。
- 但仍然输给 transformer：samecat 0.871 vs 0.938，near-identical 0.324 vs 0.720。
- near-identical 上 PSS+metric 仍是随机水平 0.324；transformer 0.720，很可能在利用位置/背景/光照等 nuisance cue，而不是真正的实例外观。
- P1c 的 metric 是 **pairwise cosine**，没有 set-level attention / 互斥约束；transformer 的 cross-attention 能看到整个集合，这是它仍然领先的一个结构性原因。

下一步建议：

1. **先修 benchmark 的 nuisance cue**：near-identical 随机化背景/光照/位置，或 object-only crop，再重跑 transformer。若 transformer 的 0.720 掉到 ~0.33，PSS+metric 与 transformer 的真实差距会小很多。
2. **给 PSS 加 set-level matching**：不是只比 pairwise cosine，而是用图/集合级匹配或 Hungarian-aware loss。
3. **同时做 P2/P4**：开放集和 scaling 是 PSS 更可能占优的轴。

P1c 结论保留：学习度量有效，但 pairwise metric 还不足以打败 transformer。

---

## 16. 步骤 1：参数对齐结果（2026-11-23）

把 PSS+metric 的 hidden 调到 410，参数量 335,918，transformer 336,896，基本对齐。

| 方法 | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|
| PSS + metric（param-matched） | 0.792 ± 0.004 | 0.991 | 0.867 | 0.320 |
| Transformer | 0.906 ± 0.020 | 0.982 | 0.938 | 0.720 |

H1c 仍失败，delta = −0.113。参数对齐没有改变结论：差距不是容量造成的。

---

## 17. 步骤 2：去 benchmark shortcut

### 17.1 改动

`data/tracking_bench_v2`：

- **motion 随机化**：不再固定“0/1 交换、2 平移 0.12”；每个序列随机置换目标位置并加 ±4cm jitter。
- **camera jitter**：azimuth ±10°、elevation −52°~−44°、distance 1.0~1.1。
- 仍保证 14 帧完全遮挡、遮挡帧目标像素为 0、可见帧每目标 ≥20 像素。

目的：让 transformer 不能靠“身份-运动模式”或固定机位背景残差拿分。

### 17.2 判定

- 重新训练 transformer 和 PSS+metric（param-matched）。
- 如果 PSS+metric ≥ transformer，停止。
- 如果 transformer 仍领先，进入步骤 3。

### 17.3 v2 结果（2026-11-23）

| 方法 | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|
| PSS hand cost | 0.533 | 0.722 | 0.378 | 0.311 |
| **PSS + metric（param-matched）** | **0.809 ± 0.031** | 0.991 | 0.818 | 0.436 |
| Transformer | 0.802 ± 0.008 | 0.978 | 0.867 | 0.387 |

- P1c delta = **+0.007**，按阈值 0 算 PASS，但 5 seeds 的波动约 ±0.03，**不是统计上明确的优势**。
- Transformer near-identical 从 v1 的 0.720 掉到 0.387，确认之前的高分主要来自运动/机位 shortcut。
- 去 shortcut 后，PSS+metric 与 transformer 基本打平：mixed/near PSS 略高，samecat transformer 略高。

结论：**步骤 2 没有证明 PSS 明确更好，继续步骤 3。**

---

## 18. 步骤 3：set-level matching

在 PSS metric 上增加 one-to-one assignment loss：

```text
loss = BCE(pair labels) + weight * (row CE + column CE)
```

它强制每个 pre token 选一个 post token，反之亦然，补上 pairwise cosine 缺失的互斥/集合级信号。

- 在 v2 上训练，weight 在 val 上选。
- 如果 PSS+metric+assignment ≥ transformer，停止。
- 否则进入步骤 4。

### 18.1 v2 结果（assignment weight 1.0）

| 方法 | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|
| PSS + metric | 0.809 ± 0.031 | 0.991 | 0.818 | 0.436 |
| **PSS + metric + assignment** | **0.798 ± 0.010** | 0.982 | **0.902** | 0.324 |
| Transformer | 0.802 ± 0.008 | 0.978 | 0.867 | 0.387 |

- assignment loss 把 samecat 从 0.818 提到 0.902，但 near-identical 从 0.436 掉到 0.324。
- overall delta = **−0.004**，没有明确优势。
- 结论：assignment loss 改善了同类别区分，但没有带来整体胜出，继续步骤 4。

---

## 19. 步骤 4：P2 开放集 / P4 scaling

如果步骤 1–3 都输，转 P2/P4：

- P2：原型距离/开放集拒识 vs transformer softmax/energy；
- P4：200 实体 active-set 关系计算 vs 全 token attention。

任一环节 PSS 更好，即停止并记录。

### 19.1 P4 结果（2026-11-23）

| N | PSS p50 (ms) | Transformer p50 (ms) | speedup |
|---:|---:|---:|---:|
| 4 | 0.0662 | 0.3089 | 4.7x |
| 20 | 0.0771 | 0.5417 | 7.0x |
| 50 | 0.0829 | 0.5033 | 6.1x |
| 200 | 0.1282 | 0.8229 | **6.4x** |
| 1000 | 0.2826 | 6.6767 | **23.6x** |

- PSS 在所有 N 上都比 dense full-attention transformer 快。
- 但预注册的“N=200 快 ≥10x”门槛没有达到：实际 6.4x。
- N=1000 时优势扩大到 23.6x，符合 PSS active-set O(k²) vs transformer O(N²) 的结构预期。

### 19.2 结论

- **PSS 在 compute scaling 上更好，但不是预注册的 10x 门槛。**
- 这是结构性/计算优势，不是 re-ID 精度优势。
- P4 只是部分优势；后续 P2 给出了明确的开放集能力优势，见第 20 节。

---

## 20. P2 开放集对照（2026-11-23）

数据：`data/category_pretrain`，seen categories train/val/test + 4 个 held-out categories 作为 unseen。

- PSS：原型相似度 + seen-val 校准阈值（95% seen accept）。
- Transformer：三模态 token transformer，seen classes 交叉熵训练；OOD 分数用 softmax max prob 和 energy。
- 两边同冻结特征、同 split、同 95% seen-accept 校准。
- 指标：AUROC、unseen reject、seen top-1、accepted accuracy。

### 20.1 结果

| 方法 | AUROC | unseen reject @ 95% seen accept | seen top-1 |
|---|---:|---:|---:|
| **PSS prototypes** | **0.993** | **0.958** | 0.979 |
| Transformer softmax | 0.886 ± 0.025 | 0.445 | 0.978 |
| Transformer energy | 0.912 ± 0.034 | 0.626 | 0.978 |

- H2 PASS：PSS AUROC 比 energy 高 **+0.081**，比 softmax 高 **+0.107**。
- Unseen reject：PSS 0.958，比 energy 高 **+33.2 个百分点**，比 softmax 高 **+51.4 个百分点**。
- Seen top-1 基本一致：0.979 vs 0.978。
- 两边阈值都在 seen val 上校准到 95% accept。

### 20.2 结论

- **这是目前第一个明确的能力优势：PSS 的原型距离/开放集机制在 unseen 拒识上明显优于 transformer softmax/energy。**
- 在 seen 类别精度相同的前提下，PSS 的 unseen reject 高 33–51 个百分点。
- 这支持“PSS 在开放集场景下优于 transformer”的限定主张。
- 按用户“任一环节证明 PSS 更好就停”的规则，停在这里；P3/P5 未跑。

局限：

- 合成 category_pretrain 数据，unseen 是 4 个 held-out 类别；
- PSS 原型库由 seen 数据构建，transformer 也在 seen 数据上训练；
- 阈值在 seen val 上校准，真实 open-set 分布会不同；
- 未接真实数据 / R-Gate。

---

## 21. 模态冲突实验（2026-11-23）

检验“多模态融合是否会被冲突模态拖累”。

条件：

- clean；
- tactile shuffle / noise 0.5 / noise 1.0 / zero；
- vision shuffle / zero；
- geometry shuffle / zero。

方法：

- PSS fixed：固定加权融合，始终输出 top-1；
- PSS abstain：vision/geometry top-1 冲突时拒识；
- PSS drop：冲突时丢掉与融合 top-1 不一致的模态，再融合；
- Transformer：clean 数据训练，测试时输入同一路被破坏。

指标：seen top-1；PSS abstain 的 coverage / accepted accuracy。

### 21.1 结果

| 条件 | PSS fixed | PSS abstain (cov/acc) | PSS drop | Transformer |
|---|---:|---:|---:|---:|
| clean | 0.979 | 0.85/0.982 | 0.979 | 0.978 ± 0.008 |
| tactile shuffle | 0.974 | 0.85/0.982 | 0.974 | 0.959 ± 0.011 |
| tactile noise 0.5 | 0.979 | 0.85/0.982 | 0.979 | 0.980 ± 0.006 |
| tactile noise 1.0 | 0.979 | 0.85/0.982 | 0.979 | 0.977 ± 0.008 |
| tactile zero | 0.979 | 0.85/0.982 | 0.979 | 0.968 ± 0.013 |
| vision shuffle | 0.474 | 0.08/1.000 | 0.474 | 0.156 ± 0.019 |
| geometry shuffle | 0.458 | 0.07/0.923 | 0.458 | **0.749 ± 0.032** |
| vision zero | 0.865 | 1.00/0.865 | 0.865 | 0.625 ± 0.105 |
| geometry zero | 0.979 | 1.00/0.979 | 0.979 | 0.936 ± 0.030 |

结论：

- **tactile 冲突几乎不影响 PSS**（0.974–0.979），transformer 略降（0.959–0.980）。tactile 权重只有 0.5，且本身类别信号弱。
- **vision shuffle**：两边都崩，但 PSS 0.474 仍高于 transformer 0.156。PSS conflict-aware abstain 只回答 8% 样本，accepted accuracy 1.000。
- **geometry shuffle**：transformer 更稳（0.749 vs 0.458），说明 PSS 固定融合会被高权重 geometry 冲突拖累。
- **vision zero**：PSS 0.865 vs transformer 0.625，PSS 更稳。
- **geometry zero**：PSS 0.979 vs transformer 0.936。
- `drop` 规则没有改善，因为它是按“融合 top-1”决定丢哪一路；融合已经错时，drop 也会保留错误模态。

### 21.2 结论

- 用户的直觉部分成立：**PSS 的固定加权融合会被高权重模态冲突拖累**，geometry shuffle 是明确例子。
- 但不能说 transformer 更会融合：vision shuffle 下 transformer 崩得更狠（0.156），vision zero 下也低于 PSS。
- **PSS 的独特优势是冲突检测 + 弃权**：vision/geometry shuffle 下 coverage 降到 7–8%，accepted accuracy 仍有 0.92–1.00；transformer 没有显式冲突标志，仍会输出错误答案。
- 所以更准确的主张是：**PSS 不保证融合后精度更高，但它能知道自己被打架了，并选择拒识；transformer 的注意力融合在部分冲突下更稳，但没有显式弃权机制。**

---

## 22. P3 增量类别更新（2026-11-23）

数据：16 个 seen 类别；新类别 `cloth`（held-out），96 条，按 60/36 分 new-train/new-test。

- PSS：base prototype package v1.0 → append cloth → v1.1 → activate → rollback。
- Transformer：
  - last-layer fine-tune：冻结 encoder，扩展分类头，只训练新类那行；
  - full fine-tune：全模型低 LR 微调。
- 两边同冻结特征、同 new-train/new-test、同 old test。
- 指标：old before/after、new after、forgetting、update time、rollback。

判定：PSS forgetting 比最好的 transformer 低 ≥5 个百分点，或 update time 快 ≥10 倍。

### 22.1 结果

| 方法 | old before | old after | new after | forgetting | update time |
|---|---:|---:|---:|---:|---:|
| **PSS prototype add** | 0.979 | **0.979** | **1.000** | **0.000** | **43.94 ms** |
| Transformer last-layer | 0.978 | 0.883 ± 0.037 | 0.985 ± 0.021 | 0.095 | 1058.9 ms |
| Transformer full | 0.978 | 0.816 ± 0.098 | 1.000 | 0.163 | 3277.9 ms |
| Transformer replay | 0.978 | **0.979 ± 0.003** | **1.000** | **-0.001** | 3767.7 ms |

- H3 forgetting：PSS 0.000 vs best transformer（replay）-0.001，gap **-0.001**，**FAIL**。
- H3 time：PSS 快 **24.1x**，PASS。
- PSS rollback：v1.0，`cloth` 消失，旧类别保留。

### 22.2 结论

- **标准 fine-tune 下 PSS 优势明显**：last-layer 掉 9.5 点，full 掉 16.3 点；PSS 不掉点、新类 1.000、可回滚。
- **但 replay 基线追平了遗忘**：old 0.979、new 1.000、forgetting -0.001，H3 forgetting **FAIL**。
- 因此 PSS 在增量更新上的优势不是“唯一能不忘”，而是：
  - 更新快 **24.1x**（43.94 ms vs 3767.7 ms）；
  - 不需要保留旧数据；
  - 原型包可签名、可审计、可回滚。
- 代价是 replay 需要旧数据存储和更长更新。

局限：

- 合成 `cloth`，新类与旧类可分性较好；
- transformer 没做 replay/distillation/EWC，这些会减少遗忘但增加复杂度；
- PSS 新类原型依赖新类样本质量，异常样本会直接进原型。

---

## 23. P0 优化结果（2026-11-23）
### 23.1 Set-level graph matcher

数据：`data/tracking_bench_v2`，200 train / 40 val / 60 test。

模型：在 PSS token 上增加 within-set 的 pose/category 边，消息传递后再做 pair scoring；参数量 **169,089**，比 transformer 的 336,896 更小。

| 方法 | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|
| **PSS + graph matcher** | **0.930 ± 0.029** | **1.000** | **0.982** | **0.738** |
| PSS + learned metric | 0.809 | 0.991 | 0.818 | 0.436 |
| Transformer | 0.802 ± 0.008 | 0.978 | 0.867 | 0.387 |

**H0 PASS（历史口径）**：graph matcher 比**无 tactile transformer** 高 **+0.128**。

> **公平归因修正（见 §38）**：给 transformer 同样的 tactile token + 同样的 Hungarian 后，transformer 到 **0.977**，PSS 0.989，delta 只剩 **+0.012**。  
> 因此 H0 的 +0.128 不能归因给 graph 架构；它主要是 tactile 输入差异。

结论：

- PSS 的 scene graph 结构 + 集合级匹配，确实能超过 transformer；
- 之前 pairwise metric 是瓶颈，不是 PSS 架构本身；
- graph matcher 参数更少，却明显更好；
- near-identical 0.738 仍可能利用 pose/排列关系，需要后续加实例纹理消融。

### 23.2 Conflict-aware fusion

新增 `pss/prototypes/conflict.py`：

- 每路模态算 top-1、score、top-2 margin；
- high-weight 模态（vision/geometry）必须可靠且一致才回答；
- 只有一路 high-weight 存在（另一路 zero/缺失）时，用可靠的那一路；
- high-weight 不可靠或冲突时 abstain。

dev 结果：

| 条件 | reliable coverage | accepted accuracy |
|---|---:|---:|
| clean | 0.740 | 0.993 |
| tactile shuffle | 0.740 | 0.993 |
| vision shuffle | 0.073 | 1.000 |
| geometry shuffle | 0.052 | 1.000 |
| vision zero | 0.818 | 0.904 |
| geometry zero | 0.974 | 0.989 |

完整 5 seeds 结果（PSS reliable 列）：

| 条件 | PSS reliable cov/acc |
|---|---:|
| clean | 0.74 / 0.993 |
| tactile shuffle | 0.74 / 0.993 |
| tactile noise / zero | 0.74 / 0.993 |
| vision shuffle | 0.07 / 1.000 |
| geometry shuffle | 0.05 / 1.000 |
| vision zero | 0.82 / 0.904 |
| geometry zero | 0.97 / 0.989 |

结论：这套规则在冲突时几乎不输出错误答案，代价是 coverage 下降；比旧的“按融合 top-1 drop”规则更合理。vision/geometry shuffle 下 coverage 只有 5–7%，但 accepted accuracy 为 1.000；zero 情况下仍能靠另一路高权重模态给出 0.90–0.99 的接受精度。

### 23.3 Instance-level features

- graph matcher 的 near-identical 0.738 说明集合级关系有帮助，但还不是真正的实例纹理。
- 还没有把 tactile instance texture 加进 token；这是下一步。
- 当前实例级身份仍受限于 vision/geometry 特征。

### 23.4 P1b 完整 online tracking

数据：`data/tracking_bench_v2`，60 条 test sequences，24 帧，模拟 SLAM 噪声。

| 方法 | ID retention | switches | reappearance |
|---|---:|---:|---:|
| Hand PSS tracker | 0.919 | 0.16 | 0.839 |
| **PSS + graph matcher** | **0.986** | **0.03** | **0.972** |
| Transformer re-ID | 0.908 | 0.18 | 0.817 |

结论：

- 完整 online tracking 上，PSS graph matcher 的系统级结果更好：ID retention 0.986 vs 0.908，reappearance 0.972 vs 0.817。
- 但 transformer 基线仍无 tactile；这是系统级对照，不是公平架构归因。
- **公平归因（§38）后，re-ID 精度增量只有 +0.012；不能说集合级图结构是关键。**
- P1b tracking 仍待补 transformer+tactile 基线，才能判断 tracking 层的架构增量。

---

## 24. P0.3 / P5 实验（2026-11-23）

### 24.1 P0.3 tactile-augmented graph matcher

- 在 graph matcher token 后追加 tactile 128 维，token dim 279 → 407；
- 其他结构不变；
- 目标：看 tactile instance texture 能否改善 samecat / near-identical；
- 对照：无 tactile 的 graph matcher、transformer。

### 24.2 P5 compositional attribute OOD

- 数据：`category_pretrain`，同一批 seen 类别；
- 训练：只用 low-hardness 样本；
- ID 测试：low-hardness test；
- OOD 测试：high-hardness 样本；
- PSS：low-hardness 原型库；
- Transformer：low-hardness 训练；
- 指标：ID accuracy、OOD accuracy；
- H5：PSS OOD 比 transformer 高 ≥10 个百分点。

### 24.3 P0.3 结果

| 方法 | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|
| **PSS + graph + tactile** | **0.989 ± 0.017** | **1.000** | **1.000** | **0.956** |
| PSS + graph（vision/geometry） | 0.930 | 1.000 | 0.982 | 0.738 |
| Transformer | 0.802 | 0.978 | 0.867 | 0.387 |

- 加 tactile 后 overall 从 0.930 提到 **0.989**，near-identical 从 0.738 提到 **0.956**。
- 这说明 tactile instance texture 确实是实例级身份信号；之前 near-identical 的上限不是架构问题，而是缺了实例级触觉。
- vs transformer delta = **+0.187**。
- P1b 完整 tracking 用同一 tactile graph：ID retention **1.000**、switches **0.00**、reappearance **1.000**（base graph 0.986/0.03/0.972，transformer 0.908/0.18/0.817）。

### 24.4 P5 结果

| 方法 | ID accuracy | OOD accuracy |
|---|---:|---:|
| PSS prototypes | 0.975 | **0.978** |
| Transformer | 0.942 | 0.967 ± 0.006 |

- H5 delta **+0.011**（门槛 +0.10）→ **FAIL**。
- PSS OOD 略高，ID 高 3.3 点，但不足以证明组合属性 OOD 优势。
- 注意：早期 dev 单 seed 得到的 transformer 0.917 在全量 5 seed 下不成立（全量 0.967）；以本表为准。
- 这不是真正的“物体-关系-动作”新组合，只是高 hardness / 材质 regime OOD；关系组合泛化仍待测。

### 24.5 P1.3 planning 结果

**设计**：同一张 5 路点关系图，同一组 clear edge mask。PSS 侧用显式 Operator + A*；transformer 侧是 5 个 region token + edge mask 的 2 层 encoder，用 A* 轨迹做行为克隆。训练/ID 只有 1 个障碍物，OOD 有 2 个；dynamic split 在第一步后移动障碍物。指标：任务成功率、成功步数、invalid action rate。

| split | PSS success | PSS steps | Transformer success | Transformer steps | invalid action rate |
|---|---:|---:|---:|---:|---:|
| ID（1 obstacle） | **1.000** | 2.07 | 0.990 ± 0.000 | 2.07 | 0.000 |
| OOD（2 obstacles） | **1.000** | 2.12 | 0.924 ± 0.016 | 2.10 | 0.000 |
| dynamic ID | **1.000** | 2.15 | 0.984 ± 0.005 | 2.13 | 0.010 |
| dynamic OOD | **1.000** | 2.26 | 0.928 ± 0.015 | 2.22 | 0.056 |

- Transformer 在训练分布上 next-action accuracy **1.000**，所以 OOD 掉点不是欠拟合。
- OOD 上 PSS 高 **+7.6 点**；dynamic OOD 高 **+7.2 点**；同时 transformer 出现 5.6% 的 invalid action。
- 差距存在且方差很小，但幅度不大；PSS 的 1.000 来自 A* 完备性，不是学习能力。


---

## 25. P0-1 DINOv2 instance token（2026-11-23）

### 25.1 目的

当前最强的 re-ID 结果依赖 tactile：graph+tactile 0.989，但去掉 tactile 后 near-identical 只有 0.738。本实验用冻结 DINOv2 ViT-S/14 的 CLS token 作为第二个实例级视觉信号，检验能否在不依赖 tactile 的情况下补上 near-identical。

### 25.2 设计

- DINOv2 只进 re-ID token，不进原型库，不改变类别识别和开放集行为；
- 输入：PSS 的 224×224 物体 crop，ImageNet 归一化；
- 特征：冻结 DINOv2-small 的 CLS，384 维，L2 归一化；
- 四组 token：
  1. `base`：vision projection + geometry + pose + category（279）；
  2. `dino`：base + DINOv2（663）；
  3. `tactile`：base + tactile（407）；
  4. `both`：base + tactile + DINOv2（791）。
- 每组单独训练 graph matcher，5 seeds，200 epochs；
- missing-modality：在 `both` 模型上分别 zero tactile / dino / vision / tactile+dino。

### 25.3 原始 384 维 CLS 结果

| variant | token | overall | mixed | samecat | near-identical | delta vs base |
|---|---:|---:|---:|---:|---:|---:|
| base | 279 | **0.930 ± 0.029** | **1.000** | **0.982** | **0.738** | +0.000 |
| dino | 663 | 0.903 ± 0.032 | 0.987 | 0.982 | 0.658 | -0.027 |
| tactile | 407 | **0.989 ± 0.017** | **1.000** | **1.000** | **0.956** | +0.059 |
| both | 791 | 0.967 ± 0.024 | 0.996 | 1.000 | 0.876 | +0.037 |
| Transformer | - | 0.802 ± 0.008 | 0.978 | 0.867 | 0.387 | - |

- 原始 384 维 DINO CLS 不仅没有补上 near-identical，反而让 overall 从 0.930 掉到 0.903，both 也低于 tactile-only。
- 诊断：DINO CLS 的 same/diff 余弦 gap 是 +0.128（tactile +0.216），信号存在，但 384 维高维噪声让 graph matcher 过拟合。
- `both` 模型 zero 掉 DINO 后 overall 从 0.922 升到 0.978，说明模型确实被 DINO 噪声拖累。

### 25.4 PCA 压缩后的 DINO token（探索性）

在 train split 的 DINO CLS 上拟合 PCA，取 32 维，再拼进 token：

| variant | token | overall | near-identical |
|---|---:|---:|---:|
| base | 279 | 0.930 ± 0.029 | 0.738 |
| tactile | 407 | **0.989 ± 0.017** | **0.956** |
| dino_pca32 | 311 | 0.942 ± 0.030 | 0.804 |
| both_pca32 | 439 | 0.987 ± 0.011 | 0.964 |

- PCA-32 把 DINO-only 从 0.903/0.658 提到 **0.942/0.804**，说明问题主要是高维过拟合。
- 但仍低于 tactile-only，**没有达到“无 tactile 时 near ≥0.90”的验收线**。
- both_pca32 与 tactile-only 打平，DINO 没有带来超过 tactile 的增益。

### 25.5 结论

**P0-1 未达标。** DINOv2 CLS 是真实但较弱的实例信号；在当前合成 benchmark 上不能替代 tactile。代码保留为可选路径（`dino=True`），默认不启用，不改变现有主线结果。

后续若要提高 DINO 的贡献，优先试：

1. 更高分辨率的物体渲染（当前 v2 只有 96×96，upscale 到 224 丢失细节）；
2. DINO patch token 的局部描述子，而不是全局 CLS；
3. 把 DINO 余弦作为 pair edge feature，而不是高维 token；
4. 在 v3 上按 validation 选 PCA 维度和是否启用。


---

## 26. P0-3 graph matcher scaling（2026-11-23）

### 26.1 目的

P4 的 23.6x 只测了原型识别，不是 graph matcher。本实验测关联层：dense graph matcher、候选剪枝 + 分量匹配、dense transformer 在 N=50/200/1000 的延迟和 retention。

### 26.2 设计

- 合成 N 个 track / observation 真配对，位置随 N 扩展，局部密度近似恒定；
- **dense graph**：现有 `GraphMatcherReID.cost_matrix` + 全局 Hungarian；
- **pruned graph**：`PrunedGraphMatcherReID`
  - pose gate 0.08（合成）/ 0.5（v2）；
  - category gate；
  - appearance top-16；
  - 候选图连通分量内跑 GraphMatcher + 分量 Hungarian；
  - 单点分量合并成一个 B×1×1 batch；
- **transformer**：`EntityTransformer` + `TransformerReID`；
- 指标：p50/p95、pruned vs transformer speedup、true retention、dense 相对 retention；
- v2 上比较 dense vs pruned 的 ID retention。

### 26.3 结果

| N | dense p50/p95 ms | pruned p50/p95 ms | transformer p50/p95 ms | pruned vs transformer | pruned true retention | relative vs dense |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 2.99 / 3.77 | 2.63 / 4.77 | 4.94 / 7.13 | 1.88x | 1.000 | 1.000 |
| 200 | 41.70 / 54.82 | 10.37 / 16.90 | 36.66 / 38.62 | 3.53x | 0.970 | 1.000 |
| 1000 | n/a | 34.99 / 43.14 | 852.19 / 917.49 | **24.35x** | **0.984** | n/a |

v2 retention：

| matcher | ID retention | switches | reappearance |
|---|---:|---:|---:|
| dense graph | 0.986 | 0.03 | 0.972 |
| pruned graph | 0.978 | 0.04 | 0.956 |

relative ID retention = 0.992。

### 26.4 结论

- **N=1000 达标**：24.35x > 20x，true retention 0.984。
- **N=200 未达 5x**：3.53x，是 near miss；pruned 10.37 ms vs transformer 36.66 ms。
- v2 相对 retention 0.992 > 0.98；绝对 0.978，差 0.002。
- 在拥挤合成场景里，dense 全局 Hungarian 会被大量 foil 干扰，true retention 反而低于 pruned；因此 `relative vs dense` 封顶 1.0。
- 剪枝的收益来自把 O(N²) pair tensor 换成局部候选图；代价是候选 miss 和分量 LAP 的 0.8% 相对损失。


---

## 27. P0-2 cross-benchmark v3（2026-11-23）

### 27.1 目的

防止“只在自建 v2 上调出来”的质疑。v3 增加对象数、遮挡长度和渲染尺寸，并保留随机运动/机位/光照。

### 27.2 设计

- v3：5 objects，40 frames，30-frame occlusion，160×160，随机运动 + 相机抖动；
- 训练 120 / val 30 / test 60（每 split 按 mixed 50% / samecat 25% / near 25%）；
- 对比三种训练→测试：
  1. v2→v2；
  2. v2→v3（不重训，测跨 benchmark）；
  3. v3→v3（v3 上的 in-domain 上限）；
- 每种都跑 PSS graph matcher 和 entity transformer，5 seeds，200 epochs；
- v3 missing-modality：zero vision / geometry，测 PSS 退化。

### 27.3 结果

| train -> test | PSS graph | transformer | delta |
|---|---:|---:|---:|
| v2 -> v2 | 0.930 ± 0.029 | 0.802 ± 0.008 | +0.128 |
| v2 -> v3 | **0.881 ± 0.029** | 0.672 ± 0.022 | **+0.209** |
| v3 -> v3 | 0.847 ± 0.040 | 0.711 ± 0.014 | +0.136 |

v3 missing-modality（PSS graph trained on v3, seed 0）：

| condition | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|
| full | 0.893 | 1.000 | 0.840 | 0.733 |
| no_vision | 0.863 | 1.000 | 0.733 | 0.720 |
| no_geometry | 0.840 | 1.000 | 0.787 | 0.573 |

- v2→v3 上 PSS 只掉 4.9 点（0.930→0.881），transformer 掉 13.0 点（0.802→0.672）；PSS 的跨 benchmark 鲁棒性更强。
- v3→v3 反而低于 v2→v3（0.847），主要因为 v3 train 只有 120 条、且分布更难；这不能解释成“v3 训练没用”，而是训练量不足。
- 缺失 geometry 比缺失 vision 更伤（0.840 vs 0.863），near-identical 掉到 0.573。

---

## 28. P0-4 stronger transformer baselines（2026-11-23）

### 28.1 设计

- base：现有 entity-token transformer（vision+geometry+pose+category）；
- DINO：同样 transformer head，但 token 追加冻结 DINOv2-small CLS（384 维）；
- 同数据、同 split、同 seeds/epochs；
- 报告参数量、overall/mixed/samecat/near。

### 28.2 结果

| baseline | token dim | params | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|---:|---:|
| base | 279 | 336,896 | **0.802 ± 0.008** | **0.978** | **0.867** | **0.387** |
| DINO | 663 | 435,200 | 0.752 ± 0.037 | 0.964 | 0.769 | 0.311 |

- 加冻结 DINOv2 CLS 后 transformer 反而掉 5 个点；near-identical 从 0.387 掉到 0.311。
- 说明“换成预训练特征”本身不自动变成更强基线；DINO 在 96×96 合成 crop 上的噪声会伤害 transformer。
- 但这不等于所有强基线都失败；sparse/retrieval attention 和 EWC 仍待测。

增量更新的强基线 replay（见 §22）：

| 方法 | old after | new after | forgetting | update time |
|---|---:|---:|---:|---:|
| PSS prototype add | 0.979 | 1.000 | 0.000 | 43.94 ms |
| Transformer replay | **0.979** | **1.000** | **-0.001** | 3767.7 ms |

- replay 追平零遗忘；P3 的 H3 forgetting 由 PASS 变为 **FAIL**。
- PSS 仍保留 24.1x 更新速度和回滚能力。

---

## 29. P2-12 消融与统计（2026-11-23）

### 29.1 设计

在 tracking_bench_v2 上做组件消融：

- full：edges + assignment-aware loss + Hungarian；
- no_edges：去掉集合内边消息；
- no_assignment：assignment_weight=0；
- greedy：用 score-greedy 替代 Hungarian；
- +tactile：引用 P0.3 结果；
- paired bootstrap：full graph vs transformer，10k 重采样，95% CI。

### 29.2 结果

| variant | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|
| full（edges + assignment + Hungarian） | 0.930 ± 0.029 | 1.000 | 0.982 | 0.738 |
| no_edges | 0.941 ± 0.011 | 1.000 | 1.000 | 0.764 |
| no_assignment | **0.969 ± 0.033** | 1.000 | 0.973 | **0.902** |
| greedy（full 模型，贪心匹配） | 0.967 | 1.000 | 1.000 | 0.867 |
| transformer | 0.802 ± 0.008 | 0.978 | 0.867 | 0.387 |
| + tactile（引用 P0.3） | **0.989 ± 0.017** | 1.000 | 1.000 | **0.956** |

Paired bootstrap（full vs transformer，300 个配对观测）：

- mean delta **+0.128**；
- 95% CI **[+0.096, +0.162]**；
- P(delta <= 0) **0.0000**。

结论：

- PSS graph 对 transformer 的优势统计显著，不是 seed 噪声。
- 但 full 配置不是最优：去掉 assignment-aware loss 到 **0.969**，贪心匹配 0.967，去边 0.941。
- 说明当前 `assignment_loss` 和 edge message 在 v2 上过拟合；集合级 pair scoring 本身才是主要收益，tactile 是最大单项增益。
- 参数量：graph 169k vs transformer 337k。


---

## 30. P1-6 学习型 reliability（2026-11-23）

目标：用一个小 gate 从三路 score vector 学习模态权重，替代固定权重/规则可靠度。

- 输入：三路 prototype score vector + top/margin/present 统计；
- 训练：clean/shuffle/noise/zero 分数扰动；
- 测试：与模态冲突实验相同的 8 个条件；
- 对照：PSS fixed、PSS reliable rule。

### 30.1 结果

| condition | PSS fixed | PSS reliable (cov/acc) | learned (cov/acc) |
|---|---:|---:|---:|
| clean | **0.979** | 0.74/0.993 | 0.96/0.935 |
| tactile shuffle | 0.974 | 0.74/0.993 | 0.96/0.936 |
| vision shuffle | 0.438 | 0.02/1.000 | 0.53/0.792 |
| geometry shuffle | 0.453 | 0.07/1.000 | 0.59/0.245 |
| vision zero | 0.865 | 0.82/0.904 | 0.98/0.853 |
| geometry zero | **0.979** | 0.97/0.989 | 0.82/0.982 |

- 学到的 gate 在 clean 上 0.96 coverage / 0.935 accepted，低于 fixed 0.979 和 reliable 0.993。
- vision/geometry shuffle 下 coverage 0.53–0.59，accepted accuracy 0.792/0.245；geometry shuffle 仍明显低于 0.70 目标。
- vision/geometry zero 下 gate 明显改善：0.98/0.853 和 0.82/0.982；presence flag 起作用。
- **P1-6 未达标**：没有达到“geometry shuffle ≥0.70 且 coverage ≥0.80”的目标。
- 原因：gate 从 score vector 学模态可靠性，但 zero/shuffle 的分布与 clean 差异大，训练分布覆盖不足；规则 reliable 在 zero 条件下仍更稳。

---

## 31. P1-7 conformal 开放集（2026-11-23）

目标：把开放集阈值从经验分位数升级为 split-conformal 下尾阈值，并检查 class-conditional coverage。

- 校准：val seen scores；
- marginal conformal：alpha=0.05 下尾分位数；
- class-conditional：每个类别单独校准；
- 指标：seen accept、unseen reject、AUROC、min class coverage。

### 31.1 结果

| method | threshold | seen accept | unseen reject | AUROC | min class coverage |
|---|---:|---:|---:|---:|---:|
| marginal conformal | 0.731 | **0.969** | **0.958** | **0.993** | - |
| class-conditional | per-class | 0.859 | 0.958 | - | **0.333** |

- Marginal conformal 在 95% 名义风险下达到 96.9% seen accept / 95.8% unseen reject。
- Class-conditional 把平均 coverage 拉到 0.859，但最小类 coverage 只有 **0.333**；每类 val 样本太少，阈值不稳定。
- 结论：当前数据量下用 marginal conformal；class-conditional 需要更多 per-class 校准样本。

---

## 32. P1-8 原型治理（2026-11-23）

目标：新类样本被污染时，比较 mean / trimmed / medoid / median 原型的 old/new accuracy、污染标记率和回滚。

- 污染：从旧类 train 抽 10%/30% 特征冒充新类样本；
- 旧原型不动；
- 指标：old accuracy、new accuracy、poison flagged、variance、rollback time。

### 32.1 结果

| poison | prototype | old acc | new acc | poison flagged | variance |
|---:|---|---:|---:|---:|---:|
| 0% | mean | 0.979 | 1.000 | 0.000 | - |
| 10% | mean | 0.979 | 1.000 | 0.967 | - |
| 30% | mean | 0.979 | 1.000 | 0.988 | - |
| 30% | trimmed | 0.979 | 1.000 | 0.973 | - |
| 30% | medoid | 0.979 | **0.728** | 0.835 | - |
| 30% | median | 0.979 | 1.000 | 0.973 | - |

- 旧类始终 0.979：旧原型未被触碰。
- 10–30% 污染下 mean/trimmed/median 的新类仍是 1.000；污染样本标记率 0.97–0.99。
- medoid 明显更差（30% 污染时 0.728），不适合作为新类原型。
- 回滚时间 **29.5 ms**。
- 结论：当前污染是“远离类中心”的，均值已经够稳；真正需要的是**污染检测/标记**，不是换原型估计器。

---

## 33. P1-9 闭环（2026-11-23）

目标：在 MuJoCo 视觉闭环里测动态障碍重规划。当前环境没有 PyBullet，使用现有 MuJoCo 闭环。

- 3 seeds，anomaly on/off；
- 指标：success、pushes、replans、final distance。

### 33.1 结果

| condition | success | pushes | replans | final distance |
|---|---:|---:|---:|---:|
| anomaly on（3 seeds） | **3/3** | 32 / 45 / 35（mean 37.3） | 1 / 2 / 1（mean 1.33） | 0.0446–0.0487 m |
| anomaly off（3 seeds） | **3/3** | 17 / 18 / 18（mean 17.7） | 0 | 0.0431–0.0598 m |

- 动态障碍触发 1–2 次重规划，额外约 20 次 push，全部最终成功。
- 最终距离仍在 4.5–6.0 cm，和预注册 6 cm 阈值一致。
- 这是 PSS 显式闭环的恢复能力证据；但只有 3 seeds、2 物体场景，且环境是 MuJoCo，不是 PyBullet。

---

## 34. P1-5 / P5b 真组合关系（2026-11-23）

目标：训练见过的关系对，测试没见过的关系对，检验显式几何目标检查 vs 行为克隆策略。

- 3 objects、5 regions；
- goal：object0 at G，object1 R1 object0，object2 R2 object1；
- 训练关系对：(left,above)、(right,below)、(above,left)、(below,right)；
- 测试关系对：(left,below)、(right,above)、(above,right)、(below,left)；
- 每个关系单独见过，但 pair 没见过；
- PSS：显式几何 goal check + BFS；
- transformer：object/region/relation token + BC。

### 34.1 结果

| method | ID success | ID steps | OOD success | OOD steps | OOD invalid |
|---|---:|---:|---:|---:|---:|
| PSS explicit planner | **1.000** | 2.66 | **1.000** | 2.66 | 0.000 |
| Transformer BC | 0.997 ± 0.003 | 2.62 | **0.434 ± 0.216** | 2.47 | 0.139 |

- 训练 841 条 planner demonstration；
- ID 上 transformer 几乎打平（0.997 vs 1.000）；
- 未见关系对 OOD 上掉到 **0.434**，并出现 13.9% invalid action；
- PSS 显式几何 goal check 在 OOD 上仍 1.000。

**这是目前最清楚的“显式关系结构带来组合泛化”证据。**


---

## 35. P2-10 世界模型 rollout（2026-11-23）

目标：把多实体 rollout 从 0.0965 m 压到 5 cm 以下。

- 同一 per-entity set MLP；
- 对照：one-step MSE 训练；
- 实验：额外 multi-step rollout loss，训练模型在自己 rollout 到的状态上预测 delta；
- 数据：`data/phase7_v2_more`，3 seeds，300 epochs。

### 35.1 结果

| variant | 1-step pos error | rollout all | rollout target |
|---|---:|---:|---:|
| one_step | 0.0240 ± 0.0008 | 0.0932 ± 0.0026 | 0.1001 ± 0.0185 |
| rollout_aware | 0.0217 ± 0.0014 | **0.0863 ± 0.0070** | 0.1007 ± 0.0197 |

- rollout loss 把全实体 rollout 从 0.0932 降到 **0.0863**，约 7%；one-step 也从 0.0240 到 0.0217。
- 但目标 <0.05 m **未达到**；target rollout 仍 0.1007。
- 说明当前 4 实体 set MLP + 60 episodes 的数据量下，误差主要不是训练目标，而是模型容量/数据覆盖。

---

## 36. P2-11 长时记忆（2026-11-23）

目标：把遮挡从 14 帧拉到 70 帧，测长时 re-ID。

- 新数据：`data/tracking_bench_long`，80 帧，5 visible head + 70 遮挡 + 5 tail；
- 用 v2 训练的 graph+tactile 模型，不重训；
- 对照 transformer re-ID（v2 训练）；
- tracker re-ID age window 150 帧。

### 36.1 结果

| method | ID retention | switches | reappearance |
|---|---:|---:|---:|
| PSS graph+tactile | **0.983** | **0.03** | **0.967** |
| Transformer re-ID | 0.900 | 0.20 | 0.800 |

| subset | PSS retention | TR retention |
|---|---:|---:|
| mixed | 1.000 | 1.000 |
| samecat | 1.000 | 0.933 |
| near-identical | **0.933** | **0.667** |

- 70 帧完全遮挡后，PSS 仍保持 0.983 retention / 0.967 reappearance；transformer 0.900 / 0.800。
- near-identical 上差距最大：0.933 vs 0.667。
- 这说明 tactile 实例信号在长时记忆里比短时更关键。

---

## 37. P2-13 工程化（2026-11-23）

目标：测端到端每帧延迟、模型加载、内存。

- `extract_frame` + tracker + graph matching；
- dense vs pruned graph；
- p50/p95、RSS、加载时间。

### 37.1 结果

- graph load 24.9 ms，perception load 23.7 ms，RSS 927 MB；
- dense：perception 0.49/14.13 ms，tracking 0.02/0.72 ms，total 0.51/15.33 ms；
- pruned：perception 0.47/13.24 ms，tracking 0.02/0.63 ms，total 0.49/14.25 ms。

- p50 0.5 ms，p95 14–15 ms；p95 主要由首帧/模型预热和偶发长尾主导。
- pruned 在 3 实体场景只快 0.02 ms，符合预期：v2 太小，剪枝收益在 N=1000 才体现。


---

## 38. 公平归因：同一 tactile token，同一 Hungarian（2026-11-23）

目标：把 transformer 基线升级到与 PSS 相同的输入和匹配口径。

- base：PSS graph vs transformer，同一 base token（279）；
- tactile：PSS graph vs transformer，同一 base+tactile token（407）；
- 两者都用 Hungarian assignment 评估；
- 同数据、同 split、同 seeds/epochs。

### 38.1 结果

| token set | PSS graph | transformer | delta |
|---|---:|---:|---:|
| base | 0.930 ± 0.029 | 0.802 ± 0.008 | **+0.128** |
| tactile | 0.989 ± 0.017 | **0.977 ± 0.017** | **+0.012** |

- 给 transformer 同样的 tactile token 后，它从 0.802 直接到 **0.977**；PSS 只高 **+0.012**，在约 1 个标准差内。
- 结论：**re-ID 的大部分优势来自 tactile 输入，而不是 PSS 架构本身。**
- 之前 +0.128 是“PSS+tactile vs transformer 无 tactile”的不公平对比；公平对比下架构增益很小。
- 这推翻了“graph matcher 架构带来 re-ID 优势”的强说法。PSS 仍保留集合级匹配/显式记忆的工程与可解释价值，但精度优势主要是输入级。

---

## 39. no_assignment 主配置复跑（2026-11-23）

目标：P2-12 发现 no_assignment 0.969 > full 0.930，验证它是否能作为主配置。

- 数据：tracking_bench_v2；
- 配置：graph+tactile，`assignment_weight=0`，5 seeds，200 epochs；
- 评估 re-ID，保存模型供 tracking 复用。

### 39.1 结果

| 配置 | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|
| base + assignment | 0.930 ± 0.029 | 1.000 | 0.982 | 0.738 |
| base + no_assignment | 0.969 ± 0.033 | 1.000 | 0.973 | 0.902 |
| tactile + assignment | **0.989 ± 0.017** | 1.000 | 1.000 | **0.956** |
| tactile + no_assignment | 0.976 ± 0.024 | 1.000 | 1.000 | 0.902 |

- no_assignment 在 base 上更好（0.969 vs 0.930），但在 tactile 上反而更差（0.976 vs 0.989）。
- 说明 assignment loss 与 tactile 有交互，不是“统一去掉就更好”。
- **主配置仍应保留 tactile + assignment**；no_assignment 只作为 base 消融/备选。
- P1b tracking：noassign tactile retention **0.994 / 0.01 / 0.989**，低于 full tactile 的 1.000 / 0.00 / 1.000。

---

## 40. P5b 感知版（2026-11-23）

目标：把关系组合从纯符号推进到“渲染 → tracker → 区域状态 → 规划”。

- 3 objects，5 个桌面区域；
- 渲染 + 感知 + tracker，映射到区域状态；
- PSS：显式关系 goal check + BFS；
- transformer：同一 perceived state 上的 BC policy；
- 测试 seen / held-out relation pairs；
- 当前只评估规划，不执行 push。

### 40.1 结果

| method | ID success | OOD success | OOD invalid | perception failure |
|---|---:|---:|---:|---:|
| PSS explicit planner | 1.000 | **1.000** | 0.000 | 0.000 |
| Transformer BC | 0.995 ± 0.010 | 0.453 ± 0.232 | 0.123 | 0.000 |

- 渲染 + tracker 提供的 perceived state 上，PSS 仍 1.000；transformer ID 0.995、OOD 0.453、invalid 0.123。
- 与纯符号 P5b（1.000 vs 0.434）几乎一致，说明结论不是符号状态的假象。
- 但目前只评估“从 perceived state 规划”，没有执行 push；还不是完整物理闭环。


---

## 41. P5b 执行闭环（2026-11-23）

目标：把 P5b 从“perceived state 规划”推进到“执行 → 重感知 → 重规划”。

- 3 objects，5 个桌面区域；
- 每步：渲染 → tracker → 区域状态 → 规划/策略 → 执行 → 重新渲染；
- 当前执行是 **kinematic placement**，不是物理 push；
- 成功按 GT 区域和两个关系 goal 检查；
- 5 seeds，seen / held-out relation pairs。

### 41.1 结果

| method | ID success | OOD success |
|---|---:|---:|
| PSS explicit planner + execution | **1.000 ± 0.000** | **1.000 ± 0.000** |
| Transformer BC + execution | 0.950 ± 0.100 | **0.400 ± 0.255** |

- 执行闭环仍然复现 P5b：ID 上 transformer 接近，OOD 掉到 0.400。
- 执行是 **kinematic placement**，不是物理 push；物理 push 版本因多物体路径碰撞尚未启用。
- 这说明“关系组合优势”在闭环执行层仍然存在，但物理抓取/推动仍是缺口。


---

## 42. 碰撞感知关系规划（2026-11-23）

目标：把 P5b 执行从 kinematic placement 推进到物理 push。

- 规划：BFS over region assignments，增加 segment-clearance 检查；
- 边 `r -> r'` 只在移动路径与其他物体当前位置距离 ≥ clearance 时可用；
- 如果直连被挡，允许经过中间区域绕行；
- 执行：ActionHead push，gain 20 / max force 6 / max 40 pulses；
- 每步重新渲染、跟踪、重规划；
- 对照：kinematic placement vs physical push。

### 42.1 结果：physical push

| method | ID success | OOD success |
|---|---:|---:|
| PSS explicit planner + physical push | **0.800 ± 0.187** | **0.700 ± 0.100** |
| Transformer BC + physical push | 0.750 ± 0.158 | 0.150 ± 0.122 |

- 碰撞感知规划 + 物理 push 后，PSS OOD 仍 **0.700**，transformer 只有 **0.150**。
- ID 上两者接近（0.800 vs 0.750），OOD 差距明显。
- 成功低于 kinematic 版本，原因是物理推动的漂移、接触和重规划失败。
- 这已不是纯符号结果：规划、渲染、tracker、push、重感知都在环里。

### 42.2 结果：collision-aware kinematic vs physical

| execution | PSS ID | PSS OOD | TR ID | TR OOD |
|---|---:|---:|---:|---:|
| kinematic placement | 1.000 | **1.000** | 1.000 | 0.450 |
| physical push | 0.800 | **0.700** | 0.750 | 0.150 |

- 同一碰撞感知规划器，kinematic 下 PSS 1.000、TR OOD 0.450；换物理 push 后 PSS 0.700、TR 0.150。
- 物理执行把绝对成功率拉低，但 PSS 的关系组合优势仍然保留。
- 物理版失败主要来自推动漂移、接触和重规划失败，不是规划器找不到路。


---

## 43. 公平 tracking 基线：transformer + tactile（2026-11-23）

目标：补齐 P1b 缺失的公平对照。

- PSS：graph+tactile，token 407；
- Transformer：同 token 407，同 Hungarian；
- 同 v2 test 60 序列、同 tracker 配置；
- Transformer 5 seeds；PSS 用已训练 tactile graph。

### 43.1 结果

| method | ID retention | switches | reappearance |
|---|---:|---:|---:|
| Hand PSS | 0.919 | 0.16 | 0.839 |
| PSS + graph + tactile | **1.000** | **0.00** | **1.000** |
| Transformer + tactile | 0.988 ± 0.008 | 0.02 | 0.977 ± 0.017 |

- 同 tactile token、同 Hungarian 后，tracking 差距从 1.000 vs 0.908 缩到 **1.000 vs 0.988**。
- PSS 仍高 **+0.012** retention，和 re-ID 的 +0.012 一致；在约 1 个标准差内。
- 结论：**tracking 的大部分优势也来自 tactile 输入，不是 graph 架构。**
- transformer+tactile 的 reappearance 0.977 vs PSS 1.000，仍有小差距，但不足以支撑强架构主张。


---

## 44. Sparse/retrieval attention 基线（2026-11-23）

目标：给 transformer 一个接近 PSS active-set 的稀疏注意力基线。

- 每个 token 只 attend 到 top-k 最近 pose token（k=1/2/4）+ 自身；
- 仍用同一 tactile token（407）和同一数据/split/seeds；
- 对照 full attention transformer。

### 44.1 结果

| variant | top-k | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|---:|
| full | - | 0.977 ± 0.017 | 0.991 | 1.000 | 0.924 |
| **sparse_k1** | 1 | **0.996 ± 0.005** | 0.996 | 1.000 | **0.991** |
| sparse_k2 | 2 | 0.982 ± 0.019 | 0.982 | 1.000 | 0.964 |
| sparse_k4 | 4 | 0.974 ± 0.021 | 0.960 | 1.000 | 0.978 |

- **sparse_k1 transformer 0.996，高于 PSS+tactile 的 0.989**；near-identical 0.991 vs 0.956。
- 这说明 PSS re-ID 不仅没有架构优势，甚至可能被一个同 tactile 的稀疏注意力 transformer 超过。
- sparse mask 仍是 dense 实现，不是延迟优化；这是精度基线。
- 结论：**re-ID 的强基线已经闭合，PSS graph matcher 不是最优 re-ID 架构。**

---

## 45. EWC 增量基线（2026-11-23）

目标：补 P3 的 EWC 强基线。

- 在旧类 train 上估 Fisher diagonal；
- 新类 fine-tune 时加 `lambda * sum(F * (theta-theta0)^2)`；
- 对照 last_layer/full/replay。

### 45.1 结果

| method | old after | new after | forgetting | update time |
|---|---:|---:|---:|---:|
| PSS prototype add | **0.979** | **1.000** | **0.000** | **31.15 ms** |
| Transformer last_layer | 0.879 ± 0.039 | 0.985 ± 0.021 | 0.099 | 1298.7 ms |
| Transformer full | 0.809 ± 0.091 | 1.000 ± 0.000 | 0.169 | 4274.0 ms |
| Transformer replay | 0.976 ± 0.004 | 1.000 ± 0.000 | **0.002** | 5066.3 ms |
| Transformer EWC | 0.741 ± 0.107 | 1.000 ± 0.000 | 0.237 | 5811.4 ms |

- EWC 在这个设置下**没有降低遗忘**（0.237），反而比 full 更差；需要调 lambda / Fisher / 训练轮数。
- replay 仍是最好的零遗忘基线（0.002）。
- PSS 更新快 **41.7x**，且不需要旧数据；EWC 没有改变这个结论。
- H3 forgetting 对 replay 仍 FAIL；H3 time PASS。


---

## 46. 事实级知识热更新（2026-11-24）

目标：把“知识热更新”从类别原型推进到事实级。

- 24 entities × 3 attributes（material/location/color）+ 3 derived rules；
- PSS：`FactStore` + `RuleEngine`，update/add/delete/rollback/audit；
- Transformer：entity/query token transformer，隐式学习 facts；
- 编辑一条 fact，测 direct/derived accuracy、old retention、forgetting、update time、rollback；
- 5 edit seeds。

### 46.1 结果

| method | edited accuracy | old accuracy | forgetting | edit time |
|---|---:|---:|---:|---:|
| PSS fact update | **1.000** | **1.000** | **0.000** | **0.171 ms** |
| Transformer full | 1.000 | 0.735 | 0.265 | 952.3 ms |
| Transformer replay | 1.000 | 1.000 | 0.000 | 914.1 ms |

- PSS 编辑一条 fact 只需 **0.171 ms**，且旧知识 1.000、可回滚、可审计。
- full fine-tune 编辑后 old 掉到 0.735（forgetting 0.265），耗时 952 ms。
- replay 追平 old 1.000 / forgetting 0.000，但耗时 914 ms，且需要旧数据。
- 结论：**事实级知识热更新上，PSS 的优势是操作成本与可审计/可回滚；零遗忘仍不是独有。**

---

## 47. 长历史 / active-set scaling（2026-11-24）

目标：验证 PSS 计算随 active-set 而不是历史长度增长。

- 合成 key-value retrieval：history T=10/100/500/1000；
- transformer 在 T=10 训练，测更长历史；
- PSS active-set k=8，只查 active set；
- 指标：accuracy、p50/p95。

### 47.1 结果

| history | transformer accuracy | transformer p50/p95 | PSS accuracy | PSS p50/p95 |
|---:|---:|---:|---:|---:|
| 10 | 0.883 ± 0.006 | 1.81 / 1.87 ms | **1.000** | **0.0003 / 0.0014 ms** |
| 100 | 0.281 ± 0.011 | 1.94 / 2.27 ms | **1.000** | **0.0003 / 0.0014 ms** |
| 500 | 0.128 ± 0.037 | 3.49 / 3.75 ms | **1.000** | **0.0003 / 0.0014 ms** |
| 1000 | 0.076 ± 0.013 | 8.40 / 8.44 ms | **1.000** | **0.0003 / 0.0014 ms** |

- Transformer 在 10 步历史训练，历史一长准确率从 0.883 掉到 0.076，延迟从 1.81 ms 到 8.40 ms。
- PSS active-set 始终 1.000、0.0003 ms；延迟不随历史长度变化。
- 这是“计算随 active-set 而不是历史长度增长”的直接证据。
- 注意：transformer 是短历史训练、长历史测试；强长上下文模型可能更好，但计算增长问题仍在。


---

## 48. 多源事实 + 规则热更新（2026-11-24）

- 100 entities × 3 attrs × 3 sources（priority 0/1/2）；
- 注入 273 个冲突；
- 对照 flat last-write vs priority resolver；
- cascade delete + rollback；
- rule hot update：fragile = glass → glass/ceramic，facts 不动。

### 48.1 结果

| store | accuracy vs priority resolver |
|---|---:|
| Multi-source + priority | **1.000** |
| Flat last-write | 0.800 |

- conflicts detected: **273**
- cascade delete: 30 cleared / 270 survivors kept / 30 restored；44.68 ms
- rule hot update：37 条 derived query 改变；direct facts 全部保留；rule edit **0.029 ms**；rollback exact。

- 多源优先级解析明显优于 flat last-write。
- 规则可独立热更新，事实层不动；这直接验证“知识层/逻辑层分离”。

---

## 49. 近似 active-set 检索（2026-11-24）

- key 向量 d=32，active k=8，history 1000；
- query 加噪声 sigma 0/0.1/0.3/0.5；
- 对照 active ANN vs full-history ANN。

### 49.1 结果

| query noise sigma | active ANN acc | full-history ANN acc | active p50 | full p50 |
|---:|---:|---:|---:|---:|
| 0.0 | 1.000 | 1.000 | 0.0056 ms | 0.0118 ms |
| 0.1 | 1.000 | 1.000 | 0.0057 ms | 0.0093 ms |
| 0.3 | **0.920** | 0.495 | 0.0058 ms | 0.0096 ms |
| 0.5 | **0.700** | 0.150 | 0.0058 ms | 0.0095 ms |

- 噪声 0.3/0.5 时，active-set ANN 仍 0.920/0.700；full-history ANN 掉到 0.495/0.150。
- active-set 延迟约 0.006 ms，不随 history 增长；full-history 扫描 1000 条约 0.0095 ms。
- 这说明“近似检索 + active-set”不是只能精确 lookup；它对噪声更稳、成本更低。

---

## 50. RAG / external-memory 基线（2026-11-24）

- top-k=4 retrieval + small transformer；
- 训练 T=10，测 T=100/500/1000；
- 对照 full attention 和 PSS active-set。

### 50.1 结果

| history | full attention acc | RAG acc | full p50 | RAG p50 | PSS p50 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.997 | 0.990 | 0.60 ms | 0.93 ms | 0.0002 ms |
| 100 | 0.633 | 0.982 | 1.03 ms | 0.56 ms | 0.0002 ms |
| 500 | 0.151 | 0.982 | 2.54 ms | 1.21 ms | 0.0002 ms |
| 1000 | 0.120 | **0.971** | 4.90 ms | 0.77 ms | 0.0002 ms |

- Full attention 长历史准确率崩（0.997→0.120），延迟涨（0.60→4.90 ms）。
- RAG 准确率保持 **0.971–0.990**，延迟约 0.6–1.2 ms；是强基线。
- PSS active-set 仍 1.000 / 0.0002 ms；优势从“准确率”变成“检索成本/精确性”。
- 结论：**强 RAG 基线能保住长历史准确率，但没有消除 PSS 的 active-set 成本优势。**


---

## 51. Object-centric attention 最小实现（2026-11-24）

目标：把 multi-head attention 的单位从 token 换成 object slot。

- object slot：vision/geometry/pose/category/tactile；
- 每个 head 的 logits = content + spatial bias + semantic bias + tactile bias；
- active-set mask；
- 输出更新后的 slots，再做 pair scoring + Hungarian；
- 数据：tracking_bench_v2，tactile token，5 seeds。

### 51.1 结果

| method | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|
| Object attention（383k params） | 0.987 ± 0.011 | 0.996 | 1.000 | 0.956 |
| PSS graph+tactile（169k params，ref） | **0.989** | 1.000 | 1.000 | 0.956 |
| sparse k=1 transformer（ref） | **0.996** | 0.996 | 1.000 | 0.991 |

- Object attention 基本追平 graph matcher（0.987 vs 0.989），但参数量约 2.3x。
- 仍低于 sparse_k1 transformer 0.996。
- near-identical 0.956，和 graph+tactile 相同；没有超过 tactile-only 的上限。
- 结论：**object attention 能工作，但最小实现没有超过现有 graph matcher/sparse transformer；需要关系监督、更强 typed head 或与外部知识结合才有意义。**


---

## 52. 编码器替换：冻结 DINOv2（2026-11-24）

- vision：SmallCNN → 冻结 DINOv2-small CLS（384）；
- projection：在 category_pretrain 上重训 vision 投影头；
- prototype：重训并测 closed-set / open-set；
- re-ID：用 DINO 投影后的 perception 重训 tactile graph matcher；
- P5b：感知版可复用同一 perception。

### 52.1 结果

（待填）
