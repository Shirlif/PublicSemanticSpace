# PSS 读出与推理优化总报告

> 日期：2026-11-27
> 范围：A 线（读出：A0 闸门、A1 打分器、A4 融合、A2 重排、A5 增量）+ B 线（推理：B0/B1 覆盖、B2 代数、B3 搜索）
> 预注册：`pss/eval/readout_reasoning_preregistration.py`
> 事实底稿：`eval/results/readout_opt/A_LINE_SUMMARY.md`、`eval/results/B_LINE_SUMMARY.md`
> 引用边界：A 线全部为 final；B 线 `relation_coverage_s3/`、`relation_algebra/`、`search_comparison/` 为 final；`relation_algebra_smoke/` 为 smoke，不引用；`relation_coverage/` 为 1-seed 诊断源，不是最终数字。

---

## 0. 摘要

1. **读出端的真实增益只有一处：先跨模态混合、再判别。** `lda_concat_whiten` 把闭集从部署原型的 **0.833 抬到 0.948（+11.5 点）**，开集保持 AUROC 0.998 / operating point 0.987，热更新 0.018 ms。学习式重排没有增量；增量刷新与联合训练持平。
2. **推理端的瓶颈不在搜索，也不在代数权重。** BFS 在 P5b 上已经 1.000；加权 A\* 通过预定义 gate，但收益只是节点数（12.3 → 4.8）。代数一致性对多数子集有效（最大 ΔOOD +0.133），却对最差子集 **BCD 完全无效（Δ=0）**。BCD 的根因已定位：缺 `dy=0` 的水平对，`above/below` 在水平对上假阳性（absence case 未监督），不是容量。补 300 条缺失方向反例后 BCD 0.571 → 1.000、ABD 0.524 → 0.952（ACG 空对照 Δ=0），因果确认；相对特征再消掉 ABD 的代价，相对 + 反例后三个子集全部 1.000。但 7×7 验证把配方降级为**条件性**：1D strip 上单轴增强牺牲 diagonal（strip_x 0.68 → 0.08），绝对位置任务上相对特征 0.367 vs 绝对 0.856；需要按 pair 类型补覆盖。pair-type 对照（§3.8）进一步确认：只补 vertical 是 0.310，补全 vertical+diagonal 到 0.967。
3. **两条线共同指向同一个结论：四因子里真正有效的是解码器替换与条件化；维度与容量都不是瓶颈。** 调完之后，最弱的一环跑到四因子之外——数据/监督覆盖与任务匹配（§4.4）。继续加头、加维度、加搜索复杂度，都不会改变这个判断。

---

## 1. 实验设置与预注册纪律

### 1.1 任务

| 线 | 任务 | 规模 | 划分 |
|---|---|---|---|
| A | `category_pretrain_instance8`，view-instance 实例判别 | 384 类 | train = views 0–5（2304）、val = view 6（384）、test = view 7（384）、held = `test_category`（768） |
| B | P5b 符号关系组合 | 3 物体 / 5 区域 | 训练关系对 4 组、held-out 组合 4 组；B0/B2: 50 episodes/pair × 3 head seeds；B3: 60 episodes/pair, 1 episode seed |

### 1.2 纪律

- 每个新 arm 在运行前登记到 `pss/eval/readout_reasoning_preregistration.py`；**失败项照实保留，bar 不修改**。
- 开集同时报 AUROC 与 operating point（95% seen accept 下的 unseen reject），不只报 AUROC。
- B 线未新增预注册 bar，B2/B3 明确标为**探索性**。
- smoke 结果（`relation_algebra_smoke/`）不进入任何结论。
- A 线的 MLP 参照为同表示 3-seed 集成；LDA 是确定性拟合，`seed_std = 0` 不代表稳定性。

---

## 2. A 线：读出

### 2.1 起点与目标

部署读出面是**分模态原型 + 固定权重 cosine 融合**：闭集 0.833，同表示 MLP 0.878。目标是收回读出税，同时不牺牲开放集、热更新与审计。总缺口按归因 v2 拆为 encoder 0.0 / projection 0.9 / reasoning 2.0 点；本线只动读出与融合。

### 2.2 A0：闸门召回

生产投影上的原型 top-k 召回（test）：

| top-1 | top-3 | top-5 | union@5 | k@95 |
|---:|---:|---:|---:|---:|
| 0.833 | 0.953 | 0.971 | 0.987 | 3 |

- val→test 掉 7.1 点（0.904 → 0.833），任何在 val 上选出的阈值都会偏乐观。
- 分模态 top-1：vision 0.727、tactile 0.263、geometry 0.206——**固定权重 1.0 与分模态能力不匹配**。
- 结论：候选不是瓶颈；重排的上限高于当前读出。

### 2.3 A1：打分器——收益在打分，损失在融合

同一份 384-d 公共表示，只换打分器：

| arm | 闭集 acc | AUROC | reject@95 |
|---|---:|---:|---:|
| `proto`（分模态 cosine 融合，部署） | 0.833 | 0.999 | 0.997 |
| `proto_mahalanobis`（分模态 Mahalanobis + 融合） | 0.763 | 0.991 | 0.947 |
| `concat_cosine` | 0.831 | 0.999 | 0.997 |
| `concat_mahalanobis`（探索性） | **0.930** | 0.997 | 0.977 |
| `concat_knn` | 0.810 | 0.999 | 0.996 |

- cosine 的融合与 concat 无差别（0.833 vs 0.831）→ **分模态结构本身不丢分**。
- Mahalanobis 在拼接表示上 +9.9 点；同一打分器塞进分模态融合掉到 0.763 → **融合吃掉收益**。
- 这一步是 A4 的直接动机：判别必须发生在跨模态混合**之后**。

### 2.4 A4：先混合、再判别（主结果）

| arm | 闭集 acc | Δ vs MLP | AUROC | reject@95 | hot ms |
|---|---:|---:|---:|---:|---:|
| `lda_concat` | 0.930 | +0.052 | 0.997 | 0.977 | 0.019 |
| **`lda_concat_whiten`** | **0.948** | **+0.070** | 0.998 | **0.987** | 0.018 |
| `lda_concat_weighted` | 0.930 | +0.052 | 0.997 | 0.982 | 0.018 |
| `lda_per_modality_fused`（负对照） | 0.763 | −0.115 | 0.991 | 0.947 | 0.017 |

- **主候选是逐模态 PCA 白化后拼接的 LDA**。受约束权重在 val 上选出 `{vision:1.5, tactile:0.5, geometry:2.0}`，test 与裸 concat 完全相同 → **权重不重要，条件化重要**。
- 负对照证明 P1 的解释：判别放在融合之前，裸 concat 0.930 → fused 0.763（−16.7）；白化版 0.948 → 0.763（−18.5）。
- 缺失模态（7 个子集各自拟合）：

| 可用模态 | LDA acc | 原型 cosine acc |
|---|---:|---:|
| vision | 0.799 | 0.727 |
| tactile | 0.266 | 0.263 |
| geometry | 0.263 | 0.206 |
| vision+tactile | 0.896 | 0.841 |
| vision+geometry | 0.901 | 0.706 |
| tactile+geometry | 0.607 | 0.492 |
| vision+tactile+geometry | 0.930 | 0.833 |

- vision 是该任务的主模态；单 tactile/geometry 很弱，但 tactile 在 v+t 中贡献 +9.7 点。
- 预注册 near-miss：`lda_concat` 的 reject@95 = 0.9766 < 0.98（差 0.0034），记失败；`lda_concat_whiten` 全项通过。

### 2.5 A2：学习式重排（负结果）

| 项 | 值 |
|---|---:|
| 闸门 `lda_concat_whiten` | 0.948 |
| 闸门 recall@1 / @3 / @5 | 0.948 / 0.995 / **1.000** |
| 重排 k=3 / k=5 / k=10 | 0.948 / 0.945 / 0.945 |
| 同表示完整 MLP | 0.807 |
| 预注册 `rerank_bilinear`（bar 0.955） | **失败（0.945）** |

- 候选不是瓶颈：recall@5 = 1.000，重排可自由选择全部候选，仍无增量。
- 学习度量初始化于 LDA 精度，768 行训练后 val 0.961 → test 0.945，过拟合 1.6 点。
- 完整 MLP 更差（0.807）：白化放大低方差方向，LDA 用池化协方差压回去，MLP 在 6 shot/类 下学不出来。**容量不是关键。**
- 按预注册规则：**A2 停止**，线性 LDA 度量已是该表示的读出上限。

### 2.6 A5：增量加类与协方差刷新

base 192 类 + 逐个加入 192 类；评估 = val+test（每类 2 个未见视角）：

| policy | base after | forgetting | added acc | overall | refits | refit s |
|---|---:|---:|---:|---:|---:|---:|
| `incr_frozen` | 0.943 | 0.031 | 0.927 | 0.935 | 1 | 0.11 |
| **`incr_refit_10`** | **0.961** | **0.013** | **0.948** | **0.954** | 20 | 2.53 |
| `incr_refit_25` | 0.956 | 0.018 | 0.948 | 0.952 | 8 | 1.02 |
| `incr_refit_all` | 0.961 | 0.013 | 0.948 | 0.954 | 193 | 24.75 |
| `cosine_whiten` | 0.802 | 0.068 | 0.768 | 0.785 | 0 | 0.00 |
| `batch_joint`（参照） | 0.961 | n/a | 0.948 | 0.954 | 1 | 0.14 |

- 刷新把冻结精度的遗忘 3.1 → 1.3 点；**1.3 是联合训练也有的加类稀释地板**，不是刷新误差。
- 频率拐点在 10：10 与每类重估精度相同（20 vs 193 次，2.5s vs 24.8s）；25 类一次掉 0.5 点。
- 全协方差不可省：cosine 版本起点 0.870、added 0.768，判别力与遗忘同时变差。
- 预注册：`incr_refit_25`（0.0182）、`incr_refit_all`（0.0130）未达 0.01 bar；因低于可达地板，记 near-miss，bar 不改。

### 2.7 A 线结论与部署建议

| 阶段 | 读出 | 闭集 acc | 相对部署原型 |
|---|---|---:|---:|
| 起点 | 分模态原型 cosine | 0.833 | — |
| A4 | `lda_concat_whiten` | **0.948** | **+11.5** |
| A2 | + 学习式重排 | 0.948 | +11.5（无增量） |
| A5 | 增量刷新 | 与联合训练持平 | — |

**部署建议：**

- 表示：逐模态 PCA 白化 → 拼接 → 池化协方差 LDA。
- 更新：类均值即时；协方差每 10 类（或类数 +5%）重估；保留特征 reservoir 或每类充分统计量。
- 缺失模态：7 个子集模型，按可用模态选择。
- 开集：LDA max score + 训练集 novelty map 阈值；不要换成 max-softmax。

---

## 3. B 线：关系覆盖、代数一致性与搜索

### 3.1 B0/B1：覆盖特征

3-seed 相关性（78 行 = 26 子集 × 3 seeds）：

| 特征 | ρ(exact-4) | ρ(OOD) | 按 seed ρ(OOD) | mean ± std |
|---|---:|---:|---|---|
| `axis_isolation` | +0.918 | **+0.843** | +0.858 / +0.862 / +0.798 | +0.839 ± 0.029 |
| `sign_pattern_seen` | +0.851 | +0.792 | +0.805 / +0.860 / +0.720 | +0.795 ± 0.058 |
| `n_regions` | +0.760 | +0.783 | +0.755 / +0.790 / +0.783 | +0.776 ± 0.015 |
| `sum_hull_distance` | −0.672 | −0.706 | −0.700 / −0.752 / −0.653 | −0.702 ± 0.040 |
| `n_outside_hull` | −0.639 | −0.690 | −0.663 / −0.719 / −0.672 | −0.685 ± 0.025 |

同尺寸对照（3/5 子集，按子集聚合 3 seeds）：

| 子集 | axis_isolation | sign_pattern_seen | hull 外点数 | exact-4 | OOD |
|---|---:|---:|---:|---:|---:|
| ADG | 2 | 0.714 | 2 | 0.976 | **0.950** |
| ACG | 2 | 0.714 | 2 | 0.929 | 0.945 |
| BCG | 2 | 0.714 | 2 | 0.881 | 0.938 |
| ABD | 1 | 0.571 | 1 | 0.524 | 0.855 |
| ACD | 1 | 0.571 | 1 | 0.524 | 0.793 |
| BDG | 2 | 0.714 | 2 | 0.833 | 0.783 |
| ABC | 1 | 0.571 | 1 | 0.548 | 0.680 |
| CDG | 1 | 0.000 | 2 | 0.405 | 0.612 |
| ABG | 1 | 0.000 | 2 | 0.405 | 0.542 |
| **BCD** | **1** | **0.571** | **1** | **0.571** | **0.335** |

**判读：**

1. `axis_isolation` 仍是相关性最强的特征，但只比区域计数 `n_regions` 高约 0.06，且两者高度相关——**不能当作因果归因**。
2. 1-seed 时“ACG 1.000 vs BCD 0.280 支持轴隔离”的结论在 3 seeds 下被削弱：同为 `axis_isolation=1`，ABD/ACD 达到 0.855/0.793，**BCD 是异常点**（0.335）。
3. 凸包假设被反证：最差子集的 hull 外点数反而更少。
4. 结论：轴隔离是**相关**，不是必要/充分条件；BCD 的诊断见 §3.4，补数据验证见 §3.5。

### 3.2 B2：关系代数一致性

weight 0.1 相对零权重臂的增量（3 seeds 均值）：

| 子集 | Δexact-4 | ΔOOD | Δviolations |
|---|---:|---:|---:|
| ABCDG | +0.000 | +0.000 | +0.0 |
| ACG | +0.071 | +0.055 | −1.7 |
| BCG | +0.024 | +0.062 | −0.7 |
| **BCD** | **+0.000** | **+0.000** | **+0.0** |
| BDG | +0.119 | **+0.133** | −2.7 |
| ABC | +0.024 | +0.042 | +1.7 |
| ACD | +0.071 | +0.083 | −2.3 |

**判读：**

1. 代数损失对多数子集有效，最大增益在 BDG（ΔOOD +0.133，violations 3.7 → 1.0）。
2. **BCD 完全不变**（exact-4 0.571 / OOD 0.335 / violations 10.0）→ 它的失败不是“谓词自相矛盾”；诊断见 §3.4，补数据验证见 §3.5。
3. violations 不是单调下降（ABC +1.7）；该计数含 inverse/transitive gap，不等于矛盾数，不能当唯一代理。
4. 满覆盖 ABCDG 两个权重都是 1.000，没有可测空间。

### 3.3 B3：BFS vs placement A\*

预定义切换条件（在空闲机重跑前写死）：

```
switchable := success >= BFS success
              AND step regret vs BFS <= 0
              AND p50 latency <= BFS p50 latency
```

p50 是 gating 的延迟统计量；p95 同时报告，作为尾部诊断，不参与 gate。

| split | planner | success | steps | nodes | p50 (ms) | p95 (ms) | regret | gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| id | bfs | 1.000 | 2.58 | 12.9 | 0.039 | 0.076 | +0.000 | baseline |
| id | astar_w1 | 1.000 | 2.58 | 9.1 | 0.048 | 0.161 | +0.000 | 未过: p50 |
| id | astar_w1.5 | 1.000 | 2.58 | 5.1 | 0.033 | 0.078 | +0.000 | 通过 |
| id | astar_w2 | 1.000 | 2.58 | 5.1 | 0.033 | 0.081 | +0.000 | 通过 |
| ood | bfs | 1.000 | 2.54 | 12.3 | 0.036 | 0.081 | +0.000 | baseline |
| ood | astar_w1 | 1.000 | 2.54 | 9.0 | 0.049 | 0.162 | +0.000 | 未过: p50 |
| ood | astar_w1.5 | 1.000 | 2.54 | 4.8 | 0.031 | 0.076 | +0.000 | 通过 |
| ood | astar_w2 | 1.000 | 2.54 | 4.8 | 0.030 | 0.075 | +0.000 | 通过 |

**判读：**

1. **BFS 保持默认**：`astar_w1` 的 p50 与 p95 都更差（heap 开销），不可切换。
2. `astar_w1.5` / `astar_w2` 通过预定义 gate：success 1.000、regret 0、p50 ≤ BFS；nodes 12.3 → 4.8（约 2.6×）。
3. **不是免费午餐**：ID 的 p95 对 w1.5/w2 略差（0.078 / 0.081 vs 0.076 ms）；OOD 的 p95 反而更好。推荐只在节点数是稀缺资源时使用加权 A\*。
4. 搜索本身不是 P5b 的瓶颈：BFS 已经 1.000。B 线可动项在谓词/覆盖，不在搜索算法。

### 3.4 BCD / ABD 的 per-label 诊断

3 子集 × 3 head seeds × 400 epochs。脚本 `run_relation_per_label.py`，报告 `bcd_diagnosis/BCD_DIAGNOSIS.md`。

训练隔离方向：

| 子集 | x 隔离对（dy=0, dx≠0） | y 隔离对（dx=0, dy≠0） |
|---|---|---|
| BCD | **否** | 是 |
| ABD | 是 | **否** |
| ACG | 是 | 是 |

Held-out exact-4（按对类型）：

| 子集 | horizontal | vertical | diagonal |
|---|---:|---:|---:|
| BCD | **0.000** | 1.000 | 1.000 |
| ABD | 0.917 | **0.000** | 0.917 |
| ACG | 0.833 | 0.917 | 1.000 |

四谓词 precision / recall / F1：

| 子集 | left | right | above | below |
|---|---:|---:|---:|---:|
| BCD | 1.000/1.000/1.000 | 1.000/1.000/1.000 | 0.635/1.000/0.776 | 0.524/1.000/0.687 |
| ABD | 0.581/1.000/0.724 | 0.578/1.000/0.724 | 0.944/1.000/0.970 | 1.000/1.000/1.000 |
| ACG | 0.944/1.000/0.970 | 1.000/1.000/1.000 | 1.000/1.000/1.000 | 0.905/1.000/0.944 |

**判读：**

1. BCD 的失败是 **above/below 在水平对上的假阳性**（precision 0.635/0.524，recall 都是 1.000），不是 left/right 缺失。
2. ABD 是对称的：left/right 在垂直对上的假阳性（precision 0.581/0.578）。
3. 机制：缺一个隔离方向 → 模型没见过“该轴 delta≈0 ⇒ 另一轴谓词为假”的 absence case。BCD 缺 `dy=0, dx≠0` 的对，ABD 缺 `dx=0, dy≠0` 的对；ACG 两者都有，四谓词没有系统性假阳性。
4. 这解释了 B2 为何对 BCD 无效：antisymmetry/exclusivity 不编码 absence case。
5. 修复方向（按成本）：补缺失方向的对；去掉绝对坐标或做坐标归一化；加 `|dy|<eps ⇒ above=below=0`、`|dx|<eps ⇒ left=right=0` 的几何约束。

### 3.5 补数据验证（因果确认）

按 §3.4 的修复方向 1：给 BCD 补 300 条 `dy=0, dx≠0` 的合成对，ABD 补 300 条 `dx=0, dy≠0` 的对，ACG 两个方向都有所以是空对照。脚本 `run_relation_per_label.py --augment 300`。

| 子集 | arm | overall | horizontal | vertical | diagonal |
|---|---|---:|---:|---:|---:|
| BCD | baseline | 0.571 | **0.000** | 1.000 | 1.000 |
| BCD | **augmented** | **1.000** | **1.000** | 1.000 | 1.000 |
| ABD | baseline | 0.524 | 0.917 | **0.000** | 0.917 |
| ABD | **augmented** | **0.952** | 0.833 | **1.000** | 1.000 |
| ACG | baseline | 0.929 | 0.833 | 0.917 | 1.000 |
| ACG | augmented（空对照） | 0.929 | 0.833 | 0.917 | 1.000 |

谓词 precision：

| 子集 | above | below | left | right |
|---|---:|---:|---:|---:|
| BCD baseline → augmented | 0.635 → **1.000** | 0.524 → **1.000** | 1.000 → 1.000 | 1.000 → 1.000 |
| ABD baseline → augmented | 0.944 → 0.889 | 1.000 → 1.000 | 0.581 → **1.000** | 0.578 → **1.000** |

**判读：**

1. **因果确认**：只补 300 条缺失方向的合成反例，BCD 的水平对 exact-4 从 0.000 到 1.000、above/below precision 从 0.635/0.524 到 1.000/1.000；ABD 的垂直对从 0.000 到 1.000、left/right precision 到 1.000。与 absence-case 机制一致。
2. **有小代价**：ABD 的水平对从 0.917 掉到 0.833（−0.083），但 overall 仍从 0.524 → 0.952。
3. **空对照干净**：ACG 没有可补的隔离方向，augmented 与 baseline 逐项相同（Δ=0），说明增强只作用在缺失轴上。
4. 因此 BCD 从“未解释异常”变成“可复现、可修复的监督缺口”；ABD 的 −0.083 代价已由相对特征消除（§3.6）。

### 3.6 去绝对坐标验证（修复方向 2）

把 `_pair_feature` 从 8 维 `[ax, ay, bx, by, dx, dy, |dx|, |dy|]` 改为 4 维 `[dx, dy, |dx|, |dy|]`（`feature_mode="relative"`，默认 `absolute` 不变），再跑同一套诊断。

| 子集 | 绝对 baseline | 相对 baseline | 绝对 + 增强 | 相对 + 增强 |
|---|---:|---:|---:|---:|
| BCD | 0.571 | 0.571 | 1.000 | **1.000** |
| ABD | 0.524 | 0.571 | 0.952 | **1.000** |
| ACG | 0.929 | **1.000** | 0.929 | **1.000** |

按对类型：相对 + 增强下，三个子集的 horizontal / vertical / diagonal 全部 1.000（3 seeds，std=0）。

**判读：**

1. **相对坐标单独不能修 BCD**：BCD baseline 仍是 0.571、horizontal 仍是 0.000；absence case 仍然缺失。补数据和去绝对坐标是互补的，不是替代。
2. **相对坐标修掉 ABD 的增强代价**：绝对 + 增强时 ABD horizontal 0.917 → 0.833（−0.083）；相对 + 增强时 1.000，代价消失。
3. **相对坐标改善 ACG 稳定性**：绝对 baseline 0.929（seed 1 只有 0.786），相对 baseline 三个 seed 全部 1.000。
4. 最终组合：**相对特征 + 缺失方向反例** → 三个子集全部 exact-4 1.000，是 B 线目前最强的闭集结果。

### 3.7 大几何与绝对位置验证（条件性配方）

7×7 网格（49 区域），3 seeds，400 epochs，800 条采样测试对；子集 block3 / strip_x / strip_y / diag。

| 实验 | 训练子集 | 标签 | feature / arm | exact-4 |
|---|---|---|---|---:|
| 绝对位置任务 | block3（3×3） | 绝对半平面 | absolute | **0.856** |
| 绝对位置任务 | block3 | 绝对半平面 | relative | **0.367** |
| 相对任务 | block3 | left/right/above/below | absolute | 0.999 |
| 相对任务 | block3 | left/right/above/below | relative | **1.000** |
| 大几何 | strip_x（水平线） | 相对 | absolute baseline | **0.586** |
| 大几何 | strip_x | 相对 | relative + 增强 | 0.310 |
| 大几何 | diag（对角线） | 相对 | relative baseline | 0.366 |
| 大几何 | diag | 相对 | relative + 增强 | **0.619** |

**判读：**

1. **绝对位置任务**：relative 0.367 vs absolute 0.856 → 相对特征是平移不变任务的匹配先验，不是通用优势。
2. **2D block3**：relative 1.000、absolute 0.999 → 训练集覆盖两类 pair 类型时，P5b 的结论随几何规模保持。
3. **1D strip**：最好的臂是 absolute baseline（0.51–0.59），不是 relative；只补缺失轴会把该轴谓词拉到 1.000（strip_x: vertical；strip_y: horizontal），但 diagonal 掉到 0.08–0.25（strip_x 0.08–0.10；strip_y 绝对 0.13、相对 0.25），net 更差。
4. **diag**：补两轴反例把 exact-4 从 0.366 抬到 0.619，但 diagonal 仍只有约 0.49。
5. **机制结论（已由 §3.8 因果验证）**：P5b 的修复成立，是因为 BCD 训练集已包含 diagonal 对（B–C、B–D）；真正需要的是**缺失 pair 类型的覆盖**，不只是补一条轴。单轴增强会修好一侧、弄坏另一侧；只有补全缺失的 pair 类型才恢复。大几何验证把“相对 + 反例 = 1.000”从通用配方降级为**条件性配方**。

### 3.8 pair-type 覆盖对照（因果验证）

在 `strip_x`（只有水平训练对）上做 4 臂对照：baseline / +vertical / +vertical+diagonal / +diagonal-only。脚本 `run_geometry_validation.py --focus pair_type`。

| feature | arm | overall | horizontal | vertical | diagonal |
|---|---|---:|---:|---:|---:|
| relative | baseline | 0.172 | 1.000 | 0.000 | 0.092 |
| relative | +vertical | 0.310 | 1.000 | 1.000 | 0.098 |
| relative | **+vertical+diagonal** | **0.967** | 1.000 | 1.000 | **0.958** |
| relative | +diagonal-only | 0.866 | 1.000 | 0.000 | 1.000 |
| absolute | baseline | 0.586 | 0.687 | 0.003 | 0.675 |
| absolute | +vertical | 0.245 | 0.510 | 1.000 | 0.077 |
| absolute | **+vertical+diagonal** | **0.884** | 0.605 | 1.000 | **0.900** |
| absolute | +diagonal-only | 0.830 | 0.630 | 0.012 | 0.999 |

**判读：**

1. **pair-type 覆盖假设得到因果支持**：相对特征下，同时补 vertical + diagonal 把 overall 从 0.172 / 0.310 抬到 **0.967**，diagonal 从 0.098 到 0.958。
2. **单轴不够**：+vertical 修好 vertical 但 diagonal 仍 0.098；+diagonal-only 修好 diagonal 但 vertical 归零。只有组合达到 0.967。
3. **绝对特征也有同样规律但更差**：+vertical+diagonal 到 0.884，但 horizontal 从 0.687 掉到 0.605；相对特征在平移不变任务上仍更优。
4. 因此修复规则是：**补“缺失的 pair 类型”，不是只补“缺失的轴”**；BCD 能到 1.000，是因为它已经包含 diagonal 对。

### 3.9 B 线结论

- 覆盖与谓词质量 > 搜索算法；最差子集 BCD 的根因已定位为“缺反例 → absence case 未监督”；补 300 条反例 + 相对特征后，BCD/ABD/ACG 全部 1.000，**不是容量**。
- 但这是**条件性配方**：7×7 验证表明必须补缺失的 pair 类型（§3.8 因果确认，单轴不够）；1D strip 上单轴增强会牺牲另一类，绝对位置任务上相对特征反而不如绝对特征。
- 代数一致性是有效的平均改善，但对异常子集无效；对 BCD 的修复是补数据 + 改特征，而不是加代数权重。
- 加权 A\* 是节点数优化，不是成功率优化；BFS 保持默认。

---

## 4. 交叉结论

### 4.1 两条可复用的架构结论

1. **混合顺序比打分器形式更重要。** 先拼接再判别（裸 concat 0.930；白化版 0.948）比先融合后判别（0.763）高 16.7–18.5 点。这不是任务细节，是判别式读出的结构性约束。
2. **显式谓词 + 组合是有效的，但上界由谓词覆盖决定；搜索本身不是瓶颈。** P5b 上 BFS 已达 1.000；代数损失能改善多数子集，却救不了 BCD。结构先验不是万能钥匙，覆盖/标注质量才是闸门。

### 4.2 哪些是架构级，哪些是任务级

| 结论 | 层级 | 依据 |
|---|---|---|
| 先混合再判别 | **架构级** | 跨 3 个打分器、同一表示；负对照差 16.7–18.5 点 |
| 线性质心/协方差度量已到读出上限 | **架构级（限该规模）** | 重排 147k 参数无增量；MLP 更差 |
| 全协方差对增量刷新必要 | **架构级（限该规模）** | cosine 版本 added 0.768 vs 0.948 |
| 绝对 0.948 | 任务级 | 合成代理、小编码器 |
| BCD 异常 | 任务级 | absence-case 监督缺口；相对特征 + 300 条反例后全部 1.000；pair-type 覆盖已由 §3.8 因果确认 |
| 加权 A\* 节点收益 | 任务级 | P5b 搜索空间 |

### 4.3 剩余短板（按优先级）

1. **BCD/ABD 的缺失方向监督**：诊断与两级修复均已完成——补 300 条反例（§3.5）+ 相对特征（§3.6）；相对 + 增强下三个子集全部 1.000，ABD 的 −0.083 代价消失。7×7 验证（§3.7）把配方降级为条件性；pair-type 对照（§3.8）因果确认：单轴增强不够，必须补缺失的 pair 类型。剩余是真实几何与真实骨干复跑。
2. **长期协方差漂移**：A5 只测了 192 个新类、无删除/合并；1.3 点地板与重估周期需要更长历史验证。
3. **真实骨干复跑**：所有 A 线数字来自合成代理与小编码器；真实 VLM 需重跑。
4. **部署化**：白化 PCA + 均值 + 精度的序列化、7 个模态子集模型、reservoir 的存储策略。

### 4.4 四因子上界的当前状态

四因子乘积是解释性模型，不是可计算公式；下表是“每个旋钮实际转了多少、现在是否卡住”的实测判断。

| 因子 | 本次做了什么 | 实测 | 现在是否瓶颈 |
|---|---|---|---|
| **最强编码器**（可换） | 未换；继续用冻结的生产专家 | encoder loss ≈ 0.0；微调 0.875 = PCA 0.875；raw MLP 0.878 | **否**（本任务）；真实骨干未测 |
| **最强解码器/读出**（可换） | 原型 cosine → 逐模态白化拼接 + 池化协方差 LDA | **0.833 → 0.948（+11.5）**；重排 0.945–0.948 无增量；同表示 MLP 0.807 | 已大幅抬升，接近该表示的局部上限 |
| **公共空间维度/无损度**（可调） | 不扩维；加逐模态 PCA 白化 | 128→512 仅 +0.8~1.1（CI 跨 0）；白化 +1.8（0.930 → 0.948） | 维度：**否**；条件化：已用上 |
| **推理层可学容量**（可调） | 学习式重排；关系代数；补反例 + 相对特征 | 重排 147k 参数无增量；代数最大 +0.133 但对 BCD 无效；补数据 + 相对特征 0.571 → 1.000；BFS 已 1.000 | **否**；瓶颈是覆盖，不是容量 |

**结论：四个旋钮里，真正有效的是“解码器替换”和“条件化”；维度和容量都不是瓶颈。** 调完之后，最弱的一环跑到四因子之外——**数据/监督覆盖与任务匹配**：

- BCD 缺水平对 → `above/below` 假阳性；补 300 条反例 → 1.000。
- 1D strip 只补一条轴是 0.310，补全 vertical + diagonal 才到 0.967。
- 绝对位置任务：相对特征 0.367 vs 绝对 0.856。

因此上界更准确的写法是：

$$
\text{PSS 上界} \approx f(\text{编码器}, \text{解码器}, \text{维度/无损度}, \text{推理容量}) \times g(\text{数据/监督覆盖}, \text{任务匹配})
$$

前四个因子调到一定程度后，乘积的下一项是 g；g 不能靠加容量补。剩余优先级：① 换真实 VLM 骨干（唯一没测的“可换”旋钮）；② 按 pair 类型补覆盖并验证真实几何；③ 维度/投影已测无益；④ 部署化与长期增量。

> **2026-11-28 复核**：`docs/experiments/PSS-架构上界复核报告.md`。U1–U4 已执行：闭集读出迁移成立（DINOv2 +12.5 pt）、开集不成立（AUROC 0.9866 / reject 0.918）；覆盖规则在 CLEVR 渲染几何上成立（single 0.419 vs combination 1.000）；**无损度不成立**——学习表示的 `L_gap = +0.193`、absence error 0.827，B3 联合上界未达到。第二阶段（U1b/U3b–U3d/U5）见 `PSS-架构上界复核-第二阶段.md`：CLIP 单头全过、DINOv2 需双头；显式几何骨架 + α=0 通过（`L_gap −0.002`、absence 0.045）。

---

## 5. 预注册失败项（如实保留）

| arm | 项 | bar | 实测 |
|---|---|---|---:|
| `proto_mahalanobis` | accuracy_delta / auroc / reject | ≥ −0.010 / 0.995 / 0.98 | −0.115 / 0.991 / 0.947 |
| `proto_knn` | accuracy_delta / auroc / reject | ≥ −0.010 / 0.99 / 0.95 | −0.039 / 0.971 / 0.846 |
| `lda_concat` | reject@95 | ≥ 0.98 | 0.9766 |
| `rerank_bilinear` | accuracy | ≥ 0.955 | 0.9453 |
| `incr_refit_25` | forgetting | ≤ 0.01 | 0.0182 |
| `incr_refit_all` | forgetting | ≤ 0.01 | 0.0130 |

口径说明：

- `lda_concat` 是 0.0034 的 near-miss；`lda_concat_whiten` 全项通过。
- `rerank_bilinear` 是方向性失败：重排没有增量。
- `incr_refit_*` 的 0.01 bar 低于联合训练地板 0.013，属预注册定得过紧，记录不修改。
- B 线未新增预注册 bar；B2/B3 为探索性，不作达标声明。

---

## 6. 不能声称

- 不能声称重排/蒸馏能进一步抬升读出；A2 已证伪。
- 不能在真实 VLM 骨干上沿用 0.948 这个绝对值；全部为合成代理 + 小编码器。
- 不能声称零遗忘；残余 1.3 点存在，且只测了新增类、未测删除/合并。
- 不能把 LDA 的 `seed_std = 0` 当作稳定性证据；它是确定性拟合。
- 不能声称轴隔离是 OOD 的必要/充分条件；3 seeds 下只是相关，BCD 是反例。
- 不能声称代数一致性普遍有效；它对 BCD 完全无效，violations 也不是单调代理。
- 不能声称加权 A\* 是免费收益；ID 的 p95 略差，BFS 仍是默认。
- 不能把 BCD 的失败归因于推理容量；诊断显示是缺反例导致的 absence-case 未监督（单任务、5 区域证据）。
- 不能把 300 条合成反例的修复外推为通用结论；样本是合成对（`y=0` 或 `x=0`），不是新区域，只验证 absence-case 机制。
- 不能把“相对特征 1.000”外推到依赖绝对位置的任务；P5b 的关系是平移不变的，相对特征是任务匹配的先验。
- 不能把 P5b 的 1.000 外推到 1D strip 或任意几何；单轴增强会牺牲另一类，必须补缺失的 pair 类型（§3.8 已因果验证）。

---

## 7. 产物与复现

### 7.1 关键产物

| 路径 | 内容 |
|---|---|
| `pss/eval/readout_reasoning_preregistration.py` | 预注册与检查器 |
| `pss/prototypes/scorers.py` | Cosine / Mahalanobis / kNN + novelty map |
| `pss/prototypes/incremental.py` | 增量 LDA 与刷新策略 |
| `pss/eval/scripts/run_attribution_grid.py --stage gate` | A0 闸门召回 |
| `pss/eval/scripts/run_readout_scorers.py` | A1 |
| `pss/eval/scripts/run_fusion_ablation.py` | A4 |
| `pss/eval/scripts/run_rerank_eval.py` | A2 |
| `pss/eval/scripts/run_incremental_eval.py` | A5 |
| `pss/reasoning/relations/coverage.py` + `run_relation_coverage.py` | B0/B1 |
| `pss/reasoning/relations/algebra.py` + `run_relation_algebra.py` | B2 |
| `pss/reasoning/planner/astar.py` + `run_search_comparison.py` | B3 |
| `run_relation_per_label.py` + `eval/results/bcd_diagnosis/` | BCD/ABD per-label 诊断 |
| `eval/results/bcd_diagnosis_aug/` | 补数据验证（`--augment 300`） |
| `eval/results/bcd_diagnosis_rel_aug/` | 相对特征验证（`--feature-mode relative`） |
| `eval/results/geometry_validation/` | 7×7 大几何与绝对位置验证 |
| `eval/results/geometry_pair_type/` | pair-type 覆盖对照（`--focus pair_type`） |
| `eval/results/readout_opt/P0–P4_REPORT.md`、`A_LINE_SUMMARY.md` | A 线明细 |
| `eval/results/B_LINE_SUMMARY.md`、`relation_coverage_s3/`、`relation_algebra/`、`search_comparison/` | B 线明细 |

### 7.2 复现命令

```bash
# A 线
python -m pss.eval.scripts.run_attribution_grid --stage gate --out eval/results/readout_opt
python -m pss.eval.scripts.run_readout_scorers --out eval/results/readout_opt
python -m pss.eval.scripts.run_fusion_ablation --out eval/results/readout_opt
python -m pss.eval.scripts.run_rerank_eval --device cpu --out eval/results/readout_opt
python -m pss.eval.scripts.run_incremental_eval --out eval/results/readout_opt

# B 线
python -m pss.eval.scripts.run_relation_coverage \
  --sizes 2 3 4 5 --epochs 400 --head-seeds 0 1 2 --device cpu \
  --out eval/results/relation_coverage_s3
python -m pss.eval.scripts.run_relation_coverage \
  --reanalyze eval/results/relation_coverage_s3/coverage.json \
  --out eval/results/relation_coverage_s3
python -m pss.eval.scripts.run_relation_algebra \
  --subsets ABCDG ACG BCG BCD BDG ABC ACD --weights 0.0 0.1 \
  --seeds 0 1 2 --epochs 400 --device cpu --out eval/results/relation_algebra
python -m pss.eval.scripts.run_search_comparison --n 60 --out eval/results/search_comparison
python -m pss.eval.scripts.run_relation_per_label --out eval/results/bcd_diagnosis
python -m pss.eval.scripts.run_relation_per_label \
  --subsets BCD ABD ACG --seeds 0 1 2 --epochs 400 --augment 300 \
  --out eval/results/bcd_diagnosis_aug
python -m pss.eval.scripts.run_relation_per_label \
  --subsets BCD ABD ACG --seeds 0 1 2 --epochs 400 --augment 300 \
  --feature-mode relative --out eval/results/bcd_diagnosis_rel_aug
python -m pss.eval.scripts.run_geometry_validation \
  --n 7 --seeds 0 1 2 --epochs 400 --out eval/results/geometry_validation
python -m pss.eval.scripts.run_geometry_validation \
  --n 7 --seeds 0 1 2 --epochs 400 --focus pair_type \
  --out eval/results/geometry_pair_type
```

---

## 8. 一句话

**读出端的答案是“先把模态混起来，再判别”，收益 +11.5 点且不牺牲开集与热更新；重排、权重搜索、更大容量都已证明没有增量。推理端的答案是“搜索不是瓶颈，谓词覆盖才是”，加权 A\* 只是节点数优化，代数一致性救不了最差子集。下一步不是加模型：BCD 的失败已定位为 absence-case 监督缺口，补 300 条缺失方向反例 + 相对特征后 BCD/ABD/ACG 全部 1.000；但这是条件性配方——7×7 验证显示需要按 pair 类型补覆盖，pair-type 对照因果确认单轴不够（strip_x 0.310 → 0.967），绝对位置任务上相对特征反而不如绝对特征。剩余是真实骨干、长期增量与真实几何复跑。**
