# Phase 30 — 把 re-ID 改成自动学习：设计与实施计划

> Date: 2026-11-24
> Scope: PSS entity layer 的 re-ID / association
> Status: 设计稿（未实现）
> Related: `pss/memory/learned_metric.py`、`pss/memory/graph_matcher.py`、`pss/memory/tracking.py`

---

## 1. 现状：已经有一半“自动学习”，但还不是自动管线

仓库里已有两条学习式路径：

| 路径 | 模型 | 输入 | 结果（tracking_bench_v2） |
|---|---|---|---|
| pairwise metric | `MetricEmbedding`（MLP → 128-d） | 每实体 token | 0.809 |
| set-level graph | `GraphMatcher`（关系边 + pair head） | 整个 pre/post 集合 | **0.930 ± 0.029** |
| transformer 参照 | `EntityTransformer` | 同 token | 0.802 ± 0.008 |
| 手写 cost | `pss_cost_matrix` | vision/geometry cosine | 0.533 |

关键事实：

- `LearnedMetricReID` 已经能作为 `TrackerConfig.metric` 接入 Tracker；
- `GraphMatcherReID` / `PrunedGraphMatcherReID` 已能作为 runtime associator；
- 但目前训练标签来自 `gt_id`，训练是离线的，runtime 默认仍是手写 cost，阈值 `metric_max_cost` 也是手调的；
- 公平归因实验显示：给 transformer 同样 tactile 后，系统级 re-ID 差距从 +0.128 缩到 **+0.012**，且 sparse-k1 transformer 0.996 反超 PSS 0.989。**不能把系统级优势全部归因给 graph matcher 架构。**

所以“改成自动学习”要解决的不是“再写一个模型”，而是四件事：

1. **表示自动学习**：不再用固定 cosine，而是学实例度量；
2. **匹配自动学习**：从逐对打分升级到集合级 assignment；
3. **监督自动生成**：没有 GT ID 时，从轨迹连续性生成伪标签；
4. **阈值/校准/更新自动化**：阈值、温度、在线更新、回滚都可审计。

---

## 2. 目标形态

```text
observations
   │
   ├─ token encoder (vision / geometry / tactile / dino / pose / category)
   │        │  learned embedding z
   │        ▼
   ├─ candidate pruning (pose gate + category gate + appearance top-k)
   │        │
   │        ▼
   ├─ set-level matcher (GraphMatcher / attention)
   │        │  match logits
   │        ▼
   ├─ Hungarian assignment + calibrated threshold
   │        │
   │        ▼
   └─ Tracker: matched / reid / new / dormant
              │
              └─ high-confidence tracklets → pseudo-label buffer → retrain
```

原则：**保留 PSS 的 scene graph / entity slots，只把“匹配”学掉。** 不把整个实体层换成一个黑盒端到端模型，否则会丢掉审计、版本、回滚和 active-set。

---

## 3. 自动学习信号

### 3.1 离线阶段：有 GT ID

用于 benchmark 和冷启动：

- 同一 `gt_id` 的 pre/post token 为正对；
- 同类别不同实例、同视频不同 track、near-identical 为硬负样本；
- 按 sequence / camera 划分 train/val/test，禁止同序列泄漏。

损失：

```text
L = BCE(match_logits, labels)
  + λ_assign * (row assignment NLL + column assignment NLL)
  + λ_metric * (triplet / ArcFace on tracklet embedding)
  + λ_consistency * temporal / permutation consistency
```

已有 `train_graph` 已实现 BCE + row/col assignment；需要补：

- hard negative mining（当前随机 batch，负样本太容易）；
- category dropout（samecat 子集不能靠类别）；
- modality dropout（避免只靠 tactile，公平归因已显示 tactile 是主要信号）；
- tracklet-level positive（不只单帧 pre/post）。

### 3.2 在线阶段：无 GT ID 的伪标签

不依赖人工标注，用“高置信轨迹连续性”生成标签：

1. 用当前 matcher + 手写 fallback 跑短时 tracker；
2. 只保留高置信关联：
   - mutual nearest（双向一致）；
   - margin 大于阈值（top1 与 top2 差距）；
   - 相邻帧/短间隔内的轨迹连续；
   - 类别兼容、pose gate 内；
3. 同一 tracklet 内采样正对；不同 tracklet / 同类别不同实例采样负对；
4. 加入 replay buffer，按 domain / category 平衡；
5. 迭代 self-training：train → re-associate → mine → train，保留 holdout 监控，防 confirmation bias。

关键：**伪标签只用于“表示/度量”更新，不直接写 fact / 不改变实体 ID 的审计语义。** 每次更新记录 model version、样本数、val retention。

### 3.3 在线自适应与治理

- 低学习率 + EWC / replay，避免灾难遗忘；
- 每次更新前在 holdout 上评估 retention / IDF1；
- 下降超过阈值就 rollback 到上一版模型；
- 低 margin 样本走 `UncertaintyService` 的 review/abstain，不强行匹配；
- 模型 checkpoint、训练数据摘要、校准阈值进入 audit。

---

## 4. 模型选择：pairwise 还是 set-level

| 方案 | 优点 | 缺点 | 建议 |
|---|---|---|---|
| `MetricEmbedding` pairwise | 轻、易接 Tracker、可解释 | 逐对独立，samecat 弱；v2 0.809 | 作为 baseline / 冷启动 |
| `GraphMatcher` set-level | 利用集合关系，v2 0.930 | O(N²)，需剪枝；架构增量在公平归因下不大 | **主路径**，配 `PrunedGraphMatcherReID` |
| Transformer | 表达强 | 需要更多数据/调参；同 tactile 下优势小 | 对照，不作为默认 |

结论：**主路径用 set-level graph matcher + pruned candidate graph**；pairwise metric 保留为轻量 fallback/蒸馏目标。

---

## 5. 需要改的代码点

1. `pss/memory/learned_metric.py`
   - `train_metric` 支持 `token_dim` / `include_tactile` / `include_dino`（当前硬编码 `TOKEN_DIM`）；
   - `LearnedMetricReID` 支持 modality flags；
   - 增加 hard-negative sampler、category/modality dropout。
2. `pss/memory/graph_matcher.py`
   - `train_graph` 增加 hard negative、tracklet positive、modality dropout；
   - `save_graph` 保存 `token_dim` / modality metadata / calibration。
3. 新增 `pss/memory/reid_trainer.py`（或 `pss/learning/reid.py`）
   - 统一训练配置、数据加载、伪标签挖掘、校准、checkpoint；
   - 输出 `reid_metric.pt` / `reid_graph.pt` + metadata。
4. `pss/memory/tracking.py`
   - `TrackerConfig` 增加 `metric_version` / `metric_fallback` / `metric_calibration`；
   - 低 margin 时回退手写 cost 或 review。
5. `pss/runtime/runtime.py`
   - `associator` 默认从 checkpoint 加载（opt-in → 验证后 default）；
   - `components()` 暴露 `reid_model`、`metric_version`、`calibration`。
6. 评测
   - P1 / P1c / P0.1 + 全在线 tracker retention；
   - TAO Charades IDF1 / retention / switches；
   - 公平归因：同 tactile transformer、同 encoder、同 Hungarian；
   - near-identical 作为压力测试，不承诺高分。

---

## 6. 分阶段实施

### Phase A：统一离线训练器

- 统一 `train_metric` / `train_graph` 配置；
- 加 hard negatives + category/modality dropout；
- 目标：v2 graph ≥ 0.930，samecat/near 不退化。

### Phase B：自动伪标签

- 从短时 tracker 生成高置信 tracklet；
- self-training + replay + holdout；
- 目标：无 GT 训练下接近有 GT 的 90%。

### Phase C：runtime 默认

- checkpoint 加载、阈值校准、低 margin fallback；
- 审计 model version / calibration；
- 目标：全在线 retention 不低于手写 cost，延迟可接受。

### Phase D：真实数据

- TAO / Charades 轨迹伪标签；
- domain randomization / 跨视频负样本；
- 目标：IDF1 不低于 detector+PSS 的 0.984（同 person 子集）。

---

## 7. 风险与边界

- **near-identical 没有身份信号**：自动学习不能创造信息；该子集应作为 abstain/压力测试，而不是必须高分。
- **tactile 依赖**：公平归因显示 tactile 是系统级 re-ID 优势的主要来源；必须做 modality dropout，否则换平台会崩。
- **伪标签 confirmation bias**：必须 holdout + margin + rollback，不能纯 self-training。
- **架构增量有限**：公平对比下 graph matcher 的增量可能只有 +0.01；自动学习的主要收益应来自表示/监督/校准，而不是“换个 matcher 名字”。
- **不能声称端到端替代 PSS**：保留实体图、版本、审计；学习只发生在匹配层。

---

## 8. 建议的第一刀

最小可行改动：

1. 把 `GraphMatcherReID` 接成 `TrackerConfig.metric` 的默认可选路径；
2. 训练时加 **hard negatives + category dropout + modality dropout**；
3. 用 val 校准 `metric_max_cost`，替代手调 0.9；
4. 记录 model version / calibration 到 audit；
5. 在 P1/P1c/P0.1 + 全在线 tracker 上复测。

如果只做一件事：**先把“训练标签”从 GT ID 扩展到高置信 tracklet 伪标签**，否则永远只是离线 benchmark 模型，不是自动学习系统。
