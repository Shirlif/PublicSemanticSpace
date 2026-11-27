# PSS 开放类别流实验计划：从“人定义覆盖”到“自己发现缺口”

> 日期：2026-11-27
> 上游：`docs/experiments/PSS-g因子优化报告.md`、`docs/plans/PSS-g因子优化实验计划.md`
> 目的：把“模型见到没见过的东西怎么办”从一句口号变成**可测、可证伪、可回滚**的实验协议。
> 状态：已执行 O0–O6（2026-11-28）；结果与失败边界见 `docs/experiments/PSS-开放类别流实验报告.md`。

---

## 0. 一句话

**开放类别流不是“多加几个类”，而是让模型在流式数据里自己完成五步：发现不知道 → 聚类成候选 → 花有限标签确认 → 增量吸收 → 自检并回滚。** 本计划用三级“新”（同分布新类 / 组合新类 / 新原语或传感器漂移）分别测这五步，主指标是**固定标签预算下的新类准确率**和**旧类遗忘**，而不是总体平均值。

---

## 1. 问题定义：三种“新”必须分开测

现在的 `HELD_OUT_CATEGORIES = ("cloth", "tool", "sponge", "sheet")` 是**封闭世界里的 open-set**：新类仍然来自同一套 primitive（cylinder/box/sphere/capsule/ellipsoid）和同一套材质分布。真实工作的“没见过”至少有三层：

| 层级 | 名称 | 例子 | 考验什么 |
|---|---|---|---|
| **Tier A** | 同分布新类 | 留出的 cloth/tool/sponge/sheet | 拒识 + 原型扩展 |
| **Tier B** | 组合新类 | 已知 primitive 的新尺寸/新材质/新颜色组合 | 组合泛化，不是全新几何 |
| **Tier C** | 新原语 / 传感器漂移 | 新几何体、复合物体、光照/触觉分布偏移 | 编码器是否还能把“新”变成可聚类的特征 |

**Tier A 是必做；Tier B 是主实验；Tier C 是 stretch。** 三层不能混在一起报一个平均数，否则“同分布新类”的好结果会掩盖“新原语”的失败。

---

## 2. 起点：已有零件与缺口

| 已有 | 位置 | 能做什么 |
|---|---|---|
| 开集拒识 | `pss/prototypes/scorers.py`、`run_p2_openset.py` | `unknown`、AUROC、`unseen_reject_at_95` |
| 原型准入 | `pss/prototypes/admission.py` | accept / review / reject |
| 增量加类 | `run_incremental_eval.py`、`MahalanobisScorer.add_class` | 加热更新、冻结/重估协方差、遗忘测量 |
| 主动选择 | `run_active_policy.py`、`pss/memory/active_policy.py` | uncertainty / margin / 多样性选择 |
| 覆盖审计 | `run_g_audit.py`、`pss/eval/g_audit.py` | 找最差 cell、组合补、响应曲线 |
| 数据生成 | `pss/sim/scripts/generate_*`、`pss/sim/scene.py` | 类别、视角、材质、触觉代理 |

**缺口**：

1. 没有“流”——现在是固定 train/val/test，不是按 episode 到来的数据。
2. 没有“自己决定学什么”——类别表、cell 定义、bar 都是人定的。
3. 没有“自己验证”——更新后是否变好，仍由脚本和人事先写死。
4. 没有“回滚”——增量更新没有自动的旧类防线。

本计划只补这四件事，不碰“通用智能”叙事。

---

## 3. 可证伪假设与预注册判据

| 假设 | 内容 | 可证伪预测 | 若成立的动作 |
|---|---|---|---|
| **O1 主动查询有效** | 聚类 + 不确定性 + 多样性的查询，比随机查询省标签 | 同预算下 `active − random ≥ 0.10` | 保留主动查询 |
| **O2 新类可吸收** | 少量标签后新类准确率接近 oracle | `active novel_acc@B ≥ 0.80` 且 ≥ 0.9×oracle | 进入 Tier C |
| **O3 自训会漂移** | 只用自己的高置信预测当标签会伤旧类 | `self_train` 旧类遗忘 ≤ −0.10，或新类 ≤ active − 0.10 | 禁止无外部校验的自训 |
| **O4 回放/回滚必要** | 无回放的增量更新会遗忘 | `no_replay` 遗忘 ≥ 0.10，`active` ≤ 0.03 | 把回放+回滚设为默认 |
| **O5 自动审计可行** | 模型自己聚出的 cell 能对上人工审计的最差 cell | top-1 命中 ≥ 0.6 或 Spearman ≥ 0.7 | 用模型 cell 替代人工 cell |
| **O6 真开放更难** | Tier C 显著难于 Tier A | Tier C 新类准确率 ≤ Tier A − 0.15，或未知 AUROC ≤ 0.80 | 把“新原语”单列为未解问题 |

**预注册主对照（≤6 个，Holm 校正）**：

| ID | 对照 | 任务 | 主指标 | 方向 |
|---|---|---|---|---|
| O1 | active vs random query | Tier B | 同预算新类 acc | active > random |
| O2 | active vs frozen | Tier B | 新类 acc@B | active ≫ frozen |
| O3 | self_train vs active | Tier B | 旧类遗忘 / 新类 acc | self_train 更差 |
| O4 | no_replay vs active | Tier B | 旧类遗忘 | no_replay 更差 |
| O5 | model_cells vs human_cells | Tier B 错误 | top-1 / Spearman | 对齐 |
| O6 | Tier C vs Tier A | 全部 | 新类 acc / 未知 AUROC | Tier C 更差 |

### 3.1 运行前写入 checker 的 bar

新增 `pss/eval/open_stream_preregistration.py`：

| arm | bar |
|---|---|
| `active_cluster` | `novel_acc_at_budget ≥ 0.80`；`old_forgetting ≤ 0.030` |
| `random_query` | `novel_acc ≤ active − 0.10`（正对照必须落后） |
| `frozen` | `novel_acc ≤ 0.20`（不学习就必须差） |
| `self_train` | `old_forgetting ≥ 0.10` 或 `novel_acc ≤ active − 0.10`（负对照必须失败） |
| `active_no_replay` | `old_forgetting ≥ 0.10`（消融必须更差） |
| `model_cells` | `top1_match ≥ 0.60` 或 `spearman ≥ 0.70` |
| `tier_c` | `unknown_auroc ≥ 0.80`；`novel_acc ≥ 0.60`（允许低于 Tier A） |

**规则与 g 计划一致**：bar 运行前固定；失败照实记录，不调 bar；smoke 不进结论；单 seed 只作诊断。**相对 bar**（如 `random_query ≤ active − 0.10`）在 checker 里实现为“同一 stream、同一预算下与 `active_cluster` 配对比较”，不是两个独立绝对值。

---

## 4. 数据与流协议

### 4.1 生成器

新增 `pss/sim/scripts/generate_open_stream.py`，复用 `pss/sim/scene.py` 的 `CATEGORY_SPECS` 和 `build_scene`：

```bash
python -m pss.sim.scripts.generate_open_stream \
  --out data/open_stream/tier_a \
  --tier a \
  --seen-categories cup,mug,bowl,plate,... \
  --novel-categories cloth,tool,sponge,sheet \
  --episodes 20 --per-episode 200 \
  --novel-rate 0.20 --drift-episode 10 --seed 0
```

每个样本的索引至少包含：

```json
{
  "episode": 0,
  "sample_id": "...",
  "path": "...",
  "label": "cup",
  "is_novel": false,
  "tier": "a",
  "drift": false,
  "split": "stream"
}
```

### 4.2 三种 tier 的构造

- **Tier A**：`seen` 取 `CATEGORY_SPECS` 的一个子集；`novel` 取 `HELD_OUT_CATEGORIES`。保证 novel 类别不出现在 encoder/projection 训练里。
- **Tier B**：从已知 primitive 生成新的尺寸/长宽比/颜色/材质组合，但**类别名仍可复用**；考察“同一类的新外观”和“已知原语的新组合”。
- **Tier C**：新 primitive（或复合物体）、光照/背景/触觉噪声的整体偏移。若 MuJoCo primitive 不够，用复合几何（两个 box/cylinder 拼）或传感器 shift；这是 stretch，允许只做“传感器漂移”子集。

### 4.3 流与标签预算

- 流长度：20 episodes；每 episode 200 个样本。
- 新类出现：episode 0–4 只出现已知类；episode 5 起按 `novel-rate` 混入新类；episode 10 起对部分已知类施加 drift。
- **drift 的层级要分清**：这里的 drift 是流内的轻度分布偏移（光照/材质/视角），用于测“更新后的鲁棒性”；**Tier C 才是原语级/传感器级的整体偏移**。两者不能混报。
- 标签预算：每 episode `B` 个查询（主实验 `B = 20`，敏感性 `B ∈ {10, 20, 50}`）。
- 查询返回 generator 的真实标签（模拟完美人工）；另设 `label_noise_10` 臂（10% 标签翻转）测鲁棒性。
- **泄露审计（O0 必做）**：检查 novel 类别/实例是否出现在 encoder、projection、原型 bank 的训练索引里；若泄露，重训 projection 时排除 novel 类别。

### 4.4 评估点

- 每个 episode 结束评估一次；
- 评估集：固定的 held-out probe（已知类 + 已出现新类 + 未出现新类），**不参与更新**；
- 新类在“首次出现后的第 1/3/5 个 episode”分别报一次，避免只报最终值。

---

## 5. 系统闭环

```
stream episode
  → 1. 表征（frozen encoder + projection）
  → 2. 拒识 / novelty score
  → 3. unknown 聚类（HDBSCAN on projected features）
  → 4. 查询策略（不确定性 + 多样性 + 聚类覆盖）
  → 5. 标签到达（oracle / noisy oracle）
  → 6. 准入（accept / review / reject）
  → 7. 更新（原型/均值；replay buffer；协方差策略）
  → 8. 自检（旧类 probe、校准、unknown AUROC）
  → 9. 通过则提交，否则回滚
  → 10. 记录 cell 提案（错误聚类）
```

关键实现复用：

- 表征：`load_projection` + `pss/projection/heads.py`；
- 拒识/打分：`MahalanobisScorer`、`fit_novelty_map`、`fuse_scores`；
- 聚类：新增 `pss/learning/open_stream.py`，用 `sklearn.cluster.HDBSCAN`（不可用时 DBSCAN）；
- 查询：新增 `uncertainty_diversity_query`，复用 `pss/memory/active_policy.py` 的 margin/特征；
- 准入：`pss/prototypes/admission.py` 的 accept/review/reject；
- 更新：`MahalanobisScorer.add_class` / `update_class`，加 replay buffer；
- 自检/回滚：新增 `UpdateGuard`，比较 probe 上的旧类 acc、ECE、unknown AUROC，超阈值回滚。

**回滚规则（运行前定）**：

| 观测 | 动作 |
|---|---|
| 旧类 probe acc 下降 > 3 pt | 回滚本次更新 |
| ECE 上升 > 0.05 | 回滚并重新校准 |
| unknown AUROC 下降 > 0.05 | 回滚，标记 novelty 漂移 |
| 新类 cluster 纯度 < 0.60 | 转 review，不自动 admit |

---

## 6. 实验设计

### O1：Tier A 封闭世界 open-set 流（先做，最便宜）

**目标**：建立“拒识 + 增量吸收”的基线，确认流协议和泄露审计没问题。

- 臂：`frozen` / `oracle_full`（全标签）/ `random_query` / `active_cluster`。
- 预算：B = 20；敏感性 B ∈ {10, 50}。
- 数据：Tier A，20 episodes，3 seeds。
- 输出：`eval/results/open_stream/o1_tier_a.json` + `O1_TIER_A.md`。
- 脚本：`run_open_stream.py --stage o1_tier_a`。

**判读**：若 `active_cluster` 新类 acc@B ≥ 0.80 且旧类遗忘 ≤ 0.03 → 进入 O2；否则先修拒识/聚类，不进入 O2。

### O2：Tier B 组合新类 + 主动查询

**目标**：测“同一原语的新外观/新组合”下，主动查询是否省标签。

- 臂：`frozen` / `oracle_full` / `random_query` / `novelty_query` / `active_cluster` / `active_no_cluster`（逐样本查询，不做聚类）/ `self_train`（负对照）/ `active_no_replay`（消融）。
- 预算：B = 20；标签效率曲线 B ∈ {10, 20, 50}。
- 指标：新类 acc@B、达到 0.9×oracle 所需标签数、旧类遗忘、cluster ARI/NMI、unknown AUROC、ECE。
- 输出：`o2_tier_b.json` + `O2_TIER_B.md`。

**停止规则**：标签效率曲线进入平台（连续两档增益 < 1 点或 CI 跨 0）。

### O3：Tier C 新原语 / 传感器漂移（stretch，可与 O2 并行）

**目标**：测“编码器没见过的分布”下，novelty 是否还可分、聚类是否还可信。

- 臂：O2 的主臂 + `encoder_swap`（DINOv2 或重新训练的 encoder）。
- 数据：Tier C；若新 primitive 成本太高，先做“光照/触觉/背景漂移”子集，并明确标注为 Tier C-partial。
- 指标：unknown AUROC、cluster ARI、新类 acc@B、拒识后的 review 率。
- 输出：`o3_tier_c.json` + `O3_TIER_C.md`。

**判读**：若 unknown AUROC < 0.80 或 ARI < 0.30 → Tier C 还不能自动聚类，转人工 review；结论单列为“未解”。

### O4：持续更新、回放与回滚

**目标**：在 O2 的流上加入 replay、协方差策略、UpdateGuard，测遗忘与回滚。

- 臂：`active_cluster`（无 replay/无回滚）/ `active_replay` / `active_replay_guard` / `oracle_full`。
- 指标：旧类遗忘、新类 acc、回滚率、false-accept（自检通过但 probe 变差的更新比例）。
- 输出：`o4_update.json` + `O4_UPDATE.md`。

**杀死标准**：若 `active_replay_guard` 的旧类遗忘仍 > 5 pt → 当前 readout 无法支持持续学习，回到 f（编码器/读出）。

### O5：自动 cell 发现 vs 人工审计

**目标**：把 g 审计的“人定 cell”换成“模型自己聚出的 cell”，看是否对得上。

- 输入：O2 的错误样本；`run_g_audit.py` 的人工 cell 作为参照。
- 方法：对错误样本的（pair_type / 区域 / 模态 / 置信度 / 聚类）做模型侧聚类，按 contribution 排序。
- 指标：top-1 最差 cell 命中、Spearman 排名相关、覆盖率（模型 cell 是否覆盖人工最差 cell 的样本）。
- 输出：`o5_cells.json` + `O5_CELLS.md`。
- 判读：top-1 ≥ 0.6 或 Spearman ≥ 0.7 → 自动审计可用；否则保留人工 cell。

### O6：最终矩阵与报告

- 汇总 O1–O5，给出：
  - Tier A/B/C 的“标签效率—新类准确率—遗忘”三角；
  - 哪些臂通过预注册 bar，哪些是近失；
  - 自动审计是否足以替代人工 cell；
  - 明确“不能声称”的清单（真实 VLM、真实人类标注成本、开放世界通用性）。
- 输出：`final_matrix.json` + `docs/experiments/PSS-开放类别流实验报告.md`。

---

## 7. 指标与统计口径

### 7.1 主指标

- **`novel_acc_at_budget`**：在固定标签预算 B 下，已出现新类的准确率。
- **`label_efficiency`**：达到 0.9×oracle 新类准确率所需的标签数。
- **`old_forgetting`**：更新后已知类在 probe 上的下降（越大越差）。

### 7.2 次要指标

- `unknown_auroc` / `unknown_reject_at_95`；
- `cluster_ari` / `cluster_nmi`（对真实新类标签）；
- `ece`（校准）；
- `update_false_accept`（自检通过但 probe 变差的更新比例）；
- `rollback_rate`；
- `review_rate`（被拒绝自动准入、转人工的比例）；
- `cell_top1_match` / `cell_spearman`。

### 7.3 统计

- 3 seeds；paired bootstrap 95% CI；主对照 Holm 校正。
- 选择只用 val/probe；test 只报一次。
- 每个更新臂必须配 `frozen` / `oracle_full` / `no-update` 对照；没有对照的增益不算。
- 标签噪声臂单独报告，不与干净标签混合。

---

## 8. 相位与依赖

| 相位 | 内容 | 依赖 | 出口判据 |
|---|---|---|---|
| **O0** | 流生成器、泄露审计、checker、smoke | — | 索引可复现；novel 不泄露；smoke 跑通 |
| **O1** | Tier A 流 + 拒识/增量基线 | O0 | `active` 新类 acc@B ≥ 0.80 或明确卡点 |
| **O2** | Tier B 主动查询 + 自训负对照 | O0/O1 | 同预算 `active − random ≥ 0.10` |
| **O3** | Tier C 新原语/漂移（stretch） | O0 | unknown AUROC ≥ 0.80 或单列未解 |
| **O4** | replay + UpdateGuard + 回滚 | O1/O2 | 旧类遗忘 ≤ 3 pt |
| **O5** | 自动 cell 发现 vs 人工审计 | O2 + g 审计 | top-1 ≥ 0.6 或 Spearman ≥ 0.7 |
| **O6** | 最终矩阵 + 报告 | O1–O5 | 分级结论 + 不可声称清单 |

**并行**：O1 与 O3 互不依赖，可并行；O2 与 O3 可并行；O4 依赖 O1/O2；O5 依赖 O2 与 g 审计结果。

---

## 9. 交付物

```
pss/sim/scripts/generate_open_stream.py       # 流生成器（tier a/b/c）
pss/learning/open_stream.py                   # 流、聚类、查询、UpdateGuard
pss/eval/scripts/run_open_stream.py           # --stage {o1_tier_a,o2_tier_b,o3_tier_c,o4_update,o5_cells,final}
pss/eval/open_stream_preregistration.py       # O1–O6 的 bar 与 checker
tests/test_open_stream.py
eval/results/open_stream/                     # o1–o5 / final_matrix
docs/experiments/PSS-开放类别流实验报告.md
```

---

## 10. 风险与边界

- **oracle 太完美**：合成标签是 generator 真值，模拟“完美人工”；`label_noise_10` 只能部分缓解，真实标注成本/歧义未建模。
- **Tier C 仍是有界开放**：新原语/漂移仍在 MuJoCo 和同一传感器模型内；“真正的外星数据”不在本计划范围。
- **编码器冻结**：如果新类在 projection 后不可分，拒识和聚类都会失败；O3 的 `encoder_swap` 是唯一出口，但也可能把问题从 g 推到 f。
- **自训确认偏差**：`self_train` 是负对照，不允许在没有外部 probe/人工抽检时启用。
- **类别数容量**：384 类已经接近当前池化协方差的舒适区；持续加类可能触发协方差漂移，必须带 replay/重估。
- **泄露风险**：novel 类别若出现在 projection/encoder 训练里，整个实验作废；O0 的泄露审计是硬门禁。
- **不能声称**：本计划不证明通用自主学习，不证明真实 VLM 上成立，不证明人类标注成本可接受。

---

## 11. 一句话

**先让模型学会说“我不知道”，再让它学会“花最少的标签把不知道变成知道”，最后让它学会“更新后自己检查、变差就回滚”；三级“新”分开报，Tier C 失败就照实写未解，不用 Tier A 的平均分掩盖。**
