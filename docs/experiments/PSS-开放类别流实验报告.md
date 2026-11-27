# PSS 开放类别流实验报告

> 日期：2026-11-28
> 上游：`docs/plans/PSS-开放类别流实验计划.md`
> 结果目录：`eval/results/open_stream/`
> 状态：O0–O6 已执行；结果见 `eval/results/open_stream/`。

---

## 0. 一句话

**先让模型说“我不知道”，再让它用有限标签把“不知道”变成“知道”，最后用 replay + 回滚防止更新把自己弄坏。** 分布内（Tier A/B）闭环可行：Tier B 上 active 查询 **0.844**（random 0.731），replay 到 **0.918**，自训 **0.000**；漂移（Tier C）暴露真正的风险：继续增量更新会把旧类准确率和 novelty 检测一起打崩（unknown 0.999 → **0.44**），冻结模型反而稳定。自动 cell 发现仍不成熟（Spearman 0.413，top1=0）。

---

## 1. 协议

### 1.1 三级“新”

| 层级 | 定义 | 本次实现 |
|---|---|---|
| **Tier A** | 同分布新类 | 16 个 seen 类别 + 4 个 held-out 类别（`cloth/tool/sponge/sheet`） |
| **Tier B** | 已知类别的新实例 | 384 个实例按 80/20 切 seen/novel（307 / 77） |
| **Tier C** | Tier A + 传感器漂移 | episode 10 起对光照/触觉施加全局 gain/bias/noise |

### 1.2 流构造

- 不重新渲染 MuJoCo，而是从现有 `data/category_pretrain` / `category_pretrain_instance8` 索引**重采样**：每个 episode 200 个样本，novel rate 0.2，novel 从 episode 5 开始。
- Tier C 对漂移 episode 的样本做后处理：光照 `rgb*gain+bias`（保持 uint8），触觉 `tac*gain+noise`。
- 每个标签按 50/50 切成 update / probe；probe 不参与查询，避免“刚标过的样本”抬高准确率。
- 再保留 **20% 的 novel 标签**作为 `unseen_only`：永不查询，只用于测量 unknown AUROC。

### 1.3 臂

| 臂 | 查询 | 更新 |
|---|---|---|
| `frozen` | 无 | 无 |
| `oracle_full` | 全部样本 | 每 episode 用 replay 重拟合 |
| `random_query` | 随机 | 增量 add/update |
| `novelty_query` | 最低分 | 增量 |
| `active_no_cluster` | 最低分 + 无聚类 | 增量 |
| `active_cluster` | 聚类 + 每簇一个 + 轮转 + 噪声单例 | 增量 |
| `self_train` | 最高分自标 | 增量（负对照） |
| `active_no_replay` | active_cluster | 增量（无 replay） |
| `active_replay` | active_cluster | replay 重拟合 |
| `active_replay_guard` | active_cluster | replay + 旧类/ECE/unknown 回滚 |

主指标：`novel_acc`（已出现新类的 probe 准确率）、`old_forgetting`（base probe 下降）、`unknown_auroc`、`cluster_ari`。

---

## 2. O0：smoke

6 episodes × 20 样本，budget 5，1 seed：

| arm | novel acc | old acc | forgetting | unknown AUROC | labels |
|---|---:|---:|---:|---:|---:|
| `frozen` | 0.000 | 0.990 | 0.000 | 1.000 | 0 |
| `active_cluster` | 0.700 | 0.990 | 0.000 | 1.000 | 30 |

链路（生成 → 特征 → 聚类 → 查询 → 增量 → probe）跑通；这个规模只作协议检查，不进结论。

---

## 3. O1：Tier A（同分布新类）

3 seeds × 20 episodes × 200 样本，budget 20；16 个 seen 类别 + 4 个 held-out 类别，其中 20% novel 标签作为 `unseen_only` 永不查询。

| arm | novel acc | old acc | forgetting | unknown AUROC | cluster ARI | 查询中新类占比 | labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| `frozen` | 0.000 | 0.990 | 0.000 | 0.996 | n/a | — | 0 |
| `oracle_full` | 1.000 | 0.995 | −0.005 | 0.999 | 0.175 | 0.117 | 7700 |
| `random_query` | 0.885 | 0.991 | −0.002 | 0.996 | 0.146 | 0.116 | 400 |
| `novelty_query` | 0.896 | 0.906 | 0.083 | 0.995 | 0.262 | 0.745 | 400 |
| `active_cluster` | **0.920** | 0.969 | **0.021** | **0.999** | **0.539** | 0.715 | 400 |

### 3.1 预注册检查

- 通过：`active_cluster.novel_acc = 0.920 ≥ 0.80`；`active_cluster.old_forgetting = 0.021 ≤ 0.03`。
- 失败：
  - `random_query.novel_acc = 0.885 > 0.70`（绝对上限）；
  - `random_query_vs_active = −0.035`，没有达到 `random ≤ active − 0.10`。

### 3.2 标签效率 sweep

| budget | arm | novel acc | old acc | forgetting | unknown AUROC |
|---:|---|---:|---:|---:|---:|
| 5 | `random_query` | 0.851 | 0.993 | −0.003 | 0.996 |
| 5 | `novelty_query` | 0.853 | 0.995 | −0.005 | 0.975 |
| 5 | `active_cluster` | 0.853 | 0.958 | 0.031 | 0.994 |
| 10 | `random_query` | 0.757 | 0.993 | −0.003 | 0.997 |
| 10 | `novelty_query` | 0.864 | 0.995 | −0.005 | 0.990 |
| 10 | `active_cluster` | **0.927** | 0.958 | 0.031 | **0.997** |
| 20 | `random_query` | 0.885 | 0.991 | −0.002 | 0.996 |
| 20 | `novelty_query` | 0.896 | 0.906 | 0.083 | 0.995 |
| 20 | `active_cluster` | 0.920 | 0.969 | 0.021 | 0.999 |

**active 的优势出现在 B=10**：0.927 vs random 0.757，差 **0.170**；B=5 时三者都接近 ceiling（~0.85），B=20 时 random 追到 0.885。B=10 的相对 bar（`random ≤ active−0.10`）**满足**；但 checker 的绝对 fallback `random ≤0.70` 仍不满足（0.757）。这再次说明预注册的绝对上限不适合“random 也很强”的 ceiling 区间。

### 3.3 判读：这是 ceiling effect，不是 active 失效

- 在 B=20（共 400 标签）下，random 已经 0.885；active 要满足“领先 10 pt”必须到 **0.985**，而 oracle 上限是 1.000。这个相对 bar 在当前预算/类别数下几乎不可满足。
- active 仍然在三个地方更好：novel 最高（0.920）、forgetting 最低（0.021）、unknown AUROC 最高（0.999）；novelty-only 虽然 novel 接近（0.896），但 forgetting 0.083，说明**聚类+多样性不只是省标签，还保护旧类边界**。
- active 查询中 71.5% 是新类，random 只有 11.6%，说明查询确实聚焦到了新类；但新类只有 4 个、每类样本很多，random 的少量新类样本也足够学好。
- 预注册的 `random ≤0.70` 和“领先 0.10”是在假设 random 较弱时写的；实测 random 接近 ceiling。**bar 不调整，记为 failure + ceiling-effect 解释**。
- 低预算 sweep（B=5/10/20）用来检验“active 的优势是否出现在标签更少时”；见 `O1_BUDGET_SWEEP.md`。

### 3.4 一个实现层面的关键修复

第一版增量更新每次只传一个样本给 `MahalanobisScorer.update_class`，而该 API 会把类均值**覆盖**为这批样本的均值，等于“最后一条标签覆盖前面所有标签”。改成按类维护 cumulative buffer 后，random 从 0.514 → 0.885、active 从 0.490 → 0.920。这个修复也影响 O3：O3 的旧数字偏悲观，修复后需要重跑。

---

## 4. O2：Tier B（已知类别的新实例）

3 seeds × 20 episodes × 200 样本，budget 20；384 个实例按 80/20 切 seen/novel（307 / 77）；base bank 只拟合 seen 307 类。

| arm | novel acc | old acc | forgetting | unknown AUROC | 查询中新类占比 | labels |
|---|---:|---:|---:|---:|---:|---:|
| `frozen` | 0.000 | 0.951 | 0.000 | 0.880 | — | 0 |
| `oracle_full` | 0.997 | 0.993 | −0.042 | 0.966 | 0.125 | 15536 |
| `random_query` | 0.731 | 0.832 | 0.119 | 0.845 | 0.133 | 400 |
| `novelty_query` | 0.820 | 0.954 | −0.003 | 0.921 | 0.482 | 400 |
| `active_no_cluster` | 0.820 | 0.954 | −0.003 | 0.921 | 0.482 | 400 |
| `active_cluster` | **0.844** | 0.941 | **0.010** | **0.923** | 0.448 | 400 |
| `self_train` | **0.000** | 0.909 | 0.042 | 0.889 | 0.000 | 400 |

### 4.1 预注册检查

- 通过：`active_cluster.novel_acc = 0.844 ≥ 0.80`；`old_forgetting = 0.010 ≤ 0.03`。
- 通过（相对）：`random_query_vs_active = 0.731 − 0.844 = −0.113 ≤ −0.10`。
- 通过（负对照）：`self_train` 新类 0.000，比 active 低 0.844 ≥ 0.10。
- **失败**：`random_query.novel_acc = 0.731 > 0.70`（绝对 fallback，差 3.1 pt）。

### 4.2 判读

- active 比 random 新类高 **11.3 pt**（0.844 vs 0.731），forgetting 从 0.119 降到 0.010；相对 bar 成立。
- `novelty_query` 与 `active_no_cluster` 完全一致（0.820），说明在这个任务上**聚类本身只贡献 +2.4 pt**（0.844−0.820）；主要收益来自“优先查低分样本”，不是聚类。
- `self_train` 一个新类都没发现（0.000），并且 old acc 从 0.951 掉到 0.909：**只用自己的高置信预测当标签，无法发现新类，还会轻微伤旧类**。负对照成立。
- 唯一的失败是 `random ≤0.70` 的绝对 fallback；paired 规则（random ≤ active−0.10）是主对照，已通过。bar 不调整，记录为“绝对 fallback 近失”。

---

## 5. O3：Tier C（漂移）

3 seeds × 20 episodes × 200 样本，budget 20。漂移从 episode 10 开始。

| arm | novel acc | old acc | forgetting | unknown AUROC | cluster ARI |
|---|---:|---:|---:|---:|---:|
| `frozen` | 0.000 | 0.990 | 0.000 | **0.999** | n/a |
| `random_query` | **0.736** | 0.911 | **0.078** | 0.459 | — |
| `novelty_query` | 0.435 | 0.630 | 0.359 | 0.451 | — |
| `active_cluster` | 0.512 | 0.750 | 0.240 | 0.441 | — |

### 5.1 关键观测

- **漂移前**（episode 8–9）：old acc ~0.95–0.99，unknown AUROC ~0.98–1.00，更新是安全的。
- **episode 10 漂移出现**：即使有 cumulative buffer，old acc 仍明显下降，unknown AUROC 从 ~1.00 掉到 **0.44–0.46**；旧类边界被漂移样本推走，novelty 分数失去校准。
- **冻结模型反而稳定**：`frozen` 全程 old 0.990、unknown 0.999；它不吸收漂移，所以不会被漂移带偏。
- **active 在漂移下反而更差**：active 的 novel 0.512 < random 0.736，forgetting 0.240 > random 0.078。它越聚焦 novelty/unknown，越容易把漂移样本吸进 bank，旧类掉得更多。
- 预注册 `tier_c` 要求 `novel_acc ≥ 0.60`、`unknown_auroc ≥ 0.80`；`active_cluster` 两项都不达标（0.512 / 0.441）。

### 5.2 判读

**Tier C 的问题不是“学不到新类”，而是“更新把旧类和 novelty 检测一起弄坏”。** cumulative buffer 修复了“均值被最后一条标签覆盖”的 bug，把 random 的 old acc 从 0.661 抬到 0.911、forgetting 从 0.328 降到 0.078；但 **unknown AUROC 仍然崩到 0.46**，说明 pooled covariance 和 novelty 校准都必须随漂移重估，单靠更新均值不够。

**动作**：Tier C 上应先做 drift 检测/协方差重估，再吸收新类；否则“自主学习”会变成“自毁”。主动查询在漂移下不是免费的：它越追 novelty，越容易把 OOD 样本写进 bank。

---

## 6. O4：replay / guard

Tier B，3 seeds × 20 episodes，budget 20；三个臂只改变更新方式，查询策略相同（active_cluster）。

| arm | novel acc | old acc | forgetting | unknown AUROC | rollback rate |
|---|---:|---:|---:|---:|---:|
| `active_no_replay` | 0.844 | 0.941 | 0.010 | 0.923 | 0.000 |
| `active_replay` | **0.918** | **0.961** | **−0.010** | **0.957** | 0.000 |
| `active_replay_guard` | 0.918 | 0.961 | −0.010 | 0.957 | 0.000 |

### 6.1 预注册检查

- 失败：`active_no_replay.old_forgetting = 0.0098 < 0.10`；`no_replay_vs_replay = 0.0195 < 0.07`。
- 即“无 replay 必须遗忘 ≥0.10”的负对照在 **Tier B 不成立**：Tier B 没有漂移，增量更新的遗忘本来就只有 1 pt。
- `active_replay_guard` 与 `active_replay` 完全相同，rollback rate 0：replay 本身没有让 probe 变差，guard 没有触发。

### 6.2 判读

- **replay 的收益是准确率，不是（在 Tier B）遗忘**：novel 0.844 → 0.918（+7.4 pt），old 0.941 → 0.961（+2.0 pt），unknown AUROC 0.923 → 0.957。原因是 replay 重估了 pooled covariance，而 no-replay 一直用旧精度。
- **遗忘负对照需要漂移才能触发**：Tier C（O3）才是 no-replay/增量更新真正崩掉的场景；在无漂移的 Tier B 上，预注册的 `old_forgetting ≥0.10` 写得太强，实测只有 0.010。bar 不调整，记录为“负对照未被激活”。
- guard 在 Tier B 是 no-op；它的价值应在 Tier C 上验证（本轮 O4 按计划只跑 Tier B，未覆盖 Tier C 的 guard）。

---

## 7. O5：模型 cell vs 人工 cell

第一次尝试用 Tier A 的 16 个类别去对比 A 线 384 实例审计，任务空间不同，得到 top1=0、Spearman=−0.32 的**伪结果**；已改为用 O2 Tier B 的 per-instance `class_acc` 对比同一套 384 实例审计（A-line `per_class`）。

| 指标 | 值 | bar |
|---|---:|---|
| model worst | `block_v022` | — |
| human worst | `block_v011` | — |
| top-1 match | **0** | ≥ 0.60 |
| Spearman | **+0.413** | ≥ 0.70 |

- 两项 bar 都不达标：自动 cell 发现**还不成熟**。
- 但 top-20 表里有多个两类误差都为 1.000 的实例（`bowl_v020`、`box_v003`、`carton_v001/002/003`、`plate_v002/006`、`rod_v012`），说明模型不是完全看不见人工审计的坏格子；是**排序不够准**，top-1 尤其脆弱（多个类并列 1.0）。
- 结论：模型侧错误聚类可以作为“候选 cell”提示，但不能替代人工/审计定义的 cell；下一步需要更细的错误归因（模态、视角、置信度），而不是只看 per-class 误差。

---

## 8. O6：最终分级

结果：`eval/results/open_stream/final_matrix.json` / `OPEN_STREAM_FINAL.md`。

### 8.1 分级

| 层级 | 主臂 | 结论 | 是否达标 |
|---|---|---|---|
| **Tier A** 同分布新类 | `active_cluster` | novel 0.920，forgetting 0.021；B=10 时 active 0.927 vs random 0.757 | 绝对 bar ✅；相对/random 绝对 fallback ❌（ceiling） |
| **Tier B** 新实例 | `active_cluster` | novel 0.844，forgetting 0.010；random 0.731/0.119；self_train 0.000 | active + 相对 bar ✅；random 绝对 fallback ❌（差 3 pt） |
| **Tier C** 漂移 | `active_cluster` | novel 0.512，forgetting 0.240，unknown 0.441；random 0.736/0.078/0.459；frozen 0.000/0.000/**0.999** | ❌ |
| **O4** replay/guard | `active_replay` | 0.918/0.961/−0.010/0.957；no-replay 0.844/0.941/0.010 | replay 收益 ✅；遗忘负对照未被激活 |
| **O5** 自动 cell | `model_cells` | top1 0，Spearman +0.413 | ❌（bar 0.60/0.70） |

### 8.2 一句话总评

**在分布内（Tier A/B），“发现不知道 → 聚类/查询 → 少量标签吸收”的闭环是可行的：**
- active 查询比 random 在新实例上高 **11.3 pt**（0.844 vs 0.731），并把 forgetting 从 0.119 压到 0.010；
- replay 再把 novel 抬到 **0.918**、old 抬到 0.961；
- 自训负对照发现 **0 个新类**，说明外部标签/验证不可省。

**在分布漂移下（Tier C），“自主更新”会自毁：**
- 漂移出现后，更新把 unknown AUROC 从 ~0.999 打到 **0.44–0.46**，old acc 也明显下降；
- active 越追 novelty 越糟（novel 0.512 < random 0.736，forgetting 0.240 > 0.078）；
- **冻结模型反而稳定**（old 0.990、unknown 0.999）：不学漂移，就不会被漂移带偏。

**自动审计还不成熟：** 模型 per-class 错误与人工 A 线审计的 Spearman 只有 **+0.413**，top-1 不一致；模型可以提示候选坏格子，但不能替代人工 cell 定义。

### 8.3 可以/不可以声称

**可以声称（限合成流、限本协议）**

- 分布内新类/新实例可以用有限标签吸收；active 查询在标签紧张时优于随机（Tier B +11.3 pt，Tier A B=10 +17 pt）。
- 自训（只用高置信预测）不能发现新类，还会轻微伤旧类。
- replay 重估协方差能同时提升新类和旧类；在无漂移的 Tier B，遗忘不是主要风险。
- 漂移下继续更新会破坏 novelty 检测；冻结是更安全的默认，直到做 drift 检测/协方差重估。
- 自动 cell 发现只能做候选提示，不能替代人工审计。

**不可以声称**

- “通用自主学习/自我迭代已实现”：只在合成流、固定编码器、完美 oracle 标签下验证。
- “真实 VLM 上同样成立”：未测。
- “replay/guard 已经解决漂移”：O4 只跑了 Tier B，Tier C 的 guard 未验证。
- “自动 cell 发现可用”：top1=0、Spearman 0.413，未达 bar。

---

## 9. 边界

- **oracle 完美**：标签是 generator 真值，模拟完美人工；真实标注成本/噪声未建模。
- **Tier C 有界**：漂移是光照/触觉 gain/bias，不是全新传感器或开放世界；结论不能外推到真实 VLM。
- **编码器冻结**：新类若在 projection 后不可分，拒识/聚类会失败；这是 f 的瓶颈，不是 g。
- **预算/规模**：20 episodes × 200 样本是 pilot 规模；更大流的曲线形状可能不同。
- **不能声称**：通用自主学习、真实人类标注成本可接受、真实 VLM 上成立。

---

## 10. 交付物

```
pss/sim/generator.py                         # generate_open_stream + build_open_stream_from_index + drift
pss/sim/scripts/generate_open_stream.py      # 生成器 CLI（MuJoCo / 索引两种路径）
pss/eval/open_stream.py                      # 流、拒识、聚类、查询、更新、guard、指标
pss/eval/open_stream_preregistration.py      # O1–O6 的 bar 与 checker
pss/eval/scripts/run_open_stream.py          # --stage o0..final
tests/test_open_stream.py
eval/results/open_stream/
```
