# PSS 读出层与推理层优化执行方案

> 日期：2026-11-27
> 依据：`docs/architecture/PSS-信息论定位.md` §5.5–5.7、`eval/results/attribution_v2/ATTRIBUTION_V2_REPORT.md`、`eval/results/version_b/VERSION_B_REPORT.md`
> 上游结论：归因 v2 把总缺口 2.9 点拆成 encoder 0.0 / projection 0.9 / **reasoning 2.0（69%）**；组合 OOD 上 PSS 1.000 vs 平坦策略 0.573，但优势受**关系头覆盖**钳制（exact-4 0.571 → OOD 0.682）。
> 状态：方案，尚未执行。

---

## 0. 一句话

**读出层的工作是"补税"（把 2 点缺口收回来，且不许拿开放集/热更新/审计去换）；推理层的工作是"兑现架构卖点"（把组合 OOD 的 1.000 从"关系头恰好覆盖测试区域"变成"谓词可组合 + 覆盖可测量"）。** 前者是工程，后者是研究赌注，资源应按这个比例分。

**不做的事（先写死）：**

- 不扩 `PUBLIC_DIM`：§5.6 已测 128→512 只 +0.8～1.1，CI 跨 0。
- 不换编码器：encoder loss 实测 0.0。
- 不用通用 MLP 直接替换原型：闭集 +0.5，开集（max-softmax 口径）−5.8，且丢掉 0.17 ms 热更新、零遗忘、分模态冲突弃权。
- 不追 re-ID 上的 sparse_k1 0.996：那是 §4 已承认的 LLM/transformer 主场，不在本方案范围。

---

## 1. 起点：三个已测事实 + 一个陷阱

| # | 事实 | 数字 | 出处 | 对方案的含义 |
|---|---|---|---|---|
| F1 | 同表示下原型打不过判别式 MLP | 0.846 vs 0.866（−2.0） | §5.6 | 但有坑，见 F1' |
| F1' | 该 −2.0 **seed 不稳** | seed0 +4.9、seed1 −1.0、均值 +2.0 | §5.6 | 任何单 checkpoint 的 4～5 点都不能当结论；方案必须 3 seed + 配对 CI |
| F2 | 开集低分是**打分器**问题 | 同 MLP 表示：max-softmax 0.935 → Mahalanobis **0.999** / kNN 0.993 | §5.5 | 第一优先级、零训练成本、收益最大 |
| F3 | 组合 OOD 的闸门是**关系头覆盖** | exact-4 1.000→OOD 1.000；0.571→OOD 0.682 | §5.7 | 先测覆盖曲线，再谈搜索升级 |
| F4 | **朴素 bilevel 有害** | 冻结投影+新头 0.876 → 联合微调 0.848 | §5.6 | 协同适应不能靠 CE 直接回传，必须加约束/蒸馏 |

> 读法：F1/F2 说明"读出"这一层有便宜可捡；F3 说明"推理"这一层的瓶颈不在搜索器而在**知识覆盖**；F4 说明"让投影一起学"这条最直觉的路已被证伪，要用蒸馏绕开。

---

## 2. 目标与预注册判据

### 2.1 目标（按优先级）

| 优先级 | 目标 | 判据 |
|---|---|---|
| P0 | 读出层"补税" | 闭集 ≥ MLP − 0.5 点，**同时**开集 AUROC ≥ 0.99、operating point（95% seen accept 下 unseen reject）≥ 0.95、热更新 ≤ 1 ms |
| P1 | 组合 OOD 去"覆盖依赖" | held-out 区域 exact-4 ≥ 0.85 时 OOD ≥ 0.95；或给出覆盖→OOD 的完整曲线，证明瓶颈已转移 |
| P2 | 全流程可复现 | 3 seed、配对 bootstrap CI、val-only 选模、失败项照实报 |

### 2.2 预注册文件

新增 `pss/eval/readout_reasoning_preregistration.py`，照 `pss/projection/calibration_preregistration.py` 的可执行写法：**先写 bar，再跑**；不达标就报失败，不许调 bar。

```python
# 建议结构（示意）
BARS = {
    "readout_closed_minus_mlp": 0.005,      # 允许落后 MLP 0.5 点
    "readout_auroc_min": 0.990,
    "readout_unseen_reject_at_95": 0.950,
    "hot_update_ms_max": 1.0,
    "seed_std_max": 0.010,                  # 治 §5.3 的 0.19 塌缩
    "composition_ood_min": 0.950,
    "relation_exact4_min": 0.850,
}
```

判据的方向必须与 §5.5 一致：**只看 operating point 和覆盖率，不用 AUROC 单指标**（拒绝一切也能刷高 AUROC）。

---

## 3. 工作流 A：读出层

> 目标不是"换掉原型"，而是**把原型当闸门、把可学读出当精排**，在保留热更新与可审计的前提下收回缺口。

### A0（先做，半天）闸门召回诊断

任何重排器的上限都是 `gate_recall@k`。先量：

- 原型 top-1/3/5 的召回（在 test split，384 类）；
- 分模态 top-k 召回（vision / tactile / geometry）。

若 `gate_recall@3 < 0.97`，A2 的收益天花板就很低，应优先 A1 而非 A2。产出：`eval/results/readout_opt/gate_diag.json`。

### A1 距离开集打分器（S，零训练）

新增 `pss/prototypes/scorers.py`：

```
CosineProtoScorer    # 现状，保留为默认
MahalanobisScorer    # 池化类内协方差 + shrinkage（λ 在 val 上选）
KNNScorer            # k ∈ {5,10,25}，距离加权
```

`recognize(..., scorer=None)` 增加可选参数，Level 4 与 `unknown` 阈值都走 scorer。`PrototypeBank` 增加可选的 `pooled_cov` / `neighbors` 字段，旧 `bank.npz` 不传仍可用。

**工程风险（必须写进报告）：** 384 类 × 6 shot 下每类协方差奇异，**只用池化协方差**；kNN 存储随类别线性增长，需要每条类别的近邻上限与压缩策略；Mahalanobis 的类更新是 O(D²)，仍可为 O(1)/类，但要实测。

依据：§5.5 同表示 Mahalanobis 0.999、projected kNN 0.998。

### A2 判别式重排（M）

新增 `pss/prototypes/rerank.py`：闸门给出 top-k 候选后，用一个小 MLP 在**候选集内**做 softmax，只有候选集内的类别参与竞争。

关键设计（否则会重蹈 §5.3 覆辙）：

- **训练分布必须模拟部署时的候选集**：用闸门实际产出的 top-k 候选去做训练样本，而不是全局 384 类 CE；否则模型学的是"全局判别"，部署时被候选集截断。
- 只在**已接受**样本上生效；被闸门拒绝的仍走弃权路径。
- 头容量要小（≤ 2 层、≤ 128 hidden），避免 §5.3 那种 seed 塌缩。

预期上限 = `gate_recall@k`（A0 已量）。收益若 < 1 点，按 §7 的杀死标准处理。

### A3 度量蒸馏（M/L，替代朴素 bilevel）

不碰部署投影（F4 已证伪），改为在**冻结公共空间之上**学一个小的残差度量 `g`，再在 `g` 空间重建原型：

```
loss = CE(候选集) + λ_kd · KL(MLP 教师) + λ_geo · ||g(zi)·g(zj) − zi·zj||
```

- `g` 训练完冻结，热更新只重算类质心（仍 O(1)/类）。
- `λ_geo` 是保几何项，防止学完的度量破坏原型的开放集性质。
- 教师用 §5.6 里最强可读的 MLP（0.866），学生不要求超过教师，只要求**在保持开集的前提下逼近**。

新增 `pss/projection/distill.py` + 脚本 stage `distill`。产出：闭集、开集 AUROC、operating point、热更新延迟、零遗忘验证。

### A4 分模态可靠度校准（M，最后做）

现状是固定权重 `{vision:1, tactile:0.5, geometry:1}`。§5.3 的自由门控塌缩（std 0.19），所以**不做自由门控**，改做受约束版本：

- 每模态分数先过 `pss/projection/confidence_map.py` 的单调校准映射 → 概率；
- 融合用 log-odds 加权，权重在 val 上估计并**限制在 [0.2, 2]**；
- 分模态权重可解释、可审计，不引入逐样本门控。

指标：分模态 ECE、融合精度、冲突弃权 accepted accuracy（现有指标）。

---

## 4. 工作流 B：推理层

> 目标不是"再刷高 P5b 的 1.000"，而是**把 1.000 的前提条件测出来**：它依赖关系头的区域覆盖；覆盖不足时 OOD 掉到 0.682。这一层的核心问题是"覆盖在哪、如何补、补到什么程度收益饱和"。

### B0 覆盖曲线（S，先做）

新增 `pss/reasoning/relations/coverage.py`：

- 统计 `(relation, region)` 单元的标注支撑数与探针置信度，输出覆盖矩阵；
- 按训练区域覆盖率 {40%, 60%, 80%, 100%} 做扫描，对应 OOD exact-4 与 planning success。

这一步回答的是："**多标数据**能买到多少 OOD？" 如果 80% 覆盖就到 0.95，那 B1 优先；如果到 100% 仍 < 0.8，瓶颈就不在覆盖，转 B2/B3。

### B1 主动补覆盖（S/M）

按覆盖矩阵的 `支撑数 × 预期 OOD 使用频次` 排序，用固定预算挑选要补的单元，生成针对性 episode。只做"补哪一格"，不改探针结构。

验收：同容量探针下 held-out 区域 exact-4 ≥ 0.85（当前 0.571）。

### B2 可组合谓词（M）

当前 held-out 区域 per-label 0.893、exact-4 0.571，说明**部分标签坏了**。加关系代数一致性约束并消融：

- 反对称：`left(a,b) = right(b,a)`；
- 传递/局部一致：`above` 在区域内的一致性；
- 帧一致性：世界系 vs 相机系谓词不混用（`Signature.frame` 已有此设计）。

复用 `pss/learning/bilevel.py` 的 `antisymmetry_loss`。指标：held-out exact-4、per-label 分解。

### B3 搜索升级（M）

现状是 BFS over placements。加两个对照臂：

- **A***：`h` = 未满足的目标约束数（可采纳），`cost` = 动作数；
- **失败重规划**：goal check 失败时把失败原子序列写回 `PlanCache`（已有"只降置信、不覆盖已验证计划"的语义），并触发重新搜索。

指标：success、invalid actions、steps、latency。**BFS 保留为默认臂**，A* 必须同时证明不劣于 BFS 才可切换。

### B4 配对 episode 复跑（S）

现报告 §3 明说"各方法使用同一分布但不是同一批 episode，seed 级未配对"。修法：固定 episode 集合，所有臂在同一批上评测，做配对 bootstrap CI。这是让"PSS − MLP +42.7 点"从"效应量很大"变成"统计上可报告"的最便宜一步。

---

## 5. 相位与依赖

| 相位 | 内容 | 依赖 | 出口判据 |
|---|---|---|---|
| **P0** | 预注册文件、A0 闸门召回、B4 配对基线 | — | 基线数字与 CI 落盘 |
| **P1** | A1 打分器、B0 覆盖曲线 | P0 | A1 开集达标；B0 给出覆盖→OOD 曲线 |
| **P2** | A2 重排、B1 补覆盖 | P1 | A2 有增量或按杀死标准停；B1 exact-4 ≥ 0.85 |
| **P3** | A3 蒸馏、B2 谓词、B3 搜索 | P2 | A3 开集不降；B2/B3 优于 BFS 基线 |
| **P4** | A4 校准、集成、报告 | P3 | 全表落盘 + 诚实边界 |

**并行建议：** A 线（读出）与 B 线（推理）互不依赖，可双线并行；但 **A0 与 B4 必须先于所有训练**，否则后面的收益无法归因。

---

## 6. 工程约束

| 约束 | 原因 |
|---|---|
| `recognize()` 签名向后兼容，默认路径行为不变 | `pss/inference.py`、`run_p2_openset.py`、baselines 都在调 |
| 新增字段全部 optional，旧 `bank.npz` 可加载 | 部署包与复现包不能同时失效 |
| 每个新模块配单测；默认路径加 golden 回归 | `tests/` 现有 `test_inference.py`、`test_heads.py` 风格 |
| 统一用 `run_attribution_grid.py` 的 `task_split` / `fused_scores` / `SEEDS` | 数字必须与 §5.6 可比 |
| 3 seed、val-only 选模、test 只报一次 | 防选择偏差（首轮已有此教训） |
| 报告必须给 operating point 与覆盖率 | §5.5 的评测口径 |

**新增文件清单：**

```
pss/eval/readout_reasoning_preregistration.py
pss/prototypes/scorers.py
pss/prototypes/rerank.py
pss/projection/distill.py
pss/reasoning/relations/coverage.py
pss/reasoning/planner/astar.py
pss/eval/scripts/run_readout_reasoning.py   # --stage {gate,scorers,rerank,distill,coverage,search,paired}
tests/test_readout_scorers.py
tests/test_readout_rerank.py
tests/test_readout_distill.py
tests/test_relation_coverage.py
tests/test_composition_search.py
```

---

## 7. 杀死标准（先定，防止事后解释）

| 观测 | 结论 | 动作 |
|---|---|---|
| A1 换打分器后开集 AUROC 仍 < 0.95 | 瓶颈不在打分器 | 退回 A3，并在报告里否定"距离打分器就够" |
| A1 + A2 合计闭集 < +1.0 且开集无增益 | 读出缺口不是容量/形式问题，是**目标错配** | 停 A2，只做 A3；结论写成"读出税只能靠蒸馏，不能靠重排" |
| A3 蒸馏闭集、开集双不达标 | "协同适应损失可被蒸馏消解"在本任务上不成立 | 如实记录，回到 §2.2 的开放式结论 |
| B0 满覆盖（100%）时 OOD 仍 < 0.80 | 瓶颈不在数据覆盖 | 跳过 B1，转 B2/B3 |
| B2 谓词一致性不提升 held-out exact-4 | 关系头是"记忆"不是"代数" | 记录为负面结果，B3 只做搜索效率 |
| B3 A* 不优于 BFS | 搜索不是瓶颈 | 保留 BFS，不引入复杂度 |

---

## 8. 预期产出

1. **一张读出层对照表**：原型 / +Mahalanobis / +kNN / 重排 / 蒸馏，列 closed-set acc、AUROC、operating point、热更新 ms、seed std。
2. **一张覆盖曲线**：训练区域覆盖率 vs held-out exact-4 vs OOD success，标注饱和点。
3. **一份预注册报告**：达标项、未达标项、按 §7 触发的分支，全部落盘到 `eval/results/readout_opt/`。
4. **对 `PSS-信息论定位.md` 的增量**：把 §5.6 的"读出税 2.0 点"更新为"可收回 / 不可收回"的实测值；把 §5.7 的"1.000 依赖覆盖"更新为覆盖曲线。

---

## 9. 一句话收尾

**读出层最多值 2 点，推理层值 40 点——但读出层的 2 点便宜、可立刻验证，推理层的 40 点依赖一个必须先量清楚的覆盖前提。** 所以顺序是：先 A0/A1 + B0 把两条曲线量出来，再决定钱花在重排、蒸馏，还是补覆盖与搜索上。

---

## 10. P0 结果（2026-11-27，已完成）

> 详细报告：`eval/results/readout_opt/P0_REPORT.md`
> 预注册：`pss/eval/readout_reasoning_preregistration.py`
> 产物：`eval/results/readout_opt/gate.json`、`eval/results/version_b_paired/`

| 项 | 结果 | 对 P1 的影响 |
|---|---|---|
| 闸门召回（生产投影，test） | top-1 0.833 / top-3 **0.953** / top-5 **0.971** / union@5 0.987 | A2 可行；主臂 k=5，对照 k=3，候选集取 union |
| val→test gap | 0.904 → 0.833（−7.1） | scorer 的 val 选参必须冻结，test 只报一次 |
| 分模态 top-1 | vision 0.727 / tactile 0.263 / geometry **0.206** | A4 第一优先：geometry 权重 1.0 明显过配 |
| 配对 OOD：PSS − MLP | **+0.490 [0.438, 0.543]** | §5.7 的结论升级为统计可报告 |
| 配对 OOD：transformer − MLP | **−0.098 [−0.168, −0.025]** | 端到端策略 OOD 显著更差，不是容量问题 |
| 配对 OOD：PSS − held-out 头 | +0.305 [0.240, 0.370] | B1/B2 要补的正是这一段 |

预注册分支：`gate_decision(0.953, 0.971) → rerank_viable_at_k5`。0.97 阈值是保守判据，未在结果后修改；口径说明见预注册模块 docstring。

---

## 11. P1 结果与方案修正（2026-11-27，已完成）

> 详细报告：`eval/results/readout_opt/P1_REPORT.md`
> 产物：`eval/results/readout_opt/scorers.json`、`eval/results/relation_coverage/`

### A1：瓶颈是融合，不是打分器

| arm | 闭集 acc | Δ vs MLP | AUROC | reject@95 | hot ms |
|---|---:|---:|---:|---:|---:|
| `proto` | 0.833 | −0.044 | 0.999 | 0.997 | 0.020 |
| `proto_mahalanobis` | 0.763 | −0.115 | 0.991 | 0.947 | 0.017 |
| `concat_cosine` | 0.831 | −0.047 | 0.999 | 0.997 | 0.028 |
| **`concat_mahalanobis`** | **0.930** | **+0.052** | 0.997 | 0.977 | 0.005 |
| `concat_knn` | 0.810 | −0.068 | 0.999 | 0.996 | 3.200 |

- 预注册检查：`proto` / `proto_mahalanobis` / `proto_knn` **全部未达标**，按协议记录为失败，不调 bar。
- `concat_mahalanobis` 不在预注册表内，只能作为探索性发现；P2 若采用必须先新增预注册。
- **方案修正：A4（融合）升为第一优先。** 打分器带来约 +10 点，但分模态融合会把它全部吃掉（−16.7）。在融合修好之前，A1 的收益无法落到部署形态。

### B0：覆盖的是方向，不是数量

| 覆盖区域数 | exact-4（均值） | OOD（均值） | OOD 范围 |
|---:|---:|---:|---|
| 2/5 | 0.372 | 0.443 | 0.130–0.750 |
| 3/5 | 0.693 | 0.740 | **0.280–1.000** |
| 4/5 | 0.825 | 0.937 | 0.835–1.000 |
| 5/5 | 1.000 | 1.000 | 1.000 |

- 同为 3/5：`ACG` 达 1.000/1.000，`BCD` 只有 0.571/0.280，且 `BCD` 跨 3 个 head seed 完全一致 → 系统性外推失败。
- **方案修正：B1 的覆盖矩阵从"区域计数"改为"方向/坐标跨度"**；优先补能覆盖坐标极值的区域对。

### P2 的调整

1. A4 先行：解决分模态融合，再谈打分器与重排。
2. A2 的候选集在 `concat_mahalanobis` 上重跑 A0 召回后再定 k。
3. `concat_mahalanobis` 作为新 arm 重新预注册（3 seeds、开集 bar、热更新口径）。
4. B2 用更多子集 × 3 seeds 验证"凸包外推"假设；若成立，谓词层应显式归一化坐标。

---

## 12. P2 / A4 结果（2026-11-27，已完成）

> 详细报告：`eval/results/readout_opt/P2_REPORT.md`
> 产物：`eval/results/readout_opt/fusion_ablation.json`、`FUSION_REPORT.md`

| arm | closed-set acc | Δ vs MLP | AUROC | reject@95 | hot ms |
|---|---:|---:|---:|---:|---:|
| `proto_cosine`（部署） | 0.833 | −0.044 | 0.999 | 0.997 | 0.000 |
| `lda_concat` | 0.930 | +0.052 | 0.997 | 0.977 | 0.019 |
| **`lda_concat_whiten`** | **0.948** | **+0.070** | 0.998 | **0.987** | 0.018 |
| `lda_concat_weighted` | 0.930 | +0.052 | 0.997 | 0.982 | 0.018 |
| `lda_per_modality_fused`（负对照） | 0.763 | −0.115 | 0.991 | 0.947 | 0.017 |

- **主候选是 `lda_concat_whiten`**：白化后再拼接，比裸 concat 高 1.8 点，开集达标。
- **受约束权重收益为零**：val 选出 `{v1.5, t0.5, g2.0}`，test 与裸 concat 相同（0.930）。条件化比可靠度加权更重要。
- **负对照确认 P1 解释**：分模态打分后融合比 whiten concat 低 18.5 点。
- 预注册 2 项 near-miss：`lda_concat` / `lda_subset_dropout` 的 reject@95 = 0.9766 < 0.98，按协议记失败，不调 bar；`lda_concat_whiten` 全部通过。
- **增量遗忘 −1.6 点 / 50 类**：不能声称零遗忘；部署需定期重估协方差。
- 缺失模态：vision 组合 0.896–0.901，单 tactile/geometry ≈ 0.26；7 个子集模型各自拟合。

**下一步：** A2 的闸门改在 `lda_concat_whiten` 上重测 top-k 召回；`lda_concat_whiten` 作为部署候选重新登记（多划分、真实骨干、长期增量）。

---

## 13. P3 / A2 结果：A 线收口（2026-11-27）

> 详细报告：`eval/results/readout_opt/P3_REPORT.md`
> 产物：`eval/results/readout_opt/rerank_eval.json`、`RERANK_REPORT.md`

| 项 | 值 |
|---|---:|
| 闸门 `lda_concat_whiten` acc | 0.948 |
| 闸门 recall@1 / @3 / @5 | 0.948 / 0.995 / **1.000** |
| 闸门 AUROC / reject@95 / hot | 0.998 / 0.987 / 0.022 ms |
| 重排 k=3 / k=5 / k=10 | 0.948 / 0.945 / 0.945 |
| 同表示完整 MLP | 0.807 |
| 预注册 `rerank_bilinear`（k=5，bar 0.955） | **失败（0.945）** |

- **候选不是瓶颈**：recall@5 = 1.000，重排可自由选择全部候选，仍无增量。
- **学习度量不胜过估计度量**：bilinear 初始化于 LDA 精度，768 行训练后 val 0.961 → test 0.945，过拟合。
- **完整 MLP 更差（0.807）**：白化放大低方差方向，LDA 用池化协方差压回去，MLP 在 6 shot/类 下学不出来。容量不是关键。
- **预注册规则触发：A2 停止。** 线性 LDA 度量已是该表示的读出上限。

### A 线收口

| 阶段 | 读出 | 闭集 acc | 相对部署原型 |
|---|---|---:|---:|
| 起点 | 分模态原型 cosine | 0.833 | — |
| A4 | `lda_concat_whiten` | **0.948** | **+11.5** |
| A2 | + 学习式重排 | 0.948 | +11.5（无增量） |

**A 线收益已全部由 A4 拿到。** 后续只做两件事：A5（长期增量/协方差刷新，解决 A4 的 −1.6 点/50 类漂移）与部署化（白化 PCA + 均值 + 精度序列化，7 个模态子集模型）；然后是真实骨干复跑。

---

## 14. P4 / A5 结果：A 线最终收口（2026-11-27）

> 详细报告：`eval/results/readout_opt/P4_REPORT.md`
> 产物：`eval/results/readout_opt/incremental_eval.json`、`INCREMENTAL_REPORT.md`

| policy | base after | forgetting | added acc | overall | refits | refit s |
|---|---:|---:|---:|---:|---:|---:|
| `incr_frozen` | 0.943 | **0.031** | 0.927 | 0.935 | 1 | 0.11 |
| **`incr_refit_10`** | **0.961** | **0.013** | **0.948** | **0.954** | 20 | 2.53 |
| `incr_refit_25` | 0.956 | 0.018 | 0.948 | 0.952 | 8 | 1.02 |
| `incr_refit_all` | 0.961 | 0.013 | 0.948 | 0.954 | 193 | 24.75 |
| `cosine_whiten` | 0.802 | 0.068 | 0.768 | 0.785 | 0 | 0.00 |
| `batch_joint`（参照） | 0.961 | n/a | 0.948 | 0.954 | 1 | 0.14 |

- **刷新消除 drift**：冻结 3.1 → 每次加 10 类重估 1.3；overall 0.935 → 0.954，与联合训练完全相同。
- **频率拐点在 10**：10 与每类重估精度相同（20 vs 193 次，2.5s vs 24.8s）；25 类一次掉 0.5 点。
- **残余 1.3 点是加类稀释**：`incr_refit_all` 与 `batch_joint` 同模型，后者也掉 0.013，故预注册 1 点 bar 低于可达地板，记 near-miss。
- **全协方差不可省**：`cosine_whiten` 起点 0.870、added 0.768，判别力与遗忘同时变差。
- 成本：重估 ≈126 ms/次（本规模）；`add_class` 测得的 3 ms/类主要是 reservoir 拷贝，不是纯均值更新。

**部署建议：** 均值即时更新；协方差每 10 类（或类数增长 5%）重估；保留特征 reservoir（或每类充分统计量）。**A 线到此收口**：原型 0.833 → LDA-whiten 0.948（+11.5），重排无增量，增量刷新与联合训练持平。





