# PSS 归因消融实验方案 v2：把"信息密度差一截"判到具体环节

> 日期：2026-11-27
> 依据：`docs/architecture/PSS-信息论定位.md` §5（含 §5.1–5.5 首轮结果）
> 目的：把 §5 的三段归因（encoder / projection / reasoning）做成**可预注册、可复跑、可证伪**的消融矩阵，并补上首轮协议的五个方法学漏洞。
> 状态：方案，尚未执行。首轮结果见 `eval/results/attribution_ablation/`、`..._view_instance/`、`eval/results/openset_scoring_diag/`。

---

## 0. 一句话

首轮已经把"128 维公共空间天生丢信息"在两个任务上打掉了一半：易任务 projection loss ≈ 0，难任务投影反而 **+10 点**（它压掉了视角 nuisance）；真正可测的短板是**读出层**与**对齐目标错配**。但首轮还不能把结论写死，因为协议里有五个漏洞：

1. `best-of-arms` 在 test 上取 max（选择偏差）；
2. raw 与 projected 的读出容量/归一化不匹配；
3. 易任务接近饱和，没有可测余量；
4. 难任务用的投影是按 **category** 对齐训练的（`checkpoints/category_pretrain_instance8_projection/metrics.json` 里 `level="category"`），却拿去考 **instance** 判别；
5. "reasoning" 臂实际只是融合头，不是显式关系/规划层。

本方案用**匹配读出 + 难度阶梯 + 预注册等价检验 + 目标匹配因子 + 随机投影对照**补齐，使"差一截到底差在哪"变成一张可判读的归因表。

---

## 1. 首轮证据盘点：哪些已判、哪些还不能判

| 结论 | 证据 | 状态 |
|---|---|---|
| 易任务上 128-d 投影几乎无损 | category：projection loss −0.005，reasoning gain −0.005（vs LR） | **已判**（限 16 类合成任务） |
| 难任务上投影是增益而非损失 | cross-view instance：projection loss −0.102（投影 +10.1 点，CI 排除 0） | **已判**（但投影目标错配，见下） |
| PSS 原型读出打不过通用 MLP | 难任务：PSS 0.833 vs proj MLP 0.878（−4.4，CI 排除 0） | **已判**（限该任务） |
| 可学融合头没有转正 reasoning gain | 门控 0.839 / 拼接 0.831 vs MLP 0.878 | **已判**（限该任务） |
| 开集低分是打分器问题，不是过拟合 | train−test gap 0.020；同表示换 Mahalanobis：0.935→0.999 | **已判**（16 类） |
| 维度/残差是否是瓶颈 | 只跑了监督瓶颈 64–512，未跑 PSS 投影的 D/residual 网格 | **未判** |
| 真正 bilevel（投影吃 reasoning loss）是否值得 | `run_bilevel` 的 joint vs sequential 不是匹配读出，且 category recognition 0.979 vs 0.984，不构成干净对照 | **未判** |
| 投影目标错配的代价 | instance8 投影 `level="category"`，从未训过 `level="instance"` | **未判（关键 confound）** |
| 编码器是否是最弱环 | 只有一次 DINOv2 换骨干，未做同读出阶梯 | **未判** |
| 信息量本身（而非精度）是否下降 | 只用了 accuracy 代理，没有 rank / Fisher / MI 估计量 | **未判（次要）** |

> **关键修正**：首轮的 `encoder_ceiling_raw = max(raw LR, raw MLP)` 不是"编码器上限"，而是**冻结算子下可读信息的下界**。真正的上限要用"微调编码器 + 同读出"来逼近。方案里统一改名为 `A_frozen(E)` 与 `A_ft(E)`。

---

## 2. 可证伪假设与预注册判据

先定记号：

- `E`：冻结编码器（专家/骨干），`P_θ(E)`：投影，`R`：读出族，`T`：任务。
- `A(E,P,R,T)`：该组合在 T 的 test 上的主指标（T2 用 closed-set acc，T3 用 top-1 re-ID）。
- `δ`：预注册等价边界；`Δ` 的 95% CI 完全落在 `[−δ, +δ]` 才算"等价于无损失/无增益"。

| 假设 | 内容 | 可证伪预测 | 若成立的动作 |
|---|---|---|---|
| **H1 维度假说** | 公共空间容量不足，128-d 装不下任务信息 | 在 T2/T3 上 `A(D=512) − A(D=128) > δ`，且随 D 单调上升 | 扩公共维度 / 上 residual / 换更强投影 |
| **H2 投影假说** | 投影/whiten 丢掉 raw 中可读的任务信息 | 同读出族下 `A(P(E)) − A(E) < −δ` | 检查 whiten 维度、residual、是否压死原始特征 |
| **H3 读出假说** | 瓶颈在 reasoning 的读出容量/形式，不在空间 | 同表示下 `A(R_MLP) − A(R_PSS) > δ`，且显式关系特征能补回 | 换可学读出 / 双头 / 加关系层，不扩维度 |
| **H4 协同适应假说** | "为对齐最优" ≠ "为任务最优" | 任务匹配投影 `A(P_instance) − A(P_category) > δ`；bilevel `A(P_joint) − A(P_align) > δ` | 做双层训练 / 任务匹配投影 |
| **H5 编码器假说** | 编码器是最弱环，DPI 真正卡住 | `A(E_DINOv2) − A(E_0) > δ`，且投影损失随骨干增强而缩小 | 换专家，别再加推理模块 |

**预注册主对照（≤6 个，Holm 校正）**：

| ID | 对照 | 任务 | 方向 |
|---|---|---|---|
| C1 | 同读出族下 `P(E) − E`（D=128） | T2 | H2 |
| C2 | `A(D=512) − A(D=128)` | T2 | H1 |
| C3 | `A(MLP) − A(PSS proto)`（同表示） | T2 | H3 |
| C4 | `A(P_instance) − A(P_category)` | T2 | H4 |
| C5 | `A(joint) − A(frozen)`（同头同数据） | T2 | H4 |
| C6 | `A(E_DINOv2) − A(E_0)`（同读出） | T2 | H5 |

其余（校准、开集、re-ID、分模态、T1）为**次要/探索性**，不进主校正，报告时明确标注。

**等价边界（先按 n≈384 定，跑前用 pilot 的 bootstrap 半宽复核）**：

- `δ_acc(T2) = 0.020`：n=384 时约 2 点的差异才稳定可测；小于此值只能报"不确定"，不能报"无差异"。
- `δ_acc(T1) = 0.010`：仅作 sanity，不用于等价结论（天花板效应）。
- `δ_reid(T3) = 0.020`。
- 若主对照 CI 宽度 > `2δ`，启动 §7 的 view-rotation 增样，而不是降低 δ。

---

## 3. 因子设计

### 3.1 因子表

| 因子 | 取值 | 说明 |
|---|---|---|
| 编码器 `E` | `E0` 现有小专家；`E1` DINOv2 ViT-S；`E2` 可选更强 VLM | 回答"公共空间 vs 高维冻结表示" |
| 投影变换 `P` | identity；PCA-whiten only；random Gaussian；linear；MLP（当前）；MLP+residual | 把"降维"与"对齐"分开 |
| 公共维度 `D` | 64 / 128 / 256 / 512 | H1 主因子 |
| residual | off / on | `ProjectionHead(residual=...)` 已支持 |
| whiten | none / 96 / 128 | 查 whiten 是否压死 geometry |
| 投影目标 | `level="category"` / `level="instance"` | H4 目标匹配 |
| 读出 `R` | LR；参数匹配 MLP；PSS 原型；可学门控/拼接；距离式（Mahalanobis/kNN） | 同表示比读出 |
| 任务 `T` | T1 category；T2 cross-view instance；T3 near-identical re-ID；T4 关系组合（可选） | 难度阶梯 |
| 估计量 `M` | accuracy / ECE / NLL / AUROC+OP / re-ID top-1 / rank·Fisher·CKA | 精度 ≠ 信息 ≠ 校准 |

### 3.2 臂清单（按优先级）

**必做（主表）**

| 臂 | 表示 | 读出 | 目的 |
|---|---|---|---|
| `raw_*` | E0 raw，仅 train 标准化 | LR / MLP | H2 参照 |
| `pca_*` | E0 → whitening 到 D=128，L2 | LR / MLP | 纯降维对照 |
| `rand_*` | E0 → 随机高斯 128-d，L2 | LR / MLP | **JL 对照**：若随机投影不掉分，说明"对齐"不是关键 |
| `lin_*` | `Linear(in,128)+L2`，监督对齐 | LR / MLP | 线性投影对照 |
| `mlp_*` | 当前 `ProjectionHead` | LR / MLP | 主投影 |
| `res_*` | `ProjectionHead(residual=True)` | LR / MLP | H2 |
| `D64/128/256/512_*` | 同目标、同 epoch、同 seed，仅 D 变 | LR / MLP | H1 |
| `proj_inst_*` | `train_projections(level="instance")` | LR / MLP | H4 目标匹配 |
| `pss_proto_*` | 当前投影 | 原型余弦融合 | H3 |
| `learned_fusion_*` | 当前投影 | 门控 / 拼接 | H3 |

**进阶（条件触发）**

| 臂 | 触发条件 | 目的 |
|---|---|---|
| `joint_*`（最小 bilevel） | C1/C4 显示投影是瓶颈 | H4：投影吃 task loss |
| `E1_dino_*` | 主表完成后 | H5 |
| `E0_ft_*`（微调编码器） | 需要真上限 | 量化 encoder loss |
| `rel_*`（显式关系特征 + 读出） | C3 显示读出是瓶颈 | 把"融合头"升级成"关系 reasoning" |
| `T4` 组合任务 | 要声称 reasoning 优势 | 与 §4 的组合/关系结论对齐 |

### 3.3 读出容量匹配（关键控制）

- 每个表示上训练 **同族** 读出：线性对线性、MLP 对 MLP；**不要**用 `max(raw LR, raw MLP)` 去比 `max(proj LR, proj MLP)`。
- MLP 隐藏层固定 256，但输入维度不同会导致参数量不同；报告实际参数量，并加一个 **capacity-capped** 变体：先 `Linear(d_in → 128)` 再共用 `128→256→C` 头，使 raw/proj/D 各臂的读出容量一致。
- 标准化：raw 用 train 拟合的 StandardScaler；projected 已 L2 归一化。主表两套都跑：`std` 与 `l2`，确认结论不依赖几何约定。
- 所有超参（LR 的 C、MLP 的 epoch/weight decay）只在 **val** 上选；test 只报告一次。

---

## 4. 归因量的定义（匹配版）

对读出族 `R` 与任务 `T`，定义（括号内为符号含义）：

```
A_frozen(E, R)        = A(E, identity, R)            # 冻结特征可读下界
A_proj(E, P, R)       = A(E, P, R)                   # 投影后可读
A_ft(E, R)            = A(finetuned E, R)            # 编码器容量参照（可选）

Δ_projection(R)       = A_proj(E, P, R) − A_frozen(E, R)      # <0 损失，>0 增益
Δ_reasoning(R)        = A_proj(E, P, R_PSS) − A_proj(E, P, R) # >0 结构读出有增量
Δ_coadapt(R)          = A_proj(E, P_joint, R) − A_proj(E, P_align, R)
Δ_target(R)           = A_proj(E, P_instance, R) − A_proj(E, P_category, R)
Δ_encoder(R)          = A_ft(E, R) − A_frozen(E, R)
```

**报告规则**：

1. 每个 `Δ` 必须带 **同读出族** 的 95% bootstrap CI；主表同时给出 linear 与 MLP 两行，不做 max。
2. `Δ_projection` 与 `Δ_reasoning` 不合并成"总损失"标量；三段分开报，避免用一段的收益掩盖另一段的损失。
3. `A_frozen` 是最弱读出的下界，不是信息上限；`A_ft` 才是接近上限的参照。若不做微调，明确写"encoder loss 未测"。
4. 所有臂一律报告，包括负结果与失败臂；不允许只报最好。

**归因表模板**（最终产出）：

| 段 | 量 | 估计 | 95% CI | δ | 判定 | 解读 |
|---|---|---:|---|---:|---|---|
| encoder | `Δ_encoder` | — | — | 0.02 | 未测 / 显著 / 等价 | 是否最弱环 |
| projection | `Δ_projection(MLP)` | — | — | 0.02 | 损失 / 增益 / 等价 | 空间是否丢信息 |
| dimension | `A(512)−A(128)` | — | — | 0.02 | 有斜率 / 平坦 | 容量是否瓶颈 |
| reasoning | `Δ_reasoning(MLP−proto)` | — | — | 0.02 | 读出瓶颈 / 无差 | 结构是否值钱 |
| co-adapt | `Δ_coadapt` | — | — | 0.02 | 显著 / 等价 | 是否值得 bilevel |
| target | `Δ_target` | — | — | 0.02 | 显著 / 等价 | 对齐目标错配的代价 |

---

## 5. 任务与数据划分（难度阶梯）

| 任务 | 数据 | 类/样本 | 主指标 | 角色 |
|---|---|---|---|---|
| **T1 category** | `data/category_pretrain` | 16 类；train 1152 / val 192 / test 192；开集 384 | acc；AUROC+OP | sanity，**饱和，不作等价结论** |
| **T2 cross-view instance** | `data/category_pretrain_instance8` | 384 实例；train v0–5 = 2304 / val v6 = 384 / test v7 = 384；开集 768 | acc | **主任务**（有 15–20 点余量） |
| **T3 near-identical re-ID** | `data/tracking_bench_v2` + 现有 pair 构建 | pair-level | top-1 / mAP | 次要；查细粒度实例信息 |
| **T4 关系/组合**（可选） | P5b 渲染关系基准 | — | 关系 F1 / 组合 acc | 仅在要声称 reasoning 优势时跑 |

**划分纪律**：

- T2 的训练/验证/测试按 **视角** 不重叠（v0–5 / v6 / v7）；同一实例跨 split 是**设计意图**（跨视角再识别），不是泄漏。
- T2 主结果先按固定 split 跑；若主对照 CI 宽度 > `2δ`，启动 **view-rotation 3-fold**：test view ∈ {5,6,7}，每个 fold 重训投影与读出，报告 fold 均值与逐 fold 值。禁止只看最好的一折。
- T1 的 `test_category` 只用于开集排序；T2 的开集用 held-out 类别的实例（768 行），作为探索性指标。
- 任何在 test 上看过结果后再改的超参，下一轮必须换一个新的 held-out 视角/种子，不能在同一 test 上反复调。

---

## 6. 度量与统计

### 6.1 度量子集

| 类别 | 指标 | 用途 |
|---|---|---|
| 判别 | accuracy、macro-F1 | 主指标 |
| 校准 | ECE、NLL、Brier | 与 accuracy 分开报（首轮已知 PSS 校准更差） |
| 开集 | AUROC、**unseen reject @95% seen accept**、coverage | 防"拒绝一切刷 AUROC" |
| 检索 | top-1、mAP | T3 |
| 表示诊断 | 线性探针、有效秩/participation ratio、Fisher ratio、raw↔proj CKA、kNN-MI 估计 | 次要；解释"信息是否真丢" |

- **有效秩**：`exp(entropy(singular values))`，回答"512 维实际用了几个自由度"。
- **Fisher ratio**：类间散度/类内散度，与读出无关，作为"可分性"的独立读数。
- **kNN-MI**：用 kNN 分类器的条件熵估计 `I(Z;Y)` 的代理；高维小样本有偏，只作方向性证据，不作主判据。
- **CKA**：raw 与 projected 的表示相似度，定位投影改动了什么。

### 6.2 统计协议

- 学习型臂（MLP / 投影 / bilevel）：**5 seeds**，报告均值 ± std；主对照用 seed 配对。
- 确定性臂（LR / 原型）：test 样本 **paired bootstrap 10,000 次**，报告 95% 百分位 CI。
- 主对照 6 个：Holm 校正，`α=0.05`。
- **等价检验**：对 H2/H1 的"无损失"结论用 TOST，CI 必须落在 `±δ` 内；"CI 跨 0"不等于"等价"。
- 多重比较、臂选择、超参选择全部记录在 `metrics.json`；报告所有臂，不只报赢家。
- 预注册文件在跑之前写死：假设、δ、主对照、seed、停止规则；跑后只追加结果。

---

## 7. 执行阶段与成本

| 阶段 | 内容 | 臂/训练量 | 通过门槛 |
|---|---|---|---|
| **A. 仪表与 smoke** | 新脚本 `run_attribution_grid.py`；在 `category_pretrain_smoke` 跑通；校验 split、参数量、无泄漏 | 1 次 smoke | 结果可复现、维度/参数打印正确 |
| **B. 投影网格** | P ∈ {pca, rand, linear, mlp, mlp+res} × D ∈ {64,128,256,512} × 目标 {category, instance} × 3 seeds；T2/T3 读出 | ~5×4×2×3 = 120 次投影训练（先用 D={128,512} 剪枝到 ~24 次） | C1/C2/C4 有 CI |
| **C. 最小 bilevel** | 投影+同读出联合训练 vs 冻结投影；同头同数据；5 seeds | 2 臂 × 5 seeds | C5 有 CI；若 C5 等价则停止 |
| **D. 编码器阶梯** | E0 vs DINOv2，同投影/同读出；可选微调上限 | 2–4 臂 × 3 seeds | C6 有 CI |
| **E. 估计量与报告** | 校准/开集/re-ID/表示诊断；生成归因表与判读 | 复用 B–D 表示 | 报告落盘、结论与判据一致 |

**成本控制**：B 阶段先跑 `D={128,512} × {mlp, res} × 目标={category,instance}`（约 8×3 seeds）；只有当 `A(512)−A(128)` 的 CI 上界 > δ/2 时，才补 64/256。C 阶段只在 C1 或 C4 显著时启动。D 阶段只在 B/C 都判"非投影/非读出"时启动。

**新脚本接口（建议）**：

```bash
python -m pss.eval.scripts.run_attribution_grid \
  --data data/category_pretrain_instance8 \
  --experts checkpoints/category_pretrain_instance8_experts \
  --regime view-instance \
  --dims 128 512 --transforms raw pca rand linear mlp res \
  --levels category instance \
  --readouts lr mlp proto \
  --seeds 0 1 2 \
  --out eval/results/attribution_v2
```

实现上直接复用：

- 特征/读出：`pss/eval/scripts/run_attribution_ablation.py` 的 `MLPClassifier`、`Bottleneck`、`build_bank_labels`、`fused_scores`、`_acc`、`_bootstrap_delta`；
- 投影训练：`pss/projection/train.py:train_projections(out_dim=..., residual=..., whiten_dim=..., level=...)`；
- 开集打分：`pss/eval/scripts/run_openset_scoring_diag.py` 的 Mahalanobis/kNN/energy；
- 编码器替换：`pss/eval/scripts/run_encoder_swap.py` 的 DINOv2 路径；
- bilevel：`pss/learning/bilevel.py`，但需要包一层"最小任务读出"而不是完整 binder/relations。

---

## 8. 判读表（预注册）

| 观测 | 结论 | 下一步 |
|---|---|---|
| C1 ≥ −δ 且 C2 ≤ δ | **投影/维度不是瓶颈**；128-d 没有丢掉可测任务信息 | 不要扩维度；查读出（C3） |
| C1 < −δ | 投影确实丢信息 | 查 whiten/residual；扩 D；必要时 bilevel |
| C2 > δ（D 越大越好） | 容量是瓶颈 | 扩公共维度 / residual / 更强投影 |
| C2 ≤ δ 但 C1 < −δ | 损失不在"维度"而在"变换" | 查 whiten 是否压死某模态；换线性/residual |
| C3 > δ（MLP 显著高于原型） | **读出层是瓶颈**，不是空间 | 双头：原型做开放集/热更新，可学头做闭集精排；或蒸馏回原型 |
| C3 ≤ δ（原型不输） | 结构读出在本任务不吃亏 | 保留原型，优势转向开放集/热更新 |
| C4 > δ（instance 投影更好） | **目标错配**是主要损失 | 任务匹配投影；把 reasoning loss 回传投影（C5） |
| C5 > δ（联合训练更高） | 协同适应损失可测 | 做双层训练；量化收益 vs 工程成本 |
| C5 ≤ δ | 在本规模下协同适应不是主因 | 不做 bilevel；把预算给编码器/关系层 |
| C6 > δ（DINOv2 更高） | **编码器是最弱环**（DPI 卡住） | 换专家；投影/读出保持 |
| C6 ≤ δ | 当前编码器不是瓶颈 | 回到 C1–C3，不要盲目换骨干 |
| 开集 AUROC 高但 OP/coverage 差 | 打分器在"拒绝一切" | 用 OP 与 coverage 作为主开集指标 |

**一票否决**：任何主对照的 CI 宽度 > `2δ` 时，不得写"无差异/等价"，只能写"本轮不可判"，并触发 view-rotation 或增加 seed。

---

## 9. 有效性威胁与对策

| 威胁 | 首轮表现 | 对策 |
|---|---|---|
| test 上取 max 造成选择偏差 | `raw best = max(LR, MLP)` | 同族读出、val 选超参、test 只报一次 |
| 读出容量不匹配 | raw 与 proj 输入维度不同 | capacity-capped 头 + 报参数量 |
| 任务饱和 | T1 0.98，delta 方差大 | 主任务改 T2/T3；报 headroom |
| 投影目标错配 | instance8 投影 `level="category"` | 增加 `level="instance"` 臂（C4） |
| 把融合头当 reasoning | 首轮 `learned_fusion_*` 只是融合 | 改名为 fusion head；T4 才叫 reasoning |
| 单一视角 split 的运气 | T2 只用了 view 7 | view-rotation 3-fold（CI 过宽时） |
| 多重比较 | 十几个臂共用 test | 6 个主对照 + Holm；其余标探索 |
| 高维 MI 估计有偏 | 未做 | MI 只作次要方向性证据；主判据用精度 + 等价检验 |
| 合成代理/小编码器 | 全部数据为模拟 | 结论限定在 synthetic proxy；预训练 VLM 需重跑 |
| 把 `A_frozen` 当上限 | 首轮叫 "encoder ceiling" | 改名；真上限需 `A_ft` |

---

## 10. 产物

| 产物 | 路径 | 内容 |
|---|---|---|
| 预注册 | `eval/prereg/attribution_v2.json` | 假设、δ、主对照、seed、停止规则、时间戳/commit |
| 脚本 | `pss/eval/scripts/run_attribution_grid.py` | 网格执行、匹配读出、CI/TOST/Holm |
| 原始指标 | `eval/results/attribution_v2/metrics.json` | 所有臂、所有 seed、所有指标 |
| 主报告 | `eval/results/attribution_v2/ATTRIBUTION_V2_REPORT.md` | 归因表 + 判读表 + 负结果 |
| 汇总 | `reports/claim_gaps/attribution_v2/REPORT.md` | 对 `PSS-信息论定位.md` §5 的结论修订建议 |

报告必须包含：

1. 主表（全部臂，含失败臂）；
2. 归因表（§4 模板）；
3. 6 个主对照的 mean + CI + Holm 校正后的 p；
4. 等价检验结论（哪些是"真无差异"，哪些是"不可判"）；
5. 与 §5.1–5.5 的差异说明（哪些旧结论被修正、哪些被加强）；
6. 明确边界：synthetic、小编码器、T2 的实例跨 split 设计、未测真实 VLM。

---

## 11. 停止规则

1. **预算停止**：B 阶段 24 次投影训练后若 C1/C2 的 CI 仍跨 `±δ`，停止扩网格，改做 view-rotation 或增加 seed；不无限加臂。
2. **结论停止**：C1 ≥ −δ 且 C2 ≤ δ 且 C3 ≤ δ 时，"空间/维度是瓶颈"在本数据上被否决；后续只做 C6（编码器）与 T4（关系），不再做投影维度消融。
3. **否证停止**：若 C1 < −δ 且 C2 > δ 且 C4 > δ 同时成立，则"差一截"落在投影与目标错配，直接进入 bilevel，不跑编码器阶梯。
4. **不可判停止**：任何主对照 CI 宽度 > `2δ` 且加 seed/加折后仍不可判，写"本轮不可判"并给出所需样本量，不硬下结论。

---

## 12. 边界（与 §6 一致）

- 本方案回答的是"在**冻结编码器 + 公共空间**这条路径上，损失落在哪一段"，不是"公共空间能否超过 LLM 隐向量"的完整答案；后者需要真实预训练 VLM/LLM 骨干重跑。
- `A_frozen` 是下界，不是信息上限；`Δ_encoder` 未做微调时标注"未测"。
- T2 的实例跨 train/test（同实例不同视角），结论是"跨视角再识别"，不是"新实例泛化"。
- kNN-MI/有效秩是诊断量，不是严格的互信息估计；高维小样本下有偏，不能单独支撑结论。
- 所有结论仅在 `data/*` 的合成代理与现有小专家上成立；换真实数据/骨干需重跑本方案。
