# PSS 架构上界复核与下一阶段计划

> 日期：2026-11-28
> 上游：`PSS-读出与推理优化总报告.md` §4.4、`PSS-g因子优化报告.md` §6–§7、`PSS-联合判别读出.md` §7
> 目的：把“每个模块是否到达上界”从一次性判断变成**分级、可证伪、带预注册 bar** 的复核计划，并给出下一阶段架构优化顺序。
> 状态：已执行 U1–U6 与审计修复（2026-11-28）；结果见 `docs/experiments/PSS-架构上界复核报告.md`、`PSS-架构上界复核-第二阶段.md`、`PSS-架构上界复核-第三阶段.md`。

---

## 0. 一句话

**当前能声称的只是“局部上界”：解码器、维度（宽度）、推理容量在当前表示与当前任务上饱和；编码器、无损度、真实几何从未测过；g 是唯一确实没饱和的因子。** 下一阶段不继续调 f，而是做两个“换基质”实验（真实骨干 A 线、真实几何 B 线）和一个“补定义”实验（公共空间组合保真度）。三者都通过，才能把 0.948/1.000 写成“上界”；否则只能写“局部上界”。

---

## 1. 判定口径

### 1.1 “上界”的三级定义

| 级别 | 定义 | 判据 | 当前状态 |
|---|---|---|---|
| **B1 局部上界** | 固定表示、固定任务，换更强的同族读出/容量没有增量 | 下一强臂 − 当前臂 ≤ 噪声（≤0.5 pt 或 CI 跨 0） | 读出/维度/容量：**已到**（0.948 vs rerank 0.945；MLP 0.807；128→512 CI 跨 0） |
| **B2 迁移上界** | 同一排序与增益在真实骨干/真实几何上保持 | U1/U2 的主 bar 通过 | **未测** |
| **B3 联合上界** | f 与 g 在同一基质上测量，`A_current ≈ A_full` 且 `L_gap ≤ 0.05` | U3 的主 bar 通过 | **未测** |

**只有 B3 才能把 `PSS 上界 ≈ f × g` 当作实测陈述；B1 只能写“该表示上的局部上界”。**

### 1.2 三种“没到上界”

| 类型 | 含义 | 模块 |
|---|---|---|
| **已到局部上界** | 再调没有增量，但不是全局上界 | 解码器、维度（宽度）、推理容量（P5b 规模） |
| **未测** | 不是瓶颈，但也没有“到上界”的证据 | 编码器、无损度、真实几何、真实 VLM |
| **未饱和** | 有可测的剩余差距 | g 覆盖 / 任务匹配 |

关键：**f×g 是乘积，非瓶颈因子的边际收益为 0。** 所以“每个模块都到上界”不是目标；目标是找到绑定约束。

---

## 2. 模块复核表

| 因子/模块 | 当前证据 | 判定 | 剩余空间 | 对应实验 |
|---|---|---|---|---|
| **编码器** | A 线“未换”；encoder loss≈0；微调 0.875 = PCA 0.875；DINOv2 只在 16 类 category 代理上 `A_current=A_full=0.969`；CLIP 无本地权重 | **未测**。报告写“否（本任务）”是“本任务非瓶颈”，不是“换编码器没有增益”——Phase 11 的换编码器测试在 re-ID/闭集任务上，结论是“投影头才是关键” | 同一 384 类 view-instance 任务换 DINOv2/CLIP 特征重跑 A 线 | U1 |
| **解码器/读出** | 0.833→0.948；rerank 0.945（bar 0.955 失败）；完整 MLP 0.807；147k 重排无增量；6 类生产 joint 0.990→0.958、开集 −7.2 | **B1 局部上界**，且任务相对：6 类生产不该整体替换，`auto`/双头是对的 | 换表示后才需要重开；shrinkage 只修开集（0.897→0.957），不抬闭集 | U1 |
| **维度/无损度** | 128→512 仅 +0.8~1.1（CI 跨 0）；白化 +1.8（0.930→0.948） | **“维度”这半已到 B1；“无损度”这半未测**——没有 reconstruction、没有 cross-modal R@1、没有“关系组合在投影后是否成立” | 补无损度的定义实验，不扩维 | U3 |
| **推理容量** | BFS 1.000；代数对 BCD Δ=0；A* 只省节点（12.3→4.8）；7×7 strip/diag 0.17–0.62 但归因是覆盖 | **P5b 规模上的 B1**；“容量”只在 3 物体/5 区域测过 | 组合深度、关系数、规划时域未测；但先测覆盖 | U2/U3 |
| **g 覆盖（g_cov）** | B 线 L1 ❌：worst-cell 0.957≥0.90 通过，但 `delta_vs_full = −0.043`，比 −0.03 差 1.3 pt；H1/H2 成立 | **未饱和** | 真实几何、严格 3 pt | U2 |
| **g 匹配（g_match）** | G3：绝对位置任务 `M_gap = +0.431`；特征不变性必须匹配任务 | **未饱和**，且表现为“换特征”而非“加数据” | 真实骨干上的匹配探针；与 U1 同一实验 | U1/U3 |
| **分解之外** | 长历史 fact active-set：learned 0.372 < hot 0.480 < full 1.000；joint re-ID 0.889–0.922 vs 成熟 pipeline 0.989–0.997；audit completeness 无统一量化；Qwen2.5-1.5B 0.857 vs ToolAgent 1.000 | **不在 f×g 里，但决定架构是否可部署** | 单独一条线，不能拿 f×g 的“饱和”掩盖 | U5 |

---

## 3. 结构性发现

### 3.1 A/B 两线不在同一基质上

- B 线 `pss/reasoning/relations/coverage.py` 的 cell 定义基于 `Coord`（符号坐标）；P5b 是 `w_category=0`、编码器不进入状态（`PSS-五优先级最终结论.md` §3.1 自己写明“P5b 对编码器构造性不变”）。
- 因此 f 的四个因子**不是在同一表示上联合测量**的；`f × g` 目前只是解释模型，不是可计算量。
- 缺的实验：**学习公共空间 → 关系谓词 → 覆盖审计** 的联合链路。这也正是“无损度”的正确度量位置。

### 3.2 上界是表示的函数，不是常数

- 0.948 是 LDA 在 384-d 白化表示上的上界；同一读出在 128-d 投影上是 0.927；6 类生产上 joint 反而更差（0.958 vs 0.990）。
- 读出的“饱和”只对当前表示成立；换编码器会移动这个上界。
- **推论：f 的下一步是换表示（编码器），不是换头（解码器）。**

### 3.3 g 应拆成 g_cov 与 g_match

| 子因子 | 机制 | 修复动作 | 已有证据 |
|---|---|---|---|
| `g_cov` | 缺 cell → absence case 未监督 | 补缺失 pair type / 反例 | BCD 0.571→1.000；strip_x single 0.310 vs combination 0.967 |
| `g_match` | 特征不变性与任务错配 | 换特征 / 换表示 | 绝对任务 absolute 0.813 vs relative 0.383，`M_gap=+0.431` |

两者修复动作相反。报告把它们合在 g 里，会让“下一步是 g”被误读为“下一步是加数据”；g_match 那一半的解法又回到编码器旋钮。

---

## 4. 可证伪假设

| 假设 | 内容 | 可证伪预测 | 若成立的动作 | 若不成立的动作 |
|---|---|---|---|---|
| **H1 读出迁移** | A4 的 +11.5 在真实骨干上保持 | `lda_whiten − proto ≥ +8 pt`，且 rerank 仍无增量（≤ +0.5 pt） | 把 0.948 写成 B2 结论 | 降级为“合成代理局部上界”，论文措辞同步 |
| **H2 覆盖规则迁移** | pair-type 组合规则在真实几何上成立 | combination worst-cell ≥ 0.90；single ≤ combination − 0.10 | 把“补缺失 pair type”写成通用规则 | 限定为 P5b/网格合成，不外推 |
| **H3 组合保真** | 学习表示的关系组合接近符号坐标 | `L_gap = exact4(symbolic) − exact4(learned) ≤ 0.05` | 无损度非瓶颈；B3 可判 | 无损度进入 f 瓶颈；不能声称公共空间支持组合 |
| **H4 上界随表示移动** | 换骨干会改变读出上界 | 真实骨干的 `A_full` 与合成不同，或最佳读出不同 | 记录“上界是表示的函数” | 读出结论可跨表示复用 |
| **H5 长期可维护** | 1.3 pt 地板在删除/合并下稳定 | 删除 10% + 合并 5% 后 `forgetting ≤ 0.03` | 部署沿用 `incr_refit_10` | 需要重估/回滚策略 |
| **H6 容量仍非瓶颈**（次要） | 真实骨干上 2× 参数仍无增量 | 2× 容量臂 ≤ 当前臂 + 0.5 pt | 停止加容量 | 重新打开容量因子 |

---

## 5. 预注册判据与 bar

### 5.1 新增字段（U0 写入 checker，运行前固定）

在 `pss/eval/readout_reasoning_preregistration.py` 的 `Expectation` 中新增：

```python
# -- upper-bound review arms -------------------------------------------
accuracy_delta_vs_proto_min: float | None = None
accuracy_delta_vs_proto_max: float | None = None
composition_gap_max: float | None = None     # L_gap = symbolic - learned
absence_cell_error_max: float | None = None
```

复用已有字段：`accuracy_min`、`auroc_min`、`unseen_reject_at_95_min`、`worst_cell_acc_min/max`、`delta_vs_full_min/max`、`forgetting_max`。

### 5.2 arm 与 bar

| arm | 实验 | bar | 性质 |
|---|---|---|---|
| `u_real_proto` | U1 | 无 bar | 参照（A_current） |
| `u_real_lda_whiten` | U1 | `accuracy_delta_vs_proto_min = +0.08`；`auroc_min = 0.99`；`unseen_reject_at_95_min = 0.95` | 主 |
| `u_real_rerank` | U1 | `accuracy_delta_vs_gate_max = +0.005` | 负对照：重排相对同一闸门必须仍无增量 |
| `u_real_dual_head` | U1b | `accuracy_delta_vs_proto_min = +0.08`；`auroc_min = 0.99`；`unseen_reject_at_95_min = 0.95` | 开集修复候选：LDA 闭集 + 原型开集 |
| `u_real_mlp` | U1 | 无 bar | 参照（同表示完整 MLP） |
| `u_geo_baseline` | U2 | 无 bar | 参照 |
| `u_geo_combination` | U2 | `worst_cell_acc_min = 0.90`；`delta_vs_full_min = −0.03` | 主 |
| `u_geo_single` | U2 | `delta_vs_full_max = −0.10` | 负对照：单轴必须落后 |
| `u_geo_noop` | U2 | `delta_vs_full_max = −0.10` | 负对照：多给已覆盖数据必须仍远低于全覆盖 |
| `u_geo_full` | U2 | 无 bar | 参照（全覆盖） |
| `u_comp_symbolic` | U3 | 无 bar | 参照（同任务符号坐标） |
| `u_comp_learned` | U3 | `composition_gap_max = 0.05`；`absence_cell_error_max = 0.20` | 主 |
| `u_comp_hybrid` | U3b | 同上 | 混合表示：embedding + 显式坐标 |
| `u_comp_branch` | U3c | 同上 | 双分支：坐标分支 + embedding 分支，logit 相加 |
| `u_comp_residual` | U3d | 同上 | 几何骨架 + α·embedding 残差，α 在训练 hold-out 上选 |
| `u_comp_patch` | U3e | 同上 | 不用显式坐标，只用 patch-token / 位置感知表示 |
| `u_real_vlm_proxy` | U1c | `accuracy_delta_vs_proto_min = 0.00`；`auroc_min = 0.99`；`unseen_reject_at_95_min = 0.95` | CLIP 双塔代理（无真实 VLM 权重） |
| `u_geom_noise` | U6 | `worst_cell_acc_min = 0.85`；`delta_vs_full_min = −0.10` | 传感器噪声压力测试（非 sim2real） |
| `u_incr_delete_merge` | U4 | `forgetting_max = 0.03`；`added_acc_min = 0.93` | 主 |

**bar 的依据（运行前写死，不因结果修改）：**

- U1 `+0.08`：合成增益 +9.4~+11.5，允许真实骨干上缩水 3.5 pt。
- U1 rerank `+0.005`：A2 的结论是“重排无增量”；若真实骨干上重排 > +0.5 pt，A2 结论被证伪，必须重开读出。
- U2 沿用 G2 的 `−0.03 / −0.10`，保持与合成实验同一把尺。
- U3 `0.05`：小于 A 线模态间差距的量级，也小于 G3 的 `M_gap`；若 `L_gap ≥ 0.10`，无损度就是 f 的真实瓶颈。`0.20` 的 absence error 是 G1 中 block3=0.000 与缺失方向=1.000 的中间带。
- U4 `0.03`：A5 的可达地板是 0.013；删除/合并各一次，允许 2.3× 地板。`0.93` 相对 A5 added 0.948 留 1.8 pt。

### 5.3 纪律

- 所有 bar 在运行前写入 checker；失败项照实保留，不调 bar。
- 声明了 bar 但未测量的 arm = 失败，不是跳过。
- 3 seeds；paired bootstrap 95% CI；主对照 Holm 校正；选择只用 val，test 只报一次。
- smoke 结果不进结论；单 seed 只作诊断。

---

## 6. 实验设计

### U1：真实骨干 A 线复跑（P0，唯一能改变 f 结论的实验）

- **任务**：与 A 线相同——`category_pretrain_instance8`，384 类，views 0–5 训练 / 6 val / 7 test。
- **骨干**：DINOv2（已有 cache）为主；CLIP 权重补齐后追加；当前生产投影作对照。
- **臂**：`proto` / `lda_concat_whiten` / `rerank_bilinear` / `mlp_whiten_concat`，全部走同一预注册 checker。
- **主指标**：闭集 test top-1 的 `delta vs proto`；同时报 AUROC 与 reject@95。
- **输出**：`eval/results/upper_bound/u1_real_backbone/`。
- **脚本**：扩 `run_fusion_ablation.py` / `run_rerank_eval.py` 加 `--backbone {dinov2,clip,production}`，或新增 `run_upper_bound_audit.py --stage u1`。
- **判读**：H1；若 LDA 增益 < 3 pt，0.948 判为代理特异，所有“读出上界”措辞降级。

### U2：真实几何覆盖审计（P0，唯一能改变 g 结论的）

- **数据**：CLEVR-CoGenT（Phase 27 已有 DINOv2 + relation head 管线，valA/valB 0.994/0.996）或一组真实空间关系数据。
- **cell 定义**：`pair_type × predicate × absence`，与 G1 相同；训练时故意留出方向/区域，测试 absence cell。
- **臂**：baseline / +single / +combination / no-op / full。
- **主指标**：worst-cell exact-4；overall 为次要。
- **输出**：`eval/results/upper_bound/u2_real_geometry/`。
- **脚本**：扩 `run_g_audit.py --stage b_coverage --dataset clevr`，或新增 `run_upper_bound_audit.py --stage u2`。
- **判读**：H2；若真实几何没有 absence-cell 效应（single ≈ combination），覆盖规则限定为合成。

### U3：公共空间组合保真度（P1，补“无损度”的定义）

- **任务**：同一批关系对，两种状态输入——符号坐标 vs 学习 embedding（DINOv2/生产投影），**同一个关系头、同一划分、同一训练预算**。
- **指标**：per-predicate F1、exact-4、absence-cell error；`L_gap = exact4(symbolic) − exact4(learned)`。
- **辅助**：cross-modal category R@1 / alignment gap（`PSS-架构评测指标.md` §3.1 已有定义，但优化报告没测）。
- **输出**：`eval/results/upper_bound/u3_composition/`。
- **脚本**：新增 `run_composition_fidelity.py`。
- **判读**：H3；`L_gap` 是“无损度”从定义空洞变成可测量的关键量。

### U4：长期增量与部署（P1，维护上界）

- **扩展 A5**：192 base + 192 add + 删除 10% + 合并 5%；3 seeds。
- **指标**：forgetting、added acc、rollback exactness、协方差条件数/漂移。
- **输出**：`eval/results/upper_bound/u4_incremental/`。
- **判读**：H5；这是“上界能否维持”，不是“上界能否抬高”。

### U5：分解之外的完整性（P2，不进主 bar）

- query-conditioned fact active-set + coverage loss（第三优先的明确负结果）；
- audit completeness 统一量化；
- 真实 LLM 接 `ToolAgent` 路径；
- joint 训练用同一 matcher 架构做公平比较。
- 这些不改变 f×g，但决定架构是否可部署；单独报告，不与 U1–U4 混表。

---

## 7. 相位与依赖

| 相位 | 内容 | 依赖 | 出口判据 |
|---|---|---|---|
| **U0** | 本计划、checker 新字段与 `u_*` arms、U1–U4 脚本骨架 | — | checker 可检查全部 arm |
| **U1** | 真实骨干 A 线复跑 | U0 | H1 判定；DINOv2 至少跑完 |
| **U2** | 真实几何覆盖审计 | U0 | H2 判定 |
| **U3** | 公共空间组合保真度 | U1/U2 的特征提取 | H3 判定 |
| **U4** | 长期增量与部署 | U0 | H5 判定 |
| **U5** | 分解外完整性 | — | 单独报告 |

**并行**：U1、U2、U4 在 U0 后可并行；U3 依赖 U1/U2 的特征但可先用 CLEVR 启动。

---

## 8. 杀死标准（先定）

| 观测 | 结论 | 动作 |
|---|---|---|
| U1：`lda_whiten − proto < +3 pt` | 0.948 是合成代理特异 | 撤回“读出上界”措辞；转 U3 查表示 |
| U1：rerank > +0.5 pt | A2 的“重排无增量”被证伪 | 重开读出线，重跑 A2 bar |
| U2：single 与 combination 差距 < 5 pt | H2 不成立 | 撤回“pair-type 组合”通用规则 |
| U2：真实几何无 absence cell | 覆盖不是真实数据的绑定约束 | g_cov 降级；转 g_match |
| U3：`L_gap ≥ 0.10` | 无损度是真实瓶颈 | 把无损度加入 f 瓶颈清单；不声称公共空间支持组合 |
| U4：合并后 forgetting > 3 pt | 1.3 pt 地板不可维护 | 增加重估频率或回滚策略 |
| 任一 arm 缺声明 bar | 记为失败 | 不补测、不调 bar，照实报告 |

---

## 9. 交付物

```
docs/plans/PSS-架构上界复核与下一阶段.md          # 本文件
pss/eval/readout_reasoning_preregistration.py     # U0：新增字段与 u_* arms
pss/eval/scripts/run_upper_bound_audit.py         # --stage {u1,u2}（或扩 run_g_audit）
pss/eval/scripts/run_composition_fidelity.py      # U3
tests/test_upper_bound_audit.py
eval/results/upper_bound/                         # u1_real_backbone / u2_real_geometry / u3_composition / u4_incremental
docs/experiments/PSS-架构上界复核报告.md            # 最终 B1–B3 分级与失败项
```

---

## 10. 风险与边界

- **真实骨干可用性**：DINOv2 已有；CLIP 无本地权重，U1 先跑 DINOv2 + 生产投影，CLIP 补齐后追加，不阻塞。
- **CLEVR 是渲染数据**：比符号坐标强，但不等于真实机器人；U2 结论写“真实几何/渲染”，不写 sim2real。
- **U3 的混淆**：`L_gap` 可能被关系头容量混淆；必须同一头、同一划分、同一训练预算，且报告头容量参照。
- **真实任务 `A_ceiling` 不可得**：只用 `A_full` 作参照，不声称“到 ceiling”。
- **不许看到结果后调 bar**；缺失测量记失败；单 seed 只作诊断。
- **不要把 B1 写成 B3**：0.948/1.000 是各自任务上的局部数字，不是 PSS 的架构上界。

---

## 11. 一句话

**先证明“局部上界”能迁移（U1/U2），再证明“f 与 g 在同一基质上可加”（U3），最后才谈维护（U4）。在这之前，0.948 与 1.000 都只是各自任务上的局部数字，不是 PSS 的架构上界。**
