# PSS 论文写作映射（基于 P0–P5 实验数据）

> 对应 `eval/EXPERIMENT_PLAN_PROPERTIES.md` §17「与论文写作的映射」。
> 数据来源：`reports/property_validation/SECTION15_RESULTS.md`、`scorecards/property/`。
> 原则：只写数据支持的 claim；把负结果和边界条件写成贡献的一部分。

---

## 0. 一句话定位

**PSS 不是一个“零样本公共空间”，而是一个 contract-level 的模块化架构：**
合法专家可以原子换入、正确路由、非干扰、可回滚；知识可以版本化热更新；推理成本有硬上界。我们提供的是一套 **property-level evaluation**，把架构贡献从专家质量里分离出来。

论文的重心应放在：

> **contract-level modularity + measurement methodology**

而不是：

> **zero-shot public semantic space**

后者已经被我们自己的数据否定。

---

## 1. 论文的核心叙事

1. **问题**：模块化具身系统需要三件事——热插拔、知识热更新、推理上界；但现有评测把架构贡献和专家质量混在一起。
2. **方法**：PSS 用公共语义空间作为接口契约，用版本化 TMS 管理知识，用预算控制器约束推理；用 L0/L1/L2 三层 + expert-quality 控制 + twin + 负控制来验证。
3. **发现**：
   - 三条契约在 L0/L1 成立；
   - property 指标专家不变，capability 指标随专家质量变化；
   - 共享空间是**可校准接口**，不是零样本公共坐标系；
   - L2 的 fresh Isaac S4-PSS 5/5 成功，但 H/K 尚未在 L2 执行。
4. **贡献**：系统 + 方法 + benchmark + 诚实的负结果。

---

## 2. Contribution bullets（建议 4–5 条）

- **C1（系统）**：一个 contract-level 的热插拔接口：合法模块原子换入、正确路由、非干扰、可回滚、非法模块拒绝；**不要求**其他专家或 runtime 重训。
- **C2（系统）**：一个版本化知识运行时：TMS-style justification、原子更新、局部传播、冲突检测、回滚、审计；flat KB 负控制失败。
- **C3（系统）**：一个有硬上界的推理运行时：预算遵守、anytime、降级 FSM、成本归因；尾延迟需要 guard。
- **C4（方法）**：一套 property-level evaluation：把 expert quality 变成控制变量，用 twin / 负控制 / TOST / two-way ANOVA 分离架构效应；N1–N8 证明测试有效。
- **C5（边界/负结果）**：共享空间是**可校准接口**而非零样本坐标系：跨专家族零样本失败；vision 1-shot 校准恢复；tactile 受专家本身限制。

> 如果期刊/会议只允许 4 条，把 C5 放进 Discussion，作为 boundary condition。

---

## 3. §17 式 claim–evidence 矩阵（修订版）

| 论文 claim | 可以写什么 | 支撑实验 | 图表 | 强度 |
|---|---|---|---|---|
| **CL1：contract-level hot-swap** | 合法模块可原子换入、正确路由、非干扰、可回滚；其他专家与 runtime 不重训 | L0 5 seeds；L1 3 seeds；H3/H4/H5；twin 15 对 | 表 1、图 1、表 5 | **强**（L0+L1） |
| **CL2：versioned KB with TMS** | 原子更新、局部性=0、传播=1、冲突检出、回滚=1、审计=1 | K1–K8；Hypothesis 4 项；N1 flat KB 失败 | 图 2、表 2、表 3 | **强**（L0+TMS） |
| **CL3：bounded inference** | BAR=1.00、MON=1.00、DCR=1.00；预算-质量曲线单调 | C1–C6；L2 bundle；S4-PSS | 图 3、表 2 | **中强**（L0/L1 强；L2 尾延迟 partial） |
| **CL4：property/capability separation** | property 指标专家不变；capability 随专家质量变化 | 表 1；表 4；表 4b；Q0–Q4 曲线 | 表 1、表 4、表 4b、图 1 | **强**（契约指标） |
| **CL5：test-suite validity** | 负控制全部按预期失败/通过 | N1–N8 | 表 3 | **强** |
| **CL6：calibrated interface** | 零样本跨族失败；vision 1-shot 恢复；tactile 受专家限制 | L1 family swap；calibration k=0/1/5/20 | 表 2、校准表 | **中强**（vision 强；tactile 弱） |
| **CL7：fresh Isaac demonstration** | S4-PSS 5/5 成功，pick→place，wall mean 2.14s | 5 seeds S4-PSS | 表 6 | **中**（oracle GT boxes） |

### 关键数字（写进 Results）

- L1：SSR=1.00，RBE=1.00，INT=0.00，RCR=1.00，跨 Q0–Q4 不变。
- ZCR：Q0 0.969 → Q1 0.878 → Q2 0.757 → Q3 0.587 → Q4 0.142。
- Q1 相对保持率：1.008。
- 跨族零样本：vision 0.20，tactile 0.067（chance 0.167）。
- 校准：vision k=1 → 1.00；tactile 最好 0.40。
- Twin：ZCR -0.717，RBE -1.0，LOC +1.0，PCR -1.0，BAR -1.0。
- L2：S4-PSS 5/5，wall mean 2.136s，BAR=1.00。
- L2 bundle：BAR mean 0.9993；2/14 trace 尾延迟 >10ms。

---

## 4. 图表计划

| 编号 | 内容 | 数据来源 | 状态 |
|---|---|---|---|
| 表 1 | Property × Expert quality（SSR/INT/RBE；ZCR 移出主表） | §15.1 | 已有 |
| 图 1 | 不变性图：property 平坦 vs ZCR 下降 | `figures/fig1_invariance.png` | 已有 |
| 图 2 | KB 状态与回滚 | `figures/fig2_kb_state.png` | 已有 |
| 图 3 | 预算-质量曲线 | `figures/fig3_budget_quality.png` | 已有 |
| 表 2 | Ablation matrix（A0–A6；A7/A8 n/a） | §15.5 | 已有 |
| 表 3 | 负控制 N1–N8 | §15.6 | 已有 |
| 表 4 | 描述性方差分解 | §15.7 | 已有 |
| 表 4b | Two-way ANOVA（balanced cells） | §15.7b | 已有 |
| 表 5 | Twin 配对差（5 seeds，15 对） | `scorecards/property/twins.md` | 已有 |
| 表 6 | L2 fresh Isaac S4-PSS（5 seeds） | `scorecards/property/l2_isaac.md` | 已有 |
| 表 7 | Calibration cost（k vs ZCR） | `l1_seed0.md` 校准节 | 已有 |
| 补充 | expert/run manifest（sha256、git、env） | `scorecards/property/expert_manifest.json`、`run_manifest.json` | 已有 |

**建议**：把 ZCR 从表 1 主表移到“Capability diagnostics”小表；表 1 只保留 property 指标。

---

## 5. 论文章节结构

### Abstract（150–200 words）

- 问题：模块化具身系统的热插拔、知识热更新、推理上界缺少可分离的评测。
- 方法：PSS + property-level evaluation。
- 结果：契约指标专家不变；能力指标随专家变化；共享空间是可校准接口；fresh Isaac 5/5。
- 结论：contract-level modularity 是可测量、可验证的；不要把它误写成零样本公共空间。

### 1. Introduction

- 模块化架构的动机：专家异构、知识演化、推理预算。
- 现有评测的混淆：end-to-end accuracy 同时受 expert/projection/prototype/runtime/task 影响。
- 本文贡献：C1–C5。
- 明确边界：不追求 SOTA；不宣称零样本公共空间。

### 2. Related Work

- Modular deep learning：Pfeiffer et al., TMLR 2023；MoE；adapter serving；model merging。
- Knowledge update：knowledge editing、RAG、temporal KG、TMS/belief revision。
- Bounded inference：dynamic NN、early exit、anytime/resource-bounded planning、metareasoning。
- Non-LLM architectures：world models（DreamerV3、V-JEPA 2）、VSA/HDC、neurosymbolic、cognitive architectures（Spaun、Thousand Brains、CoALA）。
- 定位：PSS 的贡献是 contract-level 组合 + property-level 评测，不是单个机制的首创。

### 3. Method

- Expert contract：modality/version/dim/encode/health；非法模块拒绝。
- Public semantic space：per-modality projection + L2 normalize + per-modality prototypes。
- Versioned KB：MultiSourceFactStore + TMS justification；assert/retract/update/rollback/audit。
- Runtime：EntityLayer → RelationFactLayer → Uncertainty → Planner → OutputRouter。
- Budget controller：entities/relations/expansions/retrieval levels + degradation FSM。
- 关键定义：property vs capability；hot-swap 的操作定义。

### 4. Evaluation Protocol

- L0：结构化观测 + expert stub；5 seeds。
- L1：真实 expert_v1 + projection_v1；3 seeds；Q0–Q4 质量注入；跨族 swap；k-shot 校准。
- L2：fresh Isaac S4-PSS 5 seeds；bundle trace 14 条。
- Twin：同 spec/seed，只变架构；paired delta。
- 负控制：N1–N8。
- 统计：bootstrap CI、TOST、two-way ANOVA。
- 预注册：`eval/property_preregistration.md`；偏差已记录。

### 5. Results

- **RQ1 hot-swap**：SSR/RBE/INT/RCR；H3/H4/H5；twin。
- **RQ2 KB**：K1–K8；Hypothesis；flat KB 负控制。
- **RQ3 bounded**：C1–C6；L2 bundle；S4-PSS。
- **RQ4 separation**：Table 1/4/4b；ZCR 是 capability。
- **RQ5 calibration**：跨族零样本失败；vision 1-shot；tactile 瓶颈。
- **RQ6 interactions**：HK/HC/KC/HKC pass 1.0。
- **L2**：S4-PSS 5/5；oracle boxes；H/K 未在 L2。

### 6. Discussion

- 共享空间是“公共协议”，不是“公共坐标系”。
- 校准成本是 hot-swap 的一等成本，不是失败。
- tactile 瓶颈说明：架构不能修复弱专家；需要更强传感器/专家。
- 尾延迟需要 guard；BAR 是架构指标，p99 是工程指标。
- 与 Modular DL / TMS / anytime planning 的关系：PSS 把它们组合在一个非 LLM runtime 里，并给出可分离的评测。

### 7. Limitations

- L2 只覆盖 S4-PSS，oracle GT boxes；H/K 未在 L2。
- A5 是 oracle 上界，不是真实 retrain-all；A7/A8 未接入。
- 全因子未跑；缩减设计。
- L1 只有 3 seeds；跨族只有两组专家。
- 1-shot vision 结果样本小（5 classes / 60 test）。
- 没有真机；没有非 oracle 检测器的 L2。
- TMS 尚未接入生产 `MultiSourceFactStore`。

### 8. Conclusion

- PSS 的贡献是 contract-level modularity + property-level evaluation。
- 三条契约在 L0/L1 成立；共享空间是可校准接口。
- 下一步：L2 覆盖 H/K、真实 retrain-all、更多专家族、真机。

---

## 6. 不要写的 claim

| 不要写 | 原因 | 可以写什么 |
|---|---|---|
| “PSS 比 LLM/VLM 更准” | 本实验不测 capability SOTA | “PSS 提供 contract-level modularity” |
| “零样本公共空间” | 跨族零样本失败；ZCR 随专家质量变化 | “共享空间是可校准接口” |
| “端到端真实感知外部效度” | L2 只用 oracle GT boxes；H/K 未在 L2 | “interface-level external validity” |
| “真实世界可靠” | 没有真机 | “fresh Isaac demonstration” |
| “架构主导能力” | ZCR 的 ANOVA 架构份额被 A4/A5 控制放大；VR≈1.8 | “property 指标专家不变；capability 由专家+架构共同决定” |
| “大规模/非 LLM 大模型” | 当前规模小；无 scaling 证据 | “modular semantic-space architecture” |
| “ZCR 是 property” | 它是 capability/diagnostic | “ZCR 是换入后的语义诊断；校准成本另报” |

---

## 7. Reviewer Q&A

| 审稿人可能问 | 回答 | 证据 |
|---|---|---|
| “这只是工程集成？” | 我们提供了 property test suite、负控制、TMS 形式性质和 twin/ANOVA；这是方法贡献 | N1–N8；Hypothesis；表 4b |
| “ZCR 为什么低？” | ZCR 是 capability，不是 property；契约指标 SSR/RBE/INT 不变；换入需要校准 | 表 1；校准表 |
| “为什么不走 end-to-end VLA？” | PSS 的目标是 contract-level modularity，不是端到端单体精度 | §3 Method；表 1 |
| “结果是不是 stub 撑起来的？” | L0 用 stub 隔离架构；L1 用真实 expert_v1 + projection_v1；L2 有 fresh Isaac S4-PSS | L1；表 6 |
| “能 scale 吗？” | 当前规模有限；这是 limitation；需要更多专家/数据/场景 | §7 Limitations |
| “有真机吗？” | 没有；L2 是 Isaac + oracle boxes；下一步真机 | §7 Limitations |
| “TMS 和数据库事务有什么区别？” | 语义级 justification/retraction，不是 ACID；版本化+审计是 PSS 的扩展 | K1–K8；图 2 |
| “为什么 tactile 这么低？” | 触觉专家本身 test acc≈0.245；架构不能修复弱专家 | 校准表；L1 |

---

## 8. 标题与摘要候选

### 标题

1. **Contract-Level Modularity for Embodied AI: Hot-Swap, Versioned Knowledge, and Bounded Inference in a Public Semantic Space**
2. **A Public Semantic Space Is a Calibratable Interface, Not a Zero-Shot Coordinate System**
3. **Measuring Architecture, Not Experts: Property-Level Validation for Modular Embodied Systems**

推荐：**1** 作为系统论文标题；**2** 作为 position/short paper 标题。

### Abstract 草稿（英文）

> Modular embodied systems need to swap sensory experts, update knowledge, and bound inference cost without retraining the whole stack. We present PSS, a non-LLM modular architecture whose public semantic space acts as a contract between heterogeneous experts. We evaluate it at the property level rather than end-to-end accuracy: expert quality is a controlled factor, architecture variants are paired twins, and negative controls test whether the metrics are sensitive. Across five L0 seeds and three L1 seeds with real expert checkpoints, the hot-swap contract (swap success, rollback exactness, non-interference, routing) is invariant to expert quality, while semantic compatibility is not: cross-family zero-shot swap falls to chance, vision recovers with one-shot calibration, and tactile remains limited by the expert. Knowledge updates satisfy atomicity, locality, propagation, rollback and audit under a TMS-style justification graph; bounded inference respects hard caps with a tail-latency guard. A fresh Isaac S4-PSS run succeeds in 5/5 seeds. These results support contract-level modularity and a calibrated-interface view of shared semantic spaces, not a zero-shot public-space claim.

### 关键词

modular architecture; hot-swap; knowledge update; truth maintenance; bounded inference; property-level evaluation; embodied AI; public semantic space.

---

## 9. Venue 与定位

- **系统/具身**：CoRL、ICRA、IROS（强调 hot-swap + KB + bounded inference + Isaac）。
- **神经符号/认知架构**：NeSy、IEEE TCDS、*Artificial Intelligence*（强调 TMS + semantic space + non-LLM）。
- **机器学习**：TMLR（强调 property-level evaluation + modularity；但需要更强 baseline）。
- **Position/short paper**：用标题 2，讲“公共空间是可校准接口，不是零样本坐标系”。

---

## 10. 还需要补什么（提升论文价值）

1. **A5 真实 retrain-all**：当前是 oracle 上界；审稿人会问“retrain-all 到底多贵”。
2. **L2 覆盖 H/K**：在 Isaac 上做 H5（运行中 swap）、K6（任务中更新）、C4（对抗场景）。
3. **非 oracle 检测器**：GroundingDINO/noisy 检测，证明契约在感知噪声下仍成立。
4. **更多专家族**：SqueezeNet、DINOv2、WT-MM，验证 1-shot 校准是否普遍。
5. **真机小规模**：WT-MM/LeRobot，证明不是纯仿真。
6. **tactile 改进或明确降级**：架构无法修复弱触觉；要么换专家，要么标记低可靠模态。
7. **scale 证据**：D=128→512、专家数、原型数、事实数。
8. **TMS 接入生产 `MultiSourceFactStore`**：目前是独立模块。
9. **全因子/更多 seeds**：当前是缩减设计；L1 只有 3 seeds。

---

## 11. 一页纸总结

- **写什么**：contract-level hot-swap、versioned TMS KB、bounded inference、property/capability separation、calibrated interface。
- **不写什么**：SOTA、零样本公共空间、端到端外部效度、架构主导能力、大规模/真机可靠。
- **最强证据**：L0/L1 契约不变 + N1–N8 + twin + TMS Hypothesis + S4-PSS 5/5。
- **最有价值的负结果**：跨族零样本失败、vision 1-shot 恢复、tactile 瓶颈。
- **论文定位**：不是“又一个非 LLM 大模型”，而是“模块化架构的契约与评测方法”。
