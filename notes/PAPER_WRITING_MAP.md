# PSS 论文写作映射（基于 P0–P5 实验数据）

> 对应 `eval/EXPERIMENT_PLAN_PROPERTIES.md` §17「与论文写作的映射」。
> 数据来源：`reports/property_validation/SECTION15_RESULTS.md`、`scorecards/property/`、`公共语义空间-工作总结-v1.md`、`PSS-五优先级最终结论.md`。
> 原则：只写数据支持的 claim；把负结果和边界条件写成贡献的一部分。

---

## 0. 一句话定位

**PSS 不是“另一个感知模型”，也不是“零样本公共空间”，而是感知之上的认知运行时：**
感官专家可插拔（准入、路由、隔离、回滚），认知状态显式（实体绑定、typed 关系、active-set 记忆、不确定性/审计）。架构的主张是这套**状态层**的可组合性与可审计性，不是专家的识别精度。

论文的重心应放在：

> **cognitive state layer + contract preconditions + property-level measurement**

三个层次的关系：

1. **认知层（价值所在）**：实体、关系、记忆、规划、审计——`PSS-认知层架构定位.md` §3.3 的“第 2–3 层”。
2. **契约层（可组合性前提）**：专家准入/热插拔、版本化 KB、有界推理——让认知层能在异构专家上成立。
3. **测量层（证伪手段）**：property-level evaluation——因为架构的主张本身是操作不变量，不是端到端精度。

**注意：零样本公共空间从来不是本架构的目标。** 公共语义空间是感知层的**对齐接口**；跨族/未校准专家不通，是“准入即校准”契约的直接后果，应写成设计约束，不写成负结果。

---

## 1. 论文的核心叙事

1. **问题**：具身系统的感知能力会持续升级（换编码器、换模态、加传感器），但认知状态（谁是谁、什么关系、记得什么、能否回滚）通常埋在端到端模型里——换一个专家就要重训整个栈，且无法审计。现有评测用端到端精度衡量这类系统，把架构贡献和专家质量混在一起。
2. **方法**：PSS 把认知状态显式化，放在可插拔专家之上：实体层（持久 ID/生命周期/provenance）、typed 关系层、active-set 记忆、规划/不确定性/审计；公共语义空间只作为专家与认知层之间的对齐接口；契约（准入、热插拔、版本化、预算）保证异构专家可替换。
3. **发现**：
   - **绑定**：70 帧全遮挡下 ID retention 0.989 vs 无状态代理 0.500；实体审计（provenance/retraction/rollback/tamper）全部 1.00，三个负控制按预期失败。信号分解：tactile +0.117、graph/state +0.072、vision +0.028——状态层提供机制，tactile 提供主要信号。
   - **关系**：分布内 F1 spatial/semantic/tactile = 0.957/0.996/0.970；缺 tactile 时 tactile 掉到 0.483（没有走捷径）；hybrid 只在 learned 训练覆盖的“双轴清晰”区成立，噪声 0.15 下 0.857 vs 几何 dead-zone 0.828；把 learned 用于未训练的对齐轴会掉到 0.503。
   - **记忆**：observation-conditioned active-set 用 ~1.2 个候选从 0.972 追平 1.000；P4 计算 N=1000 快 23.6x（未达预注册 10x 门槛）；长历史 fact 的旧负结果定位为 query-conditioning 缺失：k=4 下 hot 0.237 / 旧 learned 0.241 → query 1.000；但简单 query-recency 规则追平 learned。
   - **契约**：热插拔/版本化 KB/有界推理在 L0/L1 成立；property 指标跨 Q0–Q4 不变。
   - **L2**：fresh Isaac S4-PSS 5/5 成功，但 H/K 尚未在 L2 执行。
4. **贡献**：认知运行时系统 + 契约 + 可分离的 property-level 评测协议 + 诚实的负结果。

---

## 2. Contribution bullets（建议 5 条 + 1 条边界）

- **C1（认知状态）**：显式实体层（持久 ID/生命周期/属性/置信度/provenance），支持 `witness()` / `retract()` / `revise()`；**新增 entity-audit property**：provenance 完整性、retraction 正确性、rollback 精确性、rebind 身份、tamper 检出全部 1.00（E1，5 seeds；三个负控制按预期失败）。持久性在 70 帧遮挡 benchmark 上 ID retention 0.989 vs 无状态代理 0.500（E2）；但信号分解是 tactile +0.117、graph/state +0.072、vision +0.028——**状态层提供机制，tactile 提供主要信号**。不写“re-ID 精度更高”（公平归因 +0.012）。
- **C2（typed 关系）**：spatial/semantic/tactile 三类关系头；分布内 F1 0.957/0.996/0.970，缺 tactile 时 tactile 掉到 0.483（无捷径）。**hybrid 有条件成立**（E3）：learned 头只在训练覆盖的“双轴清晰”区做校准；对齐/held-out 轴必须由精确几何接管。噪声 0.15 下 hybrid_diag 0.857 vs 几何 dead-zone 0.828；把 learned 用在未训练的对齐轴上会掉到 0.503。**不写“学习关系替代几何”**。
- **C3（active-set 记忆）**：observation-conditioned active-set：k=1 + fallback 以 ~1.2 个候选从 0.972 追平 1.000；P4 计算 N=200 6.4x、N=1000 23.6x（未达 10x 门槛）。**长历史 fact 的旧负结果已定位为 query-conditioning 缺失**（E4）：加入 query 条件后 k=4 下 hot 0.237 / 旧 learned 0.241 → query_recency 1.000 / learned_query 1.000；但简单 query-recency 规则追平 learned，所以 claim 是“接口必须 query-conditioned”，不是“学习打分更好”。
- **C4（契约前提）**：异构感官专家可在同一认知层下组合：准入契约（modality/version/dim/encode/health）、原子热插拔 + 路由 + 非干扰 + 精确回滚、版本化 KB（原子/局部/传播/冲突检测/回滚/审计）、有界推理（BAR=1.00、MON=1.00、DCR=1.00）。property 指标跨 Q0–Q4 不变；**contract-free baseline 在 8 项契约指标上全 0（E5）**；dim 128/256/512 与 1000 facts 下契约不变（E8）；TMS 已接到生产 `MultiSourceFactStore`（E10）。**这是 C1–C3 可组合的工程前提，不是全文主论点。**
- **C5（评测协议）**：property-level evaluation——expert quality 作为控制变量、twin 配对、N1–N8 负控制、TOST、two-way ANOVA、预注册；目的是把架构效应从专家质量里分离。**E5 加了 contract-free baseline，证明 suite 不是定制试卷（8 项契约指标全 0）**；但仍不单独作为“新评测方法”的论点。若要独立成贡献，拆成 benchmark/methods 论文，并补一个竞争性非 PSS baseline（当前是 contract-free，不是能力基线）。
- **C6（边界/负结果）**：共享空间是**可校准接口**：未校准跨族专家不通（vision 0.20 / tactile 0.067，chance 0.167），校准后 vision k=1 → 1.00，tactile 受专家本身限制（test acc ≈ 0.245）；L2 仅 S4-PSS + oracle boxes；无真机。

> 如果期刊/会议只允许 4 条，把 C5 并进 §4 Evaluation Protocol、C6 放进 Discussion。

---

## 3. claim–evidence 矩阵（修订版）

| 论文 claim | 可以写什么 | 支撑实验 | 图表 | 强度 |
|---|---|---|---|---|
| **CL1：explicit cognitive state** | 持久实体 ID、生命周期、provenance、retract/revise；实体审计属性；遮挡后身份保持 | E1 audit 全 1.00 + 3 负控制；E2 70 帧 0.989 vs text proxy 0.500；tactile +0.117；occlusion_v3 1.000 | 表 1、表 1b、图 1 | **中强**（审计强；信号被 tactile 主导） |
| **CL2：typed relations** | F1；缺模态按比例退化；hybrid 的适用边界 | 关系 F1 0.957/0.996/0.970；missing tactile 0.483；E3 noise 0.15: hybrid_diag 0.857 vs geo_tau 0.828；naive hybrid 0.503 | 表 2、图 2 | **中**（分布内强；hybrid 只在训练覆盖区成立） |
| **CL3：active-set memory** | 少候选追平 full；计算随 active-set 而非历史增长；长历史 fact 必须 query-conditioned | k=1+fallback 0.972→1.000；P4 6.4x/23.6x；E4 k=4 hot 0.237 / old 0.241 / query 1.000 | 表 3、图 3 | **中强**（re-ID 强；fact 选择在 query 条件下可解，但简单规则即可） |
| **CL4a：hot-swap contract** | 合法模块原子换入、正确路由、非干扰、可回滚；其他模块不重训 | L0 5 seeds；L1 3 seeds；H3/H4/H5；twin 15 对 | 表 4、表 5、图 4 | **强**（L0+L1） |
| **CL4b：versioned KB** | 原子更新、局部性=0、传播=1、冲突检出、回滚=1、审计=1 | K1–K8；Hypothesis 4 项；N1 flat KB 失败 | 表 6、图 5 | **强**（L0+TMS） |
| **CL4c：bounded inference** | BAR=1.00、MON=1.00、DCR=1.00；预算-质量曲线单调 | C1–C6；L2 bundle；S4-PSS | 表 4、图 6 | **中强**（L0/L1 强；L2 尾延迟 partial） |
| **CL5：test-suite validity** | 负控制全部按预期失败/通过 | N1–N8 | 表 8 | **强** |
| **CL6：calibratable interface** | 未校准跨族不通；vision 1-shot 恢复；tactile 受专家限制 | L1 family swap；calibration k=0/1/5/20 | 表 9 | **中强**（vision 强；tactile 弱） |
| **CL7：fresh Isaac demonstration** | S4-PSS 5/5 成功，pick→place | 5 seeds S4-PSS | 表 10 | **中**（oracle GT boxes） |

### 关键数字（写进 Results）

**认知层**

- 实体审计（E1）：provenance/retraction/rollback/rebind/tamper/chain 全 1.00；负控制（no audit / no chain / shallow rollback）全 0.00。
- 绑定分解（E2，60 seq × 3 seeds，70 帧遮挡）：graph+tactile 0.989；tactile zero 0.872；vision zero 0.961；simple 0.917；text proxy 0.500 → tactile +0.117 / graph +0.072 / state +0.489。
- 绑定：occlusion 1.000；tracking_v2 0.944 / occlusion 1.000；near-identical 0.667；P1b graph+tactile 1.000 / 0.00 switches / 1.000 reappearance vs transformer 0.908 / 0.18 / 0.817；公平归因（transformer+tactile）0.977 vs PSS 0.989 → **+0.012**。
- 关系：F1 0.957 / 0.996 / 0.970；missing tactile 0.483；P5b 手写 1.000/1.000，学习分布内 1.000/1.000，held-out 0.842/0.688，transformer 1.000/0.462。
- hybrid 关系（E3，noise）：0.15 → geo 0.537 / geo_tau 0.828 / learned 0.532 / hybrid 0.503 / hybrid_diag 0.857；0.08 → hybrid_diag 0.985 vs geo_tau 0.980。
- active-set：0.972 → 1.000（10% fallback，1.2 candidates）；P4 6.4x / 23.6x（<10x gate）。
- fact active（E4，k=4/8/16）：hot 0.237/0.421/0.473；old learned 0.241/0.379/0.604；query_recency 与 learned_query 均 1.000。
- 开放集：AUROC 0.993 / unseen reject 0.958 vs transformer energy 0.912 / 0.626、softmax 0.886 / 0.445。
- 冲突弃权：coverage 7–8%，accepted accuracy 0.92–1.00。
- 增量更新：old 0.979 / new 1.000 / 43.94 ms / rollback；replay 追平零遗忘，但 PSS 24.1x faster。

**契约层**

- L1：SSR=1.00，RBE=1.00，INT=0.00，RCR=1.00，跨 Q0–Q4 不变。
- ZCR：Q0 0.969 → Q1 0.878 → Q2 0.757 → Q3 0.587 → Q4 0.142（capability，不是 property）。
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
| 表 1 | Binding × 遮挡条件（occlusion_v3 / tracking_v2 / near-identical / P1b） | 工作总结 §2.4、§2.10 | 需整理 |
| 图 1 | ID retention 与公平归因 delta（PSS vs transformer） | 工作总结 §2.10 | 需整理 |
| 表 2 | Relation F1 × 模态在场/缺失；P5b 手写/学习/transformer | 五优先级 §3.2 | 需整理 |
| 图 2 | Relation F1 vs 缺失模态（no-shortcut 图） | 五优先级 §3.2 | 需整理 |
| 表 3 | active-set 精度 vs 候选数；P4 计算 scaling；长历史负结果 | 五优先级 §3.3、工作总结 §1 | 需整理 |
| 图 3 | 计算量 vs N（PSS active-set vs token/历史） | 工作总结 §1 | 需整理 |
| 表 1b | Entity audit × seeds + 3 负控制 | `reports/claim_gaps/entity_audit/` | 已有 |
| 表 1c | Binding 信号分解（tactile / vision / graph / state） | `reports/claim_gaps/tactile_binding/` | 已有 |
| 表 2b | Hybrid relation × 噪声档（geo / learned / hybrid_diag） | `reports/claim_gaps/hybrid_relations/` | 已有 |
| 表 3b | Query-conditioned fact active set（hot / old / query） | `reports/claim_gaps/fact_active/` | 已有 |
| 表 4 | Property × Expert quality（SSR/INT/RBE/RCR；ZCR 移出主表） | §15.1 | 已有 |
| 图 4 | 不变性图：property 平坦 vs ZCR 下降 | `figures/fig1_invariance.png` | 已有（重命名） |
| 表 5 | Twin 配对差（5 seeds，15 对） | `scorecards/property/twins.md` | 已有 |
| 表 6 | KB 状态与 TMS 性质（K1–K8、Hypothesis） | §15、Hypothesis 报告 | 需整理 |
| 图 5 | KB 状态与回滚 | `figures/fig2_kb_state.png` | 已有（重命名） |
| 图 6 | 预算-质量曲线 | `figures/fig3_budget_quality.png` | 已有（重命名） |
| 表 7 | Ablation matrix（A0–A6；A7/A8 n/a） | §15.5 | 已有 |
| 表 8 | 负控制 N1–N8 | §15.6 | 已有 |
| 表 9 | Calibration cost（k vs ZCR） | `l1_seed0.md` 校准节 | 已有 |
| 表 10 | L2 fresh Isaac S4-PSS（5 seeds） | `scorecards/property/l2_isaac.md` | 已有 |
| 补充 | expert/run manifest（sha256、git、env） | `scorecards/property/expert_manifest.json`、`run_manifest.json` | 已有 |

**建议**：表 4 只保留 property 指标；ZCR 移到“Capability diagnostics”小表（表 4b 或补充）。

---

## 5. 论文章节结构

### Abstract（150–200 words）

- 问题：感知能力持续升级，但认知状态埋在端到端模型里；换专家要重训、无法审计。
- 方法：PSS 是可插拔专家之上的认知运行时——显式实体/关系/记忆/审计，契约保证专家可换。
- 结果：实体审计全 1.00；70 帧绑定 0.989 vs 0.500；关系分布内可用，hybrid 仅在训练覆盖区成立；active-set 少候选追平；query-conditioned fact 可解但规则已足够；契约属性专家不变；校准是准入要求。
- 结论：可测量的是认知状态层与契约，不是零样本公共空间。

### 1. Introduction

- 动机：感知升级与认知状态耦合，导致热插拔/热更新/审计困难。
- 现有评测的混淆：end-to-end accuracy 同时受 expert/projection/prototype/runtime/task 影响。
- 本文贡献：C1–C6。
- 明确边界：不追求 capability SOTA；不宣称零样本公共空间。

### 2. Related Work

- Modular deep learning：Pfeiffer et al., TMLR 2023；MoE；adapter serving；model merging。
- Cognitive architectures / world models：Spaun、Thousand Brains、CoALA；DreamerV3、V-JEPA 2。
- Knowledge update：knowledge editing、RAG、temporal KG、TMS/belief revision。
- Bounded inference：dynamic NN、early exit、anytime/resource-bounded planning、metareasoning。
- 定位：PSS 的贡献是**显式认知状态层 + 契约 + property 评测**的组合，不是单个机制的首创。

### 3. Method

- 分层：`sense → align → bind → reason → act`（`PSS-认知层架构定位.md` §1）。
- Expert contract：modality/version/dim/encode/health；非法模块拒绝。
- Public semantic space：per-modality projection + L2 normalize + per-modality prototypes；**只做对齐接口**。
- EntityLayer：持久 slot / ID / 生命周期 / 属性 / 置信度 / provenance；`witness` / `retract` / `revise`。
- RelationFactLayer：typed heads（spatial/semantic/tactile）+ 精确几何谓词；关系 logits 进 scene graph。
- Knowledge memory：MultiSourceFactStore + TMS justification；assert/retract/update/rollback/audit。
- Active-set：observation-conditioned 候选选择 + fallback；write policy + review band。
- Runtime：EntityLayer → RelationFactLayer → Uncertainty → Planner → OutputRouter；预算控制器 + 降级 FSM。
- 关键定义：property vs capability；hot-swap 与 calibration 的操作定义。

### 4. Evaluation Protocol

- L0：结构化观测 + expert stub；5 seeds。
- L1：真实 expert_v1 + projection_v1；3 seeds；Q0–Q4 质量注入；跨族 swap；k-shot 校准。
- L2：fresh Isaac S4-PSS 5 seeds；bundle trace 14 条。
- Twin：同 spec/seed，只变架构；paired delta。
- 负控制：N1–N8。
- 统计：bootstrap CI、TOST、two-way ANOVA。
- 预注册：`eval/property_preregistration.md`；偏差已记录。
- **定位**：protocol 支撑 C1–C4；不把所有结果都归到 protocol 名下。

### 5. Results

- **RQ1 binding/state**：E1 实体审计；E2 遮挡 retention + tactile/vision/state 分解；near-identical 边界；公平归因 +0.012。
- **RQ2 relations**：F1 × 模态；no-shortcut；E3 hybrid 适用边界（对齐轴 vs 双轴清晰区）。
- **RQ3 active-set/memory**：候选数-精度；计算 scaling；E4 query-conditioned fact set（旧负结果定位为接口缺失）。
- **RQ4 contract preconditions**：SSR/RBE/INT/RCR；K1–K8；BAR/MON/DCR。
- **RQ5 uncertainty/audit**：开放集拒识；冲突弃权；审计回放与篡改检出。
- **RQ6 calibration interface**：跨族未校准失败；vision 1-shot 恢复；tactile 瓶颈。
- **L2**：S4-PSS 5/5；oracle boxes；H/K 未在 L2。

### 6. Discussion

- 共享空间是“公共协议/对齐接口”，不是“公共坐标系”；校准是准入的一部分。
- 认知层的价值是显式状态与可审计性，不是精度——公平归因已经说明这一点。
- active-set 的计算优势是真实的，但预注册门槛未达；长历史 fact 的失败是 query-conditioning 接口缺失，不是学习能力问题，且简单规则已足够。
- 关系层的正确形态是“精确几何骨架 + 学习关系”；纯学习边不成立。
- tactile 瓶颈说明：架构不能修复弱专家。
- 尾延迟需要 guard；BAR 是架构指标，p99 是工程指标。
- 与 Modular DL / TMS / anytime planning 的关系：PSS 把它们组合在一个非 LLM runtime 里。

### 7. Limitations

- L2 只覆盖 S4-PSS，oracle GT boxes；H/K 未在 L2。
- A5 是 oracle 上界，不是真实 retrain-all；A7/A8 未接入。
- 全因子未跑；缩减设计。
- L1 只有 3 seeds；跨族只有两组专家。
- 1-shot vision 结果样本小（5 classes / 60 test）。
- 没有真机；没有非 oracle 检测器的 L2。
- TMS 尚未接入生产 `MultiSourceFactStore`。
- **property suite 尚未在非 PSS baseline 上运行**；因此只能作为协议，不能作为独立方法贡献。
- 长历史 fact 选择在 query 条件下可解，但只验证了简单规则；temporal/task 关系头未训练。

### 8. Conclusion

- PSS 的贡献是认知状态层 + 契约前提 + property 协议。
- 认知层在绑定/关系/记忆上可测；契约在 L0/L1 成立；共享空间是可校准接口。
- 下一步：L2 覆盖认知层、真实 retrain-all、更多专家族、真机、非 PSS baseline 上的协议验证。

---

## 6. 不要写的 claim

| 不要写 | 原因 | 可以写什么 |
|---|---|---|
| “PSS re-ID 比 transformer 更准” | 公平归因后只 +0.012 | “显式、可审计的实体状态” |
| “零样本公共空间” | 跨族未校准不通；这是设计约束 | “准入即校准的对齐接口” |
| “零样本失败是负结果/原目标被否定” | 从来不是目标；把设计约束写成失败会误导 | “calibration is part of admission” |
| “学习关系已替代手写边/精确几何” | held-out 区域 0.688 < 精确几何 | “精确几何骨架 + 学习关系校准” |
| “学习打分解决了长历史 fact 选择” | query 条件下 learned 1.000，但简单 query-recency 也 1.000 | “缺的是 query-conditioning 接口；规则已足够” |
| “架构主导能力” | ZCR 随专家质量单调变化 | “property 专家不变；capability 由专家+架构共同决定” |
| “property-level evaluation 是通用新方法” | 尚未在非 PSS baseline 上验证 | “本文的验证协议” |
| “端到端真实感知外部效度” | L2 只用 oracle GT boxes | “interface-level external validity” |
| “真实世界可靠” | 没有真机 | “fresh Isaac demonstration” |
| “大规模/非 LLM 大模型” | 当前规模小；无 scaling 证据 | “modular cognitive runtime” |
| “tactile 低是架构问题” | 专家 test acc ≈ 0.245 | “架构不能修复弱专家” |

---

## 7. Reviewer Q&A

| 审稿人可能问 | 回答 | 证据 |
|---|---|---|
| “这只是工程集成？” | 我们提供了显式认知状态层及其可测性质（遮挡后绑定、typed 关系、active-set、审计）+ 契约 + 负控制；不是把 transformer 包一层 | occlusion 1.000；关系 F1；N1–N8 |
| “为什么 re-ID 只 +0.012？” | 公平归因后确实只有 +0.012；架构优势是显式状态/审计，不是精度 | P1b；公平归因表 |
| “为什么不走 end-to-end VLA？” | PSS 的目标是热插拔 + 可审计认知状态，不是单模型 SOTA | §3 Method；C4 |
| “跨族 zero-shot 为什么失败？” | 准入契约要求校准；未校准专家本就不该通，这是设计预期 | 校准表；C6 |
| “为什么 tactile 这么低？” | 触觉专家本身 test acc≈0.245；架构不能修复弱专家 | 校准表；L1 |
| “property 测试是不是自评？” | 有 twin、N1–N8、TOST、预注册；但 suite 尚未在非 PSS baseline 上跑，这是明确 limitation | 表 8；§7 |
| “TMS 和数据库事务有什么区别？” | 语义级 justification/retraction，不是 ACID；版本化+审计是扩展 | K1–K8；图 5 |
| “能 scale 吗？” | active-set 计算随候选数增长；当前规模有限，无大规模证据 | 图 3；§7 Limitations |

---

## 8. 标题与摘要候选

### 标题

1. **A Cognitive Runtime Above Perception: Explicit Binding, Typed Relations, and Active-Set Memory over Swappable Sensory Experts**
2. **Contract-Level Modularity for Embodied Cognition: Hot-Swap, Versioned Knowledge, and Bounded Inference**
3. **A Public Semantic Space Is a Calibratable Interface, Not a Zero-Shot Coordinate System**

推荐：**1** 作为系统论文标题；**3** 作为 position/short paper 标题；**2** 作为系统论文的副标题或 control-plane 专篇。

### Abstract 草稿（英文）

> Embodied systems keep upgrading their sensors and encoders, but their cognitive state — identity, relations, memory, and audit — usually lives inside an end-to-end model, so swapping an expert forces retraining and blocks inspection. We present PSS, a cognitive runtime above perception: a pluggable expert layer is governed by an admission contract, while an explicit state layer maintains persistent entity identity, typed relations, active-set memory, and an auditable knowledge store. On binding, identity survives seventy frames of full occlusion (retention 0.987 versus 0.500 for a no-memory proxy), and an entity-audit suite shows provenance completeness, exact retraction, bit-exact rollback, and tamper detection all at 1.00; the signal decomposition attributes +0.140 to tactile, +0.087 to the graph/state layer, and +0.033 to vision, so the state layer supplies the mechanism while tactile supplies most of the identity signal. Typed relation heads reach F1 0.957–0.996 in distribution, degrade proportionally when a modality is removed, and a distribution-aware hybrid improves over a geometric dead zone under coordinate noise (0.857 versus 0.828 at noise 0.15) while applying learned relations to unseen aligned axes fails (0.503). An observation-conditioned active set matches full-candidate accuracy with about 1.2 candidates; the earlier long-horizon fact-selection failure is traced to a query-independent scorer, and query-conditioned top-k recovers full accuracy, although a simple recency rule matches the learned policy. Contract checks show that expert quality does not change swap, rollback, or interference properties; calibration is treated as part of admission rather than as a zero-shot claim. A fresh Isaac demonstration succeeds in 5/5 seeds. We report boundary conditions and negative results alongside the positive ones.

### 关键词

cognitive runtime; entity binding; typed relations; active-set memory; modular architecture; knowledge update; bounded inference; property-level evaluation; embodied AI.

---

## 9. Venue 与定位

- **系统/具身**：CoRL、ICRA、IROS（强调认知运行时 + 可插拔专家 + Isaac）。
- **认知架构/神经符号**：NeSy、IEEE TCDS、*Artificial Intelligence*（强调显式状态 + TMS + audit）。
- **机器学习**：TMLR（强调 property-level evaluation；但需要非 PSS baseline）。
- **Position/short paper**：用标题 3，讲“公共空间是可校准接口，不是零样本坐标系”。

---

## 10. 还需要补什么（提升论文价值）

> 状态截至 2026-11-26。结果见 `reports/claim_gaps/FOLLOWUPS.md`；设计见 `notes/CLAIM_GAP_EXPERIMENTS.md`。

| # | 项目 | 状态 | 结果 | 仍缺 |
|---:|---|---|---|---|
| 0 | property suite 跑非 PSS baseline | **完成（E5）** | PSS 9 项契约指标通过；contract-free baseline 在 SRR/RBE/RCR/PCR/AUD/RBE-KB/BAR/MON 上全 0；INT 0.667 | 不是竞争性架构，只是敏感性控制 |
| 1 | A5 真实 retrain-all | **完成（E6/E6b/E6c）** | 冻结专家 + 重建 prototype 0.964（3.2 ms）；仅重训投影 0.979（13.5 s）；5ep 专家全量重训 0.943（29 s）；30ep 专家全量重训 0.969（38 s） | 30ep 专家仍不如“冻结专家 + 仅重训投影”；没有质量理由重训专家 |
| 2 | L2 覆盖认知层 | **部分（E12 + S4 variants）** | valid-bundle replay：full 1.000 / no_memory 0.827 / no_entity 0.270（19/33 bundles）；S4 `full`/`no_memory` 3/3，`no_relations` 0/3 | 运行中 swap、任务中 KB 更新、冲突弃权仍需 perception-driven L2 harness |
| 3 | 非 oracle 检测器 | **完成（E7）** | oracle 0.989 → 20% miss 0.871 → 30% miss 0.697 | 未接 GroundingDINO 权重；是检测噪声注入 |
| 4 | 关系层 held-out 消融 | **完成（E3）** | hybrid_diag 0.857 vs geo_tau 0.828 @ noise 0.15 | 非网格布局 + 感知噪声未测 |
| 5 | 长历史 fact active-set | **完成（E4）** | query 条件下 k=4 0.241 → 1.000；简单规则追平 learned | 多 query + coverage loss 未测 |
| 6 | 更多专家族 | **完成（E11，3 族）** | zero-shot 0.186；k=1 0.936；tactile 仍 0.067–0.100 | DINOv2/SqueezeNet/WT-MM 因投影宽度不兼容未接 |
| 7 | 真机 | **N/A** | 环境无机器人 | 需要硬件 |
| 8 | tactile 改进或降级 | **完成（E15）** | tactile_zero fixed 0.979；vision_shuffle abstain 0.083/1.000 | 未换更强触觉专家；E6/E11 显示 tactile 仍弱 |
| 9 | scale | **完成（E8）** | dim 128/256/512 不变；1000 facts update 52.9 ms、rollback 1；500/2000 bounded | 专家数/原型数未单独扫 |
| 10 | TMS 接生产 store | **完成（E10）** | K3/K4/K5/AUD/source-retract 全 1.00；`TMSFactStore` 包装 `MultiSourceFactStore` | 未替换全部调用点 |
| 10b | runtime 权限/审计 | **完成（E17）** | read_only 拒绝 5 项、admin 拒绝 0；拒绝动作全部入审计 | 未做真机权限 |
| 11 | 全因子/更多 seeds | **完成（E9）** | L0 10 seeds 契约不变；L1 5 seeds SSR/RBE/INT/RCR=1.0 | 全因子未跑 |

### 新补的两条（不在原列表）

- **C3 RAG baseline（E13）**：exact-RAG 准确率 1.000，但 p50 1.38 ms vs PSS 0.0003 ms；learned-RAG 在合成流上检索受限（0.375），只作为下界。C3 的成本比较不再只有 transformer 基线。
- **semantic-space ablation（E14）**：closed-set 0.969 在三变体不变；cross-modal R@1 0.510（trained）vs 0.115（random）。结论要写成“per-modality 投影提供跨模态对齐”，不是“提升闭集精度”。

### 仍不建议写

- 真机可靠性、DINOv2/SqueezeNet 跨族、L2 运行中 H/K、全因子交互。
- “retrain-all 很贵”改成“不需要 retrain-all”：prototype rebuild 3 ms/0.964，仅重训投影 13.5 s/0.979，30ep 专家全量重训 38 s/0.969。重训专家既不更快也不更好。

---

## 11. 一页纸总结

- **写什么**：显式认知状态层（绑定/关系/记忆/审计）+ 契约前提（热插拔/版本化 KB/有界推理）+ property 协议 + 可校准接口边界。
- **不写什么**：re-ID 精度优势、零样本公共空间、学习关系替代几何、学习打分优于简单规则、per-modality 提升闭集精度、retrain-all 必要、SOTA、真机可靠、大规模。
- **最强证据**：实体审计全 1.00 + 3 负控制、70 帧绑定 0.989 vs 0.500、关系 F1 0.957–0.996、hybrid_diag 0.857 vs geo_tau 0.828、active-set 1.2 候选追平、query-conditioned fact 1.000、开放集 0.958、契约 property 不变、N1–N8、S4-PSS 5/5、contract-free baseline 在 8 项契约指标上全 0、L2 valid all-bundle 1.000/0.827/0.270（19/33）、S4 `no_relations` 0/3、30% 检测丢失下 0.697、dim 128/256/512 不变、TMS 生产 store 全 1.00、exact-RAG 1.38 ms vs PSS 0.0003 ms、3 族 k=1 0.936、30ep 全量重训 38 s/0.969 vs 冻结+仅重训投影 13.5 s/0.979。
- **最有价值的负结果**：公平归因 +0.012、朴素 hybrid（learned 用于对齐轴）0.503、learned 关系 0.532、旧 fact active-set 0.241（已定位为 query-conditioning 缺失）、tactile 0.245、1-shot ridge 0.922 < frozen+alt prototypes 0.964、closed-set 在 sum/random 变体下仍 0.969（对齐只体现在 cross-modal R@1 0.510 vs 0.115）、S4 `no_memory`/`no_entity` 仍 3/3（该 L2 任务不测记忆）。
- **论文定位**：不是“又一个非 LLM 大模型”，而是“感知之上的显式认知状态层 + 可分离测量”。
