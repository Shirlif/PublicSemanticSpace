# PSS：感知之上的认知运行时

**Public Semantic Space (PSS) — contract-level modularity for embodied cognition**

> 本文档是项目入口，说明 PSS 解决什么问题、怎么工作、证据到什么程度、边界在哪里。
> 它不是一篇对外投稿的论文；完整的 claim–evidence 映射见 `notes/PAPER_WRITING_MAP.md`，实验设计与结果见 `notes/CLAIM_GAP_EXPERIMENTS.md` 和 `reports/claim_gaps/`。

---

## 摘要

具身系统的感知能力会持续升级：换编码器、换模态、加传感器。但认知状态——谁是谁、什么关系、记得什么、能不能回滚——通常埋在端到端模型里。结果是换一个专家就要重训整个栈，知识更新无法审计，推理成本没有硬上界。

PSS 的答案不是在感知层造一个更大的模型，而是在感知之上放一个**显式、可组合、可审计的认知运行时**：

- 感官专家可以按契约换入，其他模块和 runtime 不重训；
- 认知状态（实体、关系、记忆、审计）显式存在，可版本化、可回滚；
- 知识可以原子更新，推理成本有硬上界；
- 公共语义空间只作为专家与认知层之间的**对齐接口**，不是“所有专家天然落进去的零样本坐标系”。

评测上，PSS 不追求端到端 SOTA，而是用 **property-level evaluation** 把架构贡献从专家质量里分离出来：专家质量是控制变量，架构变体是配对 twin，负控制检验指标是否敏感。

一句话：**PSS 不是“又一个更大的模型”，而是一套“让异构专家可以组合、让认知状态可以被测量”的运行时与评测方法。**

---

## 1. 问题：感知会升级，认知状态却不可替换

模块化具身系统需要三件事：

1. **热插拔**：换一个视觉或触觉专家，不应该重训整个栈；
2. **知识热更新**：事实、规则、原型可以在运行中更新、回滚、审计；
3. **推理上界**：实体、关系、规划、检索都有预算，尾延迟不能失控。

现有做法通常把这三件事混在端到端精度里：

- 换专家 → 全量微调，旧能力遗忘，无法证明“只有目标模块变了”；
- 知识更新 → 改参数或重训，更新是否局部、能否回滚、审计链是否完整，说不清；
- 推理成本 → 随上下文/历史增长，没有硬上界，也没有降级路径。

因此，一个系统的“架构贡献”和“专家质量”经常被混在一起：分数涨了，不知道是模块化做好了，还是只是换了一个更强的编码器。

PSS 的核心问题是：**能不能把认知状态从感知模型里拿出来，做成一个显式的、可替换的、可测量的层？**

---

## 2. 定位：PSS 是什么，不是什么

| PSS 是 | PSS 不是 |
|---|---|
| 感知之上的**认知运行时** | 一个新的基础模型 / foundation model |
| 一套**契约**：专家准入、热插拔、版本化 KB、有界推理 | 一个“零样本公共语义空间” |
| 一套**property-level evaluation** | 端到端 SOTA 感知系统 |
| 显式状态：实体、关系、记忆、审计 | 把所有能力压进一个端到端网络 |
| 可校准的**对齐接口** | 所有专家天然共享的坐标系统 |

两个关键判断来自实验，而不是口号：

- **公共语义空间是可校准接口，不是零样本坐标系。** 跨专家族 zero-shot 接近 chance（vision 0.186，tactile 0.067–0.100），1-shot 校准后 vision 恢复到 0.936；tactile 仍受专家本身限制。
- **显式状态层的价值不是“精度更高”。** 公平归因后 re-ID 相对 transformer 只剩 +0.012；PSS 的主张是“状态显式、可审计、可回滚”，不是“识别更准”。

---

## 3. 架构

```
传感器 / 专家（冻结或慢更新）
  vision / tactile / geometry / language
        │
        ▼
per-modality projection → 公共语义空间（对齐接口，128-d）
        │
        ▼
认知运行时（PSS）
  ├─ EntityLayer        持久实体 ID / 生命周期 / 属性 / provenance
  ├─ RelationFactLayer  typed relations / scene graph / 版本化事实
  ├─ Active-set memory  工作记忆 / 检索 / write policy
  ├─ Uncertainty        冲突 / 弃权 / 可靠性
  ├─ Planner            规则 / 算子 / A* / 异常重规划
  ├─ OutputRouter       动作 / 行为 / 语言 / 安全
  └─ Audit              版本 / 快照 / 回滚 / 权限
        │
        ▼
输出头：动作 / 行为 / 语言 / 安全
```

### 3.1 四条契约

| 契约 | 内容 | 关键证据 |
|---|---|---|
| **专家准入** | modality / version / dim / encode / health；非法模块拒绝 | SSR=1.00；无效模块全拒绝 |
| **热插拔** | 原子换入、正确路由、非干扰、精确回滚 | SSR/RBE/INT/RCR 在 Q0–Q4 上不变 |
| **版本化知识** | assert / retract / update / rollback / audit；TMS justification | K1–K8；`TMSFactStore` 接入生产 `MultiSourceFactStore` |
| **有界推理** | entity / relation / expansion / latency caps；anytime；降级 FSM | BAR=1.00、MON=1.00、DCR=1.00 |

### 3.2 三条设计规则

1. **分模态存储，禁止相加。** 每个模态的向量独立存放；相加后无法拆回单模态。公共空间是“每个模态各自的投影 + 原型”，不是一个跨模态求和向量。
2. **槽 ≠ 身份。** 工作记忆槽只是位置；实体身份在场景图里。换槽必须从 `raw` 重建 `bound`。
3. **校准是准入的一部分。** 新专家换入时需要校准；这不是“零样本失败”，而是契约要求。

### 3.3 认知层做什么，不做什么

PSS 不负责“看懂/听懂”——那是冻结专家的活。PSS 负责：

- 把观测绑成**持久实体**；
- 算**typed relations**（spatial / semantic / tactile）；
- 维护 **active-set 记忆**和版本化事实；
- 做**规划、不确定性、弃权、审计、回滚**；
- 在有预算的前提下，把状态变成动作/语言/行为。

---

## 4. 评测：为什么是 property-level

PSS 的 claim 本身是**操作不变量**，不是端到端精度：

- 换入成功（SSR）
- 换入后不污染其他模块（INT）
- 能精确回滚（RBE）
- 路由正确（RCR）
- 知识更新局部/可传播/可审计（LOC/PCR/AUD）
- 预算遵守（BAR）、降级正确（DCR）

这些指标的设计目标是**专家质量无关**。所以评测把 expert quality 作为控制变量，而不是让它在端到端分数里混淆架构效应。

### 4.1 三个层级

| 层级 | 内容 | 作用 |
|---|---|---|
| **L0** | 结构化观测 + expert stub | 隔离架构，跑全量 property 与负控制 |
| **L1** | 真实冻结专家 + 投影 + 原型 | 在真实 embedding 上验证契约 |
| **L2** | bundle replay + fresh Isaac S4-PSS | 外部效度：真实检测流、真实执行 |

### 4.2 控制手段

- **Expert quality**：Q0（oracle）→ Q4（random）。
- **Architecture variants**：A0 full、A2 flat KB、A3 unbounded、A4 naive swap、A5 retrain-all（oracle 上界）等。
- **Twin pairing**：同 spec/seed/scene，只变架构；配对差是因果分析单位。
- **Negative controls N1–N8**：flat KB、unbounded planner、naive swap、random expert、shuffled embeddings、over-budget controller 等必须按预期失败/通过。
- **统计**：bootstrap CI、TOST、two-way ANOVA、Holm–Bonferroni；预注册见 `eval/property_preregistration.md`。

### 4.3 property vs capability

| | property | capability |
|---|---|---|
| 例子 | SSR、RBE、INT、RCR、LOC、PCR、AUD、BAR | ZCR、分类准确率、re-ID、任务成功率 |
| 是否随专家质量变化 | 不应变化 | 会变化 |
| 用途 | 证明架构契约 | 诊断专家/接口质量 |

这张表是整篇工作的支点：**契约指标专家不变，能力指标随专家变化**。如果两者混在一起，就无法回答“架构到底贡献了什么”。

---

## 5. 结果

以下数字来自 `reports/claim_gaps/REPORT.md`、`reports/claim_gaps/FOLLOWUPS.md` 和 `scorecards/property/`。完整表格见 `reports/claim_gaps/INDEX.md`。

### 5.1 认知状态：显式、可审计、可回滚（C1）

- **实体审计**（E1）：provenance 完整性、retraction 正确性、rollback 精确性、rebind 身份、tamper 检出全部 **1.00**；三个负控制（禁用审计日志 / 禁用哈希链 / 浅快照回滚）目标指标全 **0.00**。
- **绑定持久性**（E2，70 帧全遮挡）：graph+tactile retention **0.989** vs 无状态代理 **0.500**。
- **信号分解**（E2）：tactile **+0.117**、graph/state **+0.072**、vision **+0.028**。状态层提供持久化机制，但身份信号主要来自 tactile。
- **L2 bundle replay**（E12，有效 19/33 bundles）：full **1.000** / no_memory **0.827** / no_entity **0.270**。
- **S4-PSS 认知变体**（E18）：`full`/`no_memory`/`no_entity` 3/3，`no_relations` **0/3**——这个 demo 测关系，不测记忆。

边界：公平归因后 re-ID 相对 transformer 只剩 **+0.012**；PSS 不宣称识别精度优势。

### 5.2 typed relations：学到的关系是校准层，不是几何的替代（C2）

- 分布内 F1：spatial **0.957**、semantic **0.996**、tactile **0.970**；去掉 tactile 时 tactile F1 掉到 **0.483**（说明没有走捷径）。
- **hybrid 有条件成立**（E3）：learned 头只在训练覆盖的“双轴清晰”区做校准；对齐/held-out 轴交给精确几何。
  - 噪声 0.15：hybrid_diag **0.857** vs 几何 dead-zone **0.828**；
  - 朴素 hybrid（learned 用于未训练的对齐轴）反而最差：**0.503**。

结论：正确形态是“精确几何骨架 + 学习关系校准”，不是“学习关系替代几何”。

### 5.3 记忆与预算：active-set 是成本优势，fact 选择需要 query 条件（C3）

- **re-ID 候选 active-set**：k=1 + fallback 用 **~1.2 个候选**从 0.972 追平 **1.000**。
- **计算**：N=200 快 **6.4x**，N=1000 快 **23.6x**（未达预注册 10x 门槛，如实报）。
- **长历史 fact 的旧负结果**：learned 0.372 < hot 0.480 < full 1.000。
- **E4 定位原因**：旧打分是 query-independent 的；加入 query 条件后，k=4 从 0.241 到 **1.000**。但简单 `query_recency` 规则也到 1.000——所以 claim 是“接口必须 query-conditioned”，不是“学习打分更好”。
- **E13 RAG 基线**：exact-RAG（top-1 精确检索）准确率 **1.000**，但 p50 **1.38 ms** vs PSS **0.0003 ms**；成本优势不是靠弱基线。

### 5.4 契约前提：可组合、可替换、可规模（C4）

- **L1 契约**：SSR=1.00、RBE=1.00、INT=0.00、RCR=1.00，跨 Q0–Q4 不变（5 seeds）。
- **contract-free baseline**（E5）：PSS 通过全部 **12 项**契约指标；flat baseline 在 **9 项**上失败（SRR/RBE/RCR/PCR/AUD/RBE-KB/BAR/MON 全 0.00，INT 0.667）。这证明 suite 不是定制试卷。
- **scale**（E8）：dim 128/256/512 契约不变；1000 facts update **52.9 ms**、rollback **1.0**；500 实体/2000 关系仍 bounded。
- **TMS 接生产 store**（E10）：K3/K4/K5/AUD/source-retract 全 **1.00**；`TMSFactStore` 包装 `MultiSourceFactStore`，TMS 不再是平行模块。
- **runtime 权限/审计**（E17）：`read_only` 拒绝 5 项动作、`admin` 拒绝 0；拒绝动作全部入审计。

### 5.5 评测协议：负控制、twin、非 PSS baseline（C5）

- N1–N8 全部按预期失败/通过。
- twin 配对差：ZCR -0.717、RBE -1.0、LOC +1.0、PCR -1.0、BAR -1.0（5 seeds，15 对）。
- E5 的 contract-free baseline 补上了“suite 是否只对 PSS 有效”的敏感性控制。

仍要注意：E5 是 contract-free baseline，不是竞争性能力基线；如果要把评测协议单独作为方法贡献，还需要一个强非 PSS 架构。

### 5.6 公共空间：可校准接口（C6）

- **跨族 zero-shot**：3 个专家族平均 vision **0.186**、tactile **0.067–0.100**（chance 0.167）。
- **1-shot 校准**：vision 平均 **0.936**（单族最高 1.000）；tactile 仍受专家本身限制。
- **retrain-all 的真实成本**（E6/E6b/E6c）：
  - 冻结专家 + 重建 prototype：**0.964 / 3.2 ms**；
  - 仅重训投影：**0.979 / 13.5 s**；
  - 30 epoch 专家全量重训：**0.969 / 38 s**。
  - 结论：**不需要 retrain-all**。重训专家既不更快，也没有稳定质量优势。

### 5.7 诚实的负结果

这些结果和正结果一样重要：

- 公平归因后 re-ID 只有 **+0.012**：优势在显式状态，不在精度。
- 朴素 hybrid（learned 用于对齐轴）**0.503**：learned 关系不能替代几何。
- 1-shot ridge **0.922** < 冻结专家 + 重建 prototype **0.964**：校准不是万能药。
- 30ep 专家全量重训 **0.969** < 冻结专家 + 仅重训投影 **0.979**：重训专家没有质量理由。
- closed-set 精度在 per-modality / summed / random 三变体下都是 **0.969**；学习投影的价值在 cross-modal R@1（**0.510 vs 0.115**），不是闭集精度。
- S4-PSS 的 `no_memory`/`no_entity` 仍然 3/3：这个 demo 不测记忆；记忆证据来自 E12。
- tactile 单模态 test acc 在 **0.09–0.36** 之间（category_pretrain ≈ 0.245，expert_v1 ≈ 0.365）：架构不能修复弱触觉。

---

## 6. 边界与限制

- **没有真机**：所有证据来自 MuJoCo 渲染、Isaac 仿真、离线 bundle replay。
- **L2 是 oracle pose**：S4-PSS 使用 GT boxes/poses；H/K（hot-swap / KB update）没有在 Isaac 任务中执行。E12/E19/E20 补了 bundle replay 和 runtime 级证据，但感知驱动的 expert swap 仍需新的 L2 bundle（当前 dev bundles 的 `obs_detections.emb` 为空）。
- **专家族有限**：只接了 3 个同宽度族；DINOv2/SqueezeNet/WT-MM 因投影宽度不兼容未接。
- **全因子未跑**：当前是缩减设计；L0 10 seeds、L1 5 seeds。
- **TMS 未替换全部调用点**：`TMSFactStore` 是新包装，生产路径未全部迁移。
- **不宣称 SOTA / 零样本公共空间 / 真机可靠 / 大规模 scaling**。

---

## 7. 快速开始

```bash
# 依赖
pip install -e ".[train,sim]"

# 全量测试
python -m pytest

# 核心 property suite
python scripts/run_property_l0.py --seed 0
python scripts/run_property_l1.py --seed 0 --skip-family --skip-calibration
python scripts/run_property_twins.py --seeds 0,1,2,3,4
python scripts/run_property_anova.py

# claim-gap 实验（第一批）
python scripts/run_claim_gaps.py --run

# §10 追加实验（示例；完整命令见 notes/CLAIM_GAP_EXPERIMENTS.md）
python -m pss.eval.scripts.run_gap_entity_audit --seeds 5
python -m pss.eval.scripts.run_gap_tactile_binding --sequences 60 --seeds 3
python -m pss.eval.scripts.run_gap_hybrid_relations --seeds 3
python -m pss.eval.scripts.run_gap_fact_active --seeds 3 --ks 4,8,16
python -m pss.eval.scripts.run_gap_nonpss --seeds 3
python -m pss.eval.scripts.run_gap_scale --seed 0
python -m pss.eval.scripts.run_gap_detect_noise --sequences 60 --seed 0
python -m pss.eval.scripts.run_gap_families --max-per-class 40 --ks 0,1,5,20
python -m pss.eval.scripts.run_gap_retrain_all --epochs 10
python -m pss.eval.scripts.run_gap_tms --seeds 5
python -m pss.eval.scripts.run_gap_l2_memory --root bundles/dev --pattern "s1_occlusion*/**/obs_detections.parquet"
python -m pss.eval.scripts.run_gap_l2_runtime_updates --seeds 3
python -m pss.eval.scripts.run_gap_rag_exact --seeds 1 --epochs 100
python -m pss.eval.scripts.run_gap_space_ablation
python scripts/run_claim_gaps_followups.py   # 汇总；先运行对应实验

# 感知层 demo（不是完整 runtime）
python -m pss.demos.public_semantic_space_demo \
  --image data/demo_input.png --tactile data/demo_tactile.npy
```

---

## 8. 仓库导航

| 路径 | 内容 |
|---|---|
| `pss/runtime/` | 认知运行时、层次、状态、变体 |
| `pss/memory/` | 场景图、追踪、active-set、typed relations |
| `pss/knowledge/` | `FactStore`、TMS、`TMSFactStore` |
| `pss/property/` | L0/L1 property suites、entity audit、契约注册表 |
| `pss/eval/scripts/` | 实验入口（L1/L2、claim-gap） |
| `reports/claim_gaps/` | claim-gap 报告、`INDEX.md`、`FOLLOWUPS.md` |
| `scorecards/property/` | 原始 scorecards、manifests |
| `notes/PAPER_WRITING_MAP.md` | claim–evidence 映射、写作口径 |
| `notes/CLAIM_GAP_EXPERIMENTS.md` | 实验设计与结果 |
| `eval/property_preregistration.md` | 预注册与偏差 |
| `AUTHORS.md` | 作者、ORCID、贡献 |
| `NOTICE` | 版权与学术归属声明 |
| `docs/README.md` | 文档总索引（按主题、阅读顺序、维护约定） |
| `docs/PRIORITY.md` | 优先权时间线、release/DOI 建议 |
| `docs/architecture/` | 架构定位、认知运行时、评测指标、推演记录、工作总结 |
| `docs/experiments/` | PSS vs Transformer 实验与五优先级结论 |
| `docs/plans/` | 落地执行计划、优化任务清单、Isaac Sim 计划 |
| `docs/phases/` | 工程阶段记录 `PHASE*_REPORT.md`（实现细节） |
| `docs/background/` | 架构背景长文 |
| `docs/releases/` | 版本发布记录 |

### 推荐阅读顺序

1. 本文档（目的、架构、结果、边界）。
2. `docs/README.md`（按主题找文档）。
3. `notes/PAPER_WRITING_MAP.md`（每个 claim 的证据、强度、不要写的 claim）。
4. `reports/claim_gaps/INDEX.md`（实验索引与关键数字）。
5. `eval/property_preregistration.md`（预注册、指标、阈值、偏差）。
6. 具体 phase report（`docs/phases/`，需要实现细节时）。

---

## 9. 相关工作与定位（简述）

- **Modular deep learning**（adapters、MoE、model merging）：关注模块训练与组合；PSS 关注模块**换入后的契约与状态**。
- **Knowledge editing / RAG / temporal KG**：关注知识更新与检索；PSS 关注更新的**原子性、局部性、回滚、审计**，以及它与认知状态的关系。
- **Anytime / resource-bounded reasoning**：关注预算下的决策；PSS 把预算做成 runtime 的硬契约与降级 FSM。
- **Cognitive architectures / neurosymbolic**：PSS 的差异是把“感知专家”与“认知状态”分层，并用 property-level evaluation 分离两者。

PSS 的贡献不是单个机制的首创，而是**在一个非 LLM runtime 里组合契约、状态、知识与预算，并给出可分离的评测**。

---

## 10. Authorship、priority 与许可

- **主要作者**：Jiahe Qi（齐嘉禾），ORCID [0009-0001-8379-631X](https://orcid.org/0009-0001-8379-631X)。
- **作者与贡献**：`AUTHORS.md`。
- **优先权记录**：`docs/PRIORITY.md`（按日期记录 release、commit、DOI/预印本）。
- **引用**：`CITATION.cff`，或 GitHub 的 “Cite this repository”。
- **许可**：Apache-2.0（`LICENSE`）；归属声明见 `NOTICE`。
- **学术使用**：如果使用本项目的想法、方法、结果、代码、数据、图表或文档，请引用本项目；请不要把本项目的成果作为他人的原创工作发表。
- 设计文档：`notes/PAPER_WRITING_MAP.md`。
- 实验索引：`reports/claim_gaps/INDEX.md`。

---

## 一句话

**PSS 不是“更大的模型”，而是“感知之上的显式认知状态层 + 让架构贡献可以被测出来的协议”。**
