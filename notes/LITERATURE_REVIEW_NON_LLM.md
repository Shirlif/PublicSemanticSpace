# 非 LLM 架构大模型：相关工作地图

> 检索日期：2026-11-25。检索源：arXiv API、OpenAlex。关键词覆盖 beyond-transformer / post-transformer、world model、JEPA、VSA/HDC、neurosymbolic、cognitive architecture、modular deep learning、MoE、adapter serving、model merging、knowledge editing、continual learning、RAG、temporal KG、truth maintenance、adaptive computation、early exit、anytime/resource-bounded planning、metareasoning、3D scene graph、TAMP、object-centric、open-set recognition、VLA/LBM。
>
> 局限：英文文献、关键词检索，不覆盖中文期刊与部分闭源工业报告；"没有找到"只代表本次检索范围内没有，不等于绝对不存在。

---

## 0. 一句话结论

**有大量人在做"非 LLM 的大模型"，但没有一个统一的"非 LLM 大模型"赛道；PSS 的三个问题分别都有成熟文献，而把三者统一在一个"公共语义空间 + 符号运行时"里的非 LLM 架构，本次检索没有找到先例。**

更准确地说：

- **问题 1（多模块热插拔）**：机制层已被"Modular Deep Learning / MoE / adapter serving / model merging"充分研究，但几乎都建立在 LLM 基座和权重级模块上；PSS 的"表征级、跨模态、无任务 ID 热插拔"是空白。
- **问题 2（知识库热更新）**：主流是 LLM 权重编辑（ROME/MEMIT 一线），与 PSS 的外部版本化 KB 是相反路线；PSS 的真正祖先是 1979 年的 Truth Maintenance System（TMS）和 belief revision，而不是现代 knowledge editing。
- **问题 3（推理复杂度可控）**：主流是网络内部的 adaptive depth / early exit / MoE routing；PSS 的"架构级硬上界 + 降级 FSM"更接近经典 AI 的 anytime / resource-bounded planning 与 rational metareasoning。

PSS 的 novelty 不在任何单个机制，而在**把"共享语义空间"当作模块热插拔的接口契约，把知识当作可版本化/可回滚的外部运行时状态，把推理成本当作有硬上界的架构资源**——这三件事在文献里是分开的。

---

## 1. 五个研究簇

### 簇 A：非 LLM 的大模型本体

这一簇最接近"非 LLM 大模型"的字面意思，但内部分裂为几条互不引用的线：

| 路线 | 代表工作 | 与 PSS 的关系 |
|---|---|---|
| 世界模型 / 模型基控制 | DreamerV3, *Nature* 2025 (10.1038/s41586-025-08744-2)；V-JEPA 2, arXiv:2506.09985；Cosmos, arXiv:2501.03575；Genie, ICML 2024, arXiv:2402.15391 | 同为非 LLM、latent space、预测式；但都是单体世界模型，没有模块热插拔/版本化 KB |
| 隐空间递归推理 | Hierarchical Reasoning Model (HRM), arXiv:2506.21734 | 非 LLM、小参数、自适应计算；但不做多模块与知识更新 |
| 概念级语言建模 | Large Concept Models, arXiv:2412.08821 | "在句子表征空间而非 token 空间建模"与公共语义空间最接近；但仍是序列生成范式 |
| 向量符号 / 超维计算 | Kleyko et al., *ACM CSUR* 2022/2023 (10.1145/3538531, 10.1145/3558000)；Aygün et al., *ACM CSUR* 2026 (10.1145/3863975) | PSS 的 Hadamard binding 直接属于这一传统；但 VSA 文献多为小规模认知任务，缺运行时/知识版本化 |
| 神经符号 | Marra et al., *Artificial Intelligence* 2024 (10.1016/j.artint.2023.104062) | 同为"神经感知 + 符号推理"；但多为可微逻辑/端到端，不是可热插拔的运行时 |
| 认知架构 | Spaun, *Science* 2012 (10.1126/science.1225266)；Thousand Brains / Monty, arXiv:2412.18354；CoALA, TMLR 2024, arXiv:2309.02427；Global Workspace, *Trends Neurosci.* 2021 (10.1016/j.tins.2021.04.005) | 精神上最像 PSS：非 LLM、多模块、认知任务。但 Spaun 无热插拔/知识版本化；Monty 无 KB；CoALA 是 LLM 中心 |
| 具身大模型 | VLA survey, *IEEE TNNLS* 2026 (10.1109/TNNLS.2025.3650584)；Large Behavior Models, *Science Robotics* 2026 (10.1126/scirobotics.aea6201) | 当前具身智能的主流；PSS 必须解释"为什么不走 end-to-end VLA" |

**判断**：这一簇没有统一的会议/期刊标签。"非 LLM 大模型"目前是一个**否定式描述**，不是领域名。写论文时用它做标题会削弱定位。

---

### 簇 B：多模块混合与热插拔

| 工作 | 核心 | 与 PSS 的差异 |
|---|---|---|
| Pfeiffer et al., *Modular Deep Learning*, TMLR 2023 (arXiv:2302.11529) | 模块三要素：可组合接口、函数、聚合规则；路由/组合/复用 | 最接近的概念框架；但模块是 LLM 上的 adapter/LoRA，不是异构感官专家；无共享语义空间契约 |
| MoE surveys: Cai et al., *IEEE TKDE* 2025 (10.1109/TKDE.2025.3554028)；Liu et al., *ACM CSUR* 2026 (10.1145/3794845) | 条件计算、稀疏路由、推理优化 | MoE 是单网络内部路由，不是外部模块热插拔；无知识版本化 |
| S-LoRA, arXiv:2311.03285；Punica | 单 GPU 服务数千 adapter，请求级换入换出 | 工程层热插拔；模块是权重增量，语义不可解释 |
| Model merging / MoErging, *ACM CSUR* 2026 (10.1145/3787849)；arXiv:2408.07057 | 静态合并/路由专家模型 | 合并后不可运行时替换；无回滚语义 |
| Parameter-isolation continual learning；prompt-based CL；MoE + prompt CL (NeurIPS 2024) | 用独立参数/提示隔离任务，避免遗忘 | 需要 task ID 或任务边界；PSS 用原型/事实而非任务 ID |
| Compound AI systems, arXiv:2502.14815；HotOS 2025 (10.1145/3713082.3730377) | 多模型/检索器/工具编排 | LLM 中心；无统一表征契约；复杂度控制靠调度而非架构 |

**空白点**：没有工作把"换掉一个感官专家/新增一个原型包，其他模块不动，且不需要 task ID"作为**一等系统属性**来定义、测量和证明。PSS 的 Phase 9（L2 原子激活/回滚）其实已经在做这件事，但文献里没有对应的 benchmark。

---

### 簇 C：知识库热更新

| 路线 | 代表工作 | 与 PSS 的关系 |
|---|---|---|
| LLM 权重编辑 | Wang et al., *ACM CSUR* 2024 (10.1145/3698590)；KnowEdit, arXiv:2401.01286；MQuAKE, EMNLP 2023 | 相反路线：改权重 vs. 改外部 KB |
| 终身编辑的失败证据 | WilKE, arXiv:2402.10987；WikiBigEdit, arXiv:2503.05683；Knowledge in Superposition, arXiv:2408.07413 | **对 PSS 最有用**：证明权重编辑在终身尺度上会退化，反衬外部版本化 KB 的必要性 |
| 持续学习 | Wang et al., *IEEE TPAMI* 2024 (10.1109/TPAMI.2024.3367329)；Zhou et al., IJCAI 2024 (10.24963/ijcai.2024/924)；Shi et al., *ACM CSUR* 2025 (10.1145/3735633) | 提供"不遗忘"的形式化与指标；PSS 的模块隔离/原型包可借用其评测 |
| RAG | Gao et al., arXiv:2312.10997；Graph RAG, *ACM TOIS* 2025 (10.1145/3777378) | 最接近"外部知识、不重训"；但 RAG 是查询时检索，不是带冲突消解/回滚的知识状态 |
| 时序知识图谱 | Zhang et al., *KBS* 2024 (10.1016/j.knosys.2024.112454) | 建模事实的时间演化；但多为 embedding/链接预测，不是运行时断言/撤销 |
| 真值维护 / 信念修正 | Doyle, *Artificial Intelligence* 1979 (10.1016/0004-3702(79)90008-0) | **PSS FactStore 的理论祖先**：justification、retraction、一致性维护。现代深度学习文献几乎不引 |
| 概念空间 | Derrac & Schockaert, *Artificial Intelligence* 2015 (10.1016/j.artint.2015.07.002)；JAIR 2018 (10.1613/jair.1.11258) | "公共语义空间"的认知科学/语义学根基；可作为 PSS 的概念合法性来源 |

**空白点**：现代文献要么改权重（不可回滚），要么检索（无状态语义），要么时序 KG（无运行时审计）。**"版本化 + 多源断言 + 冲突标记 + 原子激活/回滚 + 审计回放"这一组语义，在具身/非 LLM 系统里没有标准实现，也没有 benchmark。**

---

### 簇 D：推理复杂度可控

| 路线 | 代表工作 | 与 PSS 的关系 |
|---|---|---|
| 动态神经网络 | Han et al., *IEEE TPAMI* 2021 (10.1109/TPAMI.2021.3117837) | 输入依赖的深度/宽度/路由；PSS 是架构级选择，不是网络内自适应 |
| Early exit | *ACM CSUR* 2024 (10.1145/3698767)；*ACM CSUR* 2022 (10.1145/3527155)；NLP survey arXiv:2501.07670 | 逐样本提前退出；给期望节省，通常无硬上界 |
| MoE 推理优化 | *ACM CSUR* 2026 (10.1145/3794845) | 稀疏激活；路由本身有成本 |
| Anytime / resource-bounded | Zilberstein, *Autonomous Robots* 1996 (10.1007/BF00162466)；Zilberstein & Russell, 1992 | **PSS 的理论祖先**：计算量 vs. 解质量、可中断、随时给出当前最优 |
| Rational metareasoning | Lieder & Griffiths, *Psychological Review* 2017 (10.1037/rev0000075)；arXiv:2410.05563 | 选择"想多久/用哪个算法"；PSS 目前用固定阈值/FSM，缺一个元推理控制器 |
| Test-time compute | "When More Thinking Hurts", ACL Findings 2026 (10.18653/v1/2026.findings-acl.1199) | LLM 的"多想"路线；已出现 overthinking 反例，支持"可控"而非"越多越好" |

**空白点**：PSS 能做到**硬上界**（实体数、关系数、A* 展开节点、检索级数），而神经自适应计算大多只有期望节省、无最坏情况保证。这是一个可辩护的强主张，但需要 quality-vs-budget 曲线、尾延迟、最坏情况证明。

---

### 簇 E：具身/感知支撑（PSS 的"下半身"）

| 工作 | 作用 |
|---|---|
| Slot Attention, NeurIPS 2020 (arXiv:2006.15055)；Binding Problem, arXiv:2012.05208 | 实体/槽表征的理论与 baseline；PSS 的 "槽 ≠ 身份" 可直接对话 |
| 3D Dynamic Scene Graphs, RSS 2020 (10.15607/RSS.2020.XVI.079)；Kimera, *IJRR* 2021 (10.1177/02783649211056674)；ConceptGraphs, ICRA 2024 (10.1109/ICRA57147.2024.10610243) | PSS 的 3D 场景图有直接对标；ConceptGraphs 是开放词表场景图的最强 baseline |
| TAMP survey, *ACM CSUR* 2023 (10.1145/3583136) | PSS 的 A* 规划属于 task planning；需说明与 TAMP 的关系 |
| KnowRob, *Artificial Intelligence* 2015 (10.1016/j.artint.2015.05.010) | 机器人知识处理框架；PSS 的 FactStore/RuleEngine 与它最接近 |
| Open-set recognition survey, *IEEE TPAMI* 2020 (10.1109/TPAMI.2020.2981604) | PSS 的开放集阈值/未见拒绝有成熟评测体系 |
| Prototypical Networks, NeurIPS 2017 (arXiv:1703.05175) | PSS 原型库的最直接 baseline |
| Lifelong Robot Library Learning, ICRA 2024 (10.1109/ICRA57147.2024.10611448)；LOTUS, ICRA 2024 (10.1109/ICRA57147.2024.10611129) | 技能库/持续模仿学习；与 PSS 的模块热插拔/技能库可对比 |

---

## 2. PSS 三问逐条对标

| PSS 设计 | 最接近的工作 | PSS 的差异 | 审稿人最可能问 |
|---|---|---|---|
| 冻结异构专家 → 每模态投影 → 128-d 公共空间 → 分模态原型库 | Modular Deep Learning；Prototypical Networks；VSA/HDC；Slot Attention | 表征级、跨模态、无 task ID；共享空间是接口契约 | "换一个专家，其他模块真的不用重训吗？干扰怎么量化？" |
| 版本化 FactStore + 多源断言 + L2 原型包原子激活/回滚 + 审计 | TMS/belief revision；RAG；temporal KG；parameter-isolation CL | 外部、可审计、可回滚、带冲突标记；不是权重编辑也不是无状态检索 | "这和数据库事务/版本控制有什么区别？形式语义是什么？" |
| A* + active set + 四级检索 + 降级 FSM | anytime/resource-bounded planning；rational metareasoning；dynamic NN/early exit | 架构级硬上界 + 优雅降级；不是学习式自适应深度 | "最坏情况上界怎么证明？quality-vs-budget 曲线在哪？" |

---

## 3. 最接近 PSS 的六个工作（按"精神相似度"排序）

1. **Spaun**（Eliasmith et al., *Science* 2012）
   - 2.5M 神经元、60B 突触、非 LLM、多功能模块、8 类认知任务。
   - 像：非 LLM、模块化、向量表征（semantic pointers 属 VSA 传统）。
   - 不像：无热插拔、无版本化 KB、无运行时复杂度控制；规模是神经元规模，不是数据/模块规模。
   - 用途：论文开头最好的"非 LLM 认知架构"先例，但必须说明 PSS 在系统工程上更进一步。

2. **Thousand Brains Project / Monty**（Clay et al., arXiv:2412.18354, 2024）
   - 非 LLM、sensorimotor、参考系、模块化皮层柱。
   - 像：实体中心、模块化、非 LLM、具身。
   - 不像：无 KB/版本化、无热插拔、无显式计算预算；早期阶段。
   - 用途：PSS 的"槽/场景图/参考系"可以直接与它对话。

3. **CoALA**（Sumers et al., TMLR 2024, arXiv:2309.02427）
   - 把认知架构（记忆、动作空间、决策循环）用于 LLM agents。
   - 像：认知架构框架、模块化记忆/动作。
   - 不像：LLM 中心；无共享语义空间；无非 LLM。
   - 用途：**镜像对手**。它代表"LLM 在中心，认知架构在外围"；PSS 可以主张"语义空间/符号运行时在中心，LLM 只是可选读出"。这是很好的 positioning。

4. **Modular Deep Learning**（Pfeiffer et al., TMLR 2023）
   - 模块接口、路由、组合、冻结基座。
   - 像：热插拔、组合、局部更新、无负干扰。
   - 不像：LLM 基座、权重级模块、无 KB、无计算控制、无共享语义空间。
   - 用途：问题 1 的 canonical citation；PSS 需说明"表征级模块"与"权重级模块"的区别与优势。

5. **DreamerV3**（Hafner et al., *Nature* 2025）
   - 非 LLM、world model、通用控制、想象式规划。
   - 像：非 LLM、latent space、model-based、计算可通过想象步数调节。
   - 不像：单体世界模型、无模块/知识库/热插拔、无符号事实。
   - 用途：证明"非 LLM 大模型"能上 Nature；PSS 可引用其"固定超参跨域"作为对照。

6. **HRM**（arXiv:2506.21734, 2025）
   - 非 LLM（小参数、非自回归 token 生成）、分层递归、自适应计算。
   - 像：非 LLM 推理、计算量可控、小模型。
   - 不像：无多模块、无 KB、无具身、任务窄（Sudoku/Maze/ARC）。
   - 用途：问题 3 的现代非 LLM 参照；说明"小模型 + 可控递归"是可行路线。

---

## 4. 竞争态势：谁在抢同一个生态位

| 生态位 | 当前占据者 | 对 PSS 的威胁 |
|---|---|---|
| 非 LLM 世界模型 | JEPA/V-JEPA 2 (Meta)、DreamerV3 (DeepMind)、Cosmos (NVIDIA)、Genie (DeepMind) | 资源/声量最大；但都不做模块热插拔与版本化 KB |
| LLM 中心认知 agent | CoALA、MemGPT/Letta、各种 agent framework | 生态最大；PSS 必须证明"非 LLM 中心"在成本/可控性/可审计性上更优 |
| 端到端具身策略 | VLA（π0/OpenVLA 一线）、Large Behavior Models | 数据/算力碾压；PSS 的差异化只能是"可解释、可热插拔、可回滚、低成本" |
| 模块化/持续学习 | Modular DL、MoE、adapter serving、model merging | 机制成熟；但没有"共享语义空间"契约 |
| 知识更新 | 权重编辑、RAG、temporal KG、TMS | 各自成熟；但没有"版本化 + 回滚 + 审计"的具身实现 |
| 计算控制 | early exit、MoE routing、anytime planning、metareasoning | 理论成熟；PSS 的"硬上界"是差异化，但需证明 |

**最值得借鉴的负面结果**：WikiBigEdit、Knowledge in Superposition 等表明，权重级终身编辑在规模上会退化。这是 PSS "外部版本化 KB" 的最强论据。

**最值得借鉴的正向结果**：DreamerV3 证明非 LLM 架构可以做到 Nature 级通用性；HRM 证明小模型 + 自适应计算可以在推理任务上竞争；Monty 证明非 LLM 具身架构仍有活跃社区。

---

## 5. PSS 的可辩护 novelty claim

中文版：

> 据我们所知，PSS 是第一个把"共享语义空间"作为异构感官专家热插拔的接口契约、把知识作为可版本化/可回滚/可审计的外部运行时状态、把推理成本作为有硬上界的架构资源，并在一个非 LLM 的具身架构中同时实现这三点的系统。

English draft:

> To our knowledge, PSS is the first architecture that treats a shared semantic space as the interface contract for hot-swapping heterogeneous sensory experts, treats knowledge as an external, versioned, rollback-able and auditable runtime state, and treats inference cost as a bounded architectural resource — jointly, within a non-LLM embodied architecture.

**必须加限定**：
- "to our knowledge" + 检索日期 + 检索范围。
- 不要把"非 LLM"当唯一卖点；它是一个**设计选择**，不是贡献本身。
- 贡献应写成三条可测量的系统属性，而不是"我们做了一个新架构"。

建议的三条 contribution：

1. **Interface contract**：定义公共语义空间的交换契约（维度、归一化、分模态存储、绑定方案），并给出热插拔的兼容性条件与干扰度量。
2. **Versioned knowledge runtime**：定义事实/原型的版本、来源、冲突、激活、回滚语义，并给出审计与回放不变量。
3. **Bounded inference**：给出实体/关系/检索/规划的计算上界，以及 quality-vs-budget 的降级曲线；与 early-exit/MoE/anytime planning 对比。

---

## 6. 审稿人最可能提出的 8 个问题

1. **"Large" 在哪？** 128-d、小专家、6 类/20 类，按 2026 标准不算大。需要定义"large"= 模块数/知识量/覆盖度，或给出 scaling 实验（D、专家数、原型数、事实数）。
2. **为什么不端到端？** 需要与 VLA/LLM agent 在同等数据/算力下比较；至少要有 real VLM baseline，不能只有 mock。
3. **热插拔怎么证明？** 需要 swap benchmark：换专家/换原型包后的 zero-shot 兼容率、旧任务退化、回滚精确性、swap 延迟。
4. **知识更新怎么证明？** 需要 edit success / locality / forgetting / rollback / conflict resolution 指标；可改造 KnowEdit/MQuAKE/WikiBigEdit 到具身事实。
5. **复杂度可控怎么证明？** 需要 worst-case bound、尾延迟、quality-vs-budget 曲线；与 early-exit/MoE/anytime 对比。
6. **理论保证？** 绑定容量（Hadamard/VSA）、开放集保证、预算下规划完备性、TMS 一致性。
7. **和数据库/事务有什么区别？** 需要明确：PSS 的知识更新是**语义级**（事实/原型/规则）而非页级事务；冲突是多源证据融合，不是 ACID。
8. **和 CoALA/Modular DL 的区别？** 需要一张对照表：中心是什么（LLM vs. 语义空间）、模块是什么（权重 vs. 表征）、更新是什么（编辑 vs. 版本化）、成本是什么（token vs. 架构上界）。

---

## 7. 命名与投稿建议

- **不要**把 "non-LLM large model" 作为标题关键词；它不是领域名。
- 可用正向标签：
  - "modular semantic-space architecture"
  - "neurosymbolic cognitive architecture"
  - "external-memory embodied architecture"
  - "resource-bounded modular architecture"
- Venue 策略（一鱼多吃）：
  - **系统/架构论文**：CoRL / ICRA / IROS（具身），或 MLSys（热插拔/服务）。
  - **Position paper**：NeSy conference、arXiv（先把术语和议程打出去）。
  - **期刊**：IEEE TCDS（认知与发展系统）、*Artificial Intelligence*（知识与推理）、TMLR（模块化/持续学习）。
- 如果只能投一篇：优先 **CoRL/ICRA 的系统 track**，因为 PSS 的独特性（具身 + 热插拔 + 版本化 KB + 硬上界）在机器人系统里最容易讲清楚；纯 ML venue 会把你拉进"和 LLM 比 accuracy"的框架。

---

## 8. 参考文献（按簇）

### A. 非 LLM 架构

1. Hafner, D., Pašukonis, J., Ba, J. L., & Lillicrap, T. (2025). Mastering diverse control tasks through world models. *Nature*. DOI: 10.1038/s41586-025-08744-2.
2. Assran, M., et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. arXiv:2506.09985.
3. NVIDIA (2025). Cosmos World Foundation Model Platform for Physical AI. arXiv:2501.03575.
4. Bruce, J., et al. (2024). Genie: Generative Interactive Environments. ICML. arXiv:2402.15391.
5. Wang, G., et al. (2025). Hierarchical Reasoning Model. arXiv:2506.21734.
6. LCM Team (2024). Large Concept Models: Language Modeling in a Sentence Representation Space. arXiv:2412.08821.
7. Kleyko, D., et al. (2022). A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I. *ACM Computing Surveys*. DOI: 10.1145/3538531.
8. Kleyko, D., et al. (2023). Part II. *ACM Computing Surveys*. DOI: 10.1145/3558000.
9. Aygün, S., et al. (2026). Learning from Hypervectors: A Survey on Hyperdimensional Encoding. *ACM Computing Surveys*. DOI: 10.1145/3863975.
10. Marra, G., et al. (2024). From statistical relational to neurosymbolic artificial intelligence: A survey. *Artificial Intelligence*. DOI: 10.1016/j.artint.2023.104062.
11. Eliasmith, C., et al. (2012). A Large-Scale Model of the Functioning Brain. *Science*. DOI: 10.1126/science.1225266.
12. Clay, V., Leadholm, N., & Hawkins, J. (2024). The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence. arXiv:2412.18354.
13. Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2024). Cognitive Architectures for Language Agents. *TMLR*. arXiv:2309.02427.
14. VanRullen, R., & Kanai, R. (2021). Deep learning and the Global Workspace Theory. *Trends in Neurosciences*. DOI: 10.1016/j.tins.2021.04.005.
15. Barreiros, J. A. L., et al. (2026). A careful examination of large behavior models for multitask dexterous manipulation. *Science Robotics*. DOI: 10.1126/scirobotics.aea6201.

### B. 多模块 / 热插拔

16. Pfeiffer, J., Ruder, S., Vulić, I., & Ponti, E. M. (2023). Modular Deep Learning. *TMLR*. arXiv:2302.11529.
17. Cai, W., et al. (2025). A Survey on Mixture of Experts in Large Language Models. *IEEE TKDE*. DOI: 10.1109/TKDE.2025.3554028.
18. Liu, J., et al. (2026). A Survey on Inference Optimization Techniques for Mixture of Experts Models. *ACM Computing Surveys*. DOI: 10.1145/3794845.
19. Sheng, Y., et al. (2023). S-LoRA: Serving Thousands of Concurrent LoRA Adapters. arXiv:2311.03285.
20. Yang, E., et al. (2026). Model Merging in LLMs, MLLMs, and Beyond. *ACM Computing Surveys*. DOI: 10.1145/3787849.
21. (2024). A Survey on Model MoErging: Recycling and Routing Among Specialized Experts. arXiv:2408.07057.
22. Wang, L., et al. (2024). A Comprehensive Survey of Continual Learning: Theory, Method and Application. *IEEE TPAMI*. DOI: 10.1109/TPAMI.2024.3367329.
23. Zhou, D.-W., et al. (2024). Continual Learning with Pre-Trained Models: A Survey. IJCAI. DOI: 10.24963/ijcai.2024/924.
24. Le, M., et al. (2024). Mixture of Experts Meets Prompt-Based Continual Learning. NeurIPS.
25. Chen, L., et al. (2025). Optimizing Model Selection for Compound AI Systems. arXiv:2502.14815.
26. Chaudhry, G. I., et al. (2025). Towards Resource-Efficient Compound AI Systems. HotOS. DOI: 10.1145/3713082.3730377.

### C. 知识更新

27. Wang, S., et al. (2024). Knowledge Editing for Large Language Models: A Survey. *ACM Computing Surveys*. DOI: 10.1145/3698590.
28. Zhang, N., et al. (2024). A Comprehensive Study of Knowledge Editing for Large Language Models (KnowEdit). arXiv:2401.01286.
29. Wu, T., et al. (2024). Continual Learning for Large Language Models: A Survey. arXiv:2402.01364.
30. Shi, H., et al. (2025). Continual Learning of Large Language Models: A Comprehensive Survey. *ACM Computing Surveys*. DOI: 10.1145/3735633.
31. (2024). WilKE: Wise-Layer Knowledge Editor for Lifelong Knowledge Editing. arXiv:2402.10987.
32. (2025). WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs. arXiv:2503.05683.
33. (2024). Knowledge in Superposition: Unveiling the Failures of Lifelong Knowledge Editing. arXiv:2408.07413.
34. Gao, Y., et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997.
35. Peng, B., et al. (2025). Graph Retrieval-Augmented Generation: A Survey. *ACM TOIS*. DOI: 10.1145/3777378.
36. Zhang, Y., et al. (2024). A survey on temporal knowledge graph embedding: Models and applications. *Knowledge-Based Systems*. DOI: 10.1016/j.knosys.2024.112454.
37. Doyle, J. (1979). A truth maintenance system. *Artificial Intelligence*. DOI: 10.1016/0004-3702(79)90008-0.
38. Derrac, J., & Schockaert, S. (2015). Inducing semantic relations from conceptual spaces. *Artificial Intelligence*. DOI: 10.1016/j.artint.2015.07.002.
39. Banaee, H., et al. (2018). An Elephant in the Dark: Creating Semantic Representations of Perceived Data with Conceptual Spaces. *JAIR*. DOI: 10.1613/jair.1.11258.

### D. 计算复杂度可控

40. Han, Y., et al. (2021). Dynamic Neural Networks: A Survey. *IEEE TPAMI*. DOI: 10.1109/TPAMI.2021.3117837.
41. (2024). Early-Exit Deep Neural Network — A Comprehensive Survey. *ACM Computing Surveys*. DOI: 10.1145/3698767.
42. Matsubara, Y., et al. (2022). Split Computing and Early Exiting for Deep Learning Applications. *ACM Computing Surveys*. DOI: 10.1145/3527155.
43. Zilberstein, S. (1996). Resource-bounded sensing and planning in autonomous systems. *Autonomous Robots*. DOI: 10.1007/BF00162466.
44. Zilberstein, S., & Russell, S. (1992). Efficient Resource-Bounded Reasoning in AT-RALPH.
45. Lieder, F., & Griffiths, T. L. (2017). Strategy selection as rational metareasoning. *Psychological Review*. DOI: 10.1037/rev0000075.
46. De Sabbata, C. N., et al. (2024). Rational Metareasoning for Large Language Models. arXiv:2410.05563.
47. Zhou, S., et al. (2026). When More Thinking Hurts: Overthinking in LLM Test-Time Compute Scaling. ACL Findings. DOI: 10.18653/v1/2026.findings-acl.1199.
48. Guo, D., et al. (2026). When Do Early-Exit Networks Generalize? A PAC-Bayesian Theory of Adaptive Depth. IJCNN. DOI: 10.1109/IJCNN69854.2026.11729661.

### E. 具身 / 感知支撑

49. Locatello, F., et al. (2020). Object-Centric Learning with Slot Attention. NeurIPS. arXiv:2006.15055.
50. Greff, K., van Steenkiste, S., & Schmidhuber, J. (2020). On the Binding Problem in Artificial Neural Networks. arXiv:2012.05208.
51. Rosinol, A., et al. (2020). 3D Dynamic Scene Graphs. RSS. DOI: 10.15607/RSS.2020.XVI.079.
52. Rosinol, A., et al. (2021). Kimera: From SLAM to spatial perception with 3D dynamic scene graphs. *IJRR*. DOI: 10.1177/02783649211056674.
53. Gu, Q., et al. (2024). ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning. ICRA. DOI: 10.1109/ICRA57147.2024.10610243.
54. Guo, H., et al. (2023). Recent Trends in Task and Motion Planning for Robotics: A Survey. *ACM Computing Surveys*. DOI: 10.1145/3583136.
55. Tenorth, M., & Beetz, M. (2015). Representations for robot knowledge in the KnowRob framework. *Artificial Intelligence*. DOI: 10.1016/j.artint.2015.05.010.
56. Geng, C., et al. (2020). Recent Advances in Open Set Recognition: A Survey. *IEEE TPAMI*. DOI: 10.1109/TPAMI.2020.2981604.
57. Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical Networks for Few-shot Learning. NeurIPS. arXiv:1703.05175.
58. Ma, Y., et al. (2026). A Survey on Vision–Language–Action Models for Embodied AI. *IEEE TNNLS*. DOI: 10.1109/TNNLS.2025.3650584.
59. Tziafas, G., & Kasaei, H. (2024). Lifelong Robot Library Learning. ICRA. DOI: 10.1109/ICRA57147.2024.10611448.
60. Wan, W., et al. (2024). LOTUS: Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery. ICRA. DOI: 10.1109/ICRA57147.2024.10611129.
