# PSS 架构评测指标

> 日期：2026-11-24
> 依据：`PSS-认知层架构定位.md` §2–§8
> 原则：**固定编码器和 LLM，只评 PSS 认知层带来的增量。**
> 目的：回答“这套架构应该重点优化什么”，而不是给 LLM/编码器做通用榜单。

---

## 0. 一句话

> PSS 的评测不应问“这个模型有多强”，而应问：**在同样的感知和语言能力下，PSS 是否让实体更持久、关系更正确、记忆更可编辑、规划更可泛化、行为更可审计、上下文更省。**

---

## 1. 先划边界：哪些指标不属于 PSS

### 1.1 判定原则

一个指标是否属于 PSS，用这个实验判断：

> **冻结编码器和 LLM，只替换/消融 PSS 层；如果指标显著变化，它才是 PSS 指标。**
>
> 如果指标只随编码器或 LLM 变化，它属于上游能力，不应算作 PSS 的架构分数。

### 1.2 应剔除的指标

| 类别 | 典型指标 | 为什么剔除 |
|---|---|---|
| 语言生成 | perplexity、BLEU/ROUGE/METEOR、流畅度、chat win rate | 由 LLM decoder 决定；PSS 只负责给它结构化上下文 |
| 通用语言/推理 | MMLU、GSM8K、MATH、BBH、HellaSwag、常识 QA、阅读理解 | LLM 本身能力，不是 PSS 状态/关系/规划层 |
| 通用视觉 | ImageNet top-1、COCO mAP、VQA accuracy、captioning CIDEr、CLIP zero-shot | 编码器/检测器能力；PSS 只消费其输出 |
| 语音 | WER、MOS | 上游编码器/声学模型 |
| 通用 agent 榜单 | WebArena、通用 tool-use 总分 | 除非固定 LLM 并只比较 PSS 提供的状态/工具/审计增量，否则是 LLM+工具的混合能力 |
| 模型规模 | 参数量、FLOPs 本身 | 不是能力指标；只能作为成本约束 |
| 裸工具调用次数 | tool calls 数量 | 调用多不等于任务完成；应与成功率、citation、成本联合看 |

**可以保留的“语言相关”指标**，必须限定为 PSS 能影响的部分：

- 固定 LLM 下的 **tool selection / argument accuracy**；
- **state faithfulness**：回答是否忠实于 PSS 状态；
- **citation coverage / entity hallucination rate**；
- **tool-loop convergence**：重复调用/预算耗尽是否收敛；
- 这些不是“语言质量”，而是 **PSS 给 LLM 的上下文和工具接口是否可靠**。

---

## 2. 指标分层：一张总记分卡

| 层级 | 关注点 | 核心指标 | 性质 |
|---|---|---|---|
| T0 北星 | 任务是否完成且可审计 | **Auditable Task Success** | 结果 |
| T1 核心 | 实体、关系、记忆、规划、不确定性、审计、工具 | 见下表 | 结果 + 守卫 |
| T2 诊断 | 为什么失败 | slot purity、relation F1、retrieval recall、conflict F1 | 先导 |
| T3 效率 | 是否值得 | latency、tokens、memory、edit time | 约束 |

### 2.1 T1 核心指标

| 层 | 核心指标 | 方向 | 说明 |
|---|---|---|---|
| 实体 | IDF1、ID retention、ID switches、duplicate rate、reappearance | 越高越好 / 越低越好 | 遮挡、重入、长时身份保持 |
| 关系 | typed relation F1、edge precision/recall、consistency violation | 越高 / 越低 | spatial/semantic/tactile/temporal/task |
| 记忆 | edited accuracy、old accuracy、forgetting、rollback exactness、conflict F1 | 越高 / 越低 | fact/prototype/episode 的可编辑性 |
| 规划 | task success、valid plan rate、replan count、OOD success、recovery rate | 越高 / 越低 | 组合泛化与执行闭环 |
| 不确定性 | ECE、Brier、selective accuracy@coverage、AUROC、FPR@95TPR、abstain precision | 校准/选择性 | 知道“不知道” |
| 审计 | audit completeness、version integrity、permission violations、reproducibility | 越高 / 越低 | 版本、回滚、权限、可复现 |
| 工具/语言接口 | tool selection、argument accuracy、citation coverage、entity hallucination、state faithfulness | 越高 / 越低 | 固定 LLM 下的 grounding |
| 控制 | active-set recall@k、token compression、fallback rate、p50/p95 latency | 越高 / 越低 | 长历史成本可控 |

> 不建议把这些指标加权成一个“PSS 总分”。不同任务的风险权重不同；应报告分维度记分卡 + 明确的门槛。

---

## 3. 分层指标定义

### 3.1 投影/对齐层

PSS 不评“编码器多强”，只评 **对齐是否可用、是否校准、缺失模态是否可靠**。

| 指标 | 定义 | 用途 |
|---|---|---|
| Cross-modal category R@1 | 固定编码器，视觉/触觉/几何检索到正确类别的 top-1 | 投影是否把模态对齐到同一空间 |
| Alignment gap | 同类别跨模态距离 − 同模态距离 | 对齐是否真的发生 |
| Calibration error (ECE/Brier) | 投影后置信度与正确率的偏差 | reliability 是否可信 |
| Modality reliability AUC | 用 reliability 预测该模态是否出错 | 缺失/噪声模态的门控 |
| Missing-modality retention | 去掉一个模态后任务指标下降 | 是否过度依赖单一模态 |
| Projection stability | 输入扰动/域偏移下 embedding 漂移 | 投影是否脆弱 |

**不评**：ImageNet/COCO/VQA 等上游感知分数；这些只作为固定背景报告。

### 3.2 实体层：PSS 的第一核心

这是 PSS 最该负责的层：**对象槽、持久 ID、生命周期、属性、provenance**。

| 指标 | 定义 | 重点场景 |
|---|---|---|
| Re-ID accuracy | 遮挡前/后实体一一匹配的正确率 | 遮挡、重入、跨帧 |
| IDF1 | 全局 ID 匹配 F1 | 在线跟踪 |
| ID retention | 每个 GT track 最常被分配到的同一 pred ID 占比 | 长时保持 |
| ID switches | 每个 GT track 的 pred ID 变化次数 | 稳定性 |
| Duplicate rate | 同一 GT 实例被建成多个实体的比例 | 记忆污染 |
| Reappearance rate | 消失后重新出现仍保留原 ID 的比例 | dormant/re-ID |
| Slot purity | 实体槽与 GT 实例的匈牙利匹配纯度 | 绑定质量 |
| Attribute binding accuracy | 属性/模态是否正确挂到实体 | 多模态绑定 |
| Provenance completeness | 每个实体/属性是否有来源、时间、置信度 | 审计前提 |

**现有结果锚点**：

- P1 手写 cost：0.561；learned metric：0.809；graph matcher v2：0.930；transformer：0.802；
- TAO detector + PSS：IDF1 0.984 vs IoU 0.653；
- 但公平归因下，同 tactile 的 transformer 可到 0.996，不能把系统级优势全归因于 PSS。

**结论**：实体层要重点看 **re-ID + retention + duplicate + provenance**，不是只看平均 re-ID accuracy；near-identical 子集没有设计好的身份信号，应作为 abstain/压力测试。

### 3.3 关系层

PSS 不评“模型是否理解图片”，只评 **关系是否算对、是否一致、是否可组合**。

| 指标 | 定义 |
|---|---|
| Typed relation F1 | spatial / semantic / tactile / temporal / task 每类单独 F1 + macro |
| Edge precision/recall | scene graph 中关系边的准确率/召回 |
| Relation calibration | 关系置信度与正确率的一致性 |
| Consistency violation | 反对称、互斥、传递、置换不变性的违反率 |
| Graph edit distance | 预测图与 GT 图的结构差异 |
| Compositional relation OOD | 训练中见过的关系、未见过的关系组合 |
| Relation hallucination | 无证据边/错误谓词比例 |

**现有结果锚点**：

- spatial 0.957 / semantic 0.996 / tactile 0.970；
- P5b 组合关系 OOD：PSS explicit planner 1.000 vs transformer BC 0.434；
- 纯学习边没有超过手写边，hybrid 在缺 tactile 时更好（0.819 vs 0.791/0.696）。

**重点**：关系层不要只报一个 F1；要报 **per-predicate + OOD 组合 + consistency violation**。如果一致性 violations 高，即使 F1 高也不适合做规划。

### 3.4 记忆/知识层

这是 PSS 区别于 LLM 的第二核心：**fact/prototype/episode 可编辑、可回滚、可追溯**。

| 指标 | 定义 |
|---|---|
| Edited accuracy | 修改事实后，新事实查询正确率 |
| Old accuracy | 修改后，旧事实是否仍正确（未误伤） |
| Forgetting | old_before − old_after |
| Rollback exactness | 回滚后状态是否逐字段等于修改前 |
| Conflict detection F1 | 多源冲突是否被发现 |
| Conflict resolution accuracy | 按 priority/timestamp/source 解析是否正确 |
| Retrieval recall@k | active-set 检索是否覆盖 gold evidence |
| Coverage | 在给定 k/预算下能覆盖多少查询 |
| Edit time | 单条 fact/rule 更新时间 |
| Version integrity | 版本号单调、快照可复现、无悬挂引用 |

**现有结果锚点**：

- fact update：PSS 1.000/1.000，0.171 ms vs transformer full old 0.735 / 952 ms；
- 长历史 1000：PSS 1.000 vs transformer 0.120；RAG 0.971；
- 多源冲突：priority 1.000 vs flat last-write 0.800；
- 长历史 fact active-set 仍是负结果：learned 0.372 < hot 0.480 < full 1.000。

**重点**：**edited/old/forgetting/rollback** 四件套必须一起报；只报“新事实答对”会掩盖灾难性遗忘。

### 3.5 规划/动作层

PSS 不评“LLM 会不会做数学题”，只评 **目标、算子、计划、执行、恢复**。

| 指标 | 定义 |
|---|---|
| Task success | 目标状态是否达成（ID / OOD 分开） |
| Valid plan rate | 计划是否通过前置条件/算子校验 |
| Replan count | 执行异常后重规划次数 |
| Expansions | A*/搜索扩展节点数 |
| Plan cost ratio | 计划代价 / 最优或参考代价 |
| Execution success | 真实/仿真执行后的成功率 |
| Recovery rate | 异常注入后恢复并完成目标的比例 |
| Safety violation rate | 碰撞、越界、权限外动作 |
| Goal faithfulness | 执行动作是否真的服务于目标 |

**现有结果锚点**：

- P5b 闭环 kinematic：PSS 1.000/1.000 vs transformer 0.950/0.400；
- physical push：PSS 0.800/0.700 vs transformer 0.750/0.150；
- learned heuristic：expansions 6.29→3.28，cost/success 不变。

**重点**：**OOD success + execution success + recovery + safety** 比 ID 成功率更能说明 PSS 规划层；ID 成功率容易被记忆/规则覆盖。

### 3.6 不确定性 / 冲突 / 弃权

LLM 的 softmax 置信度不能直接用于 PSS，因为 PSS 要做 **多源融合、冲突检测、弃权/复核**。

| 指标 | 定义 |
|---|---|
| ECE / Brier | 融合置信度校准 |
| Selective accuracy @ coverage | 在覆盖率 c 下，接受样本的准确率 |
| AUROC / FPR@95TPR | 开放集/异常检测 |
| Conflict detection F1 | 模态/事实冲突是否被识别 |
| Abstain precision | 弃权样本中真正不该接受的比例 |
| Review rate | 进入人工/复核的比例 |
| Risk-coverage curve | 风险随覆盖率的变化 |

**现有结果锚点**：

- 开放集：PSS prototypes AUROC 0.993 / reject 0.958 vs transformer softmax 0.886 / 0.445；
- split-conformal：seen accept 0.969 / unseen reject 0.958；
- 模态冲突：vision shuffle 下 PSS 0.474 vs transformer 0.156，但 geometry shuffle 下 0.458 vs 0.749（互有胜负）。

**重点**：不要只报 accuracy；要报 **risk-coverage** 和 **conflict F1**。PSS 的价值是“知道什么时候不该信”，不是“永远给答案”。

### 3.7 审计 / 版本 / 权限

这是 LLM 不具备、PSS 必须自己证明的层。

| 指标 | 定义 |
|---|---|
| Audit completeness | 应记录的事件（写、拒绝、回滚、权限）实际记录比例 |
| Rollback exactness | 回滚后状态与目标版本逐字段一致 |
| Version integrity | 版本单调、无环、快照可复现 |
| Permission enforcement accuracy | 各 profile 下允许/拒绝是否符合矩阵 |
| Permission violation count | 越权动作次数（目标为 0） |
| Reproducibility | 同输入/同版本/同 seed 输出一致 |
| Audit query latency | 审计链查询耗时 |

**现有结果锚点**：

- Phase 16 消融：no_entity/no_uncertainty/no_audit/no_language 分别打在各自层；
- Phase 24：四档权限 + 拒绝审计；operator 无 rollback；
- 目前缺少统一的“audit completeness”量化，应补。

**重点**：审计层不要只做“有日志”；要能证明 **完整性、回滚精确性、权限零越权、可复现**。

### 3.8 输出 / 工具 / 语言接口

LLM 的语言能力不评；只评 PSS 给它的 **结构化状态、工具、审计** 是否让任务更可靠。

| 指标 | 定义 |
|---|---|
| Tool selection accuracy | 是否选对工具 |
| Argument accuracy | 参数/实体/版本是否正确 |
| Tool success rate | 工具调用是否成功并返回可用结果 |
| Citation coverage | 回答中的 claim 是否有可验证 citation |
| Entity hallucination rate | 回答中出现状态中不存在的实体 |
| State faithfulness | 回答是否忠实于 PSS 状态 |
| Tool-loop convergence | 重复调用/预算耗尽前是否收敛 |
| Context compression ratio | 相同任务下 tokens vs flat context |

**现有结果锚点**：

- flat JSON 0.714 vs structured state + tools 1.000；
- Qwen2.5-1.5B 0.857 vs ToolAgent 1.000；citation 0 vs 2；
- 60 物体/200 关系：3439 vs 12666 tokens（0.27×）。

**重点**：**固定 LLM**，比较“有 PSS 状态/工具” vs “无 PSS 状态/工具”的 delta；不要报 LLM 的通用语言分数。

### 3.9 控制 / 效率

| 指标 | 定义 |
|---|---|
| Active-set recall@k | 选中集合覆盖 gold evidence 的比例 |
| Coverage @ budget | 给定 token/延迟预算下的任务覆盖 |
| Token compression | 相同成功率下的 token 缩减 |
| p50/p95 latency | 端到端/分层延迟 |
| Memory footprint | scene graph/fact store/索引占用 |
| Fallback rate | 回退到更大上下文/手写规则的比例 |
| Scaling exponent | 随实体/历史增长的延迟曲线 |

**现有结果锚点**：

- 长历史 PSS 0.0003 ms vs transformer 4.90 ms；
- active ANN k=8：noise 0.3 时 0.920 vs full 0.495；
- graph matcher pruned N=1000 34.99 ms vs transformer 852 ms（24.35×）。

**重点**：效率必须和 **同等成功率** 一起报；单独报“快”没有意义。

---

## 4. 评测协议

### 4.1 Capability-controlled evaluation

固定：

- 同一个冻结编码器（DINOv2/CLIP）；
- 同一个 LLM（Qwen2.5-1.5B 或 API 模型）；
- 同一个检测器/分割器；
- 同一个 token 预算 / 工具预算；
- 同一套 prompt。

只变：

- PSS 实体层、关系层、记忆、active-set、不确定性、审计、工具接口。

报告 **delta vs 消融**，不报告“PSS 比 GPT-5 强/弱”。

### 4.2 必做消融

| 变体 | 检验 |
|---|---|
| full PSS | 完整 |
| no_entity | 实体槽/持久 ID 是否必要 |
| no_relations | 关系层是否必要 |
| no_memory | facts/prototypes/versions 是否必要 |
| no_active_set | 长历史/成本是否必要 |
| no_uncertainty | 冲突/弃权是否必要 |
| no_audit | 回滚/权限/审计是否必要 |
| flat_context | 结构化状态 vs 平铺上下文 |
| no_tools | 工具接口是否必要 |
| encoder_only / llm_only | 上游能力基线 |

### 4.3 数据与泄漏

- 按 sequence / video / camera / scene 划分 train/val/test，禁止同轨迹泄漏；
- 合成数据与官方数据分开报告；
- 关系/属性 OOD 要按组合划分，不是随机划分；
- re-ID 的 near-identical 子集单独报告，不作为总体唯一分数；
- LLM 评测固定 prompt/版本，避免模型升级污染对比。

### 4.4 统计

- 至少 3–5 seeds；
- 报 mean ± std；
- 关键指标做配对检验/置信区间；
- 小样本（如 TAO 5 视频）明确标注，不外推。

### 4.5 在线 vs 离线

- 离线：P1/P1c/P0.1、CLEVR/TAO 子任务；
- 在线：完整 tracker 的 IDF1/retention/switches/reappearance；
- 闭环：P5b + 执行/重规划/恢复；
- 长时：长历史 fact/active-set + 版本/回滚。

---

## 5. 重点指标优先级

### P0：架构声明必须证明的

1. **实体持久性**：IDF1、retention、switches、duplicate rate、reappearance；
2. **记忆可编辑性**：edited/old/forgetting/rollback/conflict；
3. **关系组合 OOD**：per-predicate F1 + consistency violation + OOD success；
4. **规划/执行 OOD**：task success、valid plan、recovery、safety；
5. **校准/弃权**：selective accuracy@coverage、conflict F1、risk-coverage；
6. **审计完整性**：audit completeness、rollback exactness、permission violations；
7. **工具/状态 grounding**：tool selection、citation coverage、entity hallucination。

### P1：解释“为什么强/为什么弱”

- slot purity、attribute binding、provenance completeness；
- active-set recall@k、fallback rate、token compression；
- relation calibration、graph edit distance；
- planning expansions/replan count；
- latency p50/p95、memory footprint。

### P2：规模化与真实数据

- 更多 TAO 视频 + 官方 HOTA；
- sim2real / 真实机器人；
- 跨域/跨相机 re-ID；
- 在线自适应/伪标签的长期稳定性；
- 多模态缺失/噪声的鲁棒性。

---

## 6. 现有结果映射（用来校准预期）

| 指标 | 现有结果 | 说明 |
|---|---|---|
| re-ID（P1） | 0.561 → learned 0.809 → graph v2 0.930 | 但公平归因同 tactile 下增量仅 +0.012 |
| TAO IDF1 | 0.984 vs IoU 0.653 | 5 视频、person-only |
| 关系 F1 | 0.957/0.996/0.970 | spatial/semantic/tactile |
| 关系 OOD | 1.000 vs transformer 0.434 | P5b explicit planner |
| 事实更新 | 1.000/1.000，0.171 ms | vs transformer full old 0.735 |
| 长历史 | 1.000 vs 0.120 | history 1000 |
| 开放集 | AUROC 0.993 / reject 0.958 | vs transformer 0.886/0.445 |
| 规划/执行 | 1.000/1.000 vs 0.950/0.400 | kinematic |
| 物理执行 | 0.800/0.700 vs 0.750/0.150 | physical push |
| 语言/工具 | 1.000 vs flat 0.714 | 7 查询，小样本 |
| 审计 | 消融全通过 | 缺统一 completeness 量化 |

---

## 7. 报告模板

每个 PSS 评测至少包含：

```text
benchmark:
layer:            encoder / projection / entity / relation / memory / planning / uncertainty / audit / output
capability:       frozen encoder = ...; frozen LLM = ...; detector = ...
variant:          full / no_entity / no_relations / no_memory / no_active_set / no_uncertainty / no_audit / flat / no_tools
metric:           <name and formula>
pss:              mean ± std (n, seeds)
baseline:         mean ± std (n, seeds)
delta:            pss - baseline (95% CI)
cost:             p50/p95 latency, tokens, memory
leakage:          sequence/video/scene split confirmed
notes:            synthetic / official; small-sample caveat
```

推荐同时报告 **leading + lagging + guardrail**：

- leading：relation F1、slot purity、retrieval recall、conflict F1；
- lagging：task success、IDF1、OOD success、edited accuracy；
- guardrail：permission violations、rollback exactness、safety violations、hallucination rate。

---

## 8. 反模式：不要这样评 PSS

1. **用 MMLU/GSM8K/ImageNet 证明 PSS**：那是编码器/LLM 的能力。
2. **把 PSS 和 LLM 端到端混在一起报总分**：无法归因。
3. **只报平均 re-ID**：samecat/near-identical 会掩盖实体层崩溃。
4. **只报新事实准确率**：不报 old/forgetting，会掩盖灾难性遗忘。
5. **只报关系 F1**：不报 consistency violation，规划层可能不可用。
6. **只报延迟**：不报同等成功率，快而无用。
7. **只报工具调用次数**：不报 tool success/citation/grounding。
8. **单 seed/单视频**：小样本结论外推。
9. **用 oracle active-set/GT box**：那是 harness，不是架构。
10. **把 near-identical 当必须高分**：该子集没有设计好的身份信号，应看 abstain/崩溃边界。

---

## 9. 结论：PSS 应该重点关注什么

> **第一优先级不是“更强的感知”，而是：实体 ID 不丢、事实可改可回滚、关系组合可泛化、规划执行可恢复、置信度可弃权、行为可审计、工具上下文可压缩。**

如果只能选五个指标作为 PSS 的“架构体检”：

1. **IDF1 / ID retention / duplicate rate**（实体持久性）
2. **edited accuracy / old accuracy / forgetting / rollback exactness**（记忆可编辑性）
3. **relation F1 + consistency violation + relation OOD**（关系可组合性）
4. **OOD task success + recovery + safety violation**（规划可执行性）
5. **selective accuracy@coverage + conflict F1 + audit completeness**（可信与可审计）

这五个指标能直接把“PSS 是 harness”与“PSS 是认知架构”区分开；LLM/编码器的通用分数不应进入这张表。
