# PSS 架构特性验证实验计划

**Property-Level Validation Plan for Public Semantic Space (PSS)**

| 项 | 内容 |
|---|---|
| 版本 | v1.0（草案，待预注册） |
| 日期 | 2026-11-25 |
| 状态 | 设计完成，未执行 |
| 关联文档 | `eval/preregistration.md`、`PHASE10_REPORT.md`、`PHASE10_HARDENING_REPORT.md`、`notes/LITERATURE_REVIEW_NON_LLM.md`、`notes/PROJECT_MAP.md` |
| 一句话目标 | 在剥离感官专家质量影响的前提下，验证 PSS 的三条架构契约：多模块热插拔、知识库热更新、推理复杂度可控。 |

---

## 0. 摘要

PSS 的端到端指标同时受五个因素影响：感官专家质量、跨模态投影、原型库、认知运行时、任务/场景难度。直接用端到端 accuracy 验证架构，会把专家质量误当成架构贡献。

本计划把验证对象从 **capability** 改为 **property**：

1. **多模块热插拔**：给定满足接口契约的专家，系统能否在不重训其它模块、不重启运行时的情况下替换它，并保持共享空间兼容、非干扰、可回滚。
2. **知识库热更新**：给定一次事实/原型更新，系统能否原子生效、局部传播、冲突可检、状态可回滚、变更可审计。
3. **推理复杂度可控**：给定计算预算，系统能否遵守硬上界、随时给出合法结果、优雅降级，并在预算-质量曲线上保持单调。

实验分三层：

- **L0 runtime sandbox**：结构化观测 + expert stub，不含感知。用于证明架构不变量。
- **L1 controlled integration**：真实/半真实专家，质量作为受控因子。用于证明性质在感知噪声下仍成立。
- **L2 real experts**：SmallCNN / SqueezeNet / DINOv2 / WT-MM。用于外部效度，不作为性质主证据。

主结论来自 L0 + L1；L2 是工程演示。

---

## 1. 动机与范围

### 1.1 问题

当前 PSS 的实验（Phase 10、S4-PSS）主要回答“系统能不能完成任务”。这会把以下因素混在一起：

```
end-to-end performance
  = f(expert quality, projection, prototype bank, runtime, task difficulty)
```

其中 expert quality 是**外部给定**的，不是 PSS 架构的贡献。要证明架构特性，必须把它变成控制变量。

### 1.2 目标

- 证明三条架构契约在专家质量变化时**保持稳定**。
- 量化架构效应与专家效应的**方差占比**。
- 建立一套可复用的 **property test suite**，进入 CI。
- 为论文提供“架构贡献”而非“专家贡献”的证据。

### 1.3 非目标

- 不追求 SOTA accuracy。
- 不宣称真机部署可靠性（L2 只做小规模验证）。
- 不验证所有任务；只选能承载三条契约的代表性场景。
- 不把 oracle 结果当作能力结论。

### 1.4 成功标准（建议阈值，需预注册确认）

| 特性 | 成功标准 |
|---|---|
| 热插拔 | 同模态零样本 swap 成功率 ≥ 0.90；跨模态 ≥ 0.70；干扰 ≤ 2 个百分点；回滚精确性 = 1.00；停机 ≤ 1% |
| 知识库 | 更新成功率 ≥ 0.99；局部性违规 = 0；回滚精确性 = 1.00；冲突检出率 ≥ 0.95；审计完整率 = 1.00 |
| 计算 | 预算遵守率 = 1.00；p99 延迟 ≤ 上界；预算-质量单调率 ≥ 0.95；降级正确率 ≥ 0.95 |
| 专家不变性 | 架构效应方差 / 专家效应方差 ≥ 3；三条 property 指标的 expert-quality 斜率在等价界内 |

---

## 2. 研究问题与假设

### 2.1 研究问题

- **RQ1（热插拔）**：在专家质量从 oracle 到 random 变化时，PSS 的 swap 成功率、兼容性、干扰、回滚是否保持稳定？
- **RQ2（知识库）**：在感知被 oracle 化的条件下，PSS 的更新原子性、局部性、传播、冲突、回滚、审计是否成立？
- **RQ3（计算）**：在场景规模和专家延迟变化时，PSS 是否遵守预算上界并保持 anytime 有效性？
- **RQ4（归因）**：三条 property 指标的方差，主要由架构变体解释，还是由专家质量解释？
- **RQ5（交互）**：三条特性同时启用时，是否出现跨特性的负干扰？

### 2.2 假设

| ID | 假设 | 零假设 | 杀死标准 |
|---|---|---|---|
| PH1 | 同模态零样本 swap 成功率与专家质量无关（斜率 ≈ 0） | 成功率随专家质量显著下降 | 在 Q1–Q4 上斜率显著为负，且 Q2 成功率 < 0.7 |
| PH2 | 跨模态 transplant 在共享空间下可行 | 共享空间是模态特定的 | 跨模态成功率 ≤ 随机水平 |
| PH3 | 版本化 KB 的局部性/回滚优于 flat KB | 两者无差异 | 版本化 KB 的局部性违规 > 0 或回滚精确性 < 1 |
| PH4 | 预算控制器在对抗场景下仍遵守上界 | 上界被突破 | 任一场景预算遵守率 < 1 或 p99 > 2× 上界 |
| PH5 | 架构效应方差 ≥ 3× 专家效应方差 | 专家效应主导 | 方差比 < 1.5，或 property 指标随专家质量单调下降 |
| PH6 | 负控制能按预期失败 | 负控制也通过 | 任一负控制通过（测试不敏感） |

### 2.3 预注册

在跑正式实验前，把 2.1–2.2、指标定义、阈值、分析计划写入：

```
eval/property_preregistration.md
```

预注册后不再改主指标和杀死标准；探索性分析单独标注。

---

## 3. 操作化定义

### 3.1 Capability vs. Property

| | capability | property |
|---|---|---|
| 问题 | 识别/规划得准不准 | 架构契约是否成立 |
| 依赖 | 专家、数据、投影、原型 | 架构机制 |
| 主指标 | top-1、R@1、AUROC、成功率 | swap 成功率、局部性、预算遵守 |
| 专家影响 | 强 | 应可分离 |
| 本计划角色 | 辅助检查（不能崩） | **主验证对象** |

### 3.2 Expert contract

所有专家（真实或 stub）必须满足同一接口：

```python
class ExpertContract(Protocol):
    modality: str          # "vision" | "tactile" | "geometry" | "audio" | ...
    version: str           # semantic version, e.g. "smallcnn-v1"
    dim: int               # must equal PUBLIC_DIM
    def encode(self, obs: Observation) -> np.ndarray:
        """Return (B, dim), L2-normalized, no NaN/Inf."""
    def health(self) -> ExpertHealth:
        """latency p50/p99, calibration, OOD score, failure flags."""
```

**硬约束**：

1. `dim == PUBLIC_DIM`（当前 128）；
2. 每行 L2 范数 = 1 ± 1e-6；
3. 无 NaN/Inf；
4. 同一 seed 下确定性；
5. 延迟 ≤ `L_max`（按模态配置）；
6. `version` 可比较，用于 version skew 测试。

违反任一条的模块，必须被 runtime **拒绝**，而不是静默产生垃圾。

### 3.3 Expert quality vector

不要用单一 accuracy 代表“专家质量”。用向量：

```
q = (accuracy, ECE, alignment_R@1, latency_p99, robustness, coverage)
```

实验中的质量等级：

| 等级 | accuracy | ECE | 说明 |
|---|---:|---:|---|
| Q0 oracle | 1.00 | 0.00 | 完美标签/嵌入，天花板 |
| Q1 high | 0.90 | 0.05 | 强专家 |
| Q2 medium | 0.75 | 0.10 | 中等 |
| Q3 low | 0.55 | 0.20 | 弱专家 |
| Q4 random | chance | — | 随机但格式合法 |
| Q5 adversarial | 系统性错误 | — | 定向错误/常数/翻转 |

Q0 只用于隔离架构，不作为能力结论。Q4/Q5 用于测试安全拒绝与优雅降级。

### 3.4 Architecture variants

| 变体 | 含义 | 用途 |
|---|---|---|
| A0 | 完整 PSS | 主条件 |
| A1 | 每模态独立空间 + late fusion（无共享空间） | H 系列 ablation |
| A2 | flat KB（无版本/无 justification） | K 系列负控制 |
| A3 | 无预算（unbounded planner） | C 系列负控制 |
| A4 | naive swap（直接替换，不校准投影） | H 系列负控制 |
| A5 | retrain-all after swap | H 系列上界基线 |
| A6 | no active set（全量实体/关系） | C 系列 ablation |
| A7 | flat retrieval（无四级检索） | C 系列 ablation |
| A8 | no degradation FSM | C 系列 ablation |

A0-* 是 A0 的子变体（如 `A0-nonorm`、`A0-randproj`、`A0-nojust`、`A0-fixed`），只在对应实验内定义和使用，不进入主条件。

所有变体共享同一专家池、同一数据、同一 seed。唯一变化是架构机制。

### 3.5 Property metrics

完整定义见 §11。缩写：

- **H 系列**：SSR、ZCR、INT、RBE、DT、RCR、SRR、SML。
- **K 系列**：USR、LOC、FOR、PCR、CDR、RES、RBE-KB、AUD、UL。
- **C 系列**：BAR、AQV、MON、p95/p99、DCR、CAT、WCB。

### 3.6 Ground truth 与 observation schema

L0 使用结构化 `Observation`，不经过检测器：

```json
{
  "frame_id": 42,
  "timestamp": 1.234,
  "entities": [
    {"id": "e1", "category": "cup", "pose": [x,y,z,qx,qy,qz,qw],
     "attributes": {"material": "ceramic", "size": "small"},
     "visibility": 0.8, "modality": "vision"}
  ],
  "relations": [
    {"subject": "e1", "predicate": "left_of", "object": "e2", "confidence": 1.0}
  ],
  "facts": [
    {"entity": "e1", "attribute": "clean", "value": "no", "source": "gt"}
  ]
}
```

L0 的 ground truth 与观测同源，因此可以精确计算架构指标。

---

## 4. 实验总体设计

### 4.1 三层

| 层 | 输入 | 专家 | 目的 | 运行成本 |
|---|---|---|---|---|
| L0 | 结构化 observation | stub | 证明架构不变量 | 低（CPU） |
| L1 | sim 观测 | 冻结真实专家 + 质量注入 | 证明性质在感知噪声下成立 | 中（GPU/Isaac） |
| L2 | sim/real 观测 | SmallCNN/SqueezeNet/DINOv2/WT-MM | 外部效度 | 高 |

**主结论只来自 L0 + L1。** L2 失败不推翻架构结论，但 L2 成功增强外部效度。

### 4.2 因子设计

```
Expert quality: Q0, Q1, Q2, Q3, Q4, Q5         (6)
Architecture:   A0, A1, ..., A8                 (per experiment subset)
Task difficulty: T1 easy, T2 medium, T3 hard    (3)
Scene/seed:     5 seeds × 10 episodes           (50)
```

全因子会爆；采用**分层全因子 + 分数因子**：

- H 系列：Q0–Q4 × {A0, A1, A4, A5} × T1–T2。
- K 系列：Q0–Q2 × {A0, A2} × T1–T3。
- C 系列：Q0–Q4 × {A0, A3, A6, A7, A8} × T1–T3。
- 交互：Q0–Q2 × {A0, A2, A3} × T2。

### 4.3 配对/反事实

每个 episode 生成一个 **twin**：

- 同一 spec hash、同一 seed、同一场景；
- 唯一差别是架构变体或 swap/update/budget 操作；
- 用配对差 `Δ = metric(treated) − metric(control)` 消除场景方差。

这是因果归因的核心。不要把不同场景的均值直接相减。

### 4.4 随机化与区组

- 场景按难度区组（T1/T2/T3）；
- 专家质量顺序随机化，避免顺序效应；
- swap/update 顺序随机化；
- 所有随机种子记录在 manifest。

### 4.5 样本量与功效

先做 pilot（每格 5 episodes），估计方差，再做 power analysis。

目标：检测 `d = 0.5` 的架构效应，power ≥ 0.80，α = 0.05（双侧，Holm 校正）。

若 pilot 方差大，优先增加 **seeds** 而不是 episodes（seed 是更干净的随机效应）。

### 4.6 预注册

正式跑之前冻结：

- 主指标（每特性 2–3 个）；
- 次要指标（探索性）；
- 杀死标准；
- 分析模型；
- 排除规则（崩溃、超时、数据损坏）；
- 多重比较校正方法。

---

## 5. 平台与数据

### 5.1 L0 runtime sandbox

**目标**：完全绕过感知，直接测 runtime。

**输入**：结构化 `Observation` 流 + expert stub。

**场景**：

- S0a：单实体、无关系（最小）；
- S0b：多实体、有遮挡（场景图/追踪）；
- S0c：多关系、有冲突（事实/规则）；
- S0d：大规模实体/关系（计算压力）；
- S0e：动态增删实体（热更新/生命周期）。

**实现**：

- `pss/experts/stubs.py`：oracle/noisy/random/constant/adversarial/delayed。
- `pss/observations/synthetic.py`：从 JSON/生成器产生 `Observation`。
- `pss/runtime/variants.py`：用 `build_runtime(variant)` 切换 A0–A8。
- `tests/properties/`：Hypothesis 生成随机操作序列。

**运行**：CPU，分钟级；可进 CI。

### 5.2 L1 controlled integration

**目标**：真实感知在回路，但专家质量受控。

**数据**：

- Isaac：`specs/s1_occlusion.yaml`、`s2_*`、`s3_*`、`s4_pss_*`、`s6_*`。
- MuJoCo：`data/expert_v1`、`data/v1`。
- 合成：`pss/sim` 生成器。

**专家池**：

- vision：SmallCNN、SqueezeNet1.1、DINOv2（若可用）。
- tactile：MLP、WT-MM CNN。
- geometry：手工特征 + 逻辑回归。
- 每个专家冻结，导出 manifest + sha256。

**质量注入**：

- 用混淆矩阵把 accuracy 标定到 Q1/Q2/Q3；
- 用温度缩放/标签平滑把 ECE 标定到目标；
- 用高斯噪声/维度扰动控制 alignment；
- 用 sleep/批大小控制延迟；
- 用 modality dropout 模拟缺失。

**注意**：质量注入必须**独立于架构**，且在实验前验证（manipulation check）。

### 5.3 L2 real experts

**目标**：外部效度。

- 直接用真实专家，不注入质量；
- 重复 H/K/C 的核心条件（A0 vs 负控制）；
- 报告真实专家的 quality vector，作为协变量。

**数据划分**：

- 训练：`expert_v1` / `data/v1`。
- 验证：留出 scenes。
- 测试：留出 scenes + 留出 expert family（如 CNN 训练、ViT 测试）。

### 5.4 可复现性

每个 run 记录：

```json
{
  "run_id": "...",
  "spec_hash": "...",
  "bundle_hash": "...",
  "expert_manifest_sha256": "...",
  "variant": "A0",
  "quality": "Q2",
  "seed": 0,
  "episode": 3,
  "git_commit": "...",
  "env": {"python": "3.13", "torch": "..."},
  "timestamp": "..."
}
```

### 5.5 新增/修改模块清单（建议）

| 路径 | 类型 | 内容 |
|---|---|---|
| `pss/experts/contract.py` | 新增 | `ExpertContract`、`validate_expert` |
| `pss/experts/stubs.py` | 新增 | Oracle/Noisy/Random/Constant/Adversarial/OOD/Delayed/Dropout |
| `pss/experts/quality.py` | 新增 | `inject_quality`、manipulation check |
| `pss/observations/synthetic.py` | 新增 | 结构化 `Observation` 生成器 |
| `pss/metrics/property.py` | 新增 | H/K/C 三组 property 指标 |
| `pss/runtime/variants.py` | 修改 | 增加 A1–A8 变体 |
| `pss/knowledge/facts.py` | 修改 | `Justification` + 依赖图（K3/K4） |
| `tests/properties/` | 新增 | Hypothesis property tests |
| `eval/property_preregistration.md` | 新增 | 预注册 |
| `scripts/run_property_experiments.py` | 新增 | 批量运行 + 配对/twin 管理 |
| `scorecards/property/` | 新增 | 结果归档 |

---

## 6. 专家质量控制系统

### 6.1 Stub family

| Stub | 行为 | 用途 |
|---|---|---|
| `OracleExpert` | 返回 ground-truth one-hot/原型 | 天花板、隔离架构 |
| `NoisyOracleExpert` | 按混淆矩阵翻转标签 | Q1–Q3 |
| `RandomExpert` | 随机单位向量 | Q4、property 负控制 |
| `ConstantExpert` | 固定单位向量 | 安全拒绝、退化测试 |
| `AdversarialExpert` | 系统性错误（固定映射） | 鲁棒性 |
| `OODExpert` | 训练分布外的嵌入 | 开放集/漂移 |
| `DelayedExpert` | 注入延迟 | 计算预算交互 |
| `DropoutExpert` | 随机返回空/零 | 缺失模态 |

### 6.2 Quality injection

```python
def inject_quality(expert, target_acc, target_ece, seed):
    # 1. 估计 expert 在 calibration set 上的 confusion matrix
    # 2. 求混合矩阵 M，使 (1-α)·I + α·M 的 top-1 ≈ target_acc
    # 3. 用 temperature scaling 调 ECE
    # 4. 返回包装后的 expert，并在 health() 报告实际质量
```

**Manipulation check**：注入后独立测量 accuracy/ECE/alignment，确认落在目标区间；否则该 run 作废。

### 6.3 Same-pool protocol

- 所有架构变体使用**同一份**专家权重（同一 sha256）。
- 所有质量等级从**同一份**权重派生（只改注入参数）。
- 不允许“A0 用更好的专家、A1 用更差的专家”。

### 6.4 Expert family hold-out

为测试 PH5（专家不变性），额外准备：

- Family F1：SmallCNN。
- Family F2：SqueezeNet。
- Family F3：DINOv2（若资源允许）。
- Family F4：oracle stub。

在 F1 上训练投影/原型，在 F2–F4 上测试 swap。这是“专家族留出”实验。

---

## 7. H 系列：多模块热插拔

### 7.1 H0：baseline（不 swap）

**目的**：建立参照。

**做法**：A0 + 固定专家，跑 T1–T3；记录所有 property 指标的 baseline 分布。

**指标**：SSR=1（无 swap）、ZCR=1、INT=0、RBE=1、DT=0。

**用途**：后续所有 Δ 的参照。

### 7.2 H1：同模态 swap

**问题**：同模态、不同骨干（CNN ↔ ViT）能否零样本互换？

**做法**：

1. 冻结 shared space（投影头 + 原型库）在 F1 上训练。
2. 换入 F2（同模态），不重训其它模块。
3. 可选：允许 k-shot 校准投影（k=0/5/20）。
4. 跑 T1–T2，记录 property 指标。

**IV**：expert family（F1→F2）、k-shot（0/5/20）。

**DV**：SSR、ZCR、INT、RBE、SML。

**控制**：同一 scenes/seeds；A0 vs A4（naive swap）vs A5（retrain-all）。

**预期**：A0 的 ZCR 显著高于 A4；A5 是上界但成本高。

**杀死**：A0 的 ZCR ≤ 随机水平，或 INT > 5 个百分点。

### 7.3 H2：跨模态 transplant

**问题**：shared space 是否真的“公共”？

**做法**：

1. 用 vision 专家建立空间。
2. 换入 tactile 或 geometry 专家（同维度、同归一化）。
3. 测量跨模态检索/识别（V-T、V-G R@1）。
4. 对照 A1（每模态独立空间 + late fusion）。

**IV**：源模态/目标模态、A0 vs A1。

**DV**：跨模态 ZCR、R@1、INT。

**预期**：A0 的跨模态 ZCR 高于 A1；但可能低于同模态。

**杀死**：A0 跨模态 ZCR ≤ A1，说明共享空间没有公共性。

### 7.4 H3：add/remove module

**问题**：新增/删除一个专家或原型包，是否需要重训其它模块？

**做法**：

1. 在 A0 上新增 audio stub（或第三模态）。
2. 只训练新投影头 + 新原型包，冻结其它模块。
3. 测量旧任务 INT、新任务 ZCR、原型库 hash 是否变化。
4. 删除该模块，测量回滚/清理。

**DV**：INT（旧任务）、ZCR（新任务）、原型库 hash、清理完整性。

**预期**：旧任务 INT ≈ 0；新任务可 zero-shot 或 few-shot 使用。

**杀死**：新增模块导致旧任务下降 > 2 个百分点，或必须重训旧模块。

### 7.5 H4：version skew

**问题**：旧/新专家版本并存时，runtime 是否能正确路由并检测冲突？

**做法**：

1. 同时注册 `vision-v1` 和 `vision-v2`。
2. 按请求/场景路由到不同版本。
3. 注入两版本输出冲突的样本。
4. 测量路由正确性、冲突检出、回滚。

**DV**：RCR、CDR（复用 K 系列）、RBE、SML。

**预期**：路由正确率 ≥ 0.95；冲突被标记而非静默覆盖。

### 7.6 H5：hot swap under load

**问题**：任务运行中换专家，是否停机/崩/丢请求？

**做法**：

1. 在 episode 中途（如 t=50%）发起 swap。
2. 记录 swap 窗口内的请求成功率、延迟、任务连续性。
3. 对照 cold restart（重跑）。
4. 注入 swap 失败（新模块非法），测量回滚。

**DV**：DT、SSR、RBE、任务成功率、p99 延迟。

**预期**：DT ≤ 1%；swap 失败可回滚；任务不崩。

**杀死**：DT > 10% 或 swap 失败导致不可恢复。

### 7.7 H6：invalid module rejection

**问题**：非法模块是否被安全拒绝？

**做法**：注入以下模块：

- 维度错误（127/129）；
- 未归一化（范数 0.1/10）；
- NaN/Inf；
- 常数向量；
- 延迟超限；
- 版本不兼容。

**DV**：SRR（应 = 1.0）、是否产生垃圾输出、是否可回滚。

**预期**：全部拒绝，运行时不受影响。

**杀死**：任一非法模块被接受并影响下游。

### 7.8 H7：freeze-space swap（最关键）

**问题**：shared space 是“公共接口”还是“专家特定”？

**做法**：

1. 在 F1 上训练 shared space（投影 + 原型）。
2. **完全冻结** shared space。
3. 换入 F2/F3/F4，不重训任何东西。
4. 测量 ZCR、R@1、INT。
5. 对照 A4（naive swap，无投影校准）。

**DV**：ZCR、R@1、INT。

**预期**：若空间公共，F2/F3 的 ZCR 应显著高于随机；F4（oracle）应接近上界。

**杀死**：所有留出专家的 ZCR ≤ 随机，说明空间是专家特定的。

### 7.9 H 系列 ablation

| Ablation | 对照 | 验证 |
|---|---|---|
| shared space vs. per-modality | A0 vs A1 | 共享空间是否带来兼容性 |
| L2 normalize vs. 不归一化 | A0 vs A0-nonorm | 归一化是否是契约的一部分 |
| trained projection vs. random | A0 vs A0-randproj | 投影是否必要 |
| prototype vs. linear classifier | A0 vs A0-lin | 原型库对热更新的作用 |
| swap vs. retrain-all | A0 vs A5 | 热插拔的成本优势 |
| swap vs. naive | A0 vs A4 | 投影/契约的必要性 |

### 7.10 H 系列指标与杀死标准

| 指标 | 定义 | 成功阈值 | 杀死 |
|---|---|---:|---|
| SSR | swap 后新模块被正确路由且输出合法 | ≥ 0.90 | < 0.70 |
| ZCR | 零样本换入后的兼容性（R@1/一致性） | 同模态 ≥ 0.90；跨模态 ≥ 0.70 | ≤ 随机 |
| INT | 换 A 前后 B 相关指标变化 | ≤ 0.02 | > 0.05 |
| RBE | swap→rollback 后输出与 swap 前一致 | = 1.00 | < 0.99 |
| DT | swap 期间被拒绝/超时请求比例 | ≤ 1% | > 10% |
| RCR | 请求被路由到正确版本 | ≥ 0.95 | < 0.85 |
| SRR | 非法模块被拒绝比例 | = 1.00 | < 1.00 |

---

## 8. K 系列：知识库热更新

**前提**：L0/L1 使用 oracle observation，把感知踢出回路。L2 再用真实观测研究感知-知识交互。

### 8.1 K1：assert-retract identity

**做法**：

1. 记录初始状态 S0。
2. `assert(F)`。
3. `retract(F)`。
4. 比较状态 S2 与 S0。

**DV**：状态差异（事实、原型、派生事实、版本号、审计）。

**预期**：S2 == S0（除审计日志和版本计数器按设计增长）。

**杀死**：任何非审计/非版本差异。

### 8.2 K2：locality

**做法**：

1. 构造事实图：F 依赖 {A,B}，G 独立。
2. 更新 F。
3. 检查 G 及其派生事实。

**DV**：LOC（无关事实变更数）。

**预期**：LOC = 0。

**杀死**：LOC > 0。

### 8.3 K3：dependency propagation

**做法**：

1. 构造规则链：A → B → C。
2. 改 A 的值。
3. 检查 B、C 是否按规则更新。
4. 回滚 A，检查 B、C 是否恢复。

**DV**：PCR（传播正确率）、回滚后 PCR。

**预期**：PCR = 1.0；回滚后 = 1.0。

**杀死**：PCR < 0.95 或回滚后不一致。

### 8.4 K4：conflict detection/resolution

**做法**：

1. Source S1 断言 `color=red`；S2 断言 `color=blue`。
2. 检查 `conflicts()` 是否检出。
3. 按 priority/timestamp 解析；检查结果符合策略。
4. 注入多源、多值、时序冲突。

**DV**：CDR、RES（解析正确率）、冲突集大小。

**预期**：CDR ≥ 0.95；RES = 1.0（按策略）。

**杀死**：漏检或解析与策略不符。

### 8.5 K5：rollback exactness

**做法**：

1. 从 v0 开始，执行 N 次 update/assert/retract。
2. 记录每个版本的快照 hash。
3. 回滚到 v_k（随机 k）。
4. 比较业务状态 hash（事实/原型/规则，不含审计日志和版本计数器）与快照。

**DV**：RBE-KB（逐位/结构一致）。

**预期**：= 1.00。

**杀死**：< 0.99。

### 8.6 K6：hot update mid-task

**做法**：

1. 任务执行到中途（如 planning 后、act 前）。
2. 更新关键事实（如“detergent available” false→true）。
3. 观察是否重规划、是否崩溃、是否违反安全约束。

**DV**：任务成功率、重规划正确率、安全违规数、UL。

**预期**：正确重规划；无安全违规。

**杀死**：崩溃、静默使用旧事实、安全违规。

### 8.7 K7：adversarial/poisoned update

**做法**：

1. 注入与硬黑名单冲突的事实。
2. 注入来源不可信/过期的事实。
3. 注入会引发矛盾规则链的事实。

**DV**：拒绝率、审计完整率、是否污染下游。

**预期**：全部被护栏拒绝或标记 review；下游不受污染。

**杀死**：毒化事实进入 active 状态并影响输出。

### 8.8 K8：multi-source priority/timestamp

**做法**：

1. 同一 (entity, attribute) 多个 source，不同 priority/timestamp。
2. 检查 resolve 结果。
3. 改变 priority/timestamp，检查单调性。

**DV**：解析正确率、时序一致性。

**预期**：符合 `(priority, timestamp)` 策略。

### 8.9 K 系列 ablation

| Ablation | 对照 | 验证 |
|---|---|---|
| versioned TMS vs. flat dict | A0 vs A2 | 版本/回滚/传播 |
| justification graph on/off | A0 vs A0-nojust | 依赖传播 |
| single-source vs. multi-source | A0 vs A0-single | 冲突检测 |
| audit on/off | A0 vs A0-noaudit | 可审计性 |
| atomic pack vs. in-place | A0 vs A0-inplace | 原子激活/回滚 |

### 8.10 K 系列指标与杀死标准

| 指标 | 定义 | 成功阈值 | 杀死 |
|---|---|---:|---|
| USR | 更新后新事实 active 且被下游使用 | ≥ 0.99 | < 0.95 |
| LOC | 无关事实变更数 | = 0 | > 0 |
| FOR | 旧事实更新后仍正确的比例 | ≥ 0.99 | < 0.95 |
| PCR | 派生事实与期望一致率 | = 1.00 | < 0.95 |
| CDR | 注入矛盾后的检出率 | ≥ 0.95 | < 0.80 |
| RES | 按策略解析正确率 | = 1.00 | < 0.95 |
| RBE-KB | 回滚后与快照一致 | = 1.00 | < 0.99 |
| AUD | 有 audit entry 的变更比例 | = 1.00 | < 1.00 |
| UL | 从提交到传播完成的时间 | ≤ 配置上界 | > 2× 上界 |

---

## 9. C 系列：推理复杂度可控

**前提**：L0/L1 使用 oracle observation，专家延迟可控。

### 9.1 C1：budget sweep

**做法**：

1. 设四类预算：实体数上限、关系数上限、A* 展开节点上限、检索级数。
2. 每类取 5 档（如 10/25/50/100/200）。
3. 每档跑 T1–T3，记录实际计算量、成功率、延迟。

**DV**：BAR、AQV、p95/p99、任务成功率。

**预期**：BAR = 1.0；质量随预算单调不降（平均）。

**杀死**：任一档 BAR < 1.0，或质量-预算曲线显著非单调。

### 9.2 C2：scene-size sweep

**做法**：

1. 实体数 10 → 50 → 200 → 1000。
2. 关系数按密度 0.1/0.5/1.0 变化。
3. 记录 active set 大小、关系查询数、规划节点数。

**DV**：延迟、内存、BAR、成功率。

**预期**：active set 控制下，延迟增长亚线性；BAR = 1.0。

**杀死**：延迟超上界或内存爆炸。

### 9.3 C3：anytime interruption

**做法**：

1. 在规划/检索的不同时间点中断（10%/25%/50%/75%/100%）。
2. 检查输出是否合法（不是半成品）。
3. 画 quality-vs-time 曲线。

**DV**：AQV、合法性、MON。

**预期**：任意中断都有合法输出；质量随时间不降。

**杀死**：中断产生非法输出，或质量显著下降。

### 9.4 C4：adversarial scene

**做法**：

1. 构造大量实体/关系/规划陷阱（如目标不可达、循环前置条件）。
2. 跑 A0 vs A3（unbounded）。
3. 记录最大延迟、最大节点数、是否超界。

**DV**：WCB、p99、BAR。

**预期**：A0 遵守上界；A3 超界（负控制）。

**杀死**：A0 超界。

### 9.5 C5：degradation FSM

**做法**：

1. 注入资源故障：超时、内存压力、专家缺失、规划失败。
2. 检查 FSM 状态转移：normal → warning → degraded → emergency → normal。
3. 记录每次转移的触发条件、输出、恢复。

**DV**：DCR、恢复时间、安全违规数。

**预期**：DCR ≥ 0.95；可从 emergency 恢复。

**杀死**：卡死、静默失败、安全违规。

### 9.6 C6：cost attribution

**做法**：

1. 给每个模块打点：专家编码、投影、检索、场景图、规划、输出。
2. 汇总每模块耗时/调用次数。
3. 与总延迟对账。

**DV**：CAT（归因误差）、模块占比。

**预期**：归因误差 ≤ 5%；能解释主要成本。

**杀死**：无法归因 > 20% 延迟。

### 9.7 C7：budget adaptation under expert latency

**做法**：

1. 专家延迟从 10ms → 100ms → 1s 变化。
2. 预算控制器是否降低检索级数/规划节点以保持总延迟上界。
3. 对照 A0 vs A3（不 adapt）。

**DV**：BAR、p99、任务成功率。

**预期**：A0 自适应；A3 超界。

### 9.8 C 系列 ablation

| Ablation | 对照 | 验证 |
|---|---|---|
| budgeted vs. unbounded | A0 vs A3 | 上界是否必要 |
| budgeted vs. fixed | A0 vs A0-fixed | 自适应是否优于固定 |
| budgeted vs. random | A0 vs A0-random | 控制器是否有效 |
| active set on/off | A0 vs A6 | 活跃集是否控制规模 |
| 4-level retrieval vs. flat | A0 vs A7 | 检索级数是否控制成本 |
| degradation on/off | A0 vs A8 | 降级是否必要 |

### 9.9 C 系列指标与杀死标准

| 指标 | 定义 | 成功阈值 | 杀死 |
|---|---|---:|---|
| BAR | 实际计算量 ≤ 配置上界 | = 1.00 | < 1.00 |
| AQV | 中断时间 t 下的解质量 | 单调不降 | 显著下降 |
| MON | 预算-质量曲线非降比例 | ≥ 0.95 | < 0.80 |
| p99 | 99 分位延迟 | ≤ 上界 | > 2× 上界 |
| DCR | FSM 转移与策略一致率 | ≥ 0.95 | < 0.80 |
| CAT | 成本归因误差 | ≤ 5% | > 20% |
| WCB | 对抗场景最大延迟/节点数 | ≤ 上界 | > 上界 |

---

## 10. 交互实验（跨特性）

三条特性不是独立的。必须测交互，否则审稿人会问“你们只测了单点”。

### 10.1 H×K：swap 后 KB 是否一致

**做法**：

1. 在 A0 上，先更新 KB（新增事实/原型）。
2. 再 swap 专家。
3. 检查 KB 是否仍一致、派生事实是否失效/重建。

**DV**：K 系列指标 + H 系列指标。

**预期**：swap 不破坏 KB；必要时触发重算。

**杀死**：swap 后出现矛盾事实或静默使用旧原型。

### 10.2 H×C：swap 延迟是否计入预算

**做法**：

1. 在紧预算下 swap 专家。
2. 测量总延迟是否超界，控制器是否降级。

**DV**：BAR、p99、DT。

**预期**：swap 延迟被计入或显式豁免；不超界。

**杀死**：swap 导致预算静默超界。

### 10.3 K×C：KB 增长是否影响规划成本

**做法**：

1. 事实数从 10 → 100 → 1000。
2. 测量 A* 节点数、延迟、BAR。
3. 检查控制器是否随事实增长调整检索/规划。

**DV**：BAR、p99、任务成功率。

**预期**：预算控制器吸收增长；BAR = 1.0。

**杀死**：事实增长导致超界。

### 10.4 H×K×C：三者同时

**做法**：

1. 紧预算下，先 swap，再 update，再执行任务。
2. 注入一个故障。
3. 检查系统是否仍满足三条契约。

**DV**：综合 property scorecard。

**预期**：三条契约同时成立；降级正确。

**杀死**：任一契约失败且不可恢复。

---

## 11. 指标定义与计算

### 11.1 H 系列

```
SSR = (# successful swaps) / (# swap attempts)
ZCR = (# correct retrievals after zero-shot swap) / (# queries)
INT = |metric_old(B | swap A) - metric_old(B | no swap)|
RBE = 1[state_after_rollback == state_before_swap]   # 逐位/结构
DT  = (# requests rejected or timed out during swap) / (# requests)
RCR = (# requests routed to correct version) / (# requests)
SRR = (# invalid modules rejected) / (# invalid modules injected)
SML = mean(t_swap_end - t_swap_start)
```

### 11.2 K 系列

```
USR = (# updates active and used downstream) / (# updates)
LOC = # facts/prototypes changed that do not depend on the updated fact
FOR = (# old facts still correct after update) / (# old facts)
PCR = (# derived facts matching expected) / (# derived facts)
CDR = (# injected conflicts detected) / (# injected conflicts)
RES = (# conflicts resolved per policy) / (# conflicts)
RBE-KB = 1[business_state_after_rollback == snapshot]   # 不含 audit/version counter
AUD = (# changes with audit entry) / (# changes)
UL  = t_propagation_done - t_update_submitted
```

### 11.3 C 系列

```
BAR = 1[actual_compute <= configured_bound]   # per query; aggregate = mean
AQV(t) = quality of output when interrupted at t
MON = (# budget pairs where quality does not decrease) / (# pairs)
p95/p99 = percentile of latency
DCR = (# FSM transitions matching policy) / (# transitions)
CAT = |sum(module_cost) - total_cost| / total_cost
WCB = max over adversarial scenes of (latency or nodes) - bound
```

### 11.4 聚合与置信区间

- 每个条件报告：mean、median、p95、p99、worst-case。
- 置信区间：bootstrap 10,000 次，95% CI。
- 配对差：先算 `Δ_i`，再对 `Δ` 做 bootstrap。
- 缺失/崩溃：单独报告，不静默丢弃；崩溃率本身是 property 指标。

### 11.5 专家质量指标

```
accuracy = top-1 on held-out set
ECE = expected calibration error (15 bins)
alignment_R@1 = cross-modal retrieval R@1
latency_p99 = 99th percentile encode time
robustness = accuracy under noise/OOD
coverage = # classes/modalities supported
```

---

## 12. 统计与因果归因

### 12.1 混合效应模型

对每个 property 指标：

```
y_ijklm = μ
        + α_i        (expert quality: Q0..Q5)
        + β_j        (architecture: A0..A8)
        + γ_k        (difficulty: T1..T3)
        + (αβ)_ij + (αγ)_ik + (βγ)_jk + (αβγ)_ijk
        + u_l        (scene random effect)
        + v_m        (seed random effect)
        + ε_ijklm
```

实现：R `lme4::lmer` 或 Python `statsmodels.formula.api.mixedlm`。

### 12.2 方差分解

报告：

- **marginal R²**：固定效应解释的方差；
- **conditional R²**：固定 + 随机；
- **方差占比**：
  ```
  share_expert   = Var(α) / (Var(α) + Var(β) + Var(αβ) + ...)
  share_arch     = Var(β) / (...)
  share_interact = Var(αβ) / (...)
  ```
- **方差比**：
  ```
  VR = Var(β) / Var(α)
  ```
  若 VR ≥ 3，支持“架构效应主导”。

### 12.3 等价性检验（TOST）

要证明 property 指标“对专家质量不敏感”，不能只用“无显著差异”。用 TOST：

- 等价界 `δ`：取 baseline 的 5% 或绝对 0.05。
- 若 90% CI 落在 `[-δ, +δ]`，拒绝“存在实质差异”，支持等价。
- 对每个 property 指标、每个架构变体分别做。

### 12.4 效应量

- 连续指标：Cohen's d、Cliff's delta。
- 比例指标：risk difference、odds ratio。
- 报告 95% bootstrap CI。

### 12.5 多重比较

- 主指标（每特性 2–3 个）：Holm–Bonferroni。
- 探索性指标：Benjamini–Hochberg FDR。
- 预注册的主指标与探索性指标分开报告。

### 12.6 功效分析

- pilot：每格 5 episodes，估计方差。
- 目标：d = 0.5，power = 0.80，α = 0.05。
- 若不足，优先增加 seeds（随机效应），再增加 episodes。
- 记录实际 power。

### 12.7 盲法与报告

- 指标自动计算，不允许手工挑选 episode。
- 报告所有预注册条件，包括不显著的。
- 失败 run 单独列表，不删除。
- 代码/配置/seed 全部归档。

---

## 13. 负控制与“测试的测试”

负控制是证明测试有效的关键。审稿人会问：“你怎么知道你的测试不是随便就能过？”

| ID | 负控制 | 预期结果 | 验证 |
|---|---|---|---|
| N1 | A2（flat KB）跑 K2/K5 | 应失败（LOC > 0 / RBE < 1） | 测试敏感性 |
| N2 | A3（unbounded）跑 C1/C4 | 应失败（BAR < 1） | 测试敏感性 |
| N3 | A4（naive swap）跑 H1/H7 | 应失败（ZCR 低） | 测试敏感性 |
| N4 | Q4（random）跑 capability | 应失败（accuracy ≈ chance） | 测试特异性 |
| N5 | Q4（random）跑 property | 应通过（架构不崩） | 性质与能力分离 |
| N6 | no-op update | 状态应完全不变 | 恒等性 |
| N7 | shuffled embeddings | capability 崩，property 不崩 | 分离 |
| N8 | 人为违反预算的控制器 | 应被 BAR 捕获 | 上界检测 |

**判据**：所有负控制必须按预期失败/通过。任一不符，先修测试，再跑正式实验。

---

## 14. 执行计划

### 14.1 阶段

| 阶段 | 周 | 内容 | 交付 |
|---|---|---|---|
| P0 | 1 | 契约、指标、stub、预注册 | `eval/property_preregistration.md`、`pss/experts/stubs.py`、`pss/metrics/property.py` |
| P1 | 2–3 | L0 sandbox | `tests/properties/` 全绿、L0 scorecard |
| P2 | 4–5 | L1 H/K/C | 三套 scorecard、初步图 |
| P3 | 6 | 交互实验 | 交互 scorecard |
| P4 | 7 | L2 真实专家 | 外部效度报告 |
| P5 | 8 | 分析、图、报告 | `reports/property_validation/` |
| P6 | 9+ | 论文草稿 | related work + experiments 节 |

### 14.2 里程碑

- **M0**：契约与指标冻结（P0 结束）。
- **M1**：L0 三条 property test suite 全绿（P1 结束）。
- **M2**：L1 主效应显著且方向正确（P2 结束）。
- **M3**：负控制全部按预期（P3 结束）。
- **M4**：L2 不推翻主结论（P4 结束）。
- **M5**：方差分解 + 等价性检验完成（P5 结束）。

### 14.3 计算预算

| 层 | 估计 | 备注 |
|---|---|---|
| L0 | < 1 GPU-hour | CPU 可跑 |
| L1 | 200–400 GPU-hours | Isaac 渲染 + 专家推理 |
| L2 | 20–50 GPU-hours | 真实专家 |
| 合计 | 250–450 GPU-hours | 可用分数因子压缩到 ~100 |

**最小可行版**：L0 + H1/H7 + K1/K2/K5 + C1/C3，约 50 GPU-hours，足以支撑三条主 claim 的 pilot。

### 14.4 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| Stub 太理想，L1 不真实 | 外部效度差 | 用真实专家误差模型校准 stub；L2 验证 |
| Shared space 训练泄漏 | H7 结论虚假 | 严格 freeze；held-out expert family |
| Isaac 不稳定 | 数据缺失 | 固定 spec hash；bundle 校验；失败重跑 |
| 专家延迟混入预算 | C 系列混淆 | 显式记录专家延迟；单独分析 |
| 样本不足 | 功效低 | pilot + power analysis；增加 seeds |
| 任务难度混杂 | 交互误判 | 分层区组；报告每层 |
| 范围蔓延 | 延期 | 以最小可行版为底线；其余为扩展 |

---

## 15. 结果报告模板

### 15.1 表 1：Property × Expert quality

| 指标 | Q0 | Q1 | Q2 | Q3 | Q4 | 斜率 | 95% CI | TOST |
|---|---:|---:|---:|---:|---:|---:|---|---|
| SSR | | | | | | | | |
| ZCR | | | | | | | | |
| INT | | | | | | | | |
| RBE | | | | | | | | |

**读法**：若斜率 ≈ 0 且 TOST 通过，支持专家不变性。

### 15.2 图 1：不变性图

```
metric
  │  ●──●──●──●   A0 (architecture ON)
  │  ○──○──○──○   A_control (architecture OFF)
  └────────────── Q0 Q1 Q2 Q3 Q4
```

### 15.3 图 2：KB 状态与回滚

- x 轴：操作序列；
- y 轴：状态 hash / 事实数；
- 标注 assert/retract/rollback；
- 证明回滚后 hash 与快照一致。

### 15.4 图 3：预算-质量曲线

- x 轴：预算（节点数/实体数/检索级数）；
- y 轴：成功率/质量；
- 多条线：A0 vs A3（unbounded）vs random；
- 标注 BAR。

### 15.5 表 2：Ablation matrix

| 变体 | SSR | ZCR | INT | RBE | USR | LOC | PCR | BAR | MON |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | | | | | | | | | |
| A1 | | | | | | | | | |
| ... | | | | | | | | | |

### 15.6 表 3：负控制

| ID | 预期 | 实际 | 通过? |
|---|---|---|---|
| N1 | fail | | |
| ... | | | |

### 15.7 表 4：方差分解

| 指标 | Var(expert) | Var(arch) | Var(interaction) | VR | 结论 |
|---|---:|---:|---:|---:|---|
| SSR | | | | | |
| ... | | | | | |

---

## 16. 威胁与局限

1. **Oracle 天花板**：Q0 只用于隔离架构，不能作为能力结论。
2. **Stub 真实性**：合成误差若与真实专家差异大，L1 的外部效度受限；用 L2 校准。
3. **Shared space 训练泄漏**：投影/原型若在目标专家上训练，H7 会高估“公共性”；必须 freeze + held-out family。
4. **专家延迟混淆**：C 系列的预算若包含专家延迟，结论会被专家速度污染；显式分离。
5. **任务难度混杂**：预算/更新在简单场景容易通过；必须分层报告。
6. **统计功效**：小样本 + 多指标容易假阳性；预注册 + 校正 + 效应量。
7. **外部效度**：L0/L1 不能直接推到真机；L2 只是小规模验证。
8. **“Property” 的构造效度**：指标是否真的代表“热插拔/热更新/可控”，需要与文献和工程直觉对齐；建议请外部 reviewer 审指标。
9. **版本/回滚的语义**：PSS 的 rollback 是状态快照，不是事务隔离；不要宣称 ACID。
10. **多源冲突**：当前 `MultiSourceFactStore` 的 resolve 是 priority/timestamp 仲裁，不是完整 TMS；K3/K4 的结论要限定在这个语义下。

---

## 17. 与论文写作的映射

> 基于当前 P0–P5 数据的完整论文映射（claim–evidence、图表、章节、Reviewer Q&A）见 `notes/PAPER_WRITING_MAP.md`。

| 论文 claim | 支撑实验 | 图表 |
|---|---|---|
| CL1：contract-level hot-swap（不是零样本语义等价） | H1、H2、H3、H4、H5、H7、twin | 表 1、图 1、表 5 |
| CL2：版本化 KB 支持原子热更新与回滚 | K1–K8、TMS Hypothesis、flat KB 负控制 | 图 2、表 2、表 3 |
| CL3：推理成本有硬上界且可优雅降级 | C1–C7、C ablation、L2 bundle、S4-PSS | 图 3、表 2、表 6 |
| CL4：property 指标对专家质量不敏感，capability 指标不是 | 方差分解、TOST、Q0–Q4 曲线 | 表 1、表 4、表 4b |
| CL5：property test suite 可作为 benchmark | 负控制 N1–N8 | 表 3 |
| CL6：共享空间是可校准接口，不是零样本坐标系 | 跨族 swap、k-shot 校准 | 表 2、校准表 |

**不要写的 claim**：

- “PSS 比 LLM/VLM 更准” —— 本计划不测 capability SOTA。
- “PSS 在真实世界可靠” —— L2 只是小规模，且 S4-PSS 用 oracle GT boxes。
- “非 LLM 一定更好” —— 这是设计选择，不是本实验结论。
- “零样本公共空间” —— 跨族零样本失败；ZCR 是 capability，不是 property。
- “端到端真实感知外部效度” —— L2 只到 interface-level；H/K 未在 L2 执行。
- “架构主导能力” —— ZCR 的 ANOVA 架构份额被 A4/A5 控制放大；VR≈1.8。

---

## 18. 附录

### A. Expert contract（完整）

```python
from typing import Protocol
import numpy as np

class ExpertHealth(Protocol):
    latency_p50: float
    latency_p99: float
    calibration_ece: float
    ood_score: float
    failure_flags: list[str]

class ExpertContract(Protocol):
    modality: str
    version: str
    dim: int

    def encode(self, obs) -> np.ndarray:
        """(B, dim), float32, L2-normalized, no NaN/Inf."""

    def health(self) -> ExpertHealth:
        ...
```

**校验函数**：

```python
def validate_expert(e, public_dim=128, tol=1e-6):
    assert e.dim == public_dim
    x = e.encode(dummy_obs)
    assert x.ndim == 2 and x.shape[1] == public_dim
    assert np.isfinite(x).all()
    norms = np.linalg.norm(x, axis=1)
    assert np.allclose(norms, 1.0, atol=tol)
```

### B. Structured observation schema

见 §3.6。补充字段：

- `provenance`: `gt | detector | replay`
- `confidence`: per-entity/relation
- `frame_id`, `episode_id`, `session_id`
- `spec_hash`, `bundle_hash`

### C. Expert stub 规格

| Stub | 构造 | 质量 | 延迟 | 用途 |
|---|---|---|---|---|
| Oracle | one-hot / 原型 | 1.0 | ~0 | 天花板 |
| NoisyOracle | confusion matrix | 可调 | ~0 | Q1–Q3 |
| Random | 随机单位向量 | chance | ~0 | Q4 |
| Constant | 固定向量 | 0 | ~0 | 安全 |
| Adversarial | 固定错误映射 | 0 | ~0 | 鲁棒 |
| OOD | 分布外嵌入 | 可调 | ~0 | 开放集 |
| Delayed | sleep | 正常 | 可调 | 预算 |
| Dropout | 空/零 | 缺失 | ~0 | 缺失模态 |

### D. 指标公式

见 §11。

### E. 事实更新序列格式

```json
{
  "sequence_id": "K5-001",
  "initial_version": 0,
  "operations": [
    {"op": "assert", "entity": "cup1", "attribute": "clean", "value": "no", "source": "gt"},
    {"op": "update", "entity": "cup1", "attribute": "clean", "value": "yes", "reason": "task"},
    {"op": "retract", "entity": "cup1", "attribute": "clean", "source": "gt"},
    {"op": "rollback", "version": 0}
  ],
  "expected_final_state_hash": "...",
  "expected_intermediate_hashes": ["...", "..."]
}
```

### F. 场景矩阵

| 场景 | spec | 难度 | 实体数 | 关系数 | 用途 |
|---|---|---|---:|---:|---|
| S0a | synthetic | T1 | 1 | 0 | 最小 |
| S0b | synthetic | T2 | 10 | 5 | 场景图 |
| S0c | synthetic | T2 | 10 | 20 | 事实/冲突 |
| S0d | synthetic | T3 | 200 | 1000 | 计算压力 |
| S0e | synthetic | T2 | 动态 | 动态 | 生命周期 |
| S1 | `s1_occlusion.yaml` | T2 | 8 | 10 | 遮挡/追踪 |
| S2 | `s2_*` | T2 | 10 | 15 | 关系 |
| S3 | `s3_*` | T3 | 20 | 50 | 复杂 |
| S4 | `s4_pss_*.yaml` | T2 | 2–4 | 5 | 抓放 |
| S6 | `s6_*` | T3 | 多 session | 多 | 连续 |

### G. 统计公式

混合效应模型见 §12.1。

方差占比：

```
share_x = Var(β_x) / Σ Var(β)
```

TOST：

```
H0: |μ_treated - μ_control| ≥ δ
Ha: |μ_treated - μ_control| < δ
若 90% CI ⊂ (-δ, δ)，拒绝 H0
```

### H. 执行检查清单

**P0**

- [x] 三条 property 的操作化定义冻结
- [x] Expert contract 实现并测试
- [x] 8 类 stub 实现
- [x] 指标库 `pss/metrics/property.py` 实现
- [x] 预注册写入 `eval/property_preregistration.md`
- [x] 负控制设计确认

**P1**

- [x] L0 sandbox 可跑
- [x] H/K/C property tests 全绿
- [x] Hypothesis 随机序列测试通过（`tests/property/test_hypothesis_sequences.py`，4 项）
- [x] 崩溃率 = 0（5 seeds 无异常；未单独打点）

**P2**

- [x] L1 专家池冻结（`expert_manifest.json`，包含 sha256）
- [x] 质量注入 manipulation check 通过
- [x] H1/H7/K1/K2/K5/C1/C3 跑完
- [x] twin 配对正确（`twins.json`，5 seeds，15 对）

**P3**

- [x] 交互实验跑完
- [x] 负控制 N1–N8 全部按预期
- [x] 方差分解完成（`anova.json`，two-way ANOVA + 方差份额）

**P4**

- [x] L2 真实专家跑完（S4-PSS 5/5 fresh Isaac；**H/K 未在 L2 执行**）
- [x] 外部效度报告（partial：oracle GT boxes；H/K 仅 L0/L1）

**P5**

- [x] 图表生成
- [x] 报告归档 `reports/property_validation/`
- [x] 代码/种子/配置归档（`run_manifest.json`：git、env、sha256、bundle spec hash）

### I. 术语表

| 术语 | 中文含义 | 英文/说明 |
|---|---|---|
| capability | 任务能力 | 依赖专家质量 |
| property | 架构契约 | 应独立于专家质量 |
| expert contract | 专家接口契约 | 专家必须满足的接口/输出/延迟约束 |
| stub | 合成专家 | 用于控制专家质量 |
| twin | 孪生配对 | 同场景/同 seed、只变架构的配对 run |
| TOST | 等价性检验 | two one-sided tests |
| BAR | 预算遵守率 | budget adherence rate：实际计算量不超过配置上界的比例 |
| ZCR | 零样本兼容率 | zero-shot compatibility rate：换入新专家后，不做校准、不加标签，直接可用的比例 |
| RBE | 回滚精确性 | rollback exactness：回滚后状态与回滚前/快照完全一致 |
| PCR | 传播正确率 | propagation correctness rate：派生事实/结论按依赖关系正确更新的比例 |
| AQV | 任意时刻质量向量 | anytime quality vector：不同中断时间点输出的质量分布 |
| DCR | 降级正确率 | degradation correctness rate：降级 FSM 状态转移与策略一致的比例 |
| VR | 方差比（架构/专家） | variance ratio：架构效应方差 ÷ 专家效应方差 |

---

## 19. 一页纸总结

- **验证对象**：三条架构契约，不是端到端 accuracy。
- **核心方法**：expert stub + 结构化观测 + 配对反事实 + 因子设计 + 方差分解。
- **主证据**：L0 property tests + L1 controlled integration。
- **外部效度**：L2 真实专家，小规模。
- **成败判据**：property 指标对专家质量不敏感（TOST + VR ≥ 3），且负控制按预期失败。
- **最小可行版**：L0 全套 + H1/H7 + K1/K2/K5 + C1/C3，约 50 GPU-hours。
- **最终交付**：property scorecard + 不变性图 + 预算-质量曲线 + 负控制表 + 预注册报告。

---

*本计划为 v1.0 草案。执行前需完成 `eval/property_preregistration.md` 的冻结；任何指标/阈值变更必须记录在案，并标注为探索性分析。*
