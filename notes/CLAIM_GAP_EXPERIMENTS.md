# Claim-Gap Experiments (v1)

> 目的：按 `notes/PAPER_WRITING_MAP.md` 的 claim 缺口重新设计可证伪实验。
> 原则：能复用现有组件就不造新数据；每条实验先写通过/失败判据，跑完如实报。
> 输出：`reports/claim_gaps/`（JSON + REPORT.md）。

---

## 0. 缺口 → 实验映射

| 缺口 | claim | 实验 | 主指标 | 通过判据 |
|---|---|---|---|---|
| C1-a：状态“可审计”没有量化指标 | 实体状态显式、可追溯 | E1 Entity audit | provenance completeness、retraction correctness、rollback exactness、tamper detection | 全部 = 1.00 |
| C1-b：绑定被 tactile 混杂 | 状态层提供持久身份 | E2 Tactile binding ablation | ID retention（tactile present / zeroed / vision zeroed） | 给出 tactile 贡献量；zeroed 显著下降 |
| C2：hybrid 关系只有设计结论 | 几何骨架 + 学习校准 | E3 Hybrid relation under noise | exact-4 relation accuracy（geometry / learned / hybrid） | hybrid ≥ max(geometry, learned) 至少一个噪声档 |
| C3：长历史 fact active-set 是负结果 | active-set 记忆 | E4 Query-conditioned fact set | fact accuracy @ k；vs hot / recency / query-independent | query-conditioned > hot heuristic |
| C5：property suite 未跑非 PSS baseline | 评测协议可迁移 | E5（deferred） | — | 需要单体基线，另立 |

---

## E1：Entity-state audit

**设计**：`EntityLedger` 记录每次 create/witness/revise/retract/rebind/evict/reload；每条日志带 `prev_hash` 形成审计链。

**指标**：

- `provenance_completeness`：所有日志条目都有 entity/op/source/frame。
- `retraction_correctness`：retract 后目标属性消失，其他属性逐位不变。
- `id_rollback_exactness`：snapshot → 操作 → rollback，状态 hash 一致。
- `rebind_identity`：rebind 后 entity id 与 raw 特征不变。
- `tamper_detection`：篡改任一日志条目后链校验失败。
- `audit_chain_valid`：未篡改时链校验通过。

**通过**：以上全部 1.00，5 seeds。

**注意**：这是 L0 结构化实验，验证的是状态层机制，不是感知。

---

## E2：Tactile binding ablation

**设计**：在 `data/tracking_bench_long`（80 帧，70 帧全遮挡）上，用同一个 graph matcher：

1. `graph+tactile`：正常观测。
2. `graph+tactile_zero`：把 `raw["tactile"]` 置零（模拟触觉缺失）。
3. `graph+vision_zero`：把 `raw["vision"]` 置零（对照）。
4. `simple_tracker`：只用 vision appearance 的 Tracker。
5. `text_state_proxy`：无持久状态的基线。

**指标**：ID retention、switches、reappearance；按 mixed/samecat/near_identical 分层。

**通过**：不是“必须不降”，而是**把 tactile 的贡献量测出来**。如果 tactile_zero 明显下降，则 C1 必须写成“机制在状态层、信号主要来自 tactile”。

---

## E3：Hybrid relation under noise

**设计**：P5b 区域网格。训练集区域 A/B/C；held-out 区域 D/G。对感知坐标加噪声，比较：

- `geometry`：精确几何谓词（dead zone ε=0.1）。
- `learned`：`SpatialRelationHead`（训练 A/B/C，测试 D/G）。
- `hybrid`：每轴独立——|dx| ≥ τ 用几何，否则用 learned；|dy| 同理。

**指标**：exact-4 关系匹配率（相对真实区域标签），噪声 0 / 0.03 / 0.08 / 0.15；3 seeds。

**通过**：至少一个噪声档 hybrid ≥ max(geometry, learned)。若没有，则 C2 的 hybrid 说法必须降级为“设计建议，未验证”。

---

## E4：Query-conditioned fact active set

**设计**：复现现有 fact active-set 负结果，并加 query 条件：

- 现有：fact 打分是 query-independent 的，选 top-k 后对所有 query 共用。
- 新：打分时加入 `query_match`（该 fact 的 key 是否等于当前 query）、query 频次、query 近因；每个 query 独立选 top-k。

**指标**：fact accuracy @ k=16，T=1000；对比 full / recent / hot / random / learned（旧）/ learned-query。

**通过**：learned-query ≥ hot heuristic（旧结果是 learned 0.372 < hot 0.480）。若仍失败，则坐实“长历史 fact 选择是开放问题”。

---

## 运行

```bash
python -m pss.eval.scripts.run_gap_entity_audit --seeds 5
python -m pss.eval.scripts.run_gap_tactile_binding --sequences 12 --seeds 2
python -m pss.eval.scripts.run_gap_hybrid_relations --seeds 3
python -m pss.eval.scripts.run_gap_fact_active --seeds 3
```

结果写入 `reports/claim_gaps/<experiment>/`，汇总见 `reports/claim_gaps/REPORT.md`。

---

## 结果（2026-11-25）

### E1 Entity audit — PASS

5 seeds，全部属性 1.00；三个负控制（no audit / no chain / shallow rollback）目标指标均为 0.00。

结论：实体状态的“可审计”现在有量化指标，不再是空口。测试：`tests/property/test_entity_audit.py`。

### E2 Tactile binding ablation — MEASURED

60 seq × 3 seeds，70 帧全遮挡：

| variant | ID retention |
|---|---:|
| graph+tactile | 0.989 |
| graph+tactile_zero | 0.872 |
| graph+vision_zero | 0.961 |
| simple_tracker | 0.917 |
| text_state_proxy | 0.500 |

贡献分解：tactile **+0.117**、graph/state **+0.072**、vision **+0.028**、state over no-memory **+0.489**。

结论：状态层提供持久化机制，但身份信号主要来自 tactile。C1 不能再写成“状态层单独保持身份”；应写“状态层 + tactile 信号，graph 提供额外 +0.072”。

### E3 Hybrid relation under noise — CONDITIONALLY SUPPORTED

held-out 区域 D/G，3 seeds：

| noise | geo ε=.1 | geo ε=.25 | learned | hybrid | hybrid_diag |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.000 | 1.000 | 0.595 | 0.595 | 1.000 |
| 0.03 | 0.987 | 1.000 | 0.580 | 0.580 | 1.000 |
| 0.08 | 0.730 | 0.980 | 0.572 | 0.567 | 0.985 |
| 0.15 | 0.537 | 0.828 | 0.532 | 0.503 | 0.857 |

结论：learned 头在训练分布内 6/6 精确，但在未见过的对齐轴（dx=0 或 dy=0）上外推错误。正确 hybrid 是**让 learned 只在训练覆盖的“双轴清晰”区工作，对齐/死区交给几何**；把 learned 用于对齐轴的朴素 hybrid 反而最差。hybrid_diag 在 0.08/0.15 略优于几何 dead-zone。

### E4 Query-conditioned fact active set — FIXED (but not by learning)

| method | k=4 | k=8 | k=16 |
|---|---:|---:|---:|
| hot | 0.237 | 0.421 | 0.473 |
| learned_old | 0.241 | 0.379 | 0.604 |
| query_recency | 1.000 | 1.000 | 1.000 |
| learned_query | 1.000 | 1.000 | 1.000 |

结论：旧负结果（learned 0.372 < hot 0.480）是 **query-independent 打分**造成的，不是学习能力不足。加入 query 条件后，简单 query-recency 规则和 learned policy 都到 1.000。因此 claim 应写成“长历史 fact 选择必须有 query-conditioned 接口”，而不是“学习打分更好”。

### 仍未闭合（更新到 2026-11-26）

- **真机**：环境无机器人。
- **L2 运行中 H/K / 冲突弃权**：S4-PSS 是 oracle pose + 无遮挡；E12 用 bundle replay 补了 L2 记忆消融，但 swap/KB 仍需新 harness。
- **E2/E7 外部效度**：仍是 MuJoCo 渲染；E7 是检测噪声注入，不是 GroundingDINO。
- **E3 网格性**：P5b 是 5-region 网格；非网格布局未测。
- **E4 多 query**：只测单 query top-k；多 query + coverage loss 未测。
- **E11 专家族**：只接了 3 个同宽度族；DINOv2/SqueezeNet/WT-MM 未接。
- **C5 竞争性 baseline**：E5 是 contract-free，不是能力基线。

---

## §10 追加实验（2026-11-26）

> 汇总：`reports/claim_gaps/FOLLOWUPS.md`。

| 实验 | §10 | 关键结果 | 结论 |
|---|---|---|---|
| E5 non-PSS baseline | 0 | PSS 契约指标 1.0；flat 在 SRR/RBE/RCR/PCR/AUD/RBE-KB/BAR/MON 全 0；INT 0.667 | suite 不是定制试卷 |
| E6 retrain-all proxy | 1 | frozen+alt prototypes 0.964 (3.2 ms)；1-shot ridge 0.922 (108 ms)；retrained projection 0.979 (13.5 s) | prototype rebuild 是主成本；retrain 只 +1.5 点；ridge 不必要 |
| E7 detection noise | 3 | 0.989 → 0.975 (10% miss) → 0.871 (20%) → 0.697 (30%) | 状态层随检测丢失平滑退化 |
| E8 scale | 9 | dim 128/256/512 契约不变；1000 facts update 52.9 ms、rollback 1；500/2000 bounded | 契约不变，延迟随数据增长 |
| E9 more seeds | 11 | L0 10 seeds 契约全 1.0；L1 5 seeds SSR/RBE/INT/RCR=1.0，ZCR 0.671±0.014 | 不是 seed 偶然 |
| E10 TMS production | 10 | K3/K4/K5/AUD/source-retract 全 1.00 | TMS 不再是与生产 store 平行的模块 |
| E11 expert families | 6 | 3 族 zero-shot 0.186；k=1 0.936；tactile 0.067–0.100 | 校准接口跨族成立；tactile 仍弱 |
| E12 L2 bundle replay | 2 | full 0.889 / no_memory 0.615 / no_entity 0.213；S4 no_relations 0/3 | L2 记忆消融成立；S4 任务测关系不测记忆 |
| E13 exact-RAG | C3 | exact RAG acc 1.000 @ 1.38 ms vs PSS 0.0003 ms；learned RAG 0.375 | 成本优势不是靠弱 RAG 基线 |
| E14 space ablation | 空间 | closed-set 0.969 三变体不变；cross-modal R@1 0.510 vs random 0.115 | 学习投影的价值在跨模态对齐 |
| E15 modality conflict | 8 | tactile_zero fixed 0.979；vision_shuffle abstain 0.083/1.000 | 触觉降级可显式处理 |
| E16 open-set | — | PSS AUROC 0.993 / reject 0.958 vs energy 0.937/0.753 | 开放集拒识仍成立 |

### 解读要点

- **E6 改写了“retrain-all 很贵”的写法**：真正便宜的是在冻结 heads 上重建 prototype（3 ms），不是全量重训。论文应说“不需要重训专家/投影”，而不是“retrain-all 很贵”。
- **E6b 把 retrain-all 的下界坐实**：5ep 专家重训 15.5 s + 投影 13.5 s ≈ 29 s，ZCR 0.943；冻结专家 + 重建 prototype 3.2 ms，ZCR 0.964。结论是“重训没有质量优势，成本高两个数量级”。
- **E6c 是 30ep 公平重训**：专家 24.6 s + 投影 13.4 s ≈ 38 s，ZCR 0.969；仍低于“冻结专家 + 仅重训投影”0.979（13.5 s），也只是略高于 prototype rebuild 0.964（3.2 ms）。结论是“没有质量理由重训专家”。
- **E14 改写了“per-modality 提升精度”的写法**：闭集精度不变；学习投影的价值在跨模态 R@1。C5/C6 应写“对齐接口”，不是“分类更强”。
- **E12 补上了 L2 记忆**：33 个 bundle 里 19 个有有效身份 trace；valid-bundle full 1.000 / no_memory 0.827 / no_entity 0.270。零 retention 的 bundle 不作为证据。
- **E18 区分了两类 L2 任务**：S4-PSS 的 `no_relations` 0/3，说明它测关系；`no_memory`/`no_entity` 3/3，说明它不测记忆。记忆证据来自 E12。
- **E17 补了 runtime 权限**：read_only 拒绝 5 项、admin 0；拒绝动作入审计。
- **E13 补上了 RAG 基线**：learned-RAG 是弱基线，exact-RAG（top-1 精确检索）是公平上界；PSS 在同等准确率下快约 4000x。

