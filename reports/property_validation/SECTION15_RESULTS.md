# 15. 结果报告（按实验计划 §15 模板）

> 数据来源：L0 sandbox（5 seeds）、L1 真实专家（3 seeds）、P3 负控制、P4 bundle trace。
> A5 是 oracle/retrain 上界，不是真实 retrain-all 运行；A7/A8 尚未接入 runtime。

## 执行清单

| 阶段 | 状态 | 说明 |
|---|---|---|
| P1 L0 | ✅ | 5 seeds；H/K/C + 交互 |
| P2 L1 | ✅ | 真实 expert_v1 + projection_v1；3 seeds |
| P3 交互/负控制 | ✅ | N1–N8 全部按预期 |
| P4 L2 | ⚠️ | 14 条 bundle trace；无新 Isaac episode |
| P5 报告 | ✅ | 本文件 |

## 与计划的偏差

| 计划项 | 实际执行 | 原因/影响 |
|---|---|---|
| Q0–Q5 × A0–A8 × T1–T3 全因子 | 缩减设计：L0 全量 + L1 3 seeds | 全因子会爆；按 §4.2 的分数因子执行 |
| A5 retrain-all | oracle 上界 | 未真实重训所有模块；A5 只作上界参考 |
| A7 flat retrieval / A8 no degradation | 未接入 runtime | 需要额外 runtime hook；不影响主 claim |
| C7 budget adaptation | 未单独运行 | C1/C3 已覆盖预算与 anytime；C7 留作扩展 |
| L2 新 Isaac episode | 只读已有 14 条 bundle trace | Isaac 3.11 环境未启动；trace 无 facts/expansions |
| H0–H2/H6/H7 | 来自 L0/L1 既有 suite | H3–H5 为补充 L0 模拟 |
| K3 依赖传播 | TMS 级验证 | 尚未接入生产 MultiSourceFactStore |

## 15.1 表 1：Property × Expert quality

| 指标 | Q0 | Q1 | Q2 | Q3 | Q4 | 斜率 | 95% CI | TOST |
|---|---|---|---|---|---|---|---|---|
| SSR | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | [-0.000, 0.000] | ✅ |
| ZCR | 0.969 | 0.878 | 0.757 | 0.587 | 0.142 | -0.194 | [-0.237, -0.132] | ❌ |
| INT | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | ✅ |
| RBE | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | [-0.000, 0.000] | ✅ |

**读法**：SSR/INT/RBE 的斜率 ≈ 0 且 TOST 通过 → 支持专家不变性。ZCR 随质量下降，是 capability，不是 property；PH1 的契约判据使用 SSR/INT/RBE。

## 15.2 图 1：不变性图

![图 1](figures/fig1_invariance.png)

左：SSR/INT/RBE 在 Q0–Q4 上保持平坦；右：ZCR 随专家质量下降，A4 naive swap 接近 chance，A5 为上界。

## 15.3 图 2：KB 状态与回滚

![图 2](figures/fig2_kb_state.png)

回滚精确性（业务状态 hash）：**1.000**。

## 15.4 图 3：预算-质量曲线

![图 3](figures/fig3_budget_quality.png)

C1 budget sweep：BAR = **1.000**，MON = **1.000**。节点数始终 ≤ 预算。

## 15.5 表 2：Ablation matrix

| 变体 | SSR | ZCR | INT | RBE | USR | LOC | PCR | BAR | MON |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 1.000 | 0.878 | 0.000 | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 1.000 |
| A1 | 1.000 | 0.167 | 0.000 | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 1.000 |
| A2 | 1.000 | 0.878 | 0.000 | 1.000 | 0.500 | 1 | 0.000 | 1.000 | 1.000 |
| A3 | 1.000 | 0.878 | 0.000 | 1.000 | 1.000 | 0 | 1.000 | 0.000 | 1.000 |
| A4 | 1.000 | 0.200 | 0.000 | 0.000 | 1.000 | 0 | 1.000 | 1.000 | 1.000 |
| A5 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 1.000 |
| A6 | 1.000 | 0.878 | 0.000 | 1.000 | 1.000 | 0 | 1.000 | 0.000 | 1.000 |
| A7 | — | — | — | — | — | — | — | — | — |
| A8 | — | — | — | — | — | — | — | — | — |

注：A1 的 ZCR 是跨模态（per-modality space，无共享几何）；A4 的 ZCR 是跨专家族零样本；A5 是 oracle 上界；A7/A8 未接入 runtime，标为 —。

## 15.6 表 3：负控制

| ID | 预期 | 实际 | 通过? |
|---|---|---|---|
| N1 | fail locality/rollback | {"rollback": false, "propagation": false} | ✅ |
| N2 | fail budget adherence | {"nodes": 56, "success": false} | ✅ |
| N3 | accept invalid module (should fail) | {"accepted": true, "active_dim": 64} | ✅ |
| N4 | fail capability (near chance) | {"accuracy": 0.21, "chance": 0.125} | ✅ |
| N5 | pass property contract | {"valid": true, "swap_success": true} | ✅ |
| N6 | identity | {"before": "bdc6345d4cfc875596bf91ad3e43bd027327103f4d5abb9fe83bac18ca875909", "after": "bdc6345d4cfc875596bf91ad3e43bd027327103f4d5abb9fe83bac18ca875909"} | ✅ |
| N7 | capability fails, contract holds | {"accuracy": 0.125, "chance": 0.125} | ✅ |
| N8 | budget metric catches violation | {"violations": ["entities", "relations"]} | ✅ |

**读法**：所有负控制按预期失败/通过，说明 property 指标对 flat KB、unbounded planner、naive swap、random expert、shuffled embeddings、over-budget 是敏感的。

## 15.7 表 4：方差分解

| 指标 | Var(expert) | Var(arch) | Var(interaction) | VR | 结论 |
|---|---|---|---|---|---|
| SSR | 0.000 | 0.000 | n/a | n/a | invariant |
| ZCR | 0.106 | 0.193 | n/a | 1.815 | capability: expert + architecture |
| INT | 0.000 | 0.000 | n/a | n/a | invariant |
| RBE | 0.000 | 0.250 | n/a | ∞ | architecture-controlled |

**读法**：SSR/INT/RBE 的专家方差为 0，支持“契约指标专家不变”。ZCR 的专家方差 0.106、架构方差 0.193，VR = 1.815 < 3：能力由专家和架构共同决定，不能声称“架构主导能力”。RBE 的专家方差为 0、架构方差 > 0，说明回滚是架构控制的。

## H 系列补充（H3–H5）

| 实验 | 关键结果 |
|---|---|
| H3 add/remove | {"INT": 0.0, "ZCR_new": 1.0, "cleanup": 1.0, "RBE": 1.0} |
| H4 version skew | {"RCR": 1.0, "CDR": 1.0, "RBE": 1.0, "SML": 0.1993000041693449, "disagreement": 0.1} |
| H5 hot swap under load | {"DT": 0.0, "SSR": 1.0, "RBE": 1.0, "requests": 200, "failures": 0} |

## 结论与边界

1. **契约层成立**：SSR/INT/RBE 对专家质量不敏感；KB 的 LOC/RBE-KB/AUD 成立；BAR/MON 成立。
2. **能力层不宣称**：ZCR 随专家质量下降；跨族零样本 swap 失败（vision 0.20、tactile 0.07）；vision 1-shot 校准可恢复。
3. **A5 只是上界**：真实 retrain-all 未运行；A7/A8 未接入 runtime。
4. **L2 部分完成**：bundle trace 未覆盖 facts/expansions，未跑新 Isaac episode；不能声称真实感知外部效度。

## 复现

```bash
python -m pytest tests/property -q
python scripts/run_property_l0.py --seed 0 --n-observations 200
python scripts/run_property_l1.py --seed 0 --max-samples 400 --max-per-class 40 --ks 0,1,5,20
python scripts/run_property_interactions.py --seeds 0,1,2,3,4
python scripts/run_property_l2.py --bundles bundles/dev --max-entities 50 --max-latency-ms 10
python scripts/run_property_section15.py
```

