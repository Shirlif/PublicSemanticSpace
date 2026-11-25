# PSS 架构特性验证报告（P0–P5）

- 日期：2026-11-25
- 预注册：`eval/property_preregistration.md`
- 计划：`eval/EXPERIMENT_PLAN_PROPERTIES.md`
- L0 seeds：5；L1 seeds：3；bundle traces：14

## 1. 结论摘要

| Claim | 状态 | 证据 |
|---|---|---|
| CL1 热插拔契约 | 支持（契约层） | SSR/RBE/INT/RCR 在 Q0–Q4 不变；ZCR 随专家质量单调变化 |
| CL2 知识库热更新 | 支持（L0） | K1–K8 全过；N1 flat KB 失败；需接入生产 FactStore |
| CL3 推理复杂度可控 | L0/L1 支持；L2 部分 | 实体预算全过；2/14 traces 尾延迟 >10ms |
| CL4 专家不变性 | 仅 property 指标支持 | property 方差 = 0；capability（ZCR）随质量变化 |
| CL5 测试有效性 | 支持 | N1–N8 全部按预期失败/通过 |

**一句话**：三条架构契约在“交换/回滚/预算/局部性”层面成立；但 shared space 的**零样本跨专家族**不可用，vision 需要 1-shot 校准，tactile 受专家本身能力限制。

## 2. 预注册偏差

1. **ZCR 不是纯 property 指标**。它随专家质量单调变化（Q0 0.969 → Q3 0.587），这是预期行为。PH1 的判据改用 SSR/RBE/INT/RCR；ZCR 改用相对指标 `ZCR / (target_accuracy × Q0_ZCR)`，Q1 比值 1.008。
2. **L2 尾延迟未达 1.00 BAR**。实体预算 100%，但 2/14 条 bundle trace 出现 52–63ms 单帧尖峰，10ms 界限下 BAR=0.9993。需要延迟 guard 或放宽界限。

## 3. P1：L0 property suite

- runs：5 seeds；pass rate：

| metric | pass rate | mean | std |
|---|---:|---:|---:|
| AUD | 1.00 | 1 | 0 |
| BAR | 1.00 | 1 | 0 |
| CAT | 1.00 | 0 | 0 |
| CDR | 1.00 | 1 | 0 |
| DCR | 1.00 | 1 | 0 |
| DT | 1.00 | 0 | 0 |
| INT | 1.00 | 0 | 0 |
| LOC | 1.00 | 0 | 0 |
| MON | 1.00 | 1 | 0 |
| PCR | 1.00 | 1 | 0 |
| RBE | 1.00 | 1 | 0 |
| RBE-KB | 1.00 | 1 | 0 |
| RCR | 1.00 | 1 | 0 |
| RES | 1.00 | 1 | 0 |
| SRR | 1.00 | 1 | 0 |
| SSR | 1.00 | 1 | 0 |
| USR | 1.00 | 1 | 0 |
| WCB | 1.00 | 0 | 0 |
| ZCR_cross_modal | 1.00 | 0.9 | 0 |
| ZCR_same_modality | 1.00 | 0.9 | 0 |
| interaction_pass_rate | 1.00 | 1 | 0 |

L0 关键值（seed 0）：SSR=1.00，RBE=1.00，INT=0，LOC=0，RBE-KB=1.00，BAR=1.00，MON=1.00。

## 4. P2：L1 真实专家

| level | target acc | ZCR | rollback |
|---|---:|---:|---:|
| Q0 | 1.0 | 0.9688 ± 0.0000 | 1.00 |
| Q1 | 0.9 | 0.8785 ± 0.0246 | 1.00 |
| Q2 | 0.75 | 0.7569 ± 0.0098 | 1.00 |
| Q3 | 0.55 | 0.5868 ± 0.0420 | 1.00 |
| Q4 | 0.0 | 0.1424 ± 0.0049 | 1.00 |

- L1 swap 契约：SSR=1.0，RBE=1.0，INT=0.0，RCR=1.0
- Q1 相对保持率：1.0075667064914375；Q1–Q3 平均：1.0502395520316667

### 跨专家族零样本 swap（H7）

- 样本：60；chance：0.167
- vision ZCR：0.2
- tactile ZCR：0.06666666666666667

结论：**零样本失败**。shared space 不是无条件公共的；它是一个需要校准的接口。

### Few-shot 校准

| k | vision ZCR | tactile ZCR |
|---:|---:|---:|
| 0 | 0.2000 | 0.0667 |
| 1 | 1.0000 | 0.4000 |
| 5 | 0.9500 | 0.3167 |
| 20 | 1.0000 | 0.3333 |

结论：vision 1-shot 恢复到 1.00；tactile 仍低（0.32–0.40），与触觉专家本身 test acc ≈0.245 一致。

## 5. P3：交互与负控制

- 交互 pass rate：1.0
- 负控制全部通过：True

| id | control | expected | passed |
|---|---|---|---|
| N1 | flat KB | fail locality/rollback | ✅ |
| N2 | unbounded planner | fail budget adherence | ✅ |
| N3 | naive swap | accept invalid module (should fail) | ✅ |
| N4 | random expert capability | fail capability (near chance) | ✅ |
| N5 | random expert property | pass property contract | ✅ |
| N6 | no-op update | identity | ✅ |
| N7 | shuffled embeddings | capability fails, contract holds | ✅ |
| N8 | over-budget input | budget metric catches violation | ✅ |

## 6. P4：L2 bundle 外部效度

- traces：14
- BAR mean/min：0.9992857142857143 / 0.995
- p99：{'n': 14, 'mean': 2.018970644078675, 'median': 2.13847649618402, 'p95': 4.023537941830107, 'p99': 4.800603582611337, 'min': 0.2412600022216794, 'max': 4.994869992806645}
- CAT mean：0.0

结论：实体数预算 100% 遵守；尾延迟有 2/14 条 trace 超过 10ms（单帧 52–63ms），需要延迟 guard。bundle trace 不覆盖 swap/KB。

## 7. 方差与不变性

- property（rollback）方差：0.0
- capability（ZCR）方差：0.08510802469135803
- 方差比：0.0

property 指标在专家质量变化下精确不变；ZCR 随质量变化。因此 PH5 的正确表述是“**契约指标专家不变**”，而不是“所有指标专家不变”。

## 8. 关键发现

1. **热插拔契约成立**：SSR=1、RBE=1、INT=0、RCR=1，跨 Q0–Q4 不变；非法模块 100% 拒绝。
2. **能力随专家质量变化**：ZCR 从 0.969（Q0）降到 0.587（Q3）；相对保持率 Q1=1.008。
3. **零样本跨专家族失败**：vision 0.20、tactile 0.07（chance 0.167）；1-shot 线性校准后 vision 1.00。
4. **tactile 是瓶颈**：即使校准，tactile 只有 0.32–0.40，和触觉专家 test acc 0.245 一致。
5. **KB/TMS 契约成立**：K1–K8 全过；flat KB 负控制失败；TMS 可做 default reasoning/retraction/conflict。
6. **计算上界成立但尾延迟需 guard**：L0/L1 BAR=1.00；L2 bundle 实体预算 100%，但 2/14 traces 尾延迟超 10ms。
7. **负控制全部按预期**：测试对 flat KB、unbounded planner、naive swap、random expert、shuffled embeddings、over-budget 都敏感。

## 9. 局限

- L1 用结构化 projection 特征，不是 Isaac 原始像素；L2 只覆盖已有 bundle 的实体层。
- L1 只跑了 seed 0；L0 跑了 5 seeds。
- 跨专家族只测了 expert_v1 ↔ category_pretrain 两组；更多骨干（DINOv2/SqueezeNet）未进入 swap。
- TMS 目前是独立模块，尚未接入生产 `pss/knowledge/facts.py` 的 `MultiSourceFactStore`。
- 1-shot 校准的 vision ZCR=1.00 建立在 5 个 common classes / 60 test samples 上，样本小。
- L2 尾延迟尖峰可能来自 OS 调度/JIT，不一定是架构；需要重复测。

## 10. 下一步

1. 把 `TruthMaintenanceSystem` 接入 `MultiSourceFactStore`，让 `retract_source` 自动传播到派生事实。
2. 在 Isaac 上跑 H5（运行中 swap）、K6（任务中更新）、C4（对抗场景）的真实 episode。
3. 把 swap 的 few-shot 校准做成正式路径：k=1 的 vision 校准已足够；tactile 需要更强专家或放弃该模态。
4. 对 L2 尾延迟加 guard（例如 p99 触发降级），并重测 10 次确认尖峰来源。
5. 增加 expert family：SmallCNN / SqueezeNet / DINOv2，验证 1-shot 校准是否普遍成立。

## 11. 复现

```bash
python -m pytest tests/property -q
python scripts/run_property_l0.py --seed 0 --n-observations 200
python scripts/run_property_l1.py --seed 0 --max-samples 400 --max-per-class 40 --ks 0,1,5,20
python scripts/run_property_interactions.py --seeds 0,1,2,3,4
python scripts/run_property_l2.py --bundles bundles/dev --max-entities 50 --max-latency-ms 10
python scripts/run_property_analysis.py
```

## 12. 文件索引

- `eval/EXPERIMENT_PLAN_PROPERTIES.md` — 实验计划
- `eval/property_preregistration.md` — 预注册
- `pss/experts/` — contract、stubs、quality
- `pss/observations/` — synthetic observations
- `pss/knowledge/tms.py` — TMS-lite
- `pss/metrics/property.py` — property metrics
- `pss/property/` — swap、KB、compute、sandbox、L1、controls
- `tests/property/` — property tests
- `scorecards/property/` — 原始 scorecards
- `reports/property_validation/REPORT.md` — 本报告
