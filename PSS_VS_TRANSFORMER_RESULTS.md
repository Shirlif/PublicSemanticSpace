# PSS vs Transformer 评测总结

> 日期：2026-11-23
> 范围：PSS 架构与 entity/modality transformer 在 6 个维度上的对比
> 状态：**结构可行性与部分结构优势已验证；通用精度优势尚未证明**
> 协议：`PSS_VS_TRANSFORMER_EXPERIMENT.md`

---

## 1. 对比维度

| 维度 | 任务 | 主要指标 |
|---|---|---|
| 闭集类别识别 | 16 个 seen 类别分类 | top-1 |
| 遮挡 re-ID | 14 帧完全遮挡后认回同一个体 | re-ID accuracy |
| 开放集拒识 | 4 个 unseen 类别出现时能否拒绝 | AUROC、unseen reject |
| 增量类别更新 | 新增 cloth 类别 | old/new accuracy、forgetting、update time、rollback |
| 模态冲突鲁棒性 | 破坏 tactile / vision / geometry | top-1、conflict abstain coverage/accuracy |
| 规划/任务成功 | 5 路点关系图，1 vs 2 障碍物 | success rate、steps、invalid actions |
| 多实体计算 scaling | N 从 4 到 1000 | p50 latency |

未对比：

- 真实数据 / R-Gate
- 真实预训练 VLM

---

## 2. 各维度结果

### 2.1 闭集类别识别：打平

| 方法 | seen top-1 |
|---|---:|
| PSS prototypes | 0.979 |
| Transformer | 0.978 |

结论：**没有证据说 PSS 闭集识别更差，也没有证据说更好。**

### 2.2 遮挡 re-ID：历史结果（transformer 基线无 tactile）

v1（固定运动/机位）：

| 方法 | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|
| PSS hand cost | 0.561 | 0.800 | 0.311 | 0.333 |
| PSS + learned metric | 0.787 | 0.976 | 0.871 | 0.324 |
| Transformer | **0.906 ± 0.020** | **0.982** | **0.938** | **0.720** |

v1 上 transformer 明显更高，但 near-identical 0.720 不真实：随机运动/机位后掉到 0.387。

v2（随机运动 + 相机抖动）：

| 方法 | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|
| PSS hand cost | 0.533 | 0.722 | 0.378 | 0.311 |
| PSS + metric（param-matched） | 0.809 ± 0.031 | 0.991 | 0.818 | 0.436 |
| PSS + metric + assignment | 0.798 ± 0.010 | 0.982 | 0.902 | 0.324 |
| PSS + graph matcher | 0.930 ± 0.029 | 1.000 | 0.982 | 0.738 |
| **PSS + graph + tactile** | **0.989 ± 0.017** | **1.000** | **1.000** | **0.956** |
| Transformer | 0.802 ± 0.008 | 0.978 | 0.867 | 0.387 |

**历史结论（注意：transformer 基线没有 tactile）**：

- 去 shortcut 后，PSS+metric 与 transformer 基本打平，delta +0.007，方差约 ±0.03，不显著。
- PSS 的手写 re-ID 是瓶颈；learned metric 把 v1 从 0.561 提到 0.787，把 v2 提到 0.809。
- assignment loss 改善 samecat（0.902），但 near-identical 掉到 0.324，overall 没有胜出。
- P0 graph matcher 把 v2 overall 提到 0.930，比**无 tactile transformer** 高 +0.128；但这只是历史对照，不是公平归因。
- P0.3 tactile token 把 overall 提到 0.989，near-identical 从 0.738 提到 0.956；tactile 是最大单项增益。

**公平归因修正（2026-11-23，5 seeds）**：

给 transformer 同样的 tactile token + 同样的 Hungarian：

| token set | PSS graph | transformer | delta |
|---|---:|---:|---:|
| base | 0.930 ± 0.029 | 0.802 ± 0.008 | +0.128 |
| tactile | 0.989 ± 0.017 | **0.977 ± 0.017** | **+0.012** |

修正后的 re-ID 结论：

1. **PSS+tactile 0.989 仍是当前系统级最好结果**，但只比 transformer+tactile 高 **+0.012**，在约 1 个标准差内。
2. **+0.128 的大差距来自 tactile 输入，不是 graph matcher 架构。**
3. **re-ID 的架构级优势不成立**；graph 边消息/assignment loss 消融也不支持（no_edges 0.941、no_assignment 0.969、greedy 0.967、full 0.930）。
4. 公平归因只覆盖 re-ID 精度；P1b tracking 的 1.000 vs 0.908 仍是**无 tactile transformer** 对照，不能直接归因给架构。需要补 transformer+tactile 的 tracking 基线后，才能下系统级结论。
5. 当前能说的是：**tactile 是实例级身份的主要信号；PSS 的 re-ID 价值在系统集成和显式记忆，不在“更聪明的匹配架构”。**

**P1b 完整 online tracking**：

| 方法 | ID retention | switches | reappearance |
|---|---:|---:|---:|
| Hand PSS tracker | 0.919 | 0.16 | 0.839 |
| PSS + graph matcher | 0.986 | 0.03 | 0.972 |
| **PSS + graph + tactile** | **1.000** | **0.00** | **1.000** |
| Transformer re-ID | 0.908 | 0.18 | 0.817 |

完整 tracking 上 PSS+tactile 是 1.000 / 0.00 / 1.000，transformer 是 0.908 / 0.18 / 0.817。**但 transformer 仍无 tactile**；这是系统级对照，不是公平架构归因。需要补 transformer+tactile 的 tracking 基线，才能判断架构增量。

**P0-1 DINOv2 instance token：未达标**：

| variant | token | overall | near-identical |
|---|---:|---:|---:|
| base | 279 | 0.930 ± 0.029 | 0.738 |
| dino（384 raw） | 663 | 0.903 ± 0.032 | 0.658 |
| tactile | 407 | **0.989 ± 0.017** | **0.956** |
| both（384 raw） | 791 | 0.967 ± 0.024 | 0.876 |
| dino_pca32 | 311 | 0.942 ± 0.030 | 0.804 |
| both_pca32 | 439 | 0.987 ± 0.011 | 0.964 |

结论：

- 原始 384 维 DINO CLS 反而拉低 overall；PCA-32 后 DINO-only 提到 0.942 / near 0.804，仍低于 tactile-only。
- DINO 是真实但较弱的实例信号；当前合成数据上不能替代 tactile，both 也只与 tactile 打平。
- P0-1 的验收线（无 tactile near ≥0.90）未达到；代码保留为可选路径，默认不启用。

### 2.3 开放集拒识：PSS 明确更好

| 方法 | AUROC | unseen reject @95% seen accept | seen top-1 |
|---|---:|---:|---:|
| **PSS prototypes** | **0.993** | **0.958** | 0.979 |
| Transformer softmax | 0.886 ± 0.025 | 0.445 | 0.978 |
| Transformer energy | 0.912 ± 0.034 | 0.626 | 0.978 |

结论：

- PSS 在 seen 精度相同的前提下，unseen reject 高 **33–51 个百分点**。
- 原因：PSS 用“到原型的距离”判断 novel；transformer softmax/energy 对 unseen 容易过度自信。
- **这是明确的开放集能力优势。**

### 2.4 增量类别更新：标准 fine-tune 下 PSS 更好，replay 追平遗忘

| 方法 | old before | old after | new after | forgetting | update time |
|---|---:|---:|---:|---:|---:|
| **PSS prototype add** | 0.979 | **0.979** | **1.000** | **0.000** | **43.94 ms** |
| Transformer last-layer | 0.978 | 0.883 | 0.985 | 0.095 | 1059 ms |
| Transformer full | 0.978 | 0.816 | 1.000 | 0.163 | 3278 ms |
| Transformer replay | 0.978 | **0.979** | **1.000** | **-0.001** | 3768 ms |

结论：

- 标准 fine-tune 下 PSS 明显更好：last-layer 掉 9.5 点，full 掉 16.3 点，PSS 不掉点。
- **但 replay 基线追平遗忘**：old 0.979、new 1.000、forgetting -0.001；H3 forgetting 从 PASS 变成 **FAIL**。
- PSS 仍快 **24.1x**，且不需要旧数据、可回滚；优势从“不会忘”变成“省存储、快更新、可审计”。
- **PSS 在增量更新上的绝对优势被 replay 削弱**。

### 2.5 模态冲突：各有胜负，PSS 多了弃权

| 条件 | PSS fixed | PSS abstain (cov/acc) | PSS reliable (cov/acc) | Transformer |
|---|---:|---:|---:|---:|
| clean | 0.979 | 0.85/0.982 | 0.74/0.993 | 0.978 |
| tactile shuffle | 0.974 | 0.85/0.982 | 0.74/0.993 | 0.959 |
| tactile noise 0.5 | 0.979 | 0.85/0.982 | 0.74/0.993 | 0.980 |
| tactile noise 1.0 | 0.979 | 0.85/0.982 | 0.74/0.993 | 0.977 |
| tactile zero | 0.979 | 0.85/0.982 | 0.74/0.993 | 0.968 |
| vision shuffle | 0.474 | 0.08/1.000 | 0.07/1.000 | 0.156 |
| geometry shuffle | 0.458 | 0.07/0.923 | 0.05/1.000 | **0.749** |
| vision zero | 0.865 | 1.00/0.865 | 0.82/0.904 | 0.625 |
| geometry zero | 0.979 | 1.00/0.979 | 0.97/0.989 | 0.936 |

结论：

- **tactile 冲突**：PSS 几乎不受影响，transformer 略降。
- **vision 冲突**：两边都崩，但 PSS 更稳（0.474 vs 0.156）。
- **geometry 冲突**：transformer 更稳（0.749 vs 0.458），PSS 固定融合被高权重 geometry 拖累。
- **PSS 独有优势**：conflict-aware abstain。vision/geometry shuffle 下 reliable 规则只回答 5–7% 样本，accepted accuracy **1.000**；transformer 没有显式弃权，仍会输出错误答案。
- 所以 PSS 不是“融合后精度必然更高”，而是**知道自己被打架，并选择拒识**。

### 2.6 组合属性 OOD：接近，未过门槛

训练只用低 hardness 样本，测试用同类高 hardness 样本（未见材质 regime）：

| 方法 | ID accuracy | OOD accuracy |
|---|---:|---:|
| PSS prototypes | 0.975 | **0.978** |
| Transformer | 0.942 | 0.967 ± 0.006 |

结论：

- PSS OOD 高 1.1 个百分点，但预注册门槛是 +10 点，**H5 FAIL**。
- ID 上 PSS 高 3.3 点，说明只在小样本低 hardness 上训练时，原型法泛化更稳；但这不足以证明组合 OOD 优势。
- 真正的“物体-关系-动作”新组合还没有测；P5 目前只是属性/材质 regime 的 OOD。

### 2.7 规划/任务成功：PSS 稳定，transformer 在 OOD 掉点

同一张 5 路点关系图、同一组 clear-edge mask。PSS 用显式算子 + A*；transformer 用 5 个 region token + edge mask 做行为克隆，训练只有 1 个障碍物。

| split | PSS success | PSS steps | Transformer success | Transformer steps | invalid rate |
|---|---:|---:|---:|---:|---:|
| ID（1 obstacle） | **1.000** | 2.07 | 0.990 ± 0.000 | 2.07 | 0.000 |
| OOD（2 obstacles） | **1.000** | 2.12 | 0.924 ± 0.016 | 2.10 | 0.000 |
| dynamic ID | **1.000** | 2.15 | 0.984 ± 0.005 | 2.13 | 0.010 |
| dynamic OOD | **1.000** | 2.26 | 0.928 ± 0.015 | 2.22 | 0.056 |

结论：

- Transformer 训练分布 next-action accuracy 1.000，OOD 掉点来自组合泛化，不是欠拟合。
- OOD 上 PSS 高 7.6 点，dynamic OOD 高 7.2 点；差距稳定但不大。
- PSS 的 1.000 是 A* 完备性保证；transformer 还会输出 blocked edge（dynamic OOD invalid 5.6%）。
- 这是“显式规划在组合布局上更可靠”的证据，但不是数量级优势。

### 2.8 多实体计算 scaling：PSS 更快

| N | PSS p50 | Transformer p50 | speedup |
|---:|---:|---:|---:|
| 4 | 0.066 ms | 0.309 ms | 4.7x |
| 20 | 0.077 ms | 0.542 ms | 7.0x |
| 50 | 0.083 ms | 0.503 ms | 6.1x |
| 200 | 0.128 ms | 0.823 ms | **6.4x** |
| 1000 | 0.283 ms | 6.677 ms | **23.6x** |

结论：

- PSS 在所有 N 上都更快。
- 但 N=200 没达到预注册的 10x 门槛；N=1000 才 23.6x。
- 优势是 **active-set O(k²) vs dense full-attention O(N²)**，不是精度。

### 2.9 P0-4 强 transformer 基线：DINO 特征没有变强

| baseline | params | overall | mixed | samecat | near-identical |
|---|---:|---:|---:|---:|---:|
| base transformer | 336,896 | **0.802 ± 0.008** | **0.978** | **0.867** | **0.387** |
| + frozen DINOv2 CLS | 435,200 | 0.752 ± 0.037 | 0.964 | 0.769 | 0.311 |

结论：

- 加冻结 DINOv2 CLS 反而掉 5 个点，near-identical 从 0.387 掉到 0.311。
- 预训练特征本身不等于更强基线；96×96 合成 crop 上 DINO 噪声会伤害 transformer。
- PSS graph 0.930 仍然明显高于两个 transformer 基线。
- sparse/retrieval attention 和 EWC 还没测。

### 2.10 消融与统计：对无 tactile 基线显著，公平归因后不显著

| variant | overall | samecat | near-identical |
|---|---:|---:|---:|
| full | 0.930 ± 0.029 | 0.982 | 0.738 |
| no_edges | 0.941 ± 0.011 | 1.000 | 0.764 |
| no_assignment | **0.969 ± 0.033** | 0.973 | **0.902** |
| greedy | 0.967 | 1.000 | 0.867 |
| transformer | 0.802 ± 0.008 | 0.867 | 0.387 |
| + tactile | **0.989 ± 0.017** | 1.000 | **0.956** |

Paired bootstrap（full vs transformer 无 tactile，300 配对）：mean delta **+0.128**，95% CI **[+0.096, +0.162]**，P(delta≤0)=**0.0000**。  
注意：这是无 tactile 基线；公平归因（同 tactile + Hungarian）后 delta 只剩 **+0.012**。

结论：

- PSS graph 对 transformer 的优势统计显著。
- 但 full 配置不是最优：去掉 assignment loss 到 0.969，贪心 0.967，去边 0.941。
- 当前 assignment loss 和 edge message 在 v2 上过拟合；集合级 pair scoring + tactile 是主要收益。
- 参数量 169k vs 337k。

### 2.11 跨 benchmark v3：PSS 掉点更少

v3 加难：5 objects、40 frames、30-frame occlusion、160×160、随机运动/机位/光照。

| train -> test | PSS graph | transformer | delta |
|---|---:|---:|---:|
| v2 -> v2 | 0.930 ± 0.029 | 0.802 ± 0.008 | +0.128 |
| v2 -> v3 | **0.881 ± 0.029** | 0.672 ± 0.022 | **+0.209** |
| v3 -> v3 | 0.847 ± 0.040 | 0.711 ± 0.014 | +0.136 |

注意：表中的 transformer 基线仍是**无 tactile**；公平归因后 re-ID 精度增量只有 +0.012。跨 benchmark 的“系统级更稳”成立，但不能归因给 graph 架构。

结论：

- v2→v3 上 PSS 掉 4.9 点，transformer 掉 13.0 点；PSS 跨 benchmark 更稳。
- v3→v3 低于 v2→v3，因为 v3 train 只有 120 条且更难；这是训练量问题，不是 v3 无效。
- v3 缺 geometry 时 overall 0.840，缺 vision 0.863；near-identical 在缺 geometry 时掉到 0.573。
- PSS 的优势不是只存在于 v2 自建集。

### 2.12 学习型 reliability：未达标

| condition | PSS fixed | PSS reliable (cov/acc) | learned (cov/acc) |
|---|---:|---:|---:|
| clean | **0.979** | 0.74/0.993 | 0.96/0.907 |
| vision shuffle | 0.438 | 0.02/1.000 | 0.84/0.695 |
| geometry shuffle | 0.453 | 0.07/1.000 | 0.85/0.273 |
| vision zero | **0.865** | 0.82/0.904 | 0.66/0.700 |
| geometry zero | **0.979** | 0.97/0.989 | 0.10/0.198 |

- 学到的 gate 在 clean 上不如 fixed/reliable；geometry zero 下几乎失效。
- 目标“geometry shuffle ≥0.70、coverage ≥0.80”未达到。
- 规则 reliable 在 zero 条件下仍更稳；学习型 gate 需要更覆盖的损坏分布或显式缺失检测。

### 2.13 conformal 开放集：marginal 可用，class-conditional 不稳

| method | seen accept | unseen reject | AUROC | min class coverage |
|---|---:|---:|---:|---:|
| marginal conformal | **0.969** | **0.958** | **0.993** | - |
| class-conditional | 0.859 | 0.958 | - | **0.333** |

- Marginal conformal 在 alpha=0.05 下达到 96.9% accept / 95.8% reject。
- Class-conditional 的平均 coverage 0.859，但最小类只有 0.333；每类 val 样本不足，阈值不稳定。
- 当前建议用 marginal conformal；class-conditional 需要更多校准数据。

### 2.14 原型治理：旧类不动，污染可标记

| poison | prototype | old acc | new acc | poison flagged |
|---:|---|---:|---:|---:|
| 0% | mean | 0.979 | 1.000 | 0.000 |
| 30% | mean | 0.979 | 1.000 | **0.988** |
| 30% | trimmed | 0.979 | 1.000 | 0.973 |
| 30% | medoid | 0.979 | **0.728** | 0.835 |
| 30% | median | 0.979 | 1.000 | 0.973 |

- 旧类始终 0.979；30% 污染下 mean/trimmed/median 新类仍 1.000，污染标记率 0.97–0.99。
- medoid 明显更差，不适合新类原型。
- 回滚 29.5 ms。
- 结论：当前污染远离类中心，均值已够稳；真正需要的是污染检测/标记，而不是换原型估计器。

### 2.15 闭环动态障碍：1–2 次重规划，全部成功

| condition | success | pushes | replans | final distance |
|---|---:|---:|---:|---:|
| anomaly on（3 seeds） | **3/3** | 37.3（mean） | 1.33（mean） | 0.0446–0.0487 m |
| anomaly off（3 seeds） | **3/3** | 17.7（mean） | 0 | 0.0431–0.0598 m |

- 动态障碍触发 1–2 次重规划，额外约 20 次 push，全部成功。
- 最终距离 4.5–6.0 cm，符合预注册阈值。
- 环境是 MuJoCo，不是 PyBullet；只有 3 seeds、2 物体。

### 2.16 真组合关系 OOD：PSS 1.000 vs transformer 0.434

| method | ID success | OOD success | OOD invalid |
|---|---:|---:|---:|
| PSS explicit planner | **1.000** | **1.000** | 0.000 |
| Transformer BC | 0.997 ± 0.003 | **0.434 ± 0.216** | 0.139 |

- 训练 4 个关系对，测试 4 个未见关系对；每个关系单独见过，但 pair 没见过。
- ID 上 transformer 几乎打平；OOD 上掉到 0.434，并出现 13.9% invalid action。
- PSS 用显式几何 goal check，OOD 仍 1.000。
- **这是显式关系结构带来组合泛化的最清楚证据。**

### 2.17 世界模型 rollout：有改善，未达 5 cm

| variant | 1-step pos error | rollout all | rollout target |
|---|---:|---:|---:|
| one_step | 0.0240 ± 0.0008 | 0.0932 ± 0.0026 | 0.1001 ± 0.0185 |
| rollout_aware | 0.0217 ± 0.0014 | **0.0863 ± 0.0070** | 0.1007 ± 0.0197 |

- rollout loss 把全实体 rollout 从 0.0932 降到 0.0863（约 7%），one-step 也略降。
- 但目标 <0.05 m 未达到；target rollout 仍 0.1007。
- 结论：当前误差主要受模型容量/数据覆盖限制，不是训练目标本身。

### 2.18 长时记忆：70 帧遮挡后 PSS 仍明显更好

| method | ID retention | switches | reappearance |
|---|---:|---:|---:|
| PSS graph+tactile | **0.983** | **0.03** | **0.967** |
| Transformer re-ID | 0.900 | 0.20 | 0.800 |

| subset | PSS | Transformer |
|---|---:|---:|
| mixed | 1.000 | 1.000 |
| samecat | 1.000 | 0.933 |
| near-identical | **0.933** | **0.667** |

- 80 帧序列、70 帧完全遮挡；PSS 仍 0.983 / 0.967，transformer 0.900 / 0.800。
- 注意：transformer 基线无 tactile；这是系统级对照，不能归因给 graph 架构。
- near-identical 差距最大；长时记忆里 tactile 实例信号更重要。
- 说明 PSS+tactile 的持久记忆不是短时 trick，但架构增量仍需公平 tracking 基线确认。

### 2.19 工程化（P2-13）

- graph load 24.9 ms，perception load 23.7 ms，RSS 927 MB；
- dense total p50/p95 0.51/15.33 ms；
- pruned total p50/p95 0.49/14.25 ms；
- p50 约 0.5 ms，p95 14–15 ms；pruned 在 3 实体场景只快 0.02 ms，符合预期。

### 2.20 P5b 感知版：渲染+tracker 后仍 1.000 vs 0.453

| method | ID success | OOD success | OOD invalid |
|---|---:|---:|---:|
| PSS explicit planner | 1.000 | **1.000** | 0.000 |
| Transformer BC | 0.995 ± 0.010 | 0.453 ± 0.232 | 0.123 |

- 3 objects 渲染到 5 个桌面区域，tracker 输出 perceived state；PSS/transformer 用同一状态。
- PSS 1.000；transformer ID 0.995、OOD 0.453。
- 与纯符号 P5b（1.000 vs 0.434）几乎一致，说明结论不是符号状态假象。
- 边界：只评估规划，不执行 push，不是完整物理闭环。

### 2.21 P5b 执行闭环：ID 0.950，OOD 0.400

| method | ID success | OOD success |
|---|---:|---:|
| PSS explicit planner + execution | **1.000 ± 0.000** | **1.000 ± 0.000** |
| Transformer BC + execution | 0.950 ± 0.100 | **0.400 ± 0.255** |

- 每步渲染 → tracker → 规划/策略 → 执行 → 重感知。
- PSS 在 seen/OOD 都 1.000；transformer ID 0.950、OOD 0.400。
- 执行是 kinematic placement，不是物理 push；物理 push 因多物体路径碰撞未启用。
- 关系组合优势进入执行闭环后仍然成立。

### 2.22 碰撞感知执行：physical push 仍保留关系优势

| execution | PSS ID | PSS OOD | TR ID | TR OOD |
|---|---:|---:|---:|---:|
| kinematic placement | 1.000 | **1.000** | 1.000 | 0.450 |
| physical push | 0.800 | **0.700** | 0.750 | 0.150 |

- 规划器加入 segment clearance，直连被挡时经中间区域绕行。
- kinematic 下 PSS 1.000、TR OOD 0.450；physical push 下 PSS 0.700、TR 0.150。
- 物理执行降低绝对成功率（推动漂移/接触/重规划失败），但 PSS 的 OOD 优势仍明显。
- 这是目前最接近“感知→规划→物理执行→重感知”闭环的结果。

### 2.23 公平 tracking 基线：tactile 也解释大部分 tracking 优势

| method | ID retention | switches | reappearance |
|---|---:|---:|---:|
| Hand PSS | 0.919 | 0.16 | 0.839 |
| PSS + graph + tactile | **1.000** | **0.00** | **1.000** |
| Transformer + tactile | 0.988 ± 0.008 | 0.02 | 0.977 ± 0.017 |

- 同 tactile token、同 Hungarian 后，tracking 差距从 1.000 vs 0.908 缩到 **1.000 vs 0.988**。
- PSS 仅高 **+0.012** retention，和 re-ID 的 +0.012 一致。
- **tracking 的大部分优势也来自 tactile 输入，不是 graph 架构。**
- P1b 之前“1.000 vs 0.908”的架构解释正式撤回。

### 2.24 EWC：没有降低遗忘

| method | old after | new after | forgetting | update time |
|---|---:|---:|---:|---:|
| PSS prototype add | **0.979** | **1.000** | **0.000** | **31.15 ms** |
| Transformer full | 0.809 | 1.000 | 0.169 | 4274 ms |
| Transformer replay | 0.976 | 1.000 | **0.002** | 5066 ms |
| Transformer EWC | 0.741 | 1.000 | 0.237 | 5811 ms |

- EWC 在当前 lambda/Fisher 设置下没有降低遗忘，反而比 full 更差。
- replay 仍是最好的零遗忘基线；PSS 仍快 41.7x。
- EWC 需要调参或换 Fisher 估计，不能作为“强基线已闭合”的证据。

### 2.25 sparse attention：同 tactile 下 transformer 反超 PSS

| variant | top-k | overall | near-identical |
|---|---:|---:|---:|
| full attention | - | 0.977 ± 0.017 | 0.924 |
| **sparse_k1** | 1 | **0.996 ± 0.005** | **0.991** |
| sparse_k2 | 2 | 0.982 ± 0.019 | 0.964 |
| sparse_k4 | 4 | 0.974 ± 0.021 | 0.978 |
| PSS graph+tactile | - | 0.989 ± 0.017 | 0.956 |

- 同 tactile token 下，**sparse_k1 transformer 0.996，高于 PSS 0.989**；near-identical 0.991 vs 0.956。
- 这进一步推翻 PSS re-ID 的架构优势：不仅没有显著增量，甚至可能被强 transformer 超过。
- sparse mask 是 dense 实现，不是延迟优化；这是精度基线。
- **re-ID 的强基线已经闭合，PSS graph matcher 不是最优 re-ID 架构。**

### 2.26 事实级知识热更新：PSS 0.171 ms，replay 追平遗忘

| method | edited accuracy | old accuracy | forgetting | edit time |
|---|---:|---:|---:|---:|
| PSS fact update | **1.000** | **1.000** | **0.000** | **0.171 ms** |
| Transformer full | 1.000 | 0.735 | 0.265 | 952.3 ms |
| Transformer replay | 1.000 | 1.000 | 0.000 | 914.1 ms |

- PSS 编辑一条 fact 0.171 ms，旧知识 1.000、可回滚、可审计。
- full fine-tune 掉到 0.735（forgetting 0.265），952 ms。
- replay 追平 old 1.000 / forgetting 0.000，但 914 ms 且需要旧数据。
- **事实级热更新：PSS 胜在操作成本、回滚、审计；零遗忘不是独有。**

### 2.27 多源事实 + 规则热更新

| store | accuracy vs priority resolver |
|---|---:|
| Multi-source + priority | **1.000** |
| Flat last-write | 0.800 |

- 273 个冲突；cascade delete 30 cleared / 270 kept / 30 restored，44.68 ms。
- 规则热更新：37 条 derived query 改变；direct facts 全保留；rule edit 0.029 ms；rollback exact。
- **知识层与逻辑层可以分别热更新，事实不被规则更新污染。**

### 2.28 近似 active-set 检索

| noise sigma | active ANN | full-history ANN | active p50 | full p50 |
|---:|---:|---:|---:|---:|
| 0.0 | 1.000 | 1.000 | 0.0056 ms | 0.0118 ms |
| 0.1 | 1.000 | 1.000 | 0.0057 ms | 0.0093 ms |
| 0.3 | **0.920** | 0.495 | 0.0058 ms | 0.0096 ms |
| 0.5 | **0.700** | 0.150 | 0.0058 ms | 0.0095 ms |

- active-set ANN 对噪声更稳、成本更低；不是只能精确 lookup。
- full-history ANN 在噪声下明显退化。

### 2.29 长历史 scaling：full attention 掉，RAG 保住准确率但慢

| history | full attention acc | RAG acc | full p50 | RAG p50 | PSS p50 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.997 | 0.990 | 0.60 ms | 0.93 ms | 0.0002 ms |
| 100 | 0.633 | 0.982 | 1.03 ms | 0.56 ms | 0.0002 ms |
| 500 | 0.151 | 0.982 | 2.54 ms | 1.21 ms | 0.0002 ms |
| 1000 | 0.120 | **0.971** | 4.90 ms | 0.77 ms | 0.0002 ms |

- Full attention 长历史准确率崩、延迟涨。
- RAG 准确率保持 0.971–0.990，是强基线；但每次仍扫描全 history，延迟 0.6–1.2 ms。
- PSS active-set 1.000 / 0.0002 ms；优势是检索成本与精确性，不是唯一能保准确率。
- **强基线没有消除 active-set 的成本优势。**

| history | transformer accuracy | transformer p50/p95 | PSS accuracy | PSS p50/p95 |
|---:|---:|---:|---:|---:|
| 10 | 0.883 | 1.81 / 1.87 ms | **1.000** | **0.0003 / 0.0014 ms** |
| 100 | 0.281 | 1.94 / 2.27 ms | **1.000** | **0.0003 / 0.0014 ms** |
| 500 | 0.128 | 3.49 / 3.75 ms | **1.000** | **0.0003 / 0.0014 ms** |
| 1000 | 0.076 | 8.40 / 8.44 ms | **1.000** | **0.0003 / 0.0014 ms** |

- Transformer 在 10 步历史训练，长历史准确率从 0.883 掉到 0.076；延迟 1.81→8.40 ms。
- PSS active-set 始终 1.000、0.0003 ms；延迟不随历史长度变化。
- 注意：transformer 是短历史训练、长历史测试；强长上下文模型可能更好，但计算增长问题仍在。

### 2.27 Object-centric attention：追平 graph matcher，未超 sparse

| method | overall | near-identical | params |
|---|---:|---:|---:|
| Object attention | 0.987 ± 0.011 | 0.956 | 383k |
| PSS graph+tactile（ref） | **0.989** | 0.956 | 169k |
| sparse k=1 transformer（ref） | **0.996** | 0.991 | - |

- Object attention 用 typed spatial/semantic/tactile bias + active mask，基本追平 graph matcher。
- 但参数更多、没有超过 sparse_k1；near-identical 仍停在 0.956。
- 最小实现证明“object-level attention 可工作”，但没有证明它是更强的架构。
- 下一步需要关系监督、更多 head 类型、或与外部知识/规划结合。

### 2.28 公平归因：tactile 解释大部分 re-ID 优势

| token set | PSS graph | transformer | delta |
|---|---:|---:|---:|
| base | 0.930 ± 0.029 | 0.802 ± 0.008 | **+0.128** |
| tactile | 0.989 ± 0.017 | **0.977 ± 0.017** | **+0.012** |

- 给 transformer 同样的 tactile token + 同样的 Hungarian 后，差距从 **+0.128 缩到 +0.012**。
- +0.012 在约 1 个标准差内，不能声称架构优势。
- **re-ID 的优势主要来自 tactile 输入，不是 graph matcher 架构。**
- 这是对之前结论最重要的修正。

### 2.24 no_assignment 复跑：base 更好，tactile 更差

| 配置 | overall | near-identical |
|---|---:|---:|
| base + assignment | 0.930 ± 0.029 | 0.738 |
| base + no_assignment | **0.969 ± 0.033** | **0.902** |
| tactile + assignment | **0.989 ± 0.017** | **0.956** |
| tactile + no_assignment | 0.976 ± 0.024 | 0.902 |

- no_assignment 在 base 上更好，但在 tactile 上反而更差。
- assignment loss 与 tactile 有交互，不能统一去掉。
- 主配置仍保留 tactile + assignment；no_assignment 只作为 base 消融/备选。
- P1b tracking：noassign tactile 0.994 / 0.01 / 0.989，低于 full tactile 1.000 / 0.00 / 1.000。

### 2.25 graph matcher scaling：N=1000 达标，N=200 差一点

P4 只测了原型识别。这里测关联层：候选剪枝 + 连通分量内 graph matcher + 分量 Hungarian。

| N | dense p50/p95 | pruned p50/p95 | transformer p50/p95 | speedup | pruned true retention |
|---:|---:|---:|---:|---:|---:|
| 50 | 2.99 / 3.77 ms | 2.63 / 4.77 ms | 4.94 / 7.13 ms | 1.88x | 1.000 |
| 200 | 41.70 / 54.82 ms | 10.37 / 16.90 ms | 36.66 / 38.62 ms | 3.53x | 0.970 |
| 1000 | n/a | 34.99 / 43.14 ms | 852.19 / 917.49 ms | **24.35x** | **0.984** |

v2 上 dense graph ID retention 0.986，pruned 0.978，相对 retention **0.992**。

结论：

- **N=1000 达标**：24.35x，true retention 0.984。
- **N=200 未达 5x**：3.53x，是 near miss。
- v2 相对 retention 0.992 > 0.98；绝对 0.978，差 0.002。
- 剪枝把 O(N²) pair tensor 换成局部候选图；代价是约 0.8% 相对 retention。
- 在拥挤合成场景里 dense 全局 Hungarian 会被 foil 干扰，pruned 的 true retention 反而更高。

---

## 3. 这些结果说明了什么

### 3.1 能得出的结论

1. **PSS 不是全面优于 transformer。**
   - 闭集识别打平；
   - 但集合级 re-ID 的优势在公平归因后只剩 +0.012，主要来自 tactile 输入；
   - 模态冲突下纯精度各有胜负。

2. **PSS 的优势集中在“显式结构”能发挥作用的地方：**
   - 完整 online tracking 的系统级结果（PSS+tactile 1.000 vs 无 tactile transformer 0.908）；但公平归因后 re-ID 精度优势主要是 tactile 输入（0.989 vs 0.977），不能归因给架构；
   - 规划/任务成功（P1.3 OOD 1.000 vs 0.924；P5b 关系组合 1.000 vs 0.434）；
   - 开放集拒识；
   - 冲突检测与弃权；
   - 增量更新：快 24x、无需旧数据、可回滚；但 replay 基线也能做到零遗忘；
   - 多实体 active-set 计算；
   - graph matcher 关联层在 N=1000 快 24x，N=200 快 3.5x。

3. **Transformer 的强项是闭集分类和部分冲突鲁棒性：**
   - 闭集分类；
   - geometry 冲突下的鲁棒性；
   - 但需要显式弃权、开放集拒识、增量更新时，PSS 更合适。

4. **PSS 的价值不是“更准”，而是“更可控”：**
   - 知道自己不知道；
   - 知道自己被打架；
   - 新知识可追加、可回滚；
   - 计算随激活实体数而不是总实体数增长。

### 3.2 不能得出的结论

- 不能说 PSS 在所有场景都优于 transformer；
- 不能说 PSS 的闭集识别精度更高；
- 不能说 graph matcher 架构带来 re-ID 精度优势：公平归因后只剩 +0.012；
- 不能说 PSS 的组合属性 OOD 更强（P5 属性 OOD delta 只有 +0.011）；P5b 关系组合另论（1.000 vs 0.434）；
- 不能把结论扩展到预训练 VLM：本轮 transformer 是小型、从零训练的基线；
- 不能把合成数据结论直接外推到真实机器人。

### 3.3 一个重要的公平性提醒

这些结论大多是对**小型、从零训练的 transformer**得出的；唯一公平的 tactile 对照显示，transformer+tactile 已到 0.977。  
真实大模型可能：

- 闭集识别更强；
- OOD 也可能更强（如果做过 outlier exposure）；
- 但训练/更新成本更高，且仍然缺少显式回滚和审计。

---

## 4. PSS 目前的问题

### 4.1 手工融合/关联仍然脆弱

- 手写 re-ID 在 v1 上只有 0.561；learned metric 才到 0.809。
- 固定加权融合会被高权重冲突模态拖累（geometry shuffle 0.458 vs transformer 0.749）。
- `drop` 规则无效，因为它是按“融合 top-1”决定丢哪一路，融合已经错时会保留错误模态。

### 4.2 set-level 匹配没有证明架构优势

- graph matcher 把 v2 从 0.809/0.802 提到 0.930，这是对**无 tactile transformer** 的历史对照。
- 公平归因：PSS+tactile 0.989 vs transformer+tactile **0.977**，delta 只有 +0.012。
- no_edges 0.941、no_assignment 0.969、greedy 0.967、full 0.930：边消息/assignment loss 不是收益来源。
- 结论只在 `tracking_bench_v2` 上成立，且架构增量不显著。

### 4.3 实例级信号仍主要靠 tactile

- DINOv2 CLS 在原始 384 维下反而有害（0.903/0.658），PCA-32 后到 0.942/0.804，仍低于 tactile 0.989/0.956。
- 这说明当前数据的实例信息主要来自触觉纹理；DINO 在 96×96 合成渲染上不够。

- near-identical 从 0.738 提到 0.956，说明 tactile instance texture 有效。
- 当前 tactile 是仿真纹理，不是真实触觉阵列；真实传感器的噪声/漂移未验证。
- DINOv2 已接过 re-ID 可选路径，但 raw 384 有害、PCA-32 仍低于 tactile；没有进默认主链。

### 4.4 闭集精度没有优势

- 0.979 vs 0.978，只是打平。
- PSS 没有证明“多模态融合能带来更高识别精度”。

### 4.5 评测范围仍窄

- 已补完整 online tracking、P5 属性 OOD、P1.3 规划对照。
- 规划对照是符号级 5 路点微基准，不是 PyBullet 闭环；没有测重规划次数和真实执行误差。
- 没接真实数据 / R-Gate。
- Transformer 基线没有预训练 VLM、没有 sparse attention、没有 replay/EWC。

### 4.6 P5 没有证明组合属性 OOD 优势

- PSS 0.978 vs transformer 0.967，delta +0.011，低于预注册 0.10。
- 当前 P5 只是高 hardness / 材质 regime，不是真正的“物体-关系-动作”新组合。

### 4.7 增量更新依赖原型质量

- PSS 新类原型直接由样本均值得到，坏样本会污染原型。
- 没有 outlier rejection、drift detection、原型重估。

---

## 5. 下一步优化路线

### P0：结构短板（已做）

1. **Set-level / graph matching（已做，但归因被修正）**
   - graph matcher 参与 tracker 匹配；公平归因后架构增量只有 +0.012；
   - 系统级 PSS+tactile 仍最好，但优势主要来自 tactile。

2. **Conflict-aware fusion（已做）**
   - reliable 规则：vision/geometry shuffle 下只回答 5–7%，accepted accuracy 1.000；
   - 还没有把“请求主动感知”接到真实控制回路。

3. **实例级特征（部分做）**
   - 已加 tactile token，near-identical 0.956；
   - DINOv2 CLS 已接入并测试：raw 384 反而有害，PCA-32 后 DINO-only 0.942 / near 0.804，仍低于 tactile；没有进默认主链。

### P1：评测范围（已做）

4. **完整 online tracking（P1b，已做）**
   - 60 序列、多帧、dormant/reappearance；
   - PSS+tactile 1.000 vs 无 tactile transformer 0.908；公平 tracking 对照待补。

5. **组合 OOD（P5，已做，未过门槛）**
   - PSS 0.978 vs transformer 0.967，delta +0.011；
   - 真正的“物体-关系-动作”新组合仍待测。

6. **规划/任务成功率（P1.3，已做符号级）**
   - OOD 1.000 vs 0.924，dynamic OOD 1.000 vs 0.928；
   - 还没接 PyBullet 闭环和真实执行误差。

### P2：真实数据

7. **RD2 / R-Gate**
   - 真实同步视/触/位姿数据；
   - 决定触觉专家/投影是否需要真实重训；
   - 这是最大的外部有效性缺口。

8. **真实 VLM 基线**
   - 用真实预训练 VLM 重跑开放集/冲突/增量；
   - 避免“小 transformer”结论被误读。

### P3：工程化

9. **原型质量治理**
   - 新类样本 outlier rejection；
   - 原型方差/漂移监控；
   - 审核门 + 回滚（Phase 9 已有基础设施）。

10. **稀疏/结构化 transformer 对照**
    - P4 只对比了 dense full attention；
    - 加 sparse/window/retrieval transformer，检验 scaling 优势是否仍成立。

---

## 6. 一句话总结

> **PSS 目前证明的不是“通用更准”，也不是“graph matcher 架构更准”。公平归因后，re-ID 精度优势主要来自 tactile；同 tactile 的 sparse_k1 transformer 甚至到 0.996，反超 PSS 0.989。PSS 真正剩下的架构级优势是显式关系组合（P5b 1.000 vs 0.434）、开放集拒识、冲突弃权和快速可审计更新。零遗忘不是独有（replay 可追平）；闭集识别和属性 OOD 与 transformer 接近。**

下一步最值得做的不是继续堆模块，而是：

1. 补 transformer+tactile 的完整 tracking 基线，闭合 re-ID 的公平归因；
2. 接真实数据 / R-Gate；
3. 用真实预训练 VLM 重跑开放集/冲突/增量；
4. 把 P5b 推进到更强关系基线、更多物体和真实机器人。

---

## 7. 相关文件

- 实验协议：`PSS_VS_TRANSFORMER_EXPERIMENT.md`
- 分层认知架构定位：`PSS-认知层架构定位.md`
- 后续优化清单：`PSS-优化任务清单.md`
- 工作总结：`公共语义空间-工作总结-v1.md`
- P1 结果：
  - `eval/results/pss_vs_transformer/p1_tracking.md`
  - `eval/results/pss_vs_transformer_v2/p1_tracking.md`
- P0.3 结果：`eval/results/pss_vs_transformer_v2/p0_tactile.md`
- P1b 结果：`eval/results/pss_vs_transformer_v2/p1b_tracking_tactile.md`
- 公平归因：`eval/results/pss_vs_transformer_v2/fair_attribution.md`
- no_assignment：`eval/results/pss_vs_transformer_v2/noassign_main.md`
- P5b 执行：`eval/results/pss_vs_transformer/physical/p5b_closed_loop.md`
- P1.3 结果：`eval/results/pss_vs_transformer_v2/p13_planning.md`
- P0-3 结果：`eval/results/pss_vs_transformer_v2/p0_scaling.md`
- P5 结果：`eval/results/pss_vs_transformer/p5_compositional.md`
- P2 结果：`eval/results/pss_vs_transformer/p2_openset.md`
- P3 结果：`eval/results/pss_vs_transformer/p3_incremental.md`
- P4 结果：`eval/results/pss_vs_transformer/p4_scaling.md`
- 模态冲突：`eval/results/pss_vs_transformer/modality_conflict.md`
- 数据：
  - `data/tracking_bench_v1/`
  - `data/tracking_bench_v2/`
  - `data/category_pretrain/`
