# PSS ContactEvent C2：多通道融合与校准（go/no-go）

> 日期：2026-11-29
> 计划：`docs/plans/PSS-ContactEvent-计划.md` §3、§6.3–6.4、§7、§8（C2）、§9
> 状态：已执行；C2 go/no-go = **H 不成立，不默认化多通道融合**；oracle 修正后 H' 仍不成立（见 §12）
> 代码：`pss/events/fusion.py`、`pss/events/arms_fusion.py`、`pss/events/sensor_models.py`（未改 C0 文件与 C1 的 `arms_channels.py`）
> 结果：`eval/results/contact_event/c2/`（clean force，上限对照）、`eval/results/contact_event/c2_noisy/`（non-oracle force）

---

## 0. 结论摘要

- **预注册 H**：`fusion` 的 event F1 ≥ 最好单通道 F1 + 0.05。
- **实测**：`fusion` F1 = **1.000 ± 0.000**；最好单通道 = **1.000 ± 0.000**（`force` 与 `geometry` 并列）；Δ = **0.000**。
- **H 判定：不成立**（需要 ≥1.05，而 F1 上限为 1.0，任何臂都不可能达到）。
- 原因不是融合实现失败，而是 **`force` 臂在本 sim 中就是 GT 的单调重表达**：
  GT 用 `normal_force ≥ 1.0 N`；`force` 通道 score = `F/(F+1)`，阈值 0.5 ⇔ `F ≥ 1.0 N`。
  两者使用同一分割参数，因此 `force` F1 必然为 1.0。这是计划 §10 风险 2 的实测确认。
- 按停止规则：**`force` 已 ≥0.95 且 fusion 无 F1 增量 → 照实写“融合在当前 sim 的 F1 上无价值”**，保留单通道。
- 但 fusion 并非处处无价值：在**事件边界**上，fusion 的 offset 误差 **6.7 ms**，显著优于 `geometry` 的 **680.7 ms** 与 `tactile` 的 **666.0 ms**，并与 `force`（0.0 ms）持平。这说明融合能学到“何时结束接触”，但这是**对 force 的复现**，不是对 force 的增量。
- **go/no-go**：不把多通道融合默认化；不以当前结果宣称多模态收益。C3 必须在 **force 退化/缺失、遮挡、冲突** 的设定下重测，否则 F1 仍会被 oracle 化的 force 封顶。

### 0.1 oracle 修正（§12 摘要）

- 新增 `NoisyForceExtractor`（σ=10% 相对噪声、延迟 1–2 帧、0.1 N 量化、5% 丢帧），GT 仍是 clean MuJoCo force。
- `force_noisy` F1 = **0.771 ± 0.021**（P=0.729，R=1.000，14 个 FP）；`fusion_noisy` F1 = **1.000 ± 0.000**。
- **H'（fusion_noisy ≥ 最好单通道 +0.05）按字面仍不成立**：clean `force` 与 clean `geometry` 仍是 1.000，F1 上限 1.0。
- 但融合相对 `force_noisy` 的增量是 **+0.229**，并把它 14 个 FP 压到 0；offset 误差 8.0 ms。失败原因是**指标/对照仍有 clean oracle 与 geometry 封顶**，不是噪声太弱、延迟、权重或通道不足。
- 事后严格 IoU≥0.5 诊断：`force_noisy` 0.771、`geometry` 0.667、`tactile` 0.556、`fusion_noisy` 1.000 → fusion_noisy 相对最好单通道 +0.229。该诊断**不是**预注册 H'。

---

## 1. 预注册与判定口径

计划 §8 C2 出口与 §9 停止规则：

| 项 | 内容 |
|---|---|
| 主指标 | event F1（同 pair + 时间 IoU ≥0.3 或 onset 差 ≤50 ms） |
| bar | `fusion` ≥ 最好单通道 + **0.05** |
| 停止规则 | `force_only` ≥0.95 且 fusion 无增量 → 融合没有价值，报告并保留单通道 |
| 调参约束 | likelihood/权重/阈值只在 train/cal 上选；test 只报一次 |

判定使用规定命令的三 seed 聚合；另外单独报告留出 seed 2，二者结论一致。

---

## 2. 实现

### 2.1 融合模型

对每个 pair，用 `make_channel` 取各通道 score 序列 `s_c(t) ∈ [0,1]`，融合为：

```text
p(contact | s) = sigmoid(b + Σ_c w_c · (logit(s_c) − μ_c) / σ_c)
```

- 特征 = 各通道 logit(score)，按训练集均值/方差标准化；
- `w_c`、`b` 在 train 上用 L2 正则 logistic（IRLS）拟合；
- 这等价于 Bayesian 形式 `p ∝ prior · ∏ LR_c` 的可校准权重版本：`b` 是先验 log-odds，`w_c` 是通道 reliability；
- 通道 score 先 clip 到 `[1e-4, 1−1e-4]`，避免 `logit(0)` 与 tactile 饱和值。

### 2.2 数据划分与校准

| 阶段 | seed | 用途 |
|---|---:|---|
| train | 0 | 拟合 `w_c, b`、特征标准化 |
| cal | 1 | 选 L2 强度 λ、事件阈值、abstain 阈值 |
| test | 2 | 只报告，不参与任何拟合/选择 |

- λ 网格：`{0.001, 0.01, 0.1, 1, 10, 100}`，按 cal frame log-loss 选择；
- 事件阈值：在 cal 上扫 `0.05..0.95` 最大化 event F1；
- `fusion_abstain`：在 cal 上再扫 abstain 阈值，丢弃峰值概率低于阈值的低置信事件。

### 2.3 事件构造

`fuse_events` 对每个 pair：

1. 计算融合概率序列；
2. 用 `segment_signal`（阈值 + hysteresis + min_duration + max_gap）得到区间；
3. 区间内峰值帧作为事件峰值：`confidence = p_peak`；
4. `intensity` 优先取 force 峰值力，否则取 tactile 峰值；`location` 取 force 接触点；
5. 若峰值帧 force 或 tactile 的原始 score ≥0.5 → `confirmed`，否则 `predicted`；
6. `evidence` 保留每个通道的 `ContactEvidence`（score/intensity/location/reliability）。

---

## 3. 运行

规定命令（**完整渲染跑，未缩减**，耗时约 18 s）：

```bash
python -u -m pss.eval.scripts.run_contact_event_bench \
  --scenarios drop collide near_miss --seeds 0 1 2 --render \
  --arms force tactile geometry vision audio fusion \
  --out eval/results/contact_event/c2
```

校准命令（train 0 / cal 1，test 2 仅报告 frame 指标）：

```bash
python -u -c "from pss.events.fusion import main; main()" \
  --train-seeds 0 --cal-seed 1 --test-seed 2 \
  --out eval/results/contact_event/c2/fusion_params.json
```

补充：

- 留出 test：`--seeds 2 --render --arms force tactile geometry vision audio fusion fusion_abstain` → `eval/results/contact_event/c2/test_seed2/`
- 三 seed 的 abstain 对照：`--arms force geometry fusion fusion_abstain` → `eval/results/contact_event/c2/abstain_3seed/`

配置：96×72，400 steps，capture_every=2，`threshold=0.5`，`gt_threshold=1.0`，`min_duration=0.02`，`max_gap=0.05`，hysteresis=0。

---

## 4. 校准结果

### 4.1 融合参数（train 0，cal 1）

| 通道 | 权重 `w_c` | reliability `|w|/max|w|` |
|---|---:|---:|
| force | **+1.0816** | 1.000 |
| tactile | ≈ 0（−2.2e−16） | ≈ 0 |
| geometry | **−0.2222** | 0.205 |
| vision | ≈ 0（−3.9e−15） | ≈ 0 |
| audio | +0.0461 | 0.043 |

- intercept `b = +6.1444`；λ = **10.0**；
- 事件阈值 = **0.05**；abstain 阈值 = **0.05**（因此 `fusion_abstain` 与 `fusion` 输出相同）；
- geometry 权重为负是校准修正：在这些场景里 geometry 在“靠近但未达到 GT 力阈值”的静置/近失阶段长期高响应（见 §6），负权重压低这些帧。只有 6 个 GT 事件，权重维度很低，**不要过度解读符号与大小**。

### 4.2 λ 选择（cal frame log-loss）

| λ | cal log-loss |
|---:|---:|
| 0.001 | 0.07251 |
| 0.01 | 0.04861 |
| 0.1 | 0.03022 |
| 1 | 0.01999 |
| **10** | **0.01509** |
| 100 | 0.01962 |

### 4.3 帧级校准

| split | n | log-loss | Brier | ECE | 正例率 |
|---|---:|---:|---:|---:|---:|
| train (seed 0) | 1200 | 0.00572 | 0.00132 | 0.00223 | 2.92% |
| cal (seed 1) | 1200 | 0.01045 | 0.00301 | 0.00536 | 3.67% |
| test (seed 2) | 1200 | 0.01480 | 0.00477 | **0.00642** | 4.08% |

test ECE = 0.006 ≤ 0.05，满足计划 §7.1 的校准 bar；但这是 force 主导下的 easy regime，不能外推到退化 force 的场景。

**阈值不可辨识**：cal 上 event F1 在整个 `0.05..0.95` 网格几乎恒为 1.0，扫描按“首个最大值”落到 0.05。事后检查（未用于调参）：drop seed 0 中真实冲击后融合概率在 0.352 s 处仍有 0.0499，刚好低于 0.05，因此没有产生额外事件；若阈值取 0.5，offset 会略短，但 F1 仍为 1.0。这是当前数据“正例极少、force 近 oracle”导致的阈值不可辨识，不是调参选择。

---

## 5. 主表：规定三 seed 聚合（seed 0/1/2，render 开）

| arm | Precision | Recall | **F1** | onset latency (s) | intensity MAE | tp/fp/fn |
|---|---:|---:|---:|---:|---:|---|
| `force` | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.000** | 0.000 | 0.000 | 6/0/0 |
| `geometry` | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.000** | 0.0127 | 0.936 | 6/0/0 |
| `fusion` | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.000** | 0.000 | 0.000 | 6/0/0 |
| `tactile` | 0.833 ± 0.000 | 1.000 ± 0.000 | 0.889 ± 0.000 | 0.000 | 0.936 | 6/3/0 |
| `audio` | 1.000 ± 0.000 | 0.333 ± 0.000 | 0.333 ± 0.000 | — | — | 0/0/6 |
| `vision` | 1.000 ± 0.000 | 0.333 ± 0.000 | 0.333 ± 0.000 | — | — | 0/0/6 |

注：

- P/R/F1 是先按场景计算再对 3 个场景求均值；`near_miss` 的“空预测 vs 空 GT”按实现约定记为 P=R=F1=1.0，会抬高所有臂的均值。
- tp/fp/fn 为三 seed 汇总；GT 共 6 个事件（drop/collide 各 3 个），`near_miss` 0 个。
- seed 0/1 分别是 train/cal，因此三 seed 聚合不是严格留出；严格留出看 §6。

---

## 6. 留出 test（seed 2）与补充诊断

### 6.1 留出 test 表

| arm | Precision | Recall | **F1** | tp/fp/fn |
|---|---:|---:|---:|---|
| `force` | 1.000 | 1.000 | **1.000** | 2/0/0 |
| `geometry` | 1.000 | 1.000 | **1.000** | 2/0/0 |
| `fusion` | 1.000 | 1.000 | **1.000** | 2/0/0 |
| `fusion_abstain` | 1.000 | 1.000 | **1.000** | 2/0/0 |
| `tactile` | 0.833 | 1.000 | 0.889 | 2/1/0 |
| `audio` | 1.000 | 0.333 | 0.333 | 0/0/2 |
| `vision` | 1.000 | 0.333 | 0.333 | 0/0/2 |

### 6.2 event F1 对“过长区间”不敏感

seed 0 的原始区间（秒）：

| 来源 | drop | collide |
|---|---|---|
| GT | 0.244–0.280 | 0.084–0.180 |
| `force` | 0.244–0.280 | 0.084–0.180 |
| `geometry` | 0.228–**1.600** | 0.080–**0.212** |
| `tactile` | 0.244–**1.600** | 0.084–0.200，**0.612–1.600（额外）** |
| `fusion` | 0.244–0.288 | 0.084–0.188 |

`geometry`/`tactile` 在 drop 中把“球停在盒子上但法向力低于 1 N”的整个静置段算成同一事件，offset 拖到 1.6 s，但按 IoU/onset 匹配**仍记为 TP**，所以 F1 仍为 1.0。这说明主指标对 offset 不敏感；计划的 offset latency 指标更严格。

### 6.3 offset 误差（drop+collide，6 个 GT）

| arm | onset 误差 | offset 误差 | 平均事件时长 | GT 平均时长 |
|---|---:|---:|---:|---:|
| `force` | 0.0 ms | 0.0 ms | 0.081 s | 0.081 s |
| `geometry` | 12.7 ms | **680.7 ms** | 0.775 s | 0.081 s |
| `tactile` | 0.0 ms | **666.0 ms** | 0.747 s | 0.081 s |
| `fusion` | 0.0 ms | **6.7 ms** | 0.088 s | 0.081 s |

在 §7.1 的 offset ≤100 ms 口径下：`force` 与 `fusion` 通过，`geometry`/`tactile` 超 6–7 倍。**这是 fusion 唯一可测到的增益**，但它是在 force 存在时的“复现”，不是多通道对 force 的超越。

---

## 7. H 判定

| 项 | 值 |
|---|---:|
| fusion event F1（三 seed / test） | 1.000 / 1.000 |
| 最好单通道 F1（三 seed / test） | 1.000 / 1.000（force、geometry 并列） |
| Δ | **0.000** |
| H 要求 | Δ ≥ +0.05 |
| **判定** | **不成立** |

补充说明：F1 的数值上限是 1.0，而 `force` 在本 sim 中是可证明的 GT 重表达，所以 **H 在该设定下先验不可满足**；这不是融合方法失败，而是 C2 的对照设计让“最好单通道”成为 oracle。若要在 F1 上检验融合，必须让 force 通道退化（噪声、带宽、丢帧、遮挡）或从臂集合去掉 force。

---

## 8. 冲突 / 失败 / 边界

1. **force 近 oracle，且是构造性的**：`F/(F+1)` 阈值 0.5 ⇔ GT 的 `F ≥ 1.0 N`，同一 pair、同一分割参数 ⇒ F1 恒 1.0。H 的上限被结构性封死。
2. **geometry/tactile 的 F1 被“过长事件”高估**：drop 中 offset 拖到 1.6 s 仍算 TP。主指标应同时看 offset；本报告用 §6.3 补上。
3. **tactile 饱和**：`SensorRig.tactile_maps()` 对每个物体按最大值归一化，导致任意非桌面接触后 score 恒为 1.0（`s/(s+1)=0.5`），丢失强度；三 seed 共 3 个 FP（collide 之外的额外接触段）。融合学到的 tactile 权重 ≈0，是正确反应。
4. **audio 实际失效**：`AudioExtractor` 是 `|Δforce|/max` 的单帧尖峰，`min_duration=0.02 s`（5 帧）过滤后输出 0 个事件，recall=0。融合给了一个很小的正权重，但在当前数据上没有可测作用。它是合成代理，不能声称声学性能。
5. **vision 是占位**：恒为 0，权重 ≈0；不能据此评价视觉通道。
6. **阈值不可辨识（§4.3）**：cal event F1 在网格上恒定，阈值落到 0.05；这是数据退化，不是调参。报告保留该冻结选择，未用 test 重选。
7. **负 geometry 权重**：是对“靠近即高响应”的校准修正，但样本只有 6 个 GT 事件，权重不稳定；不应把 `w_c` 当作物理 reliability。
8. **指标口径**：`near_miss` 的空-空匹配按 1.0 计；3 个场景的均值会抬高所有臂。真正有区分度的只有 drop/collide 各 3 个事件。
9. **train/cal 进入三 seed 聚合**：严格留出结论以 seed 2 为准；两种口径下 H 均不成立。
10. **sim 边界**：MuJoCo 图元是 proxy；force/tactile 近 oracle；audio 合成；不声称真实机器人或 sim2real。

---

## 9. go/no-go 与下一步

**C2 决策：no-go（对“多通道融合默认化”）。**

- 不把 `fusion` 设为 runtime 默认接触层；
- 保留 `force` / `geometry` 单通道作为 fallback；
- 不以本次结果宣称多模态融合收益；
- `fusion_abstain` 在当前校准下与 `fusion` 完全相同（abstain 阈值 = 事件阈值），没有额外价值。

**可在 C3 继续验证的方向（必须先改变对照设定）：**

1. **退化 force**：加噪声、降带宽、丢帧、延迟，使 force 不再等价 GT；此时再测 fusion ≥ 单通道 +0.05 才有意义。
2. **遮挡/冲突/近失为主**：geometry 失效、tactile 未覆盖、audio 误检；检验冲突处理率与 false alarm，而不是在 clean sim 上与 oracle 比 F1。
3. **补齐真实可用通道**：vision 目前是占位；audio 需要至少多帧冲击包络，否则会被 min_duration 过滤。
4. **指标**：同时报告 event F1、onset/offset 误差、IoU/时长误差与 false alarm；只用 F1 会奖励过长的 geometry/tactile 事件。

---

## 10. 复现与产物

| 路径 | 内容 |
|---|---|
| `pss/events/fusion.py` | 融合模型、校准、事件构造、校准 CLI（支持 force extractor 覆盖） |
| `pss/events/arms_fusion.py` | 注册 `fusion` / `fusion_abstain` / `force_noisy` / `fusion_noisy` 臂 |
| `pss/events/sensor_models.py` | `NoisyForceExtractor`（噪声/延迟/量化/丢帧） |
| `eval/results/contact_event/c2/fusion_params.json` | clean fusion 的冻结参数 |
| `eval/results/contact_event/c2/fusion_calibration.json` | clean λ/阈值选择、train/cal/test 帧指标 |
| `eval/results/contact_event/c2/contact_event.json` | clean 三 seed 主结果（上限对照） |
| `eval/results/contact_event/c2/CONTACT_EVENT_REPORT.md` | 基准脚本自动生成的表 |
| `eval/results/contact_event/c2/test_seed2/` | clean 留出 seed 2 结果 |
| `eval/results/contact_event/c2/abstain_3seed/` | `fusion_abstain` 三 seed 对照 |
| `eval/results/contact_event/c2_noisy/fusion_noisy_params.json` | noisy fusion 冻结参数（最终：f1_offset tie-break） |
| `eval/results/contact_event/c2_noisy/fusion_calibration.json` | noisy λ/阈值选择、train/cal/test 帧指标 |
| `eval/results/contact_event/c2_noisy/contact_event.json` | noisy 三 seed 主结果 |
| `eval/results/contact_event/c2_noisy/test_seed2/` | noisy 留出 seed 2 结果 |
| `eval/results/contact_event/c2_noisy/f1_tiebreak/` | 首次 f1-only tie-break（阈值 0.05）的冻结结果 |

未修改 `pss/events/model.py`、`extractors.py`、`segmenter.py`、`bench_api.py`、`scenarios.py`、`gt.py`、`pss/eval/scripts/run_contact_event_bench.py`，也未修改 C1 的 `pss/events/arms_channels.py`（该文件在当前工作树中已由 C1 落地，注册了 `force_tuned` 等额外臂；本报告未使用它们）。

---

## 12. Oracle 修正：noisy force 与 H'

### 12.1 动机与传感器模型

clean `force` 是 GT 的单调重表达，H 在该对照下不可满足（§7）。为此新增 `NoisyForceExtractor`（`pss/events/sensor_models.py`），只退化通道、不退化 GT：

| 参数 | 值 | 说明 |
|---|---|---|
| 相对高斯噪声 σ | **10%** | `F_noisy = F · (1 + N(0, 0.1))`，clip ≥0 |
| 延迟 | **1–2 帧** | 右移；seed 0→1 帧，seed 1→2 帧，seed 2→1 帧 |
| 量化 | **0.1 N** | `round(F/0.1)·0.1` |
| 丢帧 | **5%** | Bernoulli，丢帧读数记 0 |
| GT | **clean** | 仍是 MuJoCo `normal_force ≥ 1 N` |

噪声 RNG 由 `(seed, pair)` 稳定哈希，校准与评测看到同一传感器实现。

注册的臂：

- `force_noisy`：只用含噪 force 单通道；
- `fusion_noisy`：`force_noisy` + `geometry` + `tactile`。**不含 audio**：`AudioExtractor` 内部用 clean force 合成，若加入会把 oracle 泄漏进融合；vision 是恒零占位。

clean `force` / `fusion` 仍保留在跑批里，仅作为**上限对照**，不参与 H' 的“最好单通道”比较（见 §12.4 的说明）。

### 12.2 校准

划分与 clean 一致：train seed 0、cal seed 1、test seed 2（只报告）。通道 = `force/tactile/geometry`。

| 项 | 值 |
|---|---|
| 选中 λ | **100.0**（cal log-loss 0.1159；λ≤10 均在 0.119–0.121） |
| 权重 | force **+0.177**、tactile **+0.082**、geometry **−0.128** |
| intercept | **−0.827** |
| 事件阈值 | **0.15**（见下方 tie-break） |
| abstain 阈值 | 0.15（与事件阈值相同） |

帧级校准：

| split | n | log-loss | Brier | ECE | 正例率 |
|---|---:|---:|---:|---:|---:|
| train (seed 0) | 1200 | 0.0871 | 0.0196 | 0.0340 | 2.92% |
| cal (seed 1) | 1200 | 0.1135 | 0.0258 | 0.0378 | 3.67% |
| test (seed 2) | 1200 | 0.1069 | 0.0254 | **0.0395** | 4.08% |

test ECE = 0.0395 ≤ 0.05，仍满足校准 bar，但比 clean（0.0064）差一个量级；noisy force 下的概率校准明显更难。

**阈值 tie-break（只在 cal 上做）**：

- 首次按纯 F1 选阈值，cal F1 在整个网格上恒为 1.0，tie-break 落到 **0.05**。该版本把 drop 的冲击与静置段合并，cal offset 误差 **666 ms**；冻结在 `c2_noisy/f1_tiebreak/`。
- 随后改为 **F1 相同则取 offset 误差更小**（`--threshold-tiebreak f1_offset`，仍只用 cal），选中 **0.15**，cal offset 误差 **8 ms**。最终结果使用该版本。
- 这是对“F1 平台期阈值不可辨识”的方法修正，不是用 test 调参；两版 test 结果都在 §12.3 报告。

### 12.3 主表（规定命令，3 seeds，render 开）

```bash
python -u -m pss.eval.scripts.run_contact_event_bench \
  --scenarios drop collide near_miss --seeds 0 1 2 --render \
  --arms force force_noisy geometry tactile fusion fusion_noisy \
  --out eval/results/contact_event/c2_noisy
```

| arm | Precision | Recall | **F1** | onset latency (s) | intensity MAE | tp/fp/fn |
|---|---:|---:|---:|---:|---:|---|
| `force`（clean，上限） | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.000** | 0.000 | 0.000 | 6/0/0 |
| `force_noisy` | 0.729 ± 0.015 | 1.000 ± 0.000 | **0.771 ± 0.021** | 0.0053 | 0.086 | 6/14/0 |
| `geometry` | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.000** | 0.0127 | 0.936 | 6/0/0 |
| `tactile` | 0.833 ± 0.000 | 1.000 ± 0.000 | 0.889 ± 0.000 | 0.000 | 0.936 | 6/3/0 |
| `fusion`（clean，上限） | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.000** | 0.000 | 0.000 | 6/0/0 |
| `fusion_noisy` | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.000** | 0.0053 | 0.159 | 6/0/0 |

留出 seed 2：

| arm | Precision | Recall | **F1** | tp/fp/fn |
|---|---:|---:|---:|---|
| `force` | 1.000 | 1.000 | **1.000** | 2/0/0 |
| `force_noisy` | 0.714 | 1.000 | **0.750** | 2/6/0 |
| `geometry` | 1.000 | 1.000 | **1.000** | 2/0/0 |
| `tactile` | 0.833 | 1.000 | 0.889 | 2/1/0 |
| `fusion` | 1.000 | 1.000 | **1.000** | 2/0/0 |
| `fusion_noisy` | 1.000 | 1.000 | **1.000** | 2/0/0 |

**噪声确实造成了非 oracle 退化**：`force_noisy` 从 clean 的 1.000 掉到 0.771/0.750，14 个 FP 全部来自 drop 的静置段——法向力约 0.86 N，叠加 10% 相对噪声与 0.1 N 量化后反复越过 1 N 阈值（seed 0 的额外区间例如 0.448–0.828、0.908–1.008、1.184–1.564 s）。`fusion_noisy` 将这些 FP 全部压掉，恢复 6/0/0。

### 12.4 H' 判定

预注册 H'：`fusion_noisy` event F1 ≥ 最好单通道（含 `force_noisy`）+ 0.05。

| 比较对象 | 最好单通道 F1 | fusion_noisy F1 | Δ | 是否满足 |
|---|---:|---:|---:|---|
| 全部单通道（含 clean `force`、clean `geometry`） | 1.000 | 1.000 | 0.000 | **否** |
| 非 oracle 单通道（排除 clean `force`，保留 `geometry`） | 1.000（geometry） | 1.000 | 0.000 | **否** |
| 传感器单通道（`force_noisy`、`tactile`；排除 clean oracle 与几何预测器） | 0.889（tactile） | 1.000 | +0.111 | 是（次级判据） |
| 直接对 `force_noisy` | 0.771 | 1.000 | **+0.229** | 是（次级判据） |

**H' 按字面判定：不成立。** 原因不是 fusion_noisy 失败：

1. F1 上限是 1.0，而 clean `force` 是 oracle，clean `geometry` 在标准 event F1 下也仍是 1.000；
2. `geometry` 的 1.000 来自宽松匹配：drop 中 geometry 区间 0.228–1.600 s，offset 误差 681 ms，但 onset 差 16 ms ≤ 50 ms，按基准规则仍记 TP；事件 F1 不惩罚过长区间；
3. 因此“最好单通道”仍被 clean/oracle 或宽松指标封顶，H' 与 clean H 一样在数值上不可满足。

**诊断结论**：不是噪声太弱（force_noisy 已掉到 0.771）、不是延迟问题（onset 5.3 ms）、不是权重拟合失败（fusion_noisy 在 cal/test 都是 1.000）、也不是通道不足（三者已足够）。问题在**对照/指标设计**：标准 event F1 同时奖励了 clean geometry 的过长事件。

**事后严格指标（非预注册，仅诊断）**：把匹配改成纯 IoU ≥ 0.5（去掉 onset 容差），三 seed × 三场景：

| arm | strict P | strict R | **strict F1** | tp/fp/fn | offset 误差 |
|---|---:|---:|---:|---|---:|
| `force` | 1.000 | 1.000 | 1.000 | 6/0/0 | 0.0 ms |
| `force_noisy` | 0.729 | 1.000 | **0.771** | 6/14/0 | 5.3 ms |
| `geometry` | 0.667 | 0.667 | 0.667 | 3/3/3 | 41.3 ms |
| `tactile` | 0.500 | 0.667 | 0.556 | 3/6/3 | 12.0 ms |
| `fusion_noisy` | 1.000 | 1.000 | **1.000** | 6/0/0 | 8.0 ms |

在这个严格口径下，最好非 oracle 单通道是 `force_noisy` 0.771，`fusion_noisy` 1.000，Δ=+0.229，满足 +0.05；但它是**事后诊断**，不能替代预注册 H'。

### 12.5 fusion_noisy 的实际作用与 offset

drop+collide 共 6 个 GT 的 onset/offset 误差：

| arm | onset 误差 | offset 误差 | 平均事件时长 | GT 平均时长 |
|---|---:|---:|---:|---:|
| `force` | 0.0 ms | 0.0 ms | 0.081 s | 0.081 s |
| `force_noisy` | 5.3 ms | 5.3 ms | 0.081 s | 0.081 s |
| `geometry` | 12.7 ms | 680.7 ms | 0.775 s | 0.081 s |
| `tactile` | 0.0 ms | 666.0 ms | 0.747 s | 0.081 s |
| `fusion_noisy` | 5.3 ms | **8.0 ms** | 0.077 s | 0.081 s |

- fusion_noisy 在保留 noisy force 的 onset 延迟（5.3 ms）的同时，把 offset 误差控制在 8.0 ms；
- 首次阈值 0.05 的版本 offset 为 665 ms（把 drop 静置段并入事件），改为 cal 上的 offset tie-break 后修复；
- intensity MAE 0.159（force_noisy 0.086；clean 0.0），因为强度取自含噪 force 峰值，但远好于 geometry/tactile 的 0.936。

### 12.6 失败项与边界

1. **H' 字面不成立**，不能宣称预注册成功；只能宣称“相对 `force_noisy` 的次级比较成立”。
2. **test 被评测了两次**（f1-only 与 f1_offset 两版阈值）。两版都报告；阈值选择只在 cal 上，但这是本轮 oracle 修正的额外迭代，需在后续预注册中避免。
3. **clean `geometry` 仍是标准 F1 的封顶项**：要真正检验 H'，必须同时退化 geometry（或在遮挡/冲突场景中让 geometry 失效），或改用预注册的严格 IoU/offset-aware 指标。
4. **λ=100 收缩很强**，权重数值小；只有 6 个 GT 事件，权重的稳定性有限，不应作物理解读。
5. **ECE 0.0395**（clean 0.0064）：noisy force 下概率校准更差；仍 ≤0.05，但余量很小。
6. **audio 被排除**：合成 audio 来自 clean force，加入 fusion_noisy 会泄漏 oracle；这不是“audio 无用”的结论。
7. **仍非真实传感器**：σ/延迟/量化/丢帧是手工设定，不是实测传感器模型；不声称 sim2real。

### 12.7 结论

**在非 oracle 力传感器下，融合有增益——但 H' 按预注册字面不成立。**

- `force_noisy` F1 0.771（14 FP）→ `fusion_noisy` F1 1.000（0 FP），**+0.229**；onset 5.3 ms、offset 8.0 ms。
- 不成立的部分是“≥ 最好单通道 +0.05”：clean `geometry` 在标准 F1 下仍是 1.000，F1 上限 1.0，bar 不可满足；严格 IoU≥0.5 下 geometry 掉到 0.667、fusion_noisy 1.000，但那是事后诊断。
- 因此 C2 的结论更新为：**oracle 问题已定位并修正了 force 一侧；融合机制在 noisy force 上有效；但当前基准的 F1 对照仍不可用于检验预注册 H'，需在 C3 前先修指标/对照并重新预注册。**

---

## 13. 一句话

**clean force 下 H 不成立：fusion F1 = 1.000，最好单通道 F1 = 1.000，Δ = 0，因为 force 就是 GT 的单调重表达。oracle 修正后（σ=10%、延迟 1–2 帧、0.1 N 量化、5% 丢帧），`force_noisy` F1 = 0.771，`fusion_noisy` F1 = 1.000（+0.229，offset 8.0 ms），但 H' 按字面仍不成立：clean geometry 在宽松 event F1 下仍是 1.000，F1 上限 1.0。结论：融合机制在 non-oracle force 上有效；当前基准的 F1 对照不能检验预注册 H'。C2 go/no-go 仍不默认化多通道融合；C3 前必须先修指标/对照。**
