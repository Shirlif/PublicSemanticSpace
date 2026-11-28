# PSS ContactEvent 报告（C0–C6）：从几何碰撞到多感官事件

> 日期：2026-11-29
> 计划：`docs/plans/PSS-ContactEvent-计划.md`
> 结果：`eval/results/contact_event/{c1,c2,c2_noisy,c3,c3_conflict}/`、`eval/results/contact_active/`
> 状态：C0–C5 已执行；C6 为汇总与 go/no-go。
>
> **2026-11-29 strict 口径更正（主指标）**：事件匹配改为 strict——同 pair + **IoU ≥0.5 + onset ≤50 ms + offset ≤100 ms**；loose（IoU≥0.3 或 onset≤50 ms）只作对照。loose 会把“长区间”算对，strict 下 geometry/tactile/vision 分别掉到 **0.667 / 0.667 / 0.556**，而 fusion/force 仍 **1.000**。以下 C1–C3 数字以 strict 为准。

---

## 0. 一句话

**ContactEvent 层能工作：schema、GT、时序分割、单通道、融合、冲突、主动感知、PSS 接入都跑通了；但它不是“fusion 全面优于单通道”的胜利。** strict 口径下，clean force 仍 1.000（oracle），geometry/tactile/vision 只有 0.667/0.667/0.556；在 noisy force 下 `fusion_noisy` 1.000 vs `force_noisy` 0.771（+0.229）；在 C3 的冲突/近失场景里 `force_noisy` 本身已 1.000，融合只与它打平，但相对 `geometry` 有 +0.600 并压掉 6 个误报。**结论：ContactEvent 是 opt-in 的鲁棒层；fusion 的价值在“geometry 误报/force 退化”的场景，不是普遍精度提升。**

---

## 1. C0：共享基础

- `pss/events/model.py`：`ContactEvent` / `ContactEvidence` / `ContactFrame` / 状态。
- `gt.py`：从 MuJoCo contact force 生成 GT（过滤 table/floor，合并 chatter）。
- `segmenter.py`：onset/offset、hysteresis、gap 合并。
- `extractors.py`：force/tactile/geometry/vision/audio 通道接口。
- `scenarios.py`：drop / collide / near-miss。
- `bench_api.py`：场景/臂注册、事件匹配、P/R/F1、onset/intensity。
- 测试 8 项通过；smoke：force/geometry F1 1.0、tactile 0.889。

---

## 2. C1：单通道 baseline（strict 口径）

3 seeds × drop/collide/near_miss，400 步 @96×72 + render；strict = IoU≥0.5 + onset≤50 ms + offset≤100 ms：

| arm | strict F1 | onset | offset | 说明 |
|---|---:|---:|---:|---|
| `force` / `force_tuned` | 1.000 | 0 ms | 0 ms | clean force = oracle；无提升空间 |
| `geometry` | 0.667 | 9 ms | 41 ms | 长区间/静载接触导致 offset 超限 |
| `tactile` / `tactile_tuned` | 0.667 | 0 ms | 12 ms | 同上；strict 下不再 1.0 |
| `vision_motion` | 0.556 | 8 ms | 52 ms | 仿真 RGB-D 代理；offset 超限 |
| `audio_proxy` | 1.000 | 0 ms | 14 ms | 由 force 派生 + ring-down；循环论证 |

**判读**：strict 下只有 clean force 和 force 派生的 audio 是 1.0；geometry/tactile/vision 都因**事件区间不准**掉到 0.667/0.556。loose 的 1.0 是“长区间 + onset 容差”的 artifact，不再引用。

---

## 3. C2：融合与校准

### 3.1 clean force = oracle（H 不可测）

- GT = `normal_force ≥ 1 N`；force 通道用同一个力，F1 必然 1.0；fusion 无法超过。
- 这是计划 §10 风险 2 的实测确认。

### 3.2 noisy force 修正（关键结果）

噪声模型：相对高斯 σ=10%、延迟 1–2 帧、量化 0.1 N、5% 丢帧；GT 保持 clean。train seed 0 / cal seed 1 / test seed 2。

| arm | F1 | P | R | tp/fp/fn | offset |
|---|---:|---:|---:|---|---:|
| `force`（clean，上限） | 1.000 | 1.000 | 1.000 | 6/0/0 | 0 ms |
| **`force_noisy`** | **0.771 ± 0.021** | 0.729 | 1.000 | 6/14/0 | 5.3 ms |
| `geometry`（loose） | 1.000 | 1.000 | 1.000 | 6/0/0 | 680.7 ms |
| `geometry`（strict IoU≥0.5） | 0.667 | — | — | — | — |
| **`fusion_noisy`** | **1.000 ± 0.000** | 1.000 | 1.000 | 6/0/0 | 8.0 ms |

- 相对 `force_noisy` 的增量 **+0.229**（14 FP → 0）。
- strict IoU≥0.5：force_noisy 0.771、geometry 0.667、tactile 0.556、fusion_noisy 1.000。
- **预注册 H' 字面不成立**：clean geometry 在宽松 F1 下仍 1.000，最好单通道是 1.000；fusion 无法 +0.05。这是**指标/对照 artifact**，不是融合失败。
- 帧级 ECE 0.0064（force 主导的 easy regime）。

---

## 4. C3：遮挡、冲突、近失

5 场景（occlusion / conflict_force_only / conflict_geometry_only / near_miss_hard / near_miss_clear）×3 seeds：

| arm | loose F1 | strict F1 | tp/fp/fn |
|---|---:|---:|---|
| `force` | 1.000 | 1.000 | 6/0/0 |
| **`force_noisy`** | **1.000** | **1.000** | 6/0/0 |
| `geometry` | 0.400 | 0.400 | 3/6/3 |
| **`fusion_noisy`** | **1.000** | **1.000** | 6/0/0 |

冲突解决（4 冲突/近失场景 ×3 seeds）：

| arm | resolved |
|---|---:|
| `force` / `force_noisy` / `fusion_noisy` | 12/12 |
| `geometry` | 3/12 |

- `conflict_force_only`：geometry 0/0/3（3 FN），fusion 3/3 检出。
- `conflict_geometry_only` / `near_miss_hard`：geometry 各 3 FP，fusion 各 0。
- `near_miss_clear`：所有臂 0 FP。
- onset/offset：force 0/0、force_noisy 5.3/6.7、geometry 12.0/41.3、fusion_noisy 5.3/10.0。

**判读**：融合相对 geometry 有 **+0.600** 并压掉 6 个误报；但相对 `force_noisy` 无增量（两者都 12/12、1.000）。C2 的 +0.229 只在 drop/collide 的 strict 口径成立，不能外推到 C3 场景。

---

## 4.5 覆盖扩展：48-cell 因子网格（C2 的 +0.229 是否普遍？）

48 cells（24 coarse × 2 seeds + 24 refine × 3 seeds，共 120 cell-seed runs；60 个物理 group）；strict 口径；`force σ` 0–40%、dropout 0–20%、delay 0–4 帧、geometry mismatch 0–2×eps、occlusion、n_objects 2/4、kind drop/collide/near_miss。详见 `PSS-ContactEvent-Coverage.md`。

### 4.5.1 结果

| 指标 | 值 |
|---|---:|
| `fusion − force_noisy ≥ +0.05` 的 cell | **1/48** |
| 该 cell | drop，σ=0.20，mismatch=1×eps，dropout=0.20，delay=0，n=4，occlusion；2 seeds |
| 该 cell 的 F1 | force **0.000** → fusion **0.667**（+0.667；force 两 seed 全漏检，fusion 两 seed 都检出，各 1 FP） |
| `force_noisy ≥ fusion` 的 cell | **47/48**（43 打平，4 个 drop cell 下降） |
| 最差 Δ（cell 41） | **−0.300**（drop，σ=0.10，mismatch=1.25×eps，dropout=0.20，delay=0） |
| 总体 strict F1 | force_noisy **0.628** / fusion **0.576** / geometry **0.252** |
| fusion − geometry 范围 | max **+1.000**，min **−0.500**（collide，delay=4） |
| near-miss 误报 | geometry **40/40**，force/fusion **0/40** |

### 4.5.2 判读

- **C2 的 +0.229 不是普遍增益**：48-cell 网格里只有 1 个极端 cell 满足 +0.05，其余 47/48 是打平或下降。
- **force_noisy 是默认单通道**：它总体 0.628 > fusion 0.576；fusion 在 4 个 drop cell 反而更差。
- **fusion 的真实价值是 geometry 误报抑制**：near-miss 上 geometry 40/40 误报，force/fusion 0/40；但 force 本身已经 0 误报，所以 fused 的 +0.324 主要是“补 geometry 的短板”，不是“超过 force”。
- **delay=4 是死区**：drop 的 force 0.044、fusion 0.038；4 帧延迟 ≈ 事件长度，strict IoU 下必然失败。需要 latency-aware 对齐，不是融合权重能修。
- **occlusion 是 null factor**：本次没有 vision arm、没有 render，occluder 不影响 force/geometry。
- **阈值校准不稳**：8 个 cal cells 选出 0.10；test 事后最优 0.20 也只有 0.579，仍低于 force 0.611。

---

## 5. C4：主动感知

| condition | passive F1 | active F1 | onset |
|---|---:|---:|---:|
| `fine`（20 ms 采样） | 0.833 | **1.000** | 14.0 → 0.3 ms |
| `coarse`（60 ms 采样） | 0.167 | **1.000** | 36.0 → 0.3 ms |
| `short`（窗口早于事件） | 0.000 | **1.000** | — → 0.3 ms |
| `near_miss` | 0 误报 | 0 误报 | 不触发 |

- 18 次主动动作，每个事件 run 1 次：re-observe 或 wait；**没有物理 push/tap**。
- 代价：+31–39 帧/事件；`short` 额外 +78.7 仿真步。
- **条件性有效**：passive 足够密（4 ms）时 F1 已 1.0，active 只增加代价。

---

## 6. C5：PSS 接入

- `event_to_relation`：`predicate="contact"`、`origin="contact_event"`、`probability=confidence`；事件字段放 `attributes/metadata`，不改 state schema。
- 决策分档：`contradicted/expired` → reject；`conf<0.25` → abstain；`conf<0.5` → review；否则 accept。
- 审计：`contact_event_ingest`；`runtime.rollback` 可恢复；audit append-only。
- 测试：新增 13 项 + 旧回归 43 项 = **56 passed**。
- ContactEvent 保持 opt-in，不设为 runtime 默认。

---

## 7. 假设判定

| 假设 | bar | 实测（strict） | 判定 |
|---|---|---|---|
| H（clean force fusion） | fusion ≥ best single +0.05 | 1.000 vs 1.000 | **不可测**（force = oracle） |
| H'（noisy force fusion） | fusion_noisy ≥ best single +0.05 | C2 drop/collide：1.000 vs `force_noisy` 0.771（+0.229）；但 best single clean force 1.000 | **字面不成立；相对 noisy force 成立** |
| **覆盖 envelope** | fusion ≥ `force_noisy` +0.05 在多数 cell | **1/48**；47/48 `force_noisy` ≥ fusion；总体 0.628 vs 0.576 | **不成立**（单 cell，2 seeds，不可外推） |
| C3 冲突 | fusion ≥ best single +0.05 | `force_noisy` 已 1.000；fusion 1.000；geometry 0.400 | **不成立；相对 geometry +0.600** |
| 主动感知 | active ≥ passive | coarse 0.167→1.0、short 0.0→1.0、fine 0.833→1.0 | **条件性有效** |
| 事件 F1 绝对 bar | ≥0.95 | C2/C3 fusion 1.000 | **成立** |
| onset/offset latency | onset ≤50 ms、offset ≤100 ms | fusion 5–8 ms / 8–10 ms | **成立** |

---

## 8. Go / No-Go

### 8.1 决策

**不把 ContactEvent/fusion 设为 runtime 默认；保持 opt-in。**

- **`force_noisy` 是默认单通道**：48-cell 总体 strict F1 0.628 > fusion 0.576；47/48 cell 不输 fusion。
- `fusion_noisy` 是**窄 corner 的鲁棒层**：只在“force 已知失败（高噪声 + 高丢帧）+ geometry 仍可信”时有证据（cell 9，+0.667），且只有 2 seeds，不可外推。
- `geometry` 单独不可用于接触判定（near-miss 误报 40/40、冲突 F1 0.4）。
- 主动感知只在欠采样/窗口错位时启用，不作为默认。
- C5 的 runtime 接入保留为 opt-in；不改变既有语义。

### 8.2 为什么不是“融合赢了”

- clean 场景所有通道饱和（F1 1.0），没有区分度；
- clean force 是 GT 本身，不可作为融合上限；
- C3 的 `force_noisy` 本身 1.0，fusion 只打平；
- 48-cell 覆盖里只有 1 个 cell 有 +0.05，47/48 打平或下降；
- fusion 的真实价值是**冲突/误报抑制**（相对 geometry +0.324~+0.600）和**极端 force 失败时的恢复**（cell 9），不是普遍精度提升。

---

## 9. 失败与残余

1. clean force = oracle → H 不可测；noisy force 的 +0.229 只在 C2 drop/collide strict 成立。
2. **覆盖 48-cell：fusion 只在 1/48 cell 超过 `force_noisy` +0.05；47/48 打平或下降；C2 的 +0.229 不普遍；cell 9 只有 2 seeds。**
3. C3 的 `force_noisy` 1.0，fusion 无增量；场景依赖强。
4. clean geometry 的宽松 F1 1.0 是指标 artifact（长区间 + onset 容差）；strict 掉到 0.667。
5. delay=4 是 drop 死区（force 0.044、fusion 0.038）；4 帧延迟 ≈ 事件长度，strict IoU 下必然失败。
6. occlusion 本次是 null factor：没有 vision arm/render，occluder 不影响 force/geometry。
7. 阈值校准不稳：cal 选 0.10，test 事后最优 0.20 仍只有 0.579，低于 force 0.611。
8. tactile FP 来自 floor 未映射/被计入物体触觉图；已修，但真实触觉仍未测。
9. vision_motion 是仿真 RGB-D 代理；静载接触 offset 过长；intensity 不是力。
10. audio_proxy 由 force 派生，循环论证；真实音频未测。
11. 主动感知没有物理 probe；只做 re-observe/wait；conflict/borderline 触发未单独验证。
12. 事件 GT 依赖力阈值/合并参数；换阈值会改变 P/R。
13. 不是真实机器人/真实传感器；不声称 sim2real。

---

## 10. 下一步

1. **真实传感器**：真实力/触觉/音频/本体感觉；当前 force/tactile 近 oracle，audio 合成。
2. **strict 指标预注册**：IoU≥0.5 + offset-aware，避免宽松 F1 artifact；C6 已暴露这个问题。
3. **场景特定融合权重**：C2 的 +0.229 和 C3 的 0 + 覆盖的 1/48 说明权重应按“force 可靠性/geometry 冲突”条件化；全局 logistic 不够。
4. **latency-aware 对齐**：delay=4 的死区需要时间对齐/事件移位，不是权重能修。
5. **vision + occlusion**：本次 occlusion 是 null factor；要真正测遮挡，必须启用 vision/render，并用 vision 通道。
6. **cell 9 复现**：该 corner 只有 2 seeds；至少补到 3–5 seeds，并检查是否是场景 artifact。
7. **tactile floor 已修**：`tactile_maps` 只计 object-object 接触；真实触觉仍未测。
8. **物理 probe**：C4 只做观测型主动；真实 tap/probe 会改变物理轨迹，需要新 GT 协议。
9. **runtime 默认化**：等真实传感器 + strict 指标 + 条件权重 + cell 9 复现都过，再讨论。

---

## 11. 一句话

**ContactEvent 层能工作，但不是“fusion 全面胜出”：strict 下 clean force 仍 1.000（oracle），geometry/tactile/vision 只有 0.667/0.667/0.556；noisy force 退化时 fusion 可 +0.229，但 48-cell 覆盖里只有 1/48 有 +0.05，47/48 `force_noisy` 不输 fusion。所以 `force_noisy` 是默认单通道，fusion 只在 geometry 误报/极端 force 失败的窄 corner 作 opt-in 鲁棒层，不默认化。**
