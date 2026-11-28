# PSS ContactEvent C3：遮挡 / 冲突 / 近失（含 fusion 与 strict 指标）

- 阶段：C3（计划 §6.4、§7、§8）
- 日期：2026-11-29
- 主结果：`eval/results/contact_event/c3_conflict`（fusion 与 strict/loose 双口径）
- 辅助结果：`eval/results/contact_event/c3`（通道级 + render，用于遮挡可见性与 tactile 诊断）
- 场景：`occlusion`、`conflict_force_only`、`conflict_geometry_only`、`near_miss_hard`、`near_miss_clear`
- seeds：0/1/2；400 帧/场景（`capture_every=2`，1.6 s）
- arms：`force`、`force_noisy`、`geometry`、`fusion_noisy`（C2 已注册）
- 未新增臂、未改 bar、未改 C0/C1/C2 文件

## 0. 结论摘要

1. **loose 与 strict 下，`force` / `force_noisy` / `fusion_noisy` 的 F1 都是 1.000，`geometry` 都是 0.400。**
   - `fusion_noisy` 在 conflict resolution 上 12/12，false alarm 0%，strict IoU 0.5 下也没有掉点。
   - 但 `force_noisy` 在本 C3 场景集上同样 12/12、F1 1.000，所以**C3 不能证明 fusion 相对 force_noisy 有增量**；1.000 是 ceiling，不是融合成功的证据。
2. **fusion 相对 `geometry` 有明确增益（+0.600 F1）**：`conflict_force_only` 恢复了 geometry 的 3 个 FN，`conflict_geometry_only` 与 `near_miss_hard` 压掉了 geometry 的 6 个 FP。这个增益来自“force 证据压过 geometry 证据”，`force_noisy` 单独也能做到。
3. **clean geometry 的宽松 F1 1.0 是指标 artifact**（C2 drop/collide：geometry 区间 0.228–1.600 s，offset 误差 681 ms，但 onset 差 16 ms ≤50 ms 仍记 TP）。C3 的 `occlusion` 中 geometry loose=strict=1.0、offset 41 ms，但一旦加入冲突/近失，geometry 整体掉到 0.400。**任何 clean force/geometry 的 1.0 都不能当作融合成功。**
4. **C2 的 `fusion_noisy` +0.229（strict 0.771 → 1.000）在本 C3 场景集没有复现**：本集 `force_noisy` 已经 1.000，无 headroom。该 +0.229 只在 C2 的 drop/collide strict 口径成立，是场景相关的，不是 C3 的结论。
5. **遮挡验证成立**：GT 接触窗口内 26/34/38 帧物体可见像素全为 0；但 vision 通道是占位，未真正测试视觉遮挡下的融合。
6. **false alarm**：`near_miss_clear` 全部 0 FP；`near_miss_hard` 只有 geometry 误报（3/3 次运行，每次 1 FP），force/force_noisy/fusion_noisy 均为 0。

## 1. 运行与口径

### 1.1 主运行（fusion + strict）

```bash
python -u -m pss.eval.scripts.run_contact_conflict \
  --scenarios occlusion conflict_force_only conflict_geometry_only \
              near_miss_hard near_miss_clear \
  --seeds 0 1 2 --arms force force_noisy geometry fusion_noisy \
  --out eval/results/contact_event/c3_conflict
```

产物：

- `eval/results/contact_event/c3_conflict/contact_event.json`
- `eval/results/contact_event/c3_conflict/CONTACT_CONFLICT_REPORT.md`

指标口径：

- **loose（预注册 event 规则）**：同 pair + 时间 IoU ≥0.3 **或** onset 差 ≤50 ms。
- **strict（事后诊断，非预注册）**：同 pair + 纯 IoU ≥0.5，**去掉 onset 容差**。
- 总体 P/R/F1 为 micro：先把 5 个场景 × 3 seeds 的 tp/fp/fn 相加再算；无 GT 且无预测的场景不产生分母（显示 `—`），无 GT 但有 FP 的场景 precision=0。
- 下文引用的 C2 数字（0.771/0.667/1.000）是 C2 的**逐场景 F1 再平均**；C3 主表是 **micro**。两种聚合不要机械对比。
- 所有结果为 MuJoCo 图元 **proxy**；`force_noisy` 是模拟退化（10% 相对噪声、1–2 帧延迟、0.1 N 量化、5% dropout），不是真实传感器。

### 1.2 场景构造

| 场景 | 物理过程 | 人为“冲突”来源（只改观测模型，不改物理） | GT |
|---|---|---|---|
| `occlusion` | box 静止、ball 受 +x 力撞入；真实接触 | 相机与物体间固定 0.02×0.25×0.15 m 挡板（x=-0.40, z=0.33）；每 tick 重置位姿 | 1 事件/seed |
| `conflict_force_only` | 同 collide，真实接触 | 每帧把 ball 的 scene-model 位姿沿 +y 平移 0.50 m | 1 事件/seed |
| `conflict_geometry_only` | box/ball 静止，中心距 0.125 m；真实间隙约 0.027–0.040 m，无接触 | 每帧给两物体 scene-model 半径各 +0.05 m | 0 |
| `near_miss_hard` | 中心距 0.115 m；真实间隙约 0.017–0.030 m，**gap < eps=0.02 m** | 无；geometry 的表面间隙模型与 GT 定义不一致 | 0 |
| `near_miss_clear` | 中心距 0.140 m；真实间隙约 0.041–0.055 m，**gap > eps** | 无；真正 no-contact 对照 | 0 |

关键分数（seed 0，3 seed 一致）：

| 场景 | geometry 最大分数 | force 最大分数 | 判定 |
|---|---:|---:|---|
| occlusion | 0.980 | 0.911 | 接触 |
| conflict_force_only | 0.000 | 0.923 | 接触（geometry 被位姿偏移骗过） |
| conflict_geometry_only | 1.000 | 0.000 | 无接触（geometry 被半径膨胀骗过） |
| near_miss_hard | 0.580 | 0.000 | 无接触（0.580 > 阈值 0.5） |
| near_miss_clear | 0.109 | 0.000 | 无接触（低于阈值） |

## 2. 主结果：loose vs strict

### 2.1 总体（micro over 5 场景 × 3 seeds）

| arm | loose P | loose R | loose F1 | strict P | strict R | strict F1 | tp/fp/fn |
|---|---:|---:|---:|---:|---:|---:|---|
| `force` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6/0/0 |
| `force_noisy` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6/0/0 |
| `geometry` | 0.333 | 0.500 | 0.400 | 0.333 | 0.500 | 0.400 | 3/6/3 |
| `fusion_noisy` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6/0/0 |

判读：

- `force` 与 GT 同源，1.000 是 oracle 上界，**不能作为融合证据**。
- `force_noisy` 在本集也是 1.000；因此 `fusion_noisy` 的 1.000 相对 force_noisy 的增量是 **+0.000**，无法在此场景集上体现融合价值。
- `fusion_noisy` 相对 `geometry` 的 strict F1 增量是 **+0.600**（0.400 → 1.000），但这只是“force 证据覆盖 geometry 错误”，不是超过最好的非 oracle 单通道。

### 2.2 分场景 F1（loose = strict，因匹配集合相同）

| scenario | `force` | `force_noisy` | `geometry` | `fusion_noisy` |
|---|---:|---:|---:|---:|
| occlusion | 1.000（3/0/0） | 1.000（3/0/0） | 1.000（3/0/0） | 1.000（3/0/0） |
| conflict_force_only | 1.000（3/0/0） | 1.000（3/0/0） | —（0/0/3） | 1.000（3/0/0） |
| conflict_geometry_only | —（0/0/0） | —（0/0/0） | 0.000（0/3/0） | —（0/0/0） |
| near_miss_hard | —（0/0/0） | —（0/0/0） | 0.000（0/3/0） | —（0/0/0） |
| near_miss_clear | —（0/0/0） | —（0/0/0） | —（0/0/0） | —（0/0/0） |

括号内为 tp/fp/fn；`—` 表示该场景无 GT 且无预测（P/R/F1 无定义），不是 1.0。

### 2.3 Onset / offset 误差（匹配事件均值）

| arm | loose onset | loose offset | strict onset | strict offset | 匹配数 |
|---|---:|---:|---:|---:|---:|
| `force` | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 6 |
| `force_noisy` | 5.3 ms | 6.7 ms | 5.3 ms | 6.7 ms | 6 |
| `geometry` | 12.0 ms | 41.3 ms | 12.0 ms | 41.3 ms | 3（仅 occlusion） |
| `fusion_noisy` | 5.3 ms | 10.0 ms | 5.3 ms | 10.0 ms | 6 |

- `fusion_noisy` 保留了 noisy force 的 5.3 ms onset，offset 10.0 ms，远优于 geometry 的 41.3 ms；也远小于计划 §7 的 50/100 ms bar。
- geometry 的 offset 只在 occlusion 有匹配事件；在 `conflict_force_only` 它是 3 个 FN，没有可计算的 latency。

## 3. Conflict resolution（fusion 是否选对）

定义：对 4 个冲突/近失场景 × 3 seeds（共 12 次）：
- 有 GT（`conflict_force_only`）：只要检测到真实接触（tp ≥1）即“选对”；
- 无 GT（`conflict_geometry_only`、`near_miss_hard`、`near_miss_clear`）：只要不产生预测事件（n_pred=0）即“选对”。

| arm | resolved | conflict_force_only | conflict_geometry_only | near_miss_hard | near_miss_clear |
|---|---:|---:|---:|---:|---:|
| `force` | **12/12** | 3/3 | 3/3 | 3/3 | 3/3 |
| `force_noisy` | **12/12** | 3/3 | 3/3 | 3/3 | 3/3 |
| `geometry` | **3/12** | 0/3 | 0/3 | 0/3 | 3/3 |
| `fusion_noisy` | **12/12** | 3/3 | 3/3 | 3/3 | 3/3 |

逐条解读：

- **`conflict_force_only`：fusion_noisy 恢复 geometry 的 FN。** geometry 3/3 漏检，fusion_noisy 3/3 检出；`force_noisy` 也 3/3。fusion 的 calibrated coefficient 对 geometry 为负（C2 noisy 参数），几何“无接触”反而抬高融合概率，这是它能恢复的理论原因。
- **`conflict_geometry_only`：fusion_noisy 压掉 geometry 的 FP。** geometry 3/3 误报，fusion_noisy 3/3 静默；`force_noisy` 也静默。
- **`near_miss_hard`：fusion_noisy 压掉 geometry 的 FP。** geometry 3/3 误报（分数 0.58），fusion_noisy 3/3 静默。
- **`near_miss_clear`：所有臂都正确静默。** 这是 gap > eps 的 clean control。
- **`occlusion` 不计入 conflict resolution**：它不是 force/geometry 冲突，而是可见性场景。

**增益归属必须写清楚**：

- 相对 **geometry**：fusion 在冲突/误报上有增益（+0.600 F1；6 个 geometry 错误全部避免）。
- 相对 **force_noisy**：**没有增量**（两者 12/12、F1 1.000）。因此本次不能写“融合优于最好非 oracle 单通道”，只能写“融合避免了 geometry 的失败模式，且没有引入新的误报”。
- 由于 GT 来自 MuJoCo contact force，`force` 与 `force_noisy` 的完美表现仍带有同源/场景容易的成分；`force_noisy` 在本集没有受到 C2 drop/collide 中那种噪声 FP 的挑战。

## 4. False alarm

| arm | near_miss_clear FP / runs | clear FA 率 | near_miss_hard FP / runs | hard FA 率 |
|---|---:|---:|---:|---:|
| `force` | 0/3 | 0% | 0/3 | 0% |
| `force_noisy` | 0/3 | 0% | 0/3 | 0% |
| `geometry` | 0/3 | 0% | **3/3** | **100%** |
| `fusion_noisy` | 0/3 | 0% | 0/3 | 0% |

- `near_miss_clear`（gap > eps）所有臂 0 FP，说明 geometry 的阈值在明确无接触时不会误报。
- `near_miss_hard`（gap < eps）geometry 每 seed 1 FP，3/3 次运行；fusion_noisy 与 force/force_noisy 为 0。
- 以计划 §7 的 false alarm ≤5% 判定：`fusion_noisy` 在 near_miss 上 0%，**满足**；`geometry` 在 hard 上 100%，**不满足**。

## 5. 遮挡验证（沿用 render 运行）

`c3_conflict` 主运行为了速度未开 `--render`；遮挡可见性来自 `eval/results/contact_event/c3` 的 render 运行（场景与物理完全相同）：

| seed | GT 接触窗口 | 接触窗口帧数 | 接触窗口内物体可见像素 | 全程完全隐藏帧 | 全程最大可见像素 | occluder 中位像素 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | [0.084, 0.184] s | 26 | 0/0/0 | 191/400 | 13 | 2932 |
| 1 | [0.092, 0.224] s | 34 | 0/0/0 | 244/400 | 13 | 2932 |
| 2 | [0.088, 0.236] s | 38 | 0/0/0 | 238/400 | 14 | 2932 |

结论：接触发生的整个窗口内，两个物体在所有帧都被完全遮挡；接触后 ball 被推出桌面、越过挡板上缘，才出现 13–14 px 泄漏。**这证明遮挡确实发生，但不证明视觉通道在遮挡下的表现**——`vision_motion` 未参与本次 arms，且真实相机遮挡模型不存在。

## 6. Clean geometry 的 1.0 为什么是 artifact

C2 报告已记录（drop/collide，3 seeds）：

- `geometry` 在标准 loose event F1 下是 1.000，看似完美；
- 但 drop 中 geometry 的事件区间是 0.228–1.600 s，GT 只有约 0.08 s，**offset 误差 681 ms**；
- 由于 onset 差 16 ms ≤ 50 ms，loose 规则把整个过长的静置段记成同一个 TP；
- 换成 strict（纯 IoU ≥0.5）后，`geometry` 掉到 0.667，`force_noisy` 是 0.771，`fusion_noisy` 是 1.000（Δ=+0.229，C2 事后诊断）。

本次在同一环境复跑 C2 的 drop/collide/near_miss（3 seeds、逐场景 F1 平均），确认上述数字：

| arm | loose F1 | strict F1 | strict tp/fp/fn |
|---|---:|---:|---|
| `force_noisy` | 0.771 | 0.771 | 6/14/0 |
| `geometry` | 1.000 | 0.667 | 3/3/3 |
| `fusion_noisy` | 1.000 | 1.000 | 6/0/0 |

这复现了 +0.229，也说明它依赖 C2 场景集的噪声 FP 与 geometry 的过长区间；C3 的 contact 场景没有这些条件，故不出现同样 headroom。

在本次 C3：

- `occlusion` 中 geometry loose=strict=1.0，offset 41.3 ms，区间没有明显过长，属于真实可用的检测；
- 但加入 `conflict_*` 与 `near_miss_*` 后，geometry 整体 strict=loose=0.400；
- 因此 **C3 的 geometry 0.400 是冲突/近失导致的真实失败，不是 artifact；C2 的 clean geometry 1.000 才是 artifact。两者不要混用。**

同理，**clean `force` 的 1.000 是 oracle 上界**（GT 用同一力序列、同一阈值），`force_noisy` 的 1.000 在本集是场景太容易，二者都不能证明融合成功。

## 7. 与预注册 bar 对照（C3）

| 指标 | bar | 本次结果（fusion_noisy） | 判定 |
|---|---|---|---|
| 事件 F1（occlusion） | ≥0.85 | loose/strict 1.000 | 满足 |
| onset latency | ≤100 ms | 5.3 ms | 满足 |
| offset latency | ≤100 ms | 10.0 ms | 满足 |
| false alarm（near_miss） | ≤5% | clear 0% / hard 0% | 满足 |
| conflict resolution | ≥0.90 | 12/12 = 1.000 | 数值满足，但 `force_noisy` 也 12/12，证据强度有限 |
| intensity MAE | ≤10% | 未在本次冲突集评估（见 C2） | 未测 |

**不能据此宣布 H' 成立**：

1. H' 要求 fusion ≥ 最好非 oracle 单通道 +0.05；本集 `force_noisy` 已是 1.000，上限封顶，Δ=0.000。
2. C2 的 +0.229 是 drop/collide strict 的事后诊断，不是预注册 H'，也不能搬到 C3。
3. conflict resolution 的 12/12 同时被 `force_noisy` 达到，且 GT 与 force 同源；没有“fusion 独有的冲突解决能力”证据。

## 8. 失败项、边界与未测

1. **无法区分 fusion 与 force_noisy**：本 C3 场景集的 contact 强而单一，force_noisy 没有产生 C2 drop/collide 中的噪声误报，导致无 headroom。
2. **force 近 oracle**：clean force 的 F1=1、latency=0，是 GT 同源，不是独立性能。
3. **冲突为观测模型损坏**：位姿偏移 0.5 m、半径膨胀 0.05 m，不是真实跟踪故障/标定漂移；结论只覆盖“模型/标定错”这一类。
4. **遮挡只验证渲染可见性**：vision 通道未参与；没有真实相机遮挡或视觉融合。
5. **未测 tactile 未覆盖、音频远场误检、主动感知**；C4/C5 另有报告。
6. **tactile 的 floor 污染仍在**（通道级 `c3` 运行）：球飞出桌沿后 floor geom 未映射为 `floor`、记为 `unknown`，被逐物体触觉图计入，导致 `occlusion`/`conflict_force_only` 每 seed 多 1 个 tactile FP。本次 `fusion_noisy` 未复现该 FP（阈值/负 geometry 权重抑制），但这是 proxy 实现缺陷，修复需改 C1/C2 文件，本次未改。

## 9. 复现

主运行（fusion + strict/loose）：

```bash
python -u -m pss.eval.scripts.run_contact_conflict \
  --scenarios occlusion conflict_force_only conflict_geometry_only \
              near_miss_hard near_miss_clear \
  --seeds 0 1 2 --arms force force_noisy geometry fusion_noisy \
  --out eval/results/contact_event/c3_conflict
```

遮挡可见性（render）：

```bash
python -u -m pss.eval.scripts.run_contact_event_bench \
  --scenarios occlusion conflict_force_only conflict_geometry_only near_miss_hard \
  --seeds 0 1 2 --render --arms force tactile geometry \
  --out eval/results/contact_event/c3
```

本次新建/修改的文件：

- `pss/events/scenarios_conflict.py`（新增 `near_miss_clear`；原有 4 场景不变）
- `pss/eval/scripts/run_contact_conflict.py`（loose/strict + conflict/FA/latency 汇总）
- `docs/experiments/PSS-ContactEvent-C3-Conflict.md`（本报告）

运行产物：

- `eval/results/contact_event/c3_conflict/contact_event.json`
- `eval/results/contact_event/c3_conflict/CONTACT_CONFLICT_REPORT.md`
- `eval/results/contact_event/c3/contact_event.json`（通道级 + render）
- `eval/results/contact_event/c3/CONTACT_EVENT_REPORT.md`
