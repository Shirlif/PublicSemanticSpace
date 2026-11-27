# PSS g 因子优化报告：覆盖与任务匹配

> 日期：2026-11-27
> 上游：`docs/plans/PSS-g因子优化实验计划.md`、`docs/experiments/PSS-读出与推理优化总报告.md`
> 结果目录：`eval/results/g_audit/`
> 状态：已执行 G0–G6；G5 为 DINOv2 真实骨干代理（非 VLM）。

---

## 0. 一句话

**g 的优化按 audit → 定向补覆盖 → 不变性匹配 → 真实骨干的顺序执行。** 合成 B 线上，误差确实集中在少数 cell；单轴增强只修自己那条轴，组合补把 worst-cell 从 0.000 抬到 **0.957**，但距 `A_full` 仍有 **4.3 pt**，比预注册的 3 pt 上限多 1.3 pt，因此 L1 记为未达标。不变性匹配成立（M_gap +0.431），但 matched 臂的 absolute worst-cell 0.7973 比 0.80 低 0.27 pt，也是一项近失。A 线没有结构性空 cell，0.833 → 0.930 的差距在读出/模态表示（f），不在覆盖。真实 VLM 仍未测，L1–L3 只在合成任务上成立。

---

## 1. G0：预注册字段与 bar

在 `pss/eval/readout_reasoning_preregistration.py` 的 `Expectation` 中新增：

- `worst_cell_acc_min` / `worst_cell_acc_max`
- `delta_vs_full_min` / `delta_vs_full_max`

并注册五个 g 臂（运行前固定，失败不调 bar）：

| arm | bar |
|---|---|
| `g_response_combination` | `worst_cell_acc ≥ 0.90`；`delta_vs_full ≥ −0.03` |
| `g_response_single` | `delta_vs_full ≤ −0.10`（负对照） |
| `g_invariance_matched` | `worst_cell_acc ≥ 0.80` |
| `g_invariance_mismatched` | `worst_cell_acc ≤ 0.60`（负对照） |
| `g_a_full` | 无 bar，参照 |

`worst_cell_acc` 的定义在脚本中固定为 **pair_type 维度的最小 exact-4**（support ≥ 5）；覆盖审计的 predicate 级分解是诊断，不和主 bar 混用。

新增脚本：`pss/eval/scripts/run_g_audit.py`，stage 为 `b_audit / b_response / b_invariance / a_audit / final`；纯函数在 `pss/eval/g_audit.py`，测试在 `tests/test_g_audit.py`。

验证：`python -m pytest tests/test_g_audit.py tests/test_readout_reasoning_preregistration.py -q` → **27 passed**。

---

## 2. G1：E1 B 线覆盖审计

结果：`eval/results/g_audit/b_coverage.json` / `B_COVERAGE.md`。3 seeds × 400 epochs；P5b 用 5 区域，7×7 用 block3 / strip_x / strip_y / diag；特征为任务匹配的 `relative`。

### 2.1 总体与参照

| family | subset | overall exact-4 | worst type exact-4 | A_full | 最差 cell | contribution share |
|---|---|---:|---:|---:|---|---:|
| p5b | BCD | 0.571 ± 0.000 | 0.000 | 1.000 | `horizontal|below` | 0.833 |
| p5b | ABD | 0.571 ± 0.000 | 0.000 | 1.000 | `vertical|left` | 0.556 |
| p5b | ACG | 1.000 ± 0.000 | 1.000 | 1.000 | `diagonal|above` | 0.000 |
| grid7 | block3 | 1.000 ± 0.001 | 0.999 | 1.000 | `diagonal|above` | — |
| grid7 | strip_x | 0.172 ± 0.100 | 0.000 | 1.000 | `diagonal|below` | 0.304 |
| grid7 | strip_y | 0.129 ± 0.007 | 0.000 | 1.000 | `diagonal|left` | 0.274 |
| grid7 | diag | 0.366 ± 0.000 | 0.000 | 1.000 | `diagonal|above` | 0.196 |

### 2.2 H1：误差是否集中在少数 cell

**支持 H1。** 剩余误差最大的 cell：

| rank | subset | 最差 cell | support | error | share |
|---:|---|---|---:|---:|---:|
| 1 | p5b / BCD | `horizontal|below` | 18 | 0.833 | 0.833 |
| 2 | p5b / ABD | `vertical|left` | 18 | 0.556 | 0.556 |
| 3 | grid7 / strip_x | `diagonal|below` | 1836 | 0.472 | 0.304 |
| 4 | grid7 / strip_y | `diagonal|left` | 1827 | 0.518 | 0.274 |
| 5 | grid7 / diag | `diagonal|above` | 1776 | 0.267 | 0.196 |

- BCD 的最差 cell 是 **水平对上的 `below`**：模型没有见过 `dy = 0` 的水平对，于是把水平对误判成 above/below。ABD 对称（垂直对上的 `left`）。这与 B 线既有诊断的 absence-case 机制一致。
- 没有空 cell；每个 `pair_type × predicate` 在测试池中都有支撑。问题不是“完全没有 cell”，而是**缺失隔离方向的支撑太低**。
- `A_full` 在全部子集上都是 1.000，说明这些误差是覆盖造成的，不是任务不可约。
- strip_x 的最差 cell share 0.304，刚过 0.30 阈值；它的主要误差在 diagonal（0.436–0.472），而不是单轴自身。这直接指向 G2 的“单轴不够、要补 pair type”。

### 2.3 absence cell 的误差是 1.000

覆盖审计的 `absence` 维度把测试对分成 `dy_zero`（水平对）、`dx_zero`（垂直对）和 `general`（对角）。误差几乎完全落在训练时缺失的隔离方向：

| family / subset | dy_zero error | dx_zero error | general error |
|---|---:|---:|---:|
| p5b / BCD | **1.000**（缺水平对） | 0.000 | 0.000 |
| p5b / ABD | 0.000 | **1.000**（缺垂直对） | 0.000 |
| p5b / ACG | 0.000 | 0.000 | 0.000 |
| grid7 / strip_x | 0.000 | **1.000** | 0.908 |
| grid7 / strip_y | **1.000** | 0.000 | 0.988 |
| grid7 / diag | **1.000** | **1.000** | 0.505 |

这比“某个谓词 precision 低”更直接：**缺少哪个隔离方向，哪个 absence cell 就整格错**。block3 三个 absence cell 都是 0.000，说明两个方向都覆盖时 absence case 不是问题。

### 2.4 对 G2 的输入

- BCD 的覆盖缺口最集中（share 0.833），但它的修复已被既有 BCD 诊断覆盖（+horizontal 反例把 horizontal exact-4 从 0.000 修到 1.000）。
- G2 选择 **grid7 / strip_x** 作为预注册的 H2 对照：它同时缺 y 隔离和 diagonal，正好检验“单轴 vs 组合”。

---

## 3. G2：E2 覆盖响应曲线

结果：`eval/results/g_audit/b_response.json` / `B_RESPONSE.md`。target 为 **grid7 / strip_x**（缺 `y` 隔离与 `diagonal`，已有 `x`），3 seeds × 400 epochs；主指标 worst-cell exact-4。

### 3.1 曲线

| support | arm | augment | worst-cell exact-4 | overall exact-4 | Δ vs A_full |
|---:|---|---|---:|---:|---:|
| 300 | baseline | {} | 0.000 | 0.172 | −1.000 |
| 300 | single | {y: 300} | 0.082 | 0.298 | −0.918 |
| 300 | combination | {y: 150, diagonal: 150} | 0.947 | 0.959 | −0.053 |
| 300 | noop | {x: 300} | 0.000 | 0.134 | −1.000 |
| 500 | baseline | {} | 0.000 | 0.172 | −1.000 |
| 500 | single | {y: 500} | 0.098 | 0.310 | −0.902 |
| 500 | combination | {y: 250, diagonal: 250} | **0.958** | **0.967** | −0.042 |
| 500 | noop | {x: 500} | 0.000 | 0.155 | −1.000 |
| 1000 | baseline | {} | 0.000 | 0.172 | −1.000 |
| 1000 | single | {y: 1000} | 0.108 | 0.318 | −0.892 |
| 1000 | combination | {y: 500, diagonal: 500} | 0.957 | 0.967 | −0.043 |
| 1000 | noop | {x: 1000} | 0.000 | 0.165 | −1.000 |

`A_full`：worst-cell **1.000**，overall **1.000**。

### 3.2 按 pair type

| support | arm | horizontal | vertical | diagonal |
|---:|---|---:|---:|---:|
| 500 | baseline | 1.000 | 0.000 | 0.092 |
| 500 | single | 1.000 | 1.000 | 0.098 |
| 500 | combination | 1.000 | 1.000 | 0.958 |
| 500 | noop | 1.000 | 0.000 | 0.070 |

### 3.3 预注册检查

- 检查 arm 数 3；**通过 False**。
- 唯一 mismatch：`g_response_combination.delta_vs_full: expected >= −0.03, got −0.0430`。
- `g_response_single` 通过（delta −0.892 ≤ −0.10）；`g_a_full` 无 bar。

### 3.4 判读

- **H2 成立**：单轴 `+vertical` 把 vertical 从 0.000 修到 1.000，但 diagonal 仍停在 0.098；组合臂同时补 `+vertical+diagonal`，diagonal 到 0.958，worst-cell 到 0.957。单轴与组合的差距是 0.849（worst-cell），远大于 H2 的 0.10 阈值。
- no-op 对照加的是已覆盖的 horizontal 数据：vertical 仍是 0.000，diagonal 仍在 0.07 左右；overall 的小幅波动（0.172 → 0.165）说明“多给已覆盖数据”不能修复缺失 cell。
- 曲线在 500 后进入平台（0.958 → 0.957），满足 G2 的停止规则。
- **严格 bar 的近失**：`combination.worst_cell = 0.957 ≥ 0.90` 通过，但 `delta_vs_full = −0.043`，比 −0.03 多 1.3 pt，未达标。bar 不调整；结论是“覆盖修复接近饱和，但距全覆盖仍有 4.3 pt 的最差 cell 差距”。
- 既有 BCD 诊断（`B_LINE_SUMMARY.md`）给出同向证据：BCD 补 300 条 `dy=0` 反例后 horizontal exact-4 0.000 → 1.000，ABD 0.524 → 0.952，ACG 空对照 Δ=0。

---

## 4. G3：E3 不变性匹配

结果：`eval/results/g_audit/b_invariance.json` / `B_INVARIANCE.md`。7×7 grid、`block3` 训练、held-out pair 对半切 val/test；选择只用 val，test 只报一次。

### 4.1 task × feature（test exact-4）

| task | feature | val exact-4 | test exact-4 | worst type exact-4 |
|---|---|---:|---:|---:|
| relative | absolute | 0.999 | 0.998 | 0.987 |
| relative | relative | 0.999 | 1.000 | 1.000 |
| relative | normalized | 1.000 | 1.000 | 1.000 |
| relative | scale_invariant | 1.000 | 1.000 | 1.000 |
| absolute | absolute | 0.804 | **0.813** | **0.797** |
| absolute | relative | 0.391 | 0.383 | 0.208 |
| absolute | normalized | 0.302 | 0.290 | 0.113 |
| absolute | scale_invariant | 0.321 | 0.284 | 0.113 |
| mixed | absolute | 0.815 | **0.804** | 0.781 |
| mixed | relative | 0.379 | 0.344 | 0.149 |
| mixed | normalized | 0.242 | 0.233 | 0.113 |
| mixed | scale_invariant | 0.260 | 0.248 | 0.113 |

### 4.2 matched / mismatched

| task | matched | mismatched | M_gap [95% CI] | 敏感? |
|---|---|---|---:|---|
| relative | relative | absolute | +0.002 [+0.000, +0.004] | 否 |
| absolute | absolute | relative | **+0.431 [+0.393, +0.467]** | 是 |
| mixed | absolute | relative | **+0.460 [+0.427, +0.490]** | 是 |

### 4.3 坐标扰动（相对任务，不重训）

| feature | noise 0.03 | noise 0.10 | noise 0.20 |
|---|---:|---:|---:|
| relative | 0.995 | 0.940 | 0.922 |
| absolute | 0.993 | 0.919 | 0.906 |
| scale_invariant | 0.994 | 0.912 | 0.879 |

### 4.4 判读

- **相对任务不敏感**：absolute 也包含 delta 项，所以它和 relative 都能到 ~1.000；M_gap 只有 +0.002。不能在这个 split 上检验 H3。
- **绝对位置任务敏感**：absolute 0.813 vs relative 0.383，M_gap **+0.431**，CI 不跨 0；mixed 任务 M_gap **+0.460**。**H3 得到支持**：特征不变性必须与任务匹配。
- val 选择在 absolute / mixed 上都选中了 `absolute`，与预注册的 matched 一致。
- **预注册 bar 的近失**：`g_invariance_matched.worst_cell_acc = 0.7973`，低于 0.80 约 0.27 pt；`g_invariance_mismatched.worst_cell_acc = 0.208` 通过 ≤0.60。bar 不调整，照实记录为失败项。M_gap 结论不受影响，但“matched 臂的绝对 worst-cell”没有达到 0.80。
- 扰动下三种特征都平缓下降（0.995 → 0.922/0.906/0.879），没有出现某项特征在噪声下崩溃。

---

## 5. G4：E4 A 线模态 × 类别 × 视角审计

结果：`eval/results/g_audit/a_coverage.json` / `A_COVERAGE.md`。

### 5.1 参照

| 指标 | 值 |
|---|---:|
| `A_current`（部署 fused prototype cosine） | **0.833** |
| `A_full`（全模态 pooled-covariance LDA） | **0.930** |
| MLP strong-head proxy | 0.878 |
| `A_ceiling` | 不可得（无 oracle） |

### 5.2 模态子集

| modality mask | LDA acc |
|---|---:|
| vision | 0.799 |
| tactile | 0.266 |
| geometry | 0.263 |
| vision + tactile | 0.896 |
| vision + geometry | 0.901 |
| tactile + geometry | 0.607 |
| vision + tactile + geometry | **0.930** |

### 5.3 held-out 视图

| view | support | LDA acc | proto acc |
|---|---:|---:|---:|
| v6 | 384 | 0.958 | 0.904 |
| v7（test） | 384 | 0.930 | 0.833 |

### 5.4 结构性空 cell

**未发现结构性空 cell。** 384 个类在训练视图中都有 vision / tactile / geometry 支撑；`class × view` 的 768 个 held-out 视图 cell 全部有支撑。A 线的覆盖不是主要矛盾。

### 5.5 判读

- `A_current` 与 `A_full` 的 0.097 差距是**读出与模态表示**的差距，不是覆盖缺口。
- tactile（0.266）与 geometry（0.263）单独很弱；vision+tactile / vision+geometry 都在 0.90 左右，三者合到 0.930，说明瓶颈在**单模态表示的判别力**，不是缺数据。
- 因此 A 线的动作是 f（编码器/读出），不是继续补 cell；L3 因无 oracle 不可判定。

---

## 6. G5：E5 真实骨干（部分）

完整 E5 要求“真实 VLM 特征 + 同 E1–E4 重跑”。仓库没有真实 VLM 特征缓存或对应数据集，CLIP 权重也不在本地缓存（`eval/results/encoder_priority/encoder_priority.json` 记录 `clip` 被跳过）。因此本轮做的是 **A 线真实视觉骨干代理**：

- 骨干：冻结 DINOv2（`checkpoints/encoder_priority/dinov2/projection`）。
- 任务：`data/category_pretrain`，16 类 category 任务（**与合成 A 线的 384 类 view-instance 任务不同**，不能当作受控的 backbone 对比）。
- 这不是 VLM，B 线（合成几何）也无法用真实骨干替代。

结果：`eval/results/g_audit/e5_dinov2/a_coverage.json`。

| 指标 | 值 |
|---|---:|
| `A_current` | **0.969** |
| `A_full` | **0.969** |
| MLP proxy | 0.964 |
| 结构性空 cell | 0 |
| vision / tactile / geometry | 0.927 / 0.182 / 0.969 |
| vision+tactile / vision+geometry | 0.932 / 0.964 |
| 三模态 | 0.969 |

判读：

- 真实视觉骨干改变了尺度（合成 A_full 0.930 → DINOv2 0.969），支持 H5 的方向：**真实骨干的 `A_full` 与合成不同，g 结论不能直接外推**。
- 在这个代理任务上 `A_current = A_full`，覆盖已饱和；tactile 单独只有 0.182，瓶颈仍是表示（f）。
- **但这不是 VLM 复现**：任务、类别数、模态来源都不同；完整 E5 仍开放，L1–L3 不得对外声称已跨到真实 VLM。

---

## 7. G6：最终验证矩阵与分级

结果：`eval/results/g_audit/final_matrix.json` / `G_FINAL_REPORT.md`。

### 7.1 分级

| task | L1 覆盖饱和 | L2 匹配正确 | L3 上限对齐 | 三级同时 |
|---|---|---|---|---|
| B line（合成） | ❌ | ✅ | ✅ | 否 |
| A line（合成代理） | ✅ 无结构性空 cell | n/a | 不可判定（无 oracle） | 否 |

### 7.2 B 线为什么卡在 L1

- **L1 ❌**：审计中差距最大的子集是 `strip_y`（0.129 vs `A_full` 1.000）；响应曲线的组合臂 worst-cell 0.957 ≥ 0.90，但 `delta_vs_full = −0.043`，比预注册的 −0.03 多 1.3 pt。覆盖修复已接近饱和，但没有达到“距全覆盖 ≤3 pt”的严格定义。
- **L2 ✅**：绝对位置任务 `absolute` 0.813 vs `relative` 0.383，`M_gap = +0.431` [0.393, 0.467]，CI 不跨 0；val 也选中 `absolute`。H3 成立。
  - 但辅助 bar 近失：`g_invariance_matched.worst_cell_acc = 0.7973 < 0.80`（差 0.27 pt）。L2 的定义只看 M_gap 与 val 选择，所以 L2 判 ✅；报告同时保留这项失败，不把它藏起来。
- **L3 ✅（仅限 B 线合成任务）**：`A_full = A_ceiling = 1.000`，差距 0。
- 结论：**B 线的 g 没有达到“三级同时”**，卡在 L1 的严格 3 pt 定义；不是覆盖机制失效（H1/H2 都成立），而是组合补后的最差 cell 仍有 4.3 pt 余量。

### 7.3 A 线

- 覆盖 L1 ✅：384 类 × 8 视图的审计没有结构性空 cell；每个类在训练视图中都有三种模态支撑。
- `A_current = 0.833`，`A_full = 0.930`；差距 9.7 pt 是**读出/模态表示（f）**，不是 g。
- L3 不可判定：没有 oracle，`A_ceiling` 只能留空。
- 结论：A 线的动作是 f（更好的读出或更强的 tactile/geometry 表示），不是继续补 cell。

### 7.4 真实骨干（部分）

DINOv2 代理上 `A_current = A_full = 0.969`、无结构性空 cell、tactile 单独 0.182。它说明真实视觉骨干会改变尺度（H5 方向成立），但不构成 VLM 复现；完整 E5 仍开放。

### 7.5 可以/不可以声称

**可以声称（限合成任务）**

- B 线误差集中在缺失隔离方向的 absence cell（H1），单轴增强不够、组合补能修（H2）。
- 特征不变性必须与任务匹配：绝对位置任务的 matched − mismatched = +0.431（H3）。
- A 线没有结构性覆盖缺口；剩余差距在 f。

**不可以声称**

- “g 已达到上界水平”：B 线卡在 L1 的严格 bar，A 线 L3 无 oracle。
- “真实骨干/VLM 上同样成立”：只做了 DINOv2 代理，且任务不同。
- “单轴增强足够”：strip_x 的单轴 worst-cell 0.108 vs 组合 0.957，直接否定。

---

## 8. 交付物与复现

```
pss/eval/g_audit.py                              # 覆盖/匹配纯函数
pss/eval/scripts/run_g_audit.py                  # --stage {b_audit,b_response,b_invariance,a_audit,final}
pss/eval/readout_reasoning_preregistration.py    # g 字段与五个 arm
tests/test_g_audit.py
eval/results/g_audit/                            # b_coverage / b_response / b_invariance / a_coverage / final_matrix
eval/results/g_audit/e5_dinov2/                  # DINOv2 代理
```

复现命令：

```bash
python -m pss.eval.scripts.run_g_audit --stage b_audit \
  --out eval/results/g_audit --device cpu --seeds 0 1 2 \
  --epochs 400 --full-epochs 300 --n-per-pair 50 --p5b-n-per-pair 300 --full-repeats 5
python -m pss.eval.scripts.run_g_audit --stage b_response \
  --out eval/results/g_audit --device cpu --seeds 0 1 2 \
  --target-family grid7 --target-subset strip_x --amounts 300 500 1000
python -m pss.eval.scripts.run_g_audit --stage b_invariance \
  --out eval/results/g_audit --device cpu --seeds 0 1 2 --epochs 400 --bootstrap 2000
python -m pss.eval.scripts.run_g_audit --stage a_audit \
  --out eval/results/g_audit --device cuda --seeds 0 1 2
python -m pss.eval.scripts.run_g_audit --stage a_audit \
  --out eval/results/g_audit/e5_dinov2 --device cuda --seeds 0 1 2 --backbone dinov2
python -m pss.eval.scripts.run_g_audit --stage final --out eval/results/g_audit
```

---

## 9. 边界

- **g 是任务相对的**：换任务必须重新 audit，不能沿用 P5b 的 cell 定义。
- **合成 ceiling 不是通用上界**：P5b/7x7 的 1.000 不能外推到真实几何。
- **A 线无 oracle**：`A_ceiling` 不可得，L3 在 A 线不可判定。
- **真实 VLM 未测**：DINOv2 只是真实视觉骨干代理，不是 VLM；所有 g 结论在此之前只对合成任务成立。
- **单 seed 只作诊断**：主结果均为 3 seeds；no-op 对照用于防“多给数据就好”。
