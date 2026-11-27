# PSS g 因子优化实验计划：覆盖与任务匹配

> 日期：2026-11-27
> 上游：`PSS-信息论定位.md` §3（第五因子 g）、`PSS-读出与推理优化总报告.md` §4.4
> 目的：把 g 从“事后解释”变成**可审计、可优化、可证伪**的实验计划，并给出“达到上界水平”的操作定义与最终验证方法。
> 状态：方案，未执行。

---

## 0. 一句话

**g 不是标量，是“覆盖向量 × 匹配向量”。** 优化顺序是 audit → 定向补覆盖 → 不变性匹配 → 真实数据；“上界”不用单一数字判断，而用 `A_full`（全覆盖联合训练）与 `A_ceiling`（任务不可约上限）两个参照、外加三个停止条件来定义。

---

## 1. g 的操作化定义

$$\text{PSS 上界}\;\approx\; f(\text{编码器},\text{解码器},\text{维度/无损度},\text{推理容量})\;\times\; g(\text{覆盖},\text{匹配})$$

### 1.1 覆盖向量 $C$

对任务 $T$，把样本空间拆成互斥的 cell，记录每个 cell 的**支撑数** $n_c$ 与**错误率** $e_c$：

| 因子 | cell 定义（B 线示例） | cell 定义（A 线示例） |
|---|---|---|
| 关系/标签 | `pair_type × predicate`（h/v/d × left/right/above/below） | `class` |
| 几何/区域 | `(region_a, region_b)` | `view` |
| 反例 | `absence`：`|dx|≈0` 或 `|dy|≈0` 的样本 | `modality_mask` |
| 边界 | 死区、网格边缘、坐标极值 | 低支撑类别、近重复实例 |

**误差分解**（用于定位下一步该补哪里）：

$$E_{\text{overall}} = \sum_c \frac{n_c}{N}\, e_c$$

`contribution_c = (n_c/N) · e_c` 最大的 cell 就是下一分点的来源；`e_c` 高但 `n_c/N` 小的 cell 是“隐藏的最差格子”。

### 1.2 匹配向量 $M$

同一任务换特征，测“不变性是否与任务一致”：

| 探针 | 任务类型 | 期望 |
|---|---|---|
| absolute vs relative | 平移不变任务（left/right/above/below） | relative 更优 |
| absolute vs relative | 绝对位置任务（半平面归属） | absolute 更优 |
| raw vs normalized | 尺度敏感任务 | 按 val 选 |
| 加噪/坐标扰动 | 鲁棒性 | 性能下降应平缓 |

匹配误差：`M_gap = A(task-matched) − A(task-mismatched)`；`M_gap` 越大，说明特征选择越关键。

### 1.3 g 的仪表盘

每次实验落一个 JSON：

```json
{
  "task": "...",
  "coverage": {
    "cells": [{"factor": "...", "cell": "...", "support": 0, "error": 0.0, "contribution": 0.0}],
    "empty_cells": ["..."],
    "worst_cells": ["..."]
  },
  "matching": {"relative_vs_absolute": 0.0, "normalized_vs_raw": 0.0},
  "reference": {"a_current": 0.0, "a_full": 0.0, "a_ceiling": 0.0}
}
```

---

## 2. 起点：已知证据与未知

| 已知 | 证据 | 仍未测 |
|---|---|---|
| 缺一个 isolation 方向 → absence case 未监督 | BCD：水平对 exact-4 0.000；above/below precision 0.635/0.524 | 谓词 × 区域的全矩阵 |
| 补缺失方向反例可修复 | BCD 0.571 → 1.000；ABD 0.524 → 0.952；ACG 空对照 Δ=0 | 补多少条饱和 |
| 单轴增强会拆东墙补西墙 | strip_x：+vertical 0.310，+vertical+diagonal 0.967；diagonal-only 0.866 | 多轴组合的最优顺序 |
| 特征不变性必须匹配任务 | 绝对位置任务：relative 0.367 vs absolute 0.856 | 归一化/尺度不变性的系统扫描 |
| A 线模态不均 | vision 0.799、v+t 0.896、单 tactile 0.266 | 类别 × 视角 × 模态的 per-cell 支撑 |

---

## 3. 可证伪假设与预注册判据

| 假设 | 内容 | 可证伪预测 | 若成立的动作 |
|---|---|---|---|
| **H1 覆盖缺口** | 剩余误差集中在少数空/低支撑 cell | 最差 cell 的 contribution 占总体误差 ≥ 30% | 定向补该 cell |
| **H2 单轴不足** | 只补一条轴会牺牲另一类 | 单轴臂的 worst-cell ≤ 组合臂 − 0.10 | 必须补 pair-type 组合 |
| **H3 匹配错配** | 特征不变性与任务不一致是主要损失 | matched − mismatched ≥ 0.20（不变性敏感 split） | 按任务选特征，不按默认 |
| **H4 容量不是瓶颈** | 加容量不改善最差 cell | 同数据下 2× 参数/更强头无增量 | 停止加模型 |
| **H5 真实骨干** | 合成结论在真实 VLM 上不自动成立 | 真实骨干的 `A_full` 与合成不同 | 重跑全审计 |

**预注册主对照（≤6 个，Holm 校正）**：

| ID | 对照 | 任务 | 主指标 | 方向 |
|---|---|---|---|---|
| G1 | 补最差 cell vs 不补 | B 线 | worst-cell exact-4 | 补 > 不补 |
| G2 | 组合补 vs 单轴补 | B 线 strip | worst-cell exact-4 | 组合 > 单轴 |
| G3 | matched vs mismatched 特征 | 绝对/相对任务 | exact-4 | matched > mismatched |
| G4 | current vs `A_full` | 全部 | worst-cell exact-4 | 差距 ≤ 噪声 |
| G5 | `A_full` vs `A_ceiling` | 全部 | exact-4 | 差距 ≤ 噪声 |
| G6 | 补数据 vs 加容量 | 同数据 | worst-cell exact-4 | 补数据 ≥ 加容量 |

其余（per-label F1、OOD、operating point、延迟）为次要/探索性，不进主校正。

### 3.1 预注册 bar（运行前写入 checker）

在 `pss/eval/readout_reasoning_preregistration.py` 新增 `g` 字段：

```python
worst_cell_acc_min: float | None = None
delta_vs_full_min: float | None = None   # A_current - A_full，≥ -0.03 视为达标
delta_vs_full_max: float | None = None   # 单轴/错配臂的负对照上限
```

| arm | bar |
|---|---|
| `g_response_combination` | worst_cell_acc ≥ 0.90；delta_vs_full ≥ −0.03 |
| `g_response_single` | delta_vs_full ≤ −0.10（负对照：单轴必须落后） |
| `g_invariance_matched` | worst_cell_acc ≥ 0.80 |
| `g_invariance_mismatched` | worst_cell_acc ≤ 0.60（负对照） |
| `g_a_full` | 无 bar，参照 |

---

## 4. 实验设计

### E1：B 线覆盖审计（先做，最便宜）

**目标**：把“最差 cell 在哪”从个案（BCD）变成全矩阵。

- 因子：`pair_type × predicate × (region_a, region_b) × absence`。
- 数据：P5b 5 区域、7×7 block3、strip_x、strip_y、diag。
- 对每个训练子集，训练 3 seeds × 400 epochs，统计每个 cell 的 support / error / contribution。
- 输出：`eval/results/g_audit/b_coverage.json` + `B_COVERAGE.md`。
- 脚本：`pss/eval/scripts/run_g_audit.py --stage b_audit`（新）。

**判读**：

- 若最差 cell 的 contribution ≥ 30% → 进入 E2 定向补；
- 若误差弥散（无 cell 显著集中）→ 覆盖不是主要矛盾，转 E3 或 f。

### E2：覆盖响应曲线

**目标**：补哪一类、补多少、单补还是组合。

- 臂：baseline / +single cell / +combination / no-op control / `A_full`。
- 每组 300 / 500 / 1000 条增强，画“支撑数 → worst-cell exact-4”曲线。
- 3 seeds，paired bootstrap；主指标是 worst-cell exact-4，不是 overall。
- 脚本：扩 `run_relation_per_label.py --augment`（支持指定 cell），或 `run_g_audit.py --stage b_response`。

**停止规则**：曲线进入平台（连续两档增益 < 1 点或 CI 跨 0）。

### E3：不变性匹配探针

**目标**：判定任务需要哪种不变性，并测错配代价。

- 特征：`absolute` / `relative` / `normalized` / `scale-invariant`。
- 任务：相对任务（left/right）、绝对任务（半平面）、混合任务。
- 选择：只用 val 选，test 只报一次。
- 指标：按 pair type 的 exact-4、matched − mismatched 差值。
- 脚本：`run_g_audit.py --stage b_invariance`，或扩 `run_geometry_validation.py --focus invariance`。

### E4：A 线模态 × 类别 × 视角审计

**目标**：找 A 线的“空 cell”——某类只有 vision、某视角从未出现、某模态组合缺失。

- 因子：`modality_mask × class × view`。
- 对每个 cell 统计 support 与 LDA/原型 readout 的 acc/AUROC/reject。
- 复用 `run_fusion_ablation.py` 的 7 子集表 + `run_readout_scorers.py` 的缺失模态表；新增 per-cell 明细。
- 输出：`eval/results/g_audit/a_coverage.json`。

**判读**：若某类在 test 中出现但 train 只有 1 个模态 → 记录为“结构性空 cell”；决定补数据还是加子集模型。

### E5：真实骨干复跑（stretch）

- 同 E1–E4，在真实 VLM 特征上重跑；先建立 `A_full` 基线，再谈 bar。
- 这是唯一没测的“可换”旋钮，也是 g 结论最可能变化的地方。

---

## 5. “达到上界水平”的操作定义

严格上界不可知；采用**两个参照 + 三个停止条件 + 一个形态判据**。

### 5.1 两个参照

- **`A_full`**：全覆盖联合训练（所有 pair type / 模态 / 视角 / absence case）。
- **`A_ceiling`**：任务不可约上限（合成任务常为 1.000；真实任务用 oracle 或人工一致性估计）。

### 5.2 三个停止条件

1. **覆盖饱和**：`A_current` 与 `A_full` 的 worst-cell 差距 ≤ 噪声（≤1 点或 CI 跨 0）。
2. **边际饱和**：再补任何一类 cell，增益 ≤ 噪声（响应曲线平台）。
3. **上限对齐**：`A_full` 与 `A_ceiling` 的差距 ≤ 噪声。

### 5.3 形态判据

- **错误弥散**：最差 cell 的错误率与整体误差同量级（≤1.5×），说明覆盖不再是主要矛盾。
- 若最差 cell 仍显著高于整体，继续 E2；若 `A_full` 低于 `A_ceiling`，转 f（编码器/读出容量）。

### 5.4 达标分级

| 级别 | 条件 | 含义 |
|---|---|---|
| **L1 覆盖饱和** | `A_current ≈ A_full` | 当前数据已够，换任务需重新 audit |
| **L2 匹配正确** | matched − mismatched ≥ 0.20 且选中 matched | 特征不变性与任务一致 |
| **L3 上限对齐** | `A_full ≈ A_ceiling` | g 已不是瓶颈；剩余差距在 f 或任务不可约噪声 |

**只有三级同时满足，才能说“g 达到上界水平（限该任务）”；否则报告卡在哪一级。**

---

## 6. 最终验证方法

### 6.1 验证矩阵

| 臂 | 作用 | 数据 | 指标 |
|---|---|---|---|
| `current` | 现状 | 全部任务 | worst-cell / overall exact-4 |
| `+single` | 单轴负对照 | strip_x | worst-cell exact-4 |
| `+combination` | 组合修复 | strip_x / BCD | worst-cell exact-4 |
| `no-op control` | 防“多给数据就好” | ACG | Δ=0 |
| `mismatched` | 匹配负对照 | 绝对任务 | exact-4 |
| `A_full` | 全覆盖参照 | 全部 | exact-4 |
| `A_ceiling` | 任务上限 | 合成=1.000；真实=oracle | exact-4 |

### 6.2 统计口径

- 3 seeds；paired bootstrap 95% CI；主对照 Holm 校正。
- **主指标是 worst-cell exact-4**；overall 是次要指标，防止“平均值掩盖最差格子”。
- 选择只用 val；test 只报一次。
- 每个增强臂必须配 no-op control；没有对照的增益不算。

### 6.3 预注册与失败保留

- 所有 bar 在运行前写入 checker；失败项照实记录，不调 bar。
- 与 A/B 线一致：smoke 不进结论；单 seed 只作诊断。

### 6.4 杀死标准（先定）

| 观测 | 结论 | 动作 |
|---|---|---|
| 补最差 cell 后 worst-cell 不变 | 失败不是覆盖，是特征/标签 | 转 E3 或查标签 |
| 单轴与组合无差异 | H2 不成立 | 撤回“pair-type 组合”规则 |
| matched 不优于 mismatched | H3 不成立 | 特征不是主要 lever |
| `A_full` 显著低于 `A_ceiling` | 瓶颈在 f | 换编码器/加读出容量 |
| 加容量与补数据都无增量 | 任务已到 ceiling | 停止，记录平台 |

---

## 7. 相位与依赖

| 相位 | 内容 | 依赖 | 出口判据 |
|---|---|---|---|
| **G0** | 预注册字段与 bar、E1 审计脚本 | — | checker 可检查 g 臂 |
| **G1** | E1 B 线覆盖审计 | G0 | 输出最差/空 cell 清单 |
| **G2** | E2 覆盖响应曲线 | G1 | 组合臂达 worst-cell ≥ 0.90 或平台 |
| **G3** | E3 不变性匹配 | G0 | matched − mismatched ≥ 0.20 |
| **G4** | E4 A 线模态审计 | G0 | 输出结构性空 cell |
| **G5** | E5 真实骨干 | G2–G4 | 建立 `A_full` 基线 |
| **G6** | 最终验证矩阵 + 报告 | G5 | L1–L3 分级结论 |

**并行**：E2（B 线）与 E4（A 线）互不依赖，可并行；E3 只依赖 G0。

---

## 8. 交付物

```
pss/eval/scripts/run_g_audit.py            # --stage {b_audit,b_response,b_invariance,a_audit}
pss/eval/readout_reasoning_preregistration.py  # 新增 g 字段与 arm
tests/test_g_audit.py
eval/results/g_audit/                      # 覆盖矩阵、响应曲线、不变性探针
docs/experiments/PSS-g因子优化报告.md        # 最终 L1–L3 分级与验证矩阵
```

---

## 9. 风险与边界

- **g 是任务相对的**：换任务必须重新 audit，不能沿用 P5b 的 cell 定义。
- **cell 之间会互相挤压**：单轴增强已证会牺牲另一类；优化目标是“最差 cell 不低于整体一个量级”，不是平均值最高。
- **`A_ceiling` 在真实任务上难估**：只能用 oracle/人工一致性近似，结论要标明估计方式。
- **合成任务的 1.000 不是通用上界**：5 区域/7×7 网格的 ceiling 不能外推到真实几何。
- **真实骨干未测**：所有 g 结论在真实 VLM 上可能变化；E5 之前不能对外声称。

---

## 10. 一句话

**g 的优化不是“多给数据”，而是三步：先 audit 出最差 cell，再按 pair type 定向补（单轴不够），最后确认特征不变性与任务匹配；当 `A_current ≈ A_full ≈ A_ceiling`、边际增益进入平台、误差弥散时，才算在该任务上达到上界水平——否则报告卡在哪一级，转 f 或转真实数据。**
