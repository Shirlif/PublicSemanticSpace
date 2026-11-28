# PSS-Scene-S3-Uncertainty：三维关系 belief 与校准

> 日期：2026-11-28
> 上游：`docs/plans/PSS-三维场景建模实验计划.md` §4.4、§5、§7（H3）、§8（S3）
> 结果：`eval/results/scene_bench/s3/`（主结果，5 视角）、`eval/results/scene_bench/s3_v1/`、`s3_v3/`、`s3_v5/`（视角曲线）
> 状态：D4 / S3 的 belief 与校准部分已完成。**H3 在主口径（5 视角、全部 pair 的 bench belief ECE）成立；near 子集和 3 视角以下是限定条件，单独看不成立。** 主动视角选择不在本轮范围。

---

## 0. 一句话

**用 `solve_scene` 的线性化协方差做 200 次场景采样，5 视角的关系 belief ECE = 0.027（bar 0.05），H3 主口径成立；但校准主要来自大量容易的负对：near-boundary 子集 ECE 在 5 视角是 0.047、3 视角 0.105、1 视角 0.226。** 协方差在 3 视角整体与 χ²₃ 一致（Mahalanobis 均值 2.98），但尾部明显厚于高斯；1 视角尾部失控（29.6% 的物体超过 99.9% χ²₃ 界），不能单独使用。`same_level` / `support` 在 CLEVR 上退化（标签恒为真），它们的 ECE 不代表关系层校准。

---

## 1. 预注册（运行前）

| 项 | 内容 |
|---|---|
| **H3 主口径** | `axis_aligned_3d` 的 belief ECE ≤ 0.05 |
| ECE 定义 | bench 的 `belief_ece`：每个场景内 `binary_ece(prob, label)`（10 个等宽 bin），再对场景取均值；3 个 seed 再取均值/标准差 |
| 主运行 | `--n-views 5 --seeds 0 1 2 --arms 3d_belief oracle` |
| 视角曲线 | `--n-views 1/3/5`，同一 arm、同一 seeds |
| 判假线 | ECE > 0.10 → H3 假（计划 §9） |
| 校准 | **不在 test 上调温度**；本轮不做任何后处理校准，直接报原始协方差 belief |
| 辅助口径 | pooled ECE（全部 pair 合并）、near 子集 ECE、Brier、risk-coverage、位置 Mahalanobis 一致性 |
| 不调项 | bar、eps、采样数、协方差缩放均不因 test 结果调整 |

主口径选择说明：bench 自动计算的是**逐场景 ECE 的均值**，它对“场景内全负、概率接近 0”的情形有偏（见 §4.3）；pooled ECE 作为辅助口径一并报告，但 H3 判定仍以预注册的 bench 口径为准，不因 pooled 更好看而改判。

---

## 2. 方法

### 2.1 位置与协方差

`pss.scene.optimizer.solve_scene` 对每个物体独立做最小二乘：

- 残差：2 个重投影残差 / `sigma_px` + 深度残差 / (`sigma_depth * depth`) + 桌面先验 `(z - radius) / sigma_table`；
- `loss="soft_l1"`，`f_scale=1.0`，`max_nfev=200`；
- 协方差：`Σ = σ² (JᵀJ)^{-1}`，`σ² = Σr² / max(n_res - 3, 1)`。

本地实现（`pss/scene/uncertainty.py`）在此基础上：

- `sample_scene_positions` 从 `N(p_i, Σ_i)` 独立采样 `N=200` 个场景（与“逐物体求解”的独立性一致）；
- 用特征分解采样，先对称化、特征值裁剪到 `[1e-10, 1 m²]`，数值上非 PSD 也不会抛异常；
- 缺失、非有限、零/负定协方差 → 5 cm 各向同性 fallback，并在 `SamplingInfo.degraded_objects` 记录；
- 特征值被裁剪时记入 `clipped_objects`；
- `N=200`、3 个 seed、100 个 test scene 共 6384 个 pair。

**重要限制**：这是线性化高斯后验，不是精确后验。`result.jac` 是原始残差 Jacobian，不是 robust loss 重加权后的 Jacobian；物体间误差按独立处理。这两点都会影响尾部与 differential uncertainty，见 §6、§8。

### 2.2 关系查询与 belief

在每一个采样场景上计算布尔查询，再对 200 个样本取均值：

| 查询 | 定义 | 参数 |
|---|---|---|
| `axis_aligned` | `\|Δx\| ≤ eps_xy` 或 `\|Δy\| ≤ eps_xy` | `eps_xy = 0.12` |
| `same_level` | `\|(z_b - r_b) - (z_a - r_a)\| ≤ eps_z` | `eps_z = 0.05` |
| `contact` | `norm(p_b - p_a) ≤ r_a + r_b + eps_c` | `eps_c = 0.02` |
| `collision` | `norm(p_b - p_a) < r_a + r_b` | — |
| `support` | `\|(z_i - r_i) - table_z\| ≤ eps_s` | `eps_s = 0.05` |

`3d_belief` arm（`pss/scene/arms_uncertainty.py`）返回：

- `positions`：`solve_scene` 的点估计（**与 `3d_ls` 完全相同**，belief 不改变点估计）；
- `probabilities`：`axis_aligned` 的 MC 概率，供 bench 计算 `belief_ece`；
- `covariances`：原协方差。

采样种子由 `stable_seed("3d_belief", config.seed, scene_id)` 生成，可复现且逐场景独立。

### 2.3 观测设置

| 项 | 值 |
|---|---|
| 数据 | CLEVR CoGenT：train `traina` 50 scenes / test `vala` 100 scenes |
| 相机 | 环绕圆环，半径 10，高度 6，focal 400，方位角均匀 |
| 噪声 | `sigma_px=2.0`，`sigma_depth=0.01`（相对深度） |
| 位姿噪声 | 0（本轮不测） |
| 离群 | 0（本轮不测） |
| 关联 | 已知 object identity |
| seeds | 0 / 1 / 2 |

---

## 3. 主表

`3d_belief` 在 1 / 3 / 5 视角；`oracle`（真值位姿）只作上限参照。均值 ± 标准差（跨 3 个 seed）。

| 视角 | near-aligned | overall | pose median | pose p90 | same_level | contact | collision | support |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5865 ± 0.0235 | 0.9554 ± 0.0038 | 0.0890 ± 0.0028 | 0.1482 ± 0.0035 | 0.8968 ± 0.0156 | 1.0000 | 1.0000 | 0.9803 ± 0.0030 |
| 3 | 0.6557 ± 0.0162 | 0.9721 ± 0.0042 | 0.0516 ± 0.0005 | 0.0764 ± 0.0020 | 0.9161 ± 0.0079 | 1.0000 | 1.0000 | 0.9718 ± 0.0067 |
| 5 | **0.7612 ± 0.0190** | **0.9825 ± 0.0023** | **0.0403 ± 0.0001** | **0.0606 ± 0.0003** | 0.9334 ± 0.0051 | 1.0000 | 1.0000 | 0.9895 ± 0.0046 |
| oracle (5) | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

- 3 视角 near 0.6557 与 D1 的 `3d_ls` 一致（同一位置估计），说明 belief 层没有改变点估计，只是附上概率。S1 的 `3d_3view` 使用 `optimizer_v2` 且相机子集口径不同（16 视角环中取前 3 个），其 near 0.686 不能与本表直接混比。
- `contact` / `collision` 的 1.000 是退化结果：CLEVR 物体都放在桌面上、互不接触，标签全为 0，估计也全为 0（见 §8）。
- `same_level` / `support` 的标签在 CLEVR 上恒为真，这里的数值是“估计是否仍落在阈值内”，不是平衡分类。

守卫对照（计划 §7.2，5 视角）：

| 守卫 | bar | 本轮 | 判定 |
|---|---|---:|---|
| overall relation accuracy | ≥0.95 | 0.9825 | ✓ |
| `same_level` accuracy | ≥0.95 | 0.9334 | ✗（1/3 视角更低：0.8968 / 0.9161） |
| `contact` / `collision` accuracy | ≥0.95 | 1.0000 / 1.0000 | ✓（但数据集退化，见 §8） |
| 位置误差 median | ≤1 cm | 4.03 cm | ✗ |
| 位置误差 p90 | ≤3 cm | 6.06 cm | ✗ |
| belief ECE | ≤0.05 | 0.0274 | ✓ |
| 单场景更新延迟（CPU） | ≤50 ms | median 11.9 ms / p90 18.2 ms | ✓ |

位置误差与 `same_level` 未达 bar，与 S1/S2 的结论一致：在相机半径 10 m、1% 深度噪声下，逐物体最小二乘的位置精度上限就是厘米级，`eps_z=0.05` 与 z 误差同量级；S3 的 belief 层不改变位置估计，也没有为了碰 guard 去调估计器或改 eps。**未达项照实记录，不作为 H3 的判定依据。** 延迟为 100 个 test scene 的单场景 median/p90（含 `solve_scene` + 200 次采样 + 关系查询，测得 5 视角 11.9/18.2 ms）。

---

## 4. ECE 与校准

### 4.1 主校准表

| 视角 | bench `belief_ece`（主口径） | pooled ECE | near 子集 ECE | Brier | hard-belief bench ECE | hard-belief pooled ECE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0486 ± 0.0007 | 0.0182 | **0.2262** | 0.0370 | 0.0446 ± 0.0038 | 0.0504 |
| 3 | 0.0369 ± 0.0027 | 0.0056 | **0.1054** | 0.0198 | 0.0279 ± 0.0042 | 0.0287 |
| 5 | **0.0274 ± 0.0005** | **0.0042** | **0.0473** | 0.0148 | 0.0175 ± 0.0023 | 0.0205 |

- near 子集 = 真实 `|Δx|` 或 `|Δy|` 落在 `[0.5, 1.5]·eps_xy` 的 pair；3 个 seed 共 447 对，正例率 0.436。
- `hard-belief` 是对照：把点估计的 `axis_aligned` 直接当 0/1 概率。它不是注册 arm，只用于判断 MC 概率是否真的比硬判好。
- pooled 与 Brier 上 MC 概率明显优于 hard（5 视角 Brier 0.0148 vs 0.0205）；bench 的逐场景 ECE 上 MC 反而略高，原因见 §4.3。

### 4.2 5 视角可靠性分箱（全部 pair）

| 概率 bin | n | 平均置信 | 实际准确率 |
|---|---:|---:|---:|
| [0.0, 0.1) | 5731 | 0.002 | 0.002 |
| [0.1, 0.2) | 90 | 0.151 | 0.078 |
| [0.2, 0.3) | 73 | 0.248 | 0.192 |
| [0.3, 0.4) | 45 | 0.346 | 0.311 |
| [0.4, 0.5) | 51 | 0.455 | 0.412 |
| [0.5, 0.6) | 51 | 0.557 | 0.490 |
| [0.6, 0.7) | 39 | 0.655 | 0.615 |
| [0.7, 0.8) | 67 | 0.761 | 0.746 |
| [0.8, 0.9) | 75 | 0.849 | 0.907 |
| [0.9, 1.0] | 162 | 0.961 | 0.975 |

高置信区（>0.8）略欠自信，0.6–0.8 区间略过自信；整体误差远小于 0.05。但 90% 的 pair 落在 [0, 0.1) 的“容易负对”，主 ECE 被这一箱主导。

### 4.3 为什么 bench ECE 与 pooled ECE 不一致

bench 的 `belief_ece` 是**逐场景**计算再平均。CLEVR 单个场景通常只有 3–10 个物体（约 3–45 个 pair），正例率仅 6.1%；许多场景一个正例都没有。此时：

- MC 概率在负对上给出接近 0 但非 0 的值（如 0.02），`binary_ece` 记为 0.02；
- 硬判概率恰好是 0，`binary_ece` 记为 0。

于是逐场景 ECE 系统性奖励“在容易负对上过度自信”，惩罚诚实的非零概率。pooled ECE 与 Brier 没有这个偏差，显示 MC 概率更好。这也是 hard-belief 的逐场景 ECE（0.0175）低于 MC（0.0274）的原因。

**H3 判定仍按预注册的 bench 口径**，不因 pooled 更低而改判；但解读时必须同时看 near 子集：near 子集上 MC 校准明显更差（5 视角 0.047、3 视角 0.105、1 视角 0.226）。

### 4.4 退化关系的“校准”

| 视角 | same_level ECE | support ECE | contact ECE | collision ECE | same/support 正例率 | contact/collision 正例率 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3409 | 0.1867 | 0.0009 | 0.0007 | 1.0 / 1.0 | 0.0 / 0.0 |
| 3 | 0.3572 | 0.1916 | 0.0001 | 0.0000 | 1.0 / 1.0 | 0.0 / 0.0 |
| 5 | 0.3011 | 0.1459 | 0.0000 | 0.0000 | 1.0 / 1.0 | 0.0 / 0.0 |

这些数字**不能**当作关系层校准的证据：

- `same_level`：所有物体 bottom_z 都是桌面高度，标签恒真；逐物体独立协方差把 `z_a - z_b` 的差分不确定性高估了（两个物体其实共享同一桌面/同一标定），于是概率只有 ~0.6–0.7，ECE 必然大。这暴露的是“独立高斯后验”的模型误设，不是关系定义错。
- `support`：同理，标签恒真，概率偏低。
- `contact` / `collision`：标签恒假、概率接近 0，ECE≈0 只是因为负例平凡，没有正例可供检验。

---

## 5. Risk-coverage 与 abstention

按置信度 `max(p, 1-p)` 从高到低保留 pair。6384 个 pair，3 个 seed 合并。

| 视角 | 覆盖率 10% 风险 | 25% | 50% | 75% | 90% | 100% |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0031 | 0.0031 | 0.0044 | 0.0044 | 0.0157 | 0.0525 |
| 3 | 0.0000 | 0.0000 | 0.0003 | 0.0002 | 0.0028 | 0.0288 |
| 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0012 | 0.0210 |

给定风险预算时的最大覆盖率（`abstention_summary`）：

| 视角 | 风险预算 1%：覆盖率 / 阈值 / 实际风险 | 2%：覆盖率 / 阈值 / 实际风险 | 5%：覆盖率 / 阈值 / 实际风险 |
|---:|---|---|---|
| 1 | 86.8% / 0.940 / 0.0099 | 91.7% / 0.815 / 0.0200 | 99.5% / 0.525 / 0.0499 |
| 3 | 94.9% / 0.735 / 0.0099 | 98.0% / 0.600 / 0.0200 | 100% / 0.500 / 0.0288 |
| 5 | 97.4% / 0.680 / 0.0100 | 99.7% / 0.520 / 0.0199 | 100% / 0.500 / 0.0210 |

- 选择性预测有效：3 视角下只要弃权约 5%（1% 风险预算），风险从 2.88% 降到 0.99%；5 视角弃权 2.6% 即可到 1%。
- 1 视角曲线两极分化：最自信的 10% 几乎全对（风险 0.3%），但整体风险 5.25%；弃权 13.2% 可把风险压到 0.99%，说明少数 pair 的概率接近 0.5 且实际错误，靠 abstention 可以挡住。
- 5% 预算下 3/5 视角保留全部 pair，因为整体风险已低于 5%；这不表示不需要 abstention，只表示本轮 clean 设置下误差率本身较低。

---

## 6. 协方差一致性（位置层）

对每个物体计算 `(p_est - p_true)ᵀ Σ⁻¹ (p_est - p_true)`，与 χ²₃ 比较（均值 3、99.9% 界约 16.27）。1995 个物体 = 100 scenes × 3 seeds。

| 视角 | Mahalanobis 均值 | p50 | p90 | p99 | max | >16.27 比例 | >1000 比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.09×10⁵ | 4.72 | 199.5 | 2.44×10⁴ | 5.92×10⁸ | 29.6% | 5.1% |
| 3 | 2.978 | 1.636 | 6.77 | 22.17 | 65.7 | 2.0% | 0.0% |
| 5 | 2.230 | 1.475 | 4.89 | 12.39 | 25.0 | 0.4% | 0.0% |

- **3 视角均值 2.98 ≈ 3**，整体尺度正确；但 p90/p99 都超界，尾部比高斯厚，说明线性化协方差低估了少数困难物体的误差。
- **5 视角均值 2.23 < 3**，略保守（过覆盖），与 near ECE 0.047 一致。
- **1 视角不可用**：30% 的物体落在 99.9% 界外，5% 的 Mahalanobis >1000；少数协方差接近奇异，误差远大于预测。`sample_scene_positions` 的 PSD 裁剪阈值（1e-10）没有触发，因为矩阵数学上是 PSD，只是数值极小；这类“合法但过度自信”的协方差无法靠 PSD 检查发现。
- 本轮 1995 个物体中 `degraded_objects=0`、`clipped_objects=0`；fallback 路径已实现但未使用。

这些尾部误差直接解释 §4 的 near 子集 ECE：near pair 的概率对位置差的尾部敏感，高斯后验在尾部给的概率过低，于是 near ECE 远大于全量 ECE。

---

## 7. H3 判定

| 判定口径 | 5 视角 | 3 视角 | 1 视角 | 结论 |
|---|---:|---:|---:|---|
| **主口径：bench `belief_ece`（axis_aligned，全部 pair）** | **0.0274** | 0.0369 | 0.0486 | **≤0.05，H3 成立** |
| pooled ECE | 0.0042 | 0.0056 | 0.0182 | 成立 |
| near 子集 ECE | 0.0473 | **0.1054** | **0.2262** | 3/1 视角不成立 |
| Brier | 0.0148 | 0.0198 | 0.0370 | 优于 hard 0.0205 / 0.0287 / 0.0504 |

**结论：H3 在主口径成立。** 5 视角 `belief_ece = 0.0274 ± 0.0005`，低于 0.05，也远低于 0.10 的判假线。3 视角 0.0369、1 视角 0.0486 也都在 bar 内（1 视角余量很小）。

**限定条件（必须同时写进结论）：**

1. 全量 ECE 被 90% 的容易负对主导；near-boundary 子集在 3 视角是 0.105，超过 0.05，在 5 视角是 0.047 刚好压线。因此不能宣称“在 near 关系上已校准”，只能宣称“全量 belief 在主口径下校准”。
2. 1 视角虽然 ECE 0.0486 过线，但协方差尾部失控，near ECE 0.226，risk-coverage 两极分化；不应把 1 视角 belief 用于下游决策。
3. 逐场景 ECE 对“全负场景 + 诚实小概率”有偏；pooled 与 Brier 更好，但判定没有改用 pooled 来放宽标准。

没有调温度，没有改 bar，没有用 test 做任何后处理。

---

## 8. 边界

- **协方差是线性化近似，不是精确后验。** `Σ = σ²(JᵀJ)⁻¹` 依赖高斯噪声、局部线性、`soft_l1` 的 Jacobian 未被稳健重加权；物体之间按独立处理。`same_level` / `support` 的高 ECE 正是独立假设的代价：两个物体共享桌面和相机标定，差分误差小于独立协方差预测。
- **1 视角协方差重尾。** 29.6% 物体超出 99.9% χ²₃ 界，5.1% 的 Mahalanobis >1000；PSD 检查发现不了这类过度自信。若要在 1 视角使用，需要 bootstrap / Student-t / 重尾似然，或直接拒绝低视角 belief。
- **CLEVR 关系退化。** 所有物体在桌面上：`same_level` 与 `support` 标签恒真，`contact` / `collision` 恒假。这些查询在本数据集上只能验证“估计是否落在阈值内”，不能验证平衡校准；要评估它们需要带堆叠/接触的场景（如 CLEVR-Hans 或自建堆叠场景）。
- **观测前端是受控模拟。** 已知关联、无位姿噪声、无离群、无遮挡；真实检测器/标定误差会进一步破坏校准，本轮不覆盖。
- **只测固定环形视角。** S3 计划中的主动视角选择（按信息增益选第 2/3 视角）未实现；本报告的视角曲线只说明固定视角数的影响。
- **bench `belief_ece` 是逐场景均值**，小样本 bin 偏差明显；报告同时给出 pooled ECE 和 Brier，避免单一口径误读。
- **oracle 的 `belief_ece = 0.941` 不是反例**：oracle 对所有 pair 给概率 1.0，而 CLEVR 的 axis_aligned 基础正例率约 6%，所以 ECE 必然接近 0.94。它只作位置上限，不是校准 baseline。
- **未做 cal 划分的温度/isotonic 校准。** 本轮没有 cal split；按停止规则，也不允许在 test 上调温度。

---

## 9. 复现

```bash
python -u -m pss.eval.scripts.run_scene_bench \
  --train-scenes 50 --test-scenes 100 --n-views 5 --seeds 0 1 2 \
  --arms 3d_belief oracle --out eval/results/scene_bench/s3

for n in 1 3 5; do
  python -u -m pss.eval.scripts.run_scene_bench \
    --train-scenes 50 --test-scenes 100 --n-views $n --seeds 0 1 2 \
    --arms 3d_belief --out eval/results/scene_bench/s3_v$n
done
```

产物：

| 路径 | 内容 |
|---|---|
| `eval/results/scene_bench/s3/scene_bench.json` | 主运行：5 视角，`3d_belief` + `oracle`，3 seeds |
| `eval/results/scene_bench/s3/SCENE_BENCH_REPORT.md` | bench 自动生成的表 |
| `eval/results/scene_bench/s3_v1/`、`s3_v3/`、`s3_v5/` | 视角曲线 |

新增代码：

| 文件 | 内容 |
|---|---|
| `pss/scene/uncertainty.py` | 协方差采样、关系概率、risk-coverage、abstention、可靠协方差处理 |
| `pss/scene/arms_uncertainty.py` | 注册 `3d_belief`；`three_d_belief_output` 供分析复用 |
| `docs/experiments/PSS-Scene-S3-Uncertainty.md` | 本报告 |

未修改 `optimizer.py` / `bench_api.py` / `model.py` / `observation.py` / `metrics.py` / `run_scene_bench.py` / `arms_baseline.py` / runtime。

---

## 10. 一句话结论

**三维场景后验可以给出可用的关系 belief：5 视角全量 ECE 0.027，H3 主口径成立；但这份校准由容易负对支撑，near 子集在 3 视角下 ECE 0.105，1 视角协方差的重尾更使其不可用于决策。** 下一步若要宣称 near 关系校准，需要修协方差（重尾/相关误差/共享桌面），而不是调温度或调 bar。
