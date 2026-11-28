# PSS 三维场景建模 S2：支撑/不穿透约束与消融

- 计划：`docs/plans/PSS-三维场景建模实验计划.md` §4.2、§6、§7.3（S2）、§8（S2）、§9
- 代码：`pss/scene/constraints.py`（新增）、`pss/scene/arms_constraints.py`（新增）
- 数据：CLEVR-CoGenT v1.0，train `traina` 50 场景 / test `vala` 100 场景
- 配置：`n_views=5`、`σ_px=2.0`、`σ_depth=0.01`、`sigma_pose=0`、`outlier_frac=0`、`seeds=0,1,2`
- 结果：`eval/results/scene_bench/s2/`、`eval/results/scene_bench/s2_outliers/`
- 说明：臂名沿用计划 §6 的 `3d_3view_*` 命名；本轮按 D3 指令统一跑 **5 视角**，名字里的 `3view` 只是臂标识。

## 1. 实现

`constraints.py` 提供：

| 因子 | 残差 | 权重（σ） |
|---|---|---|
| 支撑 | `bottom_z - table_z`，`bottom_z = z - radius` | `sigma_support = 0.02` |
| 不穿透 | `max(0, r_i + r_j - ||p_i - p_j||)` | `sigma_penetration = 0.01` |
| 接触（可选，默认关） | `abs(||p_i - p_j|| - (r_i + r_j))` | `sigma_contact = 0.02` |

- `constraint_residual(positions, radii, config)` 的签名与 `optimizer.solve_scene` 的 `constraints` 钩子一致，便于后续接入；本轮不依赖该钩子。
- `joint_refine(...)`：先用 `solve_scene` 得到逐物体初值，再把**所有物体位置联合**放入一个 `least_squares`，残差由重投影（按 `σ_px` 归一化、乘观测权重）+ 深度（按 `σ_depth` 归一化）+ 约束组成，`loss="soft_l1"`、`f_scale=1.0`、`max_nfev=200`。输出逐物体位置和线性化协方差块。
- **未修改** `optimizer.py` / `bench_api.py` / `model.py` / `observation.py` / `metrics.py` / `run_scene_bench.py` / `arms_baseline.py`。

臂注册（`arms_constraints.py`）：

| arm | 估计器 |
|---|---|
| `3d_3view_noconstraint` | 仅 `solve_scene`（逐物体最小二乘，含其内置桌面先验） |
| `3d_3view_constraints` | `joint_refine`：联合重投影 + 深度 + 支撑 + 不穿透 |
| `3d_3view_nodepth` | `solve_scene(SolveOptions(use_depth=False))` |
| `3d_3view_outliers` | 与 `3d_3view_constraints` **完全相同**；离群条件由 `--outlier-frac 0.1` 施加 |

关键实现语境：D1 的 `solve_scene` 已默认 `use_table=True`（`(z-r)/0.05`），所以“无约束”臂并非真的没有支撑先验。S2 的新增信息主要是**不穿透**和**更紧的支撑权重**（0.05 → 0.02）。这直接决定了 H2 的预期上限。

## 2. 主表（clean，3 seeds × 100 scenes）

均值 ± 跨 seed 标准差；near 只在含 near-boundary 对的场景上统计（149 个 near 对，2128 个总对）。

| arm | near-aligned | overall | pose median (m) | pose p90 (m) | same_level | contact | collision | support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `3d_3view_noconstraint` | 0.761 ± 0.019 | 0.983 ± 0.002 | 0.0403 ± 0.0001 | 0.0606 ± 0.0003 | 0.933 ± 0.005 | 1.000 | 1.000 | 0.990 ± 0.005 |
| `3d_3view_constraints` | **0.765 ± 0.016** | **0.983 ± 0.002** | **0.0362 ± 0.0001** | **0.0570 ± 0.0005** | **1.000 ± 0.000** | 1.000 | 1.000 | **1.000 ± 0.000** |
| `3d_3view_nodepth` | 0.748 ± 0.018 | 0.981 ± 0.002 | 0.0433 ± 0.0011 | 0.0646 ± 0.0007 | 0.926 ± 0.008 | 1.000 | 1.000 | 0.986 ± 0.004 |
| `oracle` | 1.000 | 1.000 | 0.0000 | 0.0000 | 1.000 | 1.000 | 1.000 | 1.000 |

逐 seed（near-aligned / overall / pose median）：

| seed | noconstraint | constraints | nodepth |
|---|---|---|---|
| 0 | 0.762 / 0.981 / 0.040 | 0.766 / 0.982 / 0.036 | 0.737 / 0.980 / 0.044 |
| 1 | 0.738 / 0.980 / 0.040 | 0.746 / 0.980 / 0.036 | 0.733 / 0.979 / 0.044 |
| 2 | 0.784 / 0.986 / 0.040 | 0.784 / 0.986 / 0.036 | 0.773 / 0.985 / 0.042 |

## 3. 消融

### 3.1 约束（H2 对照）

| 指标 | noconstraint | constraints | Δ |
|---|---:|---:|---:|
| near-aligned | 0.7612 | 0.7654 | **+0.0042** |
| overall | 0.9825 | 0.9826 | +0.0001 |
| pose median | 0.0403 m | 0.0362 m | −0.0041 m |
| pose p90 | 0.0606 m | 0.0570 m | −0.0036 m |
| same_level | 0.9334 | 0.9998 | **+0.0664** |
| support | 0.9895 | 1.0000 | +0.0105 |

- 约束没有伤害 clean 指标：overall +0.01 pt，near +0.42 pt，位置误差中位数改善 4.1 mm。
- 三维独有查询受益明显：`same_level` +6.6 pt（0.933 → 1.000），`support` 到 1.000。原因是联合优化让底面高度一致、支撑残差被显式清零。

### 3.2 去深度（H1 对照，S2 附带）

| 指标 | noconstraint | nodepth | Δ |
|---|---:|---:|---:|
| near-aligned | 0.7612 | 0.7477 | **−0.0135** |
| overall | 0.9825 | 0.9814 | −0.0011 |
| pose median | 0.0403 m | 0.0433 m | +0.0030 m |
| pose p90 | 0.0606 m | 0.0646 m | +0.0040 m |

去掉深度后 near 掉 1.35 pt、位置误差升 3 mm；5 视角的像素约束能部分补偿深度，但深度仍有可测增益。

### 3.3 离群（`--outlier-frac 0.1`）

同一 `3d_3view_constraints` 估计器，10% 观测带 20–40 px 像素偏移 + 20% 深度偏差。

| 指标 | clean | 10% outliers | Δ |
|---|---:|---:|---:|
| near-aligned | 0.7654 ± 0.0155 | **0.7135 ± 0.0417** | −0.0519 |
| overall | 0.9826 | 0.9789 | −0.0037 |
| pose median | 0.0362 m | 0.0384 m | +0.0022 m |
| pose p90 | 0.0570 m | 0.0616 m | +0.0046 m |
| same_level / support | 1.000 | 1.000 | 0 |

逐 seed near：0.681 / 0.687 / 0.772。near 在离群下掉 5.2 pt 且跨 seed 波动大（std 4.2 pt，seed 2 明显更好），说明 near-boundary 对的判别对离群观测敏感；整体和位置误差仍稳健（overall 仅 −0.37 pt，soft-L1 + 观测权重 0.2 起作用）。离群臂与 clean 臂代码相同，差异全部来自观测生成，符合预注册。

## 4. H2 判定

**预注册**：`3d_3view_constraints` 的 near-aligned ≥ `3d_3view_noconstraint` + 0.03；若去掉约束后不变则 H2 假。

- 实测 Δ near-aligned = **+0.0042**（0.7612 → 0.7654），远低于 +0.03 门槛；0.42 pt 也落在跨 seed 标准差（约 1.6–1.9 pt）之内。
- **结论：H2 不成立。** 支撑/不穿透约束在 near-boundary 主指标上没有带来预注册要求的增益。按计划 §9，不调 bar、不换指标。
- 解释（不是事后调参）：`solve_scene` 本身已有桌面先验，支撑约束并非新增信息；不穿透主要修正真正重叠的少数对，而 near-boundary 对是“接近对齐”而非“接近碰撞”，两者几何上不重合，因此约束对 near 的杠杆很小。
- 约束**不是负面结果**：它没有让 anything 变差，且显著改善 `same_level` / `support` / 位置误差。因此保留 `3d_3view_constraints` 作为几何一致性更好的估计器，但不宣称 H2。

## 5. 守卫项与停止规则

| 计划守卫 | bar | 本轮（constraints） | 判定 |
|---|---|---|---|
| 位置误差 median | ≤ 1 cm | 3.62 cm | **未达** |
| 位置误差 p90 | ≤ 3 cm | 5.70 cm | **未达** |
| overall | ≥ 0.95 | 0.983 | 达标 |
| same_level / contact / collision | ≥ 0.95 | 1.000 / 1.000 / 1.000 | 达标 |
| clean 不低于 1D 基线 −2 pt | ≥ 基线 −2 pt | 本轮未含 1D 臂；只与 D1 参考值对照（D1 为 3 视角，本轮 5 视角，不可直接比较） | 未直接验证 |
| 约束伤害 clean/overall > 2 pt | — | +0.0001 overall，无伤害 | 未触发 |
| `3d_3view` 与 oracle 差距 > 0.10 | 诊断 | near 差 0.235 | 已知估计器上限，见下 |

- 约束**未触发**“伤害 clean/overall >2 pt”的停止规则，无需回滚。
- 位置误差两项未达计划 bar，属于 D1 已存在的估计器精度问题（相机半径 10 m、深度 1% → 单视角约 10 cm 深度噪声，5 视角平均后约 3–4 cm），不是 S2 引入的。S2 的约束把 median 从 4.03 cm 改善到 3.62 cm，方向正确但不足以跨过 1 cm bar。
- near 与 oracle 的 23.5 pt 差距同样来自深度/像素噪声，而非约束缺失；本轮不新增臂、不改谓词。

## 6. 结论

1. S2 的支撑 + 不穿透联合优化**可用且无害**：overall +0.01 pt、near +0.42 pt、pose median −4.1 mm、`same_level` +6.6 pt、`support` +1.1 pt。
2. **H2 判定为假**：near-aligned 仅 +0.0042，未达到预注册的 +0.03。照实记录，不调 bar。
3. 约束没有伤害 clean 或 overall（均 <0.1 pt 变化），因此不触发回滚。
4. 去深度：near −1.35 pt、pose median +3 mm，深度信息有可测但有限的贡献。
5. 10% 离群：overall 仅 −0.37 pt、pose 误差基本不变，但 near −5.2 pt 且跨 seed 波动大；near-boundary 判别是最脆弱的一环。
6. 位置误差未达计划守卫（median 3.62 cm vs 1 cm），属于观测噪声导致的估计器上限，留给后续阶段（更多视角/更好相机布局/去噪）而非 S2 约束解决。

## 7. 复现命令

```bash
python -u -m pss.eval.scripts.run_scene_bench \
  --train-scenes 50 --test-scenes 100 --n-views 5 --seeds 0 1 2 \
  --arms 3d_3view_noconstraint 3d_3view_constraints 3d_3view_nodepth oracle \
  --out eval/results/scene_bench/s2

python -u -m pss.eval.scripts.run_scene_bench \
  --train-scenes 50 --test-scenes 100 --n-views 5 --outlier-frac 0.1 --seeds 0 1 2 \
  --arms 3d_3view_constraints --out eval/results/scene_bench/s2_outliers
```
