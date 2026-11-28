# PSS-Boundary-Canonical：empirical-Bayes 边界在 U6 canonical 3D 噪声上的复测

> 日期：2026-11-28
> 上游：`docs/experiments/PSS-Boundary-Report.md` §5 残余 3、§7 下一步 1
> 脚本：`pss/eval/scripts/run_boundary_canonical.py`
> 结果：`eval/results/boundary_canonical/boundary_canonical.json`
> 状态：**预注册 `u_boundary_soft` 在 3°+1% 档失败（0.693 < 0.80）。canonical 不是像素特例，而是揭示了像素版 0.948 是“干净输入上改查表 σ”的度量假象。**

---

## 0. 一句话

把 grid empirical-Bayes 边界接到 U6 的 canonical 3D→桌面 XY 投影后，**3°+1% 档 near-worst 0.693±0.018，5°+2% 档 0.461±0.028**，均未达到 0.80；与 U6 hard baseline 基本持平（3° 档 −0.008，5° 档 −0.054）。用全量训练先验的精确 kernel-Bayes 在同一批噪声观测上也只有 **0.708 / 0.456**，说明 0.80 不是 grid 矩近似的误差，而是“经验边缘先验 + 高斯似然 + 逐轴判别”这套规则在 near-aligned 稀有格上的上限。进一步复跑像素代理发现：PSS-Boundary-Report 的 0.948 是在**干净测试坐标**上只改内部查表 σ 得到的；喂真实加噪坐标后，同一像素 split 的 grid Bayes 只有 **0.658**，kernel 0.646。canonical 的 0.693 与“正确度量下的像素 0.658”同档，因此 **canonical 并不是像素特例——像素版本身就没有在噪声输入上成立**。

---

## 1. 设置

| 项 | 值 |
|---|---|
| 数据 | CLEVR CoGenT `traina_8000.pt` / `vala_3000.pt` |
| pairs | `run_upper_bound_u6._pair_indices`，train 30000 / test 10000 ordered pairs |
| 配对与采样 | 与 U3/U6 逐字节一致（seed train 0 / test 1） |
| 3D 点 | `_xyz`（CLEVR `3d_coords` 世界坐标） |
| 相机 | `_fit_camera` DLT，4000 点，rms **4.84 px**，det(R) **+1.000**，C ≈ `[6.25, −5.33, 4.41]` |
| 归一化 | train clean pair 端点的 min/max：lo `[−3.000, −3.000]`，hi `[3.000, 3.000]`（span≈6） |
| 先验 | `GridBayesBoundary(eps=0.02, bins=257)`，fit 在 train clean canonical `(dx,dy)` |
| σ 校准 | **train split** 上按 seed/level 从 pair-difference 噪声估计，测试标签/坐标不参与 |
| 噪声档 | clean、`pose3_depth1`（3°+1%）、`pose5_depth2`（5°+2%）；另报 `pose1_depth0.5` |
| 噪声种子 | 测试噪声 `seed*100000 + j + 17`，与 U6 combined canonical **完全相同**；calibration `seed*100000 + j + 7` |
| seeds | 0, 1, 2 |
| 标签/格 | 来自 **clean canonical**；只有边界输入加噪（与 U6 一致） |
| 指标 | overall、h/v/d exact-4、near-worst = min(h,v)、absence error = max(1−h, 1−v) |
| 预注册 | `u_boundary_soft`（`worst_cell_acc_min=0.80`），读 `pose3_depth1` 的 near-worst 均值 |

测试噪声与 U6 使用同一 RNG 流、同一相机、同一归一化，因此本报告的 noisy 几何就是 U6 打分用的那批几何，对照是逐样本可比的。

---

## 2. 诊断：尺度、先验与噪声

### 2.1 canonical 投影尺度

世界 XY 落在约 `[−3, 3]²`，归一化后 `[0,1]`，pair 差 `(dx,dy)` 落在 `[−1,1]`，正好覆盖 `GridBayesBoundary` 的固定范围。先验直方图拟合没有越界或退化问题。

### 2.2 train 校准的 pair-difference 噪声（归一化单位）

| level | σ_x | σ_y | pooled σ | bias_x | bias_y |
|---|---:|---:|---:|---:|---:|
| pose1_depth0.5 | 0.0077 | 0.0067 | 0.0073 | ~1e−4 | ~−1e−4 |
| pose3_depth1 | 0.0155 | 0.0136 | 0.0146 | ~1e−4 | ~1e−4 |
| pose5_depth2 | 0.0310 | 0.0274 | 0.0293 | ~3e−4 | ~3e−4 |

- pair 差噪声的零偏 < 3e−4，高斯零均值似然假设在本轴上成立；不是“投影有系统偏移”。
- 3°+1% 的 pooled σ=0.0146 与 eps=0.02 的比值 0.73，和像素契约（per-object σ=0.01 → pair σ≈0.0141，σ/eps≈0.71）几乎相同。**两套几何的固有难度相同**，差异不来自尺度。

### 2.3 先验结构

train canonical pair 类型：horizontal 956 / vertical 981 / diagonal 28063；test：307 / 331 / 9362。near-aligned 两格合计约 6.5%，diagonal 占 93.5%。这个稀有性对 worst-cell 指标很关键：以 overall 为目标的 Bayes 规则会把边界推向多数类 diagonal。

---

## 3. 主表：grid empirical-Bayes 边界（3 seeds，mean ± std）

| level | overall | horizontal | vertical | diagonal | near-worst | absence error |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.998 | 0.980 | 0.967 | 1.000 | **0.967** | 0.033 |
| pose1_depth0.5（1°+0.5%） | 0.983 ± 0.001 | 0.864 | 0.868 | 0.991 | 0.859 ± 0.014 | 0.141 |
| pose3_depth1（3°+1%） | 0.964 ± 0.001 | 0.708 | 0.704 | 0.982 | **0.693 ± 0.018** | 0.307 |
| pose5_depth2（5°+2%） | 0.936 ± 0.002 | 0.467 | 0.473 | 0.967 | **0.461 ± 0.028** | 0.539 |

clean 是确定性的，只评一次；其余为 3 seeds。

### 3.1 同一批噪声观测上的精确 kernel-Bayes（全量 30000 先验）

| level | grid near-worst | kernel near-worst | grid overall | kernel overall |
|---|---:|---:|---:|---:|
| pose1_depth0.5 | 0.859 | 0.862 | 0.983 | 0.983 |
| pose3_depth1 | 0.693 | **0.708** | 0.964 | 0.964 |
| pose5_depth2 | 0.461 | 0.456 | 0.936 | 0.936 |

- kernel 列与 `run_boundary_ceiling` 的判别规则同源（经验先验样本的核加权投票），但这里是**真实加噪观测**上的逐 pair 预测，并算到了 near-worst。
- 3°+1% 档 kernel 只比 grid 高 1.5 pt，**仍远低于 0.80**；5°+2% 档二者都在 0.46。near-worst 的失败不是 grid 矩近似造成的。
- kernel 的 overall（0.964 / 0.936）与 `ceiling.json` 的 0.971（pair-diff σ=0.01，overall）同量级；**ceiling 报的是 overall，不是 near-worst**。用 overall 上限去支持 near-worst 目标 0.80 是跨指标比较。

### 3.2 为什么 hard MLP 有时还更高

U6 的 hard MLP 在 5°+2% 档 worst 0.516，高于 kernel 的 0.456。这不是矛盾：kernel/grid 用边缘先验做逐轴 Bayes，优化的是 overall；hard MLP 学习的是 clean 标签的硬 eps 阈值，在稀有 near-aligned 格上反而更“保守”。要提升 worst-cell，需要**代价敏感/按格校准**的决策规则，而不是提高 overall Bayes 的精度。

---

## 4. 与 U6 hard baseline 并排

U6 hard 值取 `eval/results/upper_bound/u6/u6.json` 的 mean（U6 `worst_cell_acc` 是 min(h,v,d)；Bayes 列为 min(h,v)，因 diagonal 恒高，二者可直接对比）。

| level | U6 hard worst-cell | Bayes grid near-worst | Δ (Bayes − hard) |
|---|---:|---:|---:|
| clean | 0.962 | 0.967 | **+0.005** |
| pose1_depth0.5 | 0.868 | 0.859 | −0.008 |
| pose3_depth1（3°+1%） | 0.701 | 0.693 | **−0.008** |
| pose5_depth2（5°+2%） | 0.516 | 0.461 | **−0.054** |

clean 上 Bayes 略高（+0.5 pt），噪声档全部持平或更低；5°+2% 档低 5.4 pt。**empirical-Bayes 边界在 canonical 上没有带来 hard baseline 之外的收益。**

---

## 5. 关键对照：像素版 0.948 是“干净输入”读数

`run_boundary_bayes.py` 在测试端**没有给坐标加噪**：`dx_te/dy_te` 来自 clean 归一化坐标，只把 `sigma` 传给内部查表。而 `run_boundary_bench.py` 的 hard/soft 臂是在**真实加噪坐标**上评估。用同一像素 split 复跑：

| 输入 | 判别规则 | near-worst | overall |
|---|---|---:|---:|
| clean，查表 σ=0.01 | grid | **0.948** | 0.997 |
| 真实加噪（per-object σ=0.01），查表 σ=0.01 | grid | **0.658 ± 0.022** | 0.968 |
| 真实加噪（per-object σ=0.01），σ=0.01 | kernel（全量先验） | **0.646 ± 0.019** | 0.967 |

`boundary_bench.json` 的契约档对照：`hard_clean` 0.685、`hard_noise_aug` 0.645、`soft_estimated` 0.682。

- 0.948 只出现在“输入干净、模型以为有噪声”的行；一旦输入真的加噪，grid/kernel 都落到 0.65 左右，与 `soft_estimated` 0.682 / `hard_clean` 0.685 同档。
- 因此像素版的 0.948 ≥ 0.80 的 GO，**不是噪声鲁棒性证据**。本报告不改动 `PSS-Boundary-Report.md`，只在此记录该度量差异。
- canonical 的 0.693 与“正确度量下的像素 0.658”一致；**canonical 不是异常特例，而是把像素版的度量问题暴露出来的同一现象**。

---

## 6. 预注册 `u_boundary_soft`

| 项 | 值 |
|---|---|
| 读数档 | `pose3_depth1`（3°+1%） |
| 指标 | near-worst = min(horizontal, vertical) |
| 实测 | **0.6927 ± 0.0177**（seeds 0/1/2：0.707 / 0.668 / 0.704） |
| bar | `worst_cell_acc_min = 0.80` |
| 判定 | **失败（NO-GO）** |

`check_readout_reasoning_preregistration` 输出：

```
u_boundary_soft.worst_cell_acc: expected >= 0.8, got 0.6927
```

bar 未调整；失败照实保留。即使改用全量先验的 kernel-Bayes，3°+1% 也只有 0.708，仍低于 0.80。

---

## 7. 判读

1. **canonical 投影本身没有额外损伤信息。** 相机 DLT rms 4.84 px、det(R)=1，pair-diff 噪声零偏、σ 与像素契约同量级；失败不能用“3D 投影尺度/归一化”解释。
2. **0.80 near-worst 在这套经验先验 + 高斯似然下不可达。** grid 0.693、全量 kernel 0.708，均低于 0.80；5°+2% 只有 0.46。目标与 `u_geom_noise` 的 0.85 一样，属于当前边界层的残余，不是可调的 bar。
3. **PSS-Boundary-Report 的 0.948 来自干净输入。** 真实加噪像素 grid 0.658、kernel 0.646；canonical 0.693 与之一致。**像素代理的 GO 需要撤回或重测**，而不是把 canonical 当成特例。
4. **ceiling 的 0.971 是 overall exact-4，不是 near-worst。** 同一规则在真实噪声下的 overall 仍有 0.964（canonical 3° 档），但 near-worst 只有 0.708。overall 被 94% 的 diagonal 托住，不能用来证明 dead-zone 边界可达。
5. **下一步不是更大的模型，而是目标函数与后验形式。** 要提升 worst-cell，需要 (a) 代价敏感/按格校准的决策规则，或 (b) 多帧/深度融合把 pair σ 压到 0.005 以下（对应 near-worst 上限更高），或 (c) 用更细的后验（KDE/mixture/条件先验）并直接优化 min(h,v)。

---

## 8. 边界

- **不是真实 RGB-D 标定。** CLEVR 是渲染几何；位姿/深度误差是各向同性高斯代理，不是从真实传感器误差分布拟合的模型。
- **不是 sim2real。** 无真实相机内参、无真实深度噪声模型；canonical 只是把 CLEVR 的 `3d_coords` 经 DLT 相机投到世界 XY。
- 噪声按场景共享位姿、按物体独立深度；真实标定误差可能相关。
- 测试标签与 pair 类型来自 clean 几何，本报告度量的是**决策鲁棒性**，不是噪声标签鲁棒性。
- `GridBayesBoundary` 使用逐轴后验均值/方差的高斯矩近似；kernel 诊断用全量经验先验、但仍假设高斯似然。
- 结果只覆盖 `traina/vala` 与 CLEVR 的关系对分布；部署换场景需重估先验。

---

## 9. 产物与复现

```
pss/eval/scripts/run_boundary_canonical.py          # 新建：canonical grid-Bayes + 像素/ kernel 对照
eval/results/boundary_canonical/boundary_canonical.json
eval/results/boundary_canonical/BOUNDARY_CANONICAL_RESULTS.md
```

复现（CPU 即可，约 3 分钟，主要耗在全量先验的 kernel 诊断）：

```bash
python -u -m pss.eval.scripts.run_boundary_canonical \
  --out eval/results/boundary_canonical
```

关键参数：`--train-pairs 30000 --test-pairs 10000 --bins 257 --seeds 0 1 2 --seed 0 --camera-points 4000`；`--kernel-prior-samples 0` 表示 kernel 诊断使用全量训练先验。未修改 `bayes_boundary.py`、`soft_deadzone.py`、U6 脚本、预注册或 runtime 代码。

---

## 10. 一句话

**empirical-Bayes 边界在 canonical 3D 噪声下 3°+1% near-worst 0.693、5°+2% 0.461，预注册失败；精确 kernel-Bayes 也只有 0.708/0.456。canonical 与“正确加噪的像素”同档（0.658），真正的问题是像素版 0.948 量在干净输入上、ceiling 0.971 量的是 overall。**
